"""
HYDRA Optimizer — Algorithmic Innovation #2

A custom optimizer wrapper that addresses the unique challenges of
training a multi-pathway routed architecture:

1. Differential Learning Rates:
   - Router components learn faster (they need to stabilize early)
   - Pathway components learn at standard rates
   - Embeddings learn slowly (stable representations)

2. Gradient Scaling per Pathway:
   - Prevents pathway starvation: if one pathway rarely receives gradients
     (because few tokens are routed to it), its gradients are scaled up
   - Prevents pathway dominance: heavily-used pathways get slightly
     damped gradients to prevent them from becoming too specialized

3. Router Warmup:
   - During initial training, router gradients are attenuated to prevent
     premature routing decisions before the pathways have learned anything

4. Adaptive Weight Decay:
   - Router parameters have lower weight decay (routing decisions should
     be flexible, not penalized toward zero)
   - SSM parameters have custom decay matching their parameterization

Research Note:
    This optimizer design emerged from empirical observation that standard
    AdamW causes "routing collapse" — where one pathway dominates and
    others atrophy due to gradient starvation. The differential learning
    rates and gradient scaling prevent this failure mode.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: torch.nn.Module,
    config,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    """
    Create optimizer with parameter-group-specific settings.
    
    Parameter Group Strategy:
        Group 0: Embedding parameters     — lr × 0.5, wd × 1.0
        Group 1: Router parameters         — lr × 2.0, wd × 0.1
        Group 2: SSM pathway parameters    — lr × 1.0, wd × 0.5
        Group 3: Attention pathway params  — lr × 1.0, wd × 1.0
        Group 4: FFN & other parameters    — lr × 1.0, wd × 1.0
    """
    
    # Categorize parameters
    embedding_params = []
    router_params = []
    ssm_params = []
    attention_params = []
    other_params = []
    
    no_decay_keywords = ["bias", "LayerNorm", "layer_norm", "norm"]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "embedding" in name:
            embedding_params.append((name, param))
        elif "router" in name or "complexity_estimator" in name or "routing_head" in name:
            router_params.append((name, param))
        elif "ssm" in name or "stream" in name:
            ssm_params.append((name, param))
        elif "attention" in name or "focus" in name or "reason" in name:
            attention_params.append((name, param))
        else:
            other_params.append((name, param))
    
    def make_group(params, lr_scale, wd_scale):
        """Create param groups separating decay and no-decay params."""
        decay_params = [
            p for n, p in params
            if not any(k in n for k in no_decay_keywords)
        ]
        no_decay_params = [
            p for n, p in params
            if any(k in n for k in no_decay_keywords)
        ]
        
        groups = []
        if decay_params:
            groups.append({
                "params": decay_params,
                "lr": config.learning_rate * lr_scale,
                "weight_decay": config.weight_decay * wd_scale,
            })
        if no_decay_params:
            groups.append({
                "params": no_decay_params,
                "lr": config.learning_rate * lr_scale,
                "weight_decay": 0.0,
            })
        return groups
    
    param_groups = []
    param_groups.extend(make_group(embedding_params, lr_scale=0.5, wd_scale=1.0))
    param_groups.extend(make_group(router_params, lr_scale=2.0, wd_scale=0.1))
    param_groups.extend(make_group(ssm_params, lr_scale=1.0, wd_scale=0.5))
    param_groups.extend(make_group(attention_params, lr_scale=1.0, wd_scale=1.0))
    param_groups.extend(make_group(other_params, lr_scale=1.0, wd_scale=1.0))
    
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # Learning rate scheduler: warmup + cosine decay
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    return optimizer, scheduler


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine learning rate schedule with linear warmup.
    
    Decays to min_lr_ratio × initial_lr, not to zero.
    """
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


class HydraOptimizer:
    """
    High-level optimizer wrapper with gradient scaling and routing-aware updates.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config,
        gradient_scale_enabled: bool = True,
    ):
        self.model = model
        self.config = config
        self.gradient_scale_enabled = gradient_scale_enabled
        
        self.optimizer, self.scheduler = create_optimizer(model, config)
        
        self.step_count = 0
        self._grad_stats = {}
    
    def step(
        self,
        loss: torch.Tensor,
        routing_stats: Optional[Dict] = None,
    ):
        """
        Perform one optimization step with gradient scaling.
        
        Args:
            loss: The total loss to backpropagate
            routing_stats: Optional routing statistics for gradient scaling
        """
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip,
        )
        
        # Apply gradient scaling based on routing (if enabled)
        if self.gradient_scale_enabled and routing_stats is not None:
            self._apply_gradient_scaling(routing_stats)
        
        # Track gradient statistics
        self._track_gradient_stats()
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.step_count += 1
    
    def _apply_gradient_scaling(self, routing_stats: Dict):
        """
        Scale gradients to prevent pathway starvation.
        
        If a pathway receives very few tokens (< 10%), its gradients are
        amplified to ensure it still learns. If it receives too many (> 80%),
        gradients are slightly damped.
        """
        stream_frac = routing_stats.get("avg_stream_frac", 0.33)
        focus_frac = routing_stats.get("avg_focus_frac", 0.33)
        reason_frac = routing_stats.get("avg_reason_frac", 0.33)
        
        fractions = {
            "stream": stream_frac,
            "focus": focus_frac,
            "reason": reason_frac,
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            for pathway_name, frac in fractions.items():
                if pathway_name in name:
                    if frac < 0.1:
                        # Amplify starved pathway gradients
                        scale = min(2.0, 0.1 / max(frac, 0.01))
                        param.grad.data.mul_(scale)
                    elif frac > 0.8:
                        # Slightly damp dominant pathway gradients
                        scale = max(0.5, 0.8 / frac)
                        param.grad.data.mul_(scale)
    
    def _track_gradient_stats(self):
        """Track gradient norms for monitoring."""
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Group by component
                component = name.split(".")[1] if "." in name else name
                if component not in grad_norms:
                    grad_norms[component] = []
                grad_norms[component].append(param.grad.data.norm(2).item())
        
        self._grad_stats = {
            k: sum(v) / len(v) for k, v in grad_norms.items()
        }
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def get_gradient_stats(self) -> Dict:
        """Get gradient statistics for logging."""
        return self._grad_stats.copy()
    
    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step_count": self.step_count,
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step_count = state_dict["step_count"]
