"""
Pathway Router — The Brain of HYDRA

This is the core innovation: a differentiable routing mechanism that decides,
for each token, which computational pathway to use.

Key Research Contributions:
    1. Token-level routing across computational PARADIGMS (not just experts)
    2. Gumbel-Softmax with temperature annealing for differentiable discrete routing
    3. Multi-signal complexity estimation using:
       - Hidden state magnitude
       - Local entropy estimation
       - Gradient signal (during training)
    4. Load balancing via auxiliary losses
    5. Compute budget regularization to prevent the model from always choosing
       the most expensive pathway

Innovation Insight:
    Unlike Mixture of Experts (MoE) where all experts have similar compute cost,
    HYDRA's pathways have DIFFERENT computational costs. This creates a unique
    optimization challenge: the model must learn not just WHICH pathway is best,
    but also when the accuracy gain of an expensive pathway justifies its cost.
    
    The router solves this with the "Compute Budget Loss" — a regularization
    term that penalizes total compute beyond a learnable budget threshold.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexityEstimator(nn.Module):
    """
    Estimates per-token "complexity" to inform routing decisions.
    
    Uses multiple signals:
    1. Content-based: MLP on hidden states
    2. Context-based: Rolling statistics of local neighborhood
    3. Entropy-based: Approximation of local information density
    
    The estimator is trained end-to-end — it learns what "complex" means
    in the context of the model's task, rather than relying on hand-crafted
    complexity heuristics.
    """
    
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        
        # Content-based complexity estimation
        self.content_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Context-based: lightweight 1D conv to capture local statistics
        self.context_conv = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim // 2, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Entropy approximation: variance of local window
        self.entropy_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
        )
        
        # Fusion: combine all signals
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            complexity_features: (batch, seq_len, hidden_dim)
        """
        B, L, D = x.shape
        
        # Signal 1: Content-based
        content_feat = self.content_mlp(x)  # (B, L, hidden_dim)
        
        # Signal 2: Context-based (needs transposition for Conv1d)
        x_t = x.transpose(1, 2)  # (B, D, L)
        context_feat = self.context_conv(x_t).transpose(1, 2)  # (B, L, hidden_dim)
        
        # Signal 3: Entropy approximation
        # Compute local variance as a proxy for information density
        # High variance → likely more complex / information-dense
        window_size = min(7, L)
        if L >= window_size:
            # Unfold to get local windows
            x_unfold = x.unfold(1, window_size, 1)  # (B, L-w+1, D, w)
            local_var = x_unfold.var(dim=-1)  # (B, L-w+1, D)
            # Pad to original length
            pad_left = (window_size - 1) // 2
            pad_right = window_size - 1 - pad_left
            local_var = F.pad(local_var, (0, 0, pad_left, pad_right), mode='replicate')
        else:
            local_var = x.var(dim=1, keepdim=True).expand_as(x)
        
        entropy_feat = self.entropy_proj(local_var)  # (B, L, hidden_dim)
        
        # Fuse all signals
        combined = torch.cat([content_feat, context_feat, entropy_feat], dim=-1)
        complexity = self.fusion(combined)
        complexity = self.norm(complexity)
        
        return complexity


class PathwayRouter(nn.Module):
    """
    Differentiable router that assigns each token to a computational pathway.
    
    Routing Mechanism:
        1. ComplexityEstimator produces rich per-token features
        2. Linear projection produces logits for each pathway
        3. Gumbel-Softmax produces differentiable one-hot routing decisions
        4. During inference, argmax is used (hard routing)
    
    Auxiliary Losses:
        - Load Balance Loss: Encourages even distribution across pathways
          (prevents mode collapse to one pathway)
        - Compute Budget Loss: Penalizes total compute exceeding a target budget
          (prevents always using expensive pathways)
    
    Temperature Annealing:
        Gumbel-Softmax temperature starts high (soft routing, easy gradients)
        and anneals to low (hard routing, precise allocation).
    """
    
    NUM_PATHWAYS = 3  # Stream, Focus, Reason
    PATHWAY_NAMES = ["Stream (SSM)", "Focus (Windowed)", "Reason (Global)"]
    PATHWAY_COSTS = [1.0, 3.0, 10.0]  # Relative compute costs
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Complexity estimator
        self.complexity_estimator = ComplexityEstimator(
            d_model=config.d_model,
            hidden_dim=config.router_hidden_dim,
        )
        
        # Routing head: maps complexity features to pathway logits
        self.routing_head = nn.Sequential(
            nn.Linear(config.router_hidden_dim, config.router_hidden_dim),
            nn.GELU(),
            nn.Linear(config.router_hidden_dim, self.NUM_PATHWAYS),
        )
        
        # Learnable pathway cost weights (allows the model to learn
        # its own notion of "expensive")
        self.pathway_costs = nn.Parameter(
            torch.tensor(self.PATHWAY_COSTS, dtype=torch.float32),
            requires_grad=False,  # Fixed costs; learned routing adapts to them
        )
        
        # Target ratios for load balancing
        self.register_buffer(
            "target_ratios",
            torch.tensor([
                config.target_stream_ratio,
                config.target_focus_ratio,
                config.target_reason_ratio,
            ]),
        )
        
        # Temperature for Gumbel-Softmax (annealed during training)
        self.temperature = config.router_temperature
        
        # Loss weights
        self.aux_loss_weight = config.router_aux_loss_weight
        self.budget_loss_weight = config.router_budget_loss_weight
        
        # Tracking
        self._routing_stats = {}
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route each token to a pathway.
        
        Args:
            x: (batch, seq_len, d_model) — input hidden states
            training: Whether we're in training mode
            
        Returns:
            routing_weights: (batch, seq_len, 3) — soft/hard pathway assignments
            router_info: dict containing:
                - "logits": raw routing logits
                - "load_balance_loss": auxiliary loss for even distribution
                - "compute_budget_loss": auxiliary loss for compute efficiency
                - "pathway_fractions": actual fraction of tokens per pathway
                - "complexity_scores": per-token complexity estimates
        """
        B, L, D = x.shape
        
        # Step 1: Estimate per-token complexity
        complexity_features = self.complexity_estimator(x)  # (B, L, hidden_dim)
        
        # Step 2: Compute routing logits
        logits = self.routing_head(complexity_features)  # (B, L, 3)
        
        # Step 3: Apply Gumbel-Softmax or hard routing
        if training:
            routing_weights = F.gumbel_softmax(
                logits, tau=self.temperature, hard=True, dim=-1
            )
            # "hard=True" produces one-hot in forward pass but smooth gradients
            # in backward pass (straight-through estimator)
        else:
            # Hard routing during inference
            indices = logits.argmax(dim=-1)
            routing_weights = F.one_hot(indices, self.NUM_PATHWAYS).float()
        
        # Step 4: Compute auxiliary losses
        router_info = self._compute_aux_losses(logits, routing_weights)
        
        # Step 5: Track routing statistics
        with torch.no_grad():
            fractions = routing_weights.mean(dim=(0, 1))
            self._routing_stats = {
                f"frac_{name}": frac.item()
                for name, frac in zip(self.PATHWAY_NAMES, fractions)
            }
            router_info["pathway_fractions"] = fractions
            router_info["complexity_scores"] = complexity_features.norm(dim=-1)
        
        return routing_weights, router_info
    
    def _compute_aux_losses(
        self,
        logits: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for routing optimization.
        
        Two losses work together:
        
        1. Load Balance Loss:
           Encourages the actual routing distribution to match target ratios.
           Without this, the model tends to collapse all tokens to one pathway.
           
           L_balance = sum_i (f_i - target_i)^2
           where f_i is the fraction of tokens routed to pathway i.
        
        2. Compute Budget Loss:
           Penalizes total compute cost exceeding a target budget.
           This prevents the model from routing everything to Reason (most powerful
           but most expensive).
           
           L_budget = max(0, total_cost - target_cost)^2
           where total_cost = sum(fractions * pathway_costs)
        """
        B, L, P = logits.shape
        
        # Actual fractions (differentiable through soft logits)
        soft_fractions = F.softmax(logits, dim=-1).mean(dim=(0, 1))  # (3,)
        
        # === Load Balance Loss ===
        # We use the soft fractions for gradient flow, even though routing is hard
        load_balance_loss = ((soft_fractions - self.target_ratios) ** 2).sum()
        
        # Also add a "diversity" term: penalize if any pathway gets < 5%
        min_fraction_penalty = F.relu(0.05 - soft_fractions).sum()
        load_balance_loss = load_balance_loss + min_fraction_penalty
        
        # === Compute Budget Loss ===
        # Total compute cost under current routing
        total_cost = (soft_fractions * self.pathway_costs).sum()
        
        # Target cost: weighted by target ratios
        target_cost = (self.target_ratios * self.pathway_costs).sum()
        
        # Only penalize if over budget
        compute_budget_loss = F.relu(total_cost - target_cost * 1.1) ** 2
        
        # Combined auxiliary loss
        total_aux_loss = (
            self.aux_loss_weight * load_balance_loss
            + self.budget_loss_weight * compute_budget_loss
        )
        
        return {
            "logits": logits,
            "load_balance_loss": load_balance_loss,
            "compute_budget_loss": compute_budget_loss,
            "total_aux_loss": total_aux_loss,
            "total_compute_cost": total_cost.detach(),
            "soft_fractions": soft_fractions.detach(),
        }
    
    def set_temperature(self, temperature: float):
        """Set the Gumbel-Softmax temperature (for annealing)."""
        self.temperature = max(temperature, self.config.router_min_temperature)
    
    def get_routing_stats(self) -> Dict[str, float]:
        """Return latest routing statistics for logging."""
        return self._routing_stats.copy()
    
    def get_routing_visualization_data(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate data for routing visualization.
        
        Returns complexity scores and routing decisions for analysis.
        Useful for understanding what the model considers "complex."
        """
        with torch.no_grad():
            complexity_features = self.complexity_estimator(x)
            logits = self.routing_head(complexity_features)
            
            probs = F.softmax(logits, dim=-1)
            decisions = logits.argmax(dim=-1)
            complexity_norm = complexity_features.norm(dim=-1)
            
            return {
                "complexity_scores": complexity_norm,
                "routing_probabilities": probs,
                "routing_decisions": decisions,
                "logits": logits,
            }
