"""
Adaptive Compute Budget Loss — Algorithmic Innovation #1

A novel loss function that trains the model to be "compute-aware":
it learns to allocate expensive computation only when the accuracy
gain justifies the cost.

Three Components:
    1. Task Loss: Standard cross-entropy for language modeling
    2. Efficiency Loss: Penalizes using expensive pathways for "easy" tokens
    3. Consistency Loss: Encourages similar tokens to be routed similarly

Mathematical Formulation:
    L_total = L_task + α·L_efficiency + β·L_consistency + γ·L_aux
    
    L_efficiency = E_tokens[(cost(pathway) × (1 - confidence))]
    - High cost + low confidence = heavily penalized
    - High cost + high confidence = moderately penalized (justified expense)
    - Low cost + any confidence = minimal penalty
    
    L_consistency = E_layers[KL(routing_l || routing_{l-1})]
    - Neighboring layers should have similar routing patterns
    - Prevents chaotic layer-to-layer routing oscillation

Research Note:
    The efficiency loss creates an interesting training dynamic: early in
    training, the model routes everything to SSM (cheapest). As task loss 
    dominates, it gradually learns to escalate compute for tokens that
    genuinely need it. The curriculum scheduler works with this dynamic
    to produce stable routing emergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AdaptiveComputeBudgetLoss(nn.Module):
    """
    Multi-component loss for compute-aware routing optimization.
    """
    
    def __init__(
        self,
        efficiency_weight: float = 0.1,
        consistency_weight: float = 0.05,
        confidence_threshold: float = 0.8,
        cost_normalization: str = "mean",  # "mean" or "max"
    ):
        super().__init__()
        self.efficiency_weight = efficiency_weight
        self.consistency_weight = consistency_weight
        self.confidence_threshold = confidence_threshold
        self.cost_normalization = cost_normalization
        
        # Pathway costs (relative compute cost of SSM / Windowed / Global)
        self._pathway_costs_values = [1.0, 3.0, 10.0]
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full adaptive compute budget loss.
        
        Args:
            model_output: dict from HydraModel.forward() containing:
                - "loss": task loss
                - "aux_loss": router auxiliary losses
                - "block_info": per-block routing information
                
        Returns:
            dict with individual and total losses
        """
        block_info = model_output.get("block_info", [])
        # Determine device from model output tensors
        if block_info and len(block_info) > 0:
            _dev = block_info[0]["routing_weights"].device
        elif "loss" in model_output:
            _dev = model_output["loss"].device
        else:
            _dev = torch.device("cpu")
        task_loss = model_output.get("loss", torch.tensor(0.0, device=_dev))
        aux_loss = model_output.get("aux_loss", torch.tensor(0.0, device=_dev))
        
        # Compute efficiency loss
        efficiency_loss = self._compute_efficiency_loss(block_info)
        
        # Compute consistency loss
        consistency_loss = self._compute_consistency_loss(block_info)
        
        # Total loss
        total_loss = (
            task_loss
            + aux_loss
            + self.efficiency_weight * efficiency_loss
            + self.consistency_weight * consistency_loss
        )
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "efficiency_loss": efficiency_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
        }
    
    def _compute_efficiency_loss(
        self, block_info: list
    ) -> torch.Tensor:
        """
        Efficiency loss: penalizes expensive computation for easy tokens.
        
        For each token, compute:
            L_eff = cost(selected_pathway) × (1 - max_routing_probability)
        
        Intuition:
        - If a token is routed to Reason (cost=10) with high confidence (prob=0.95),
          the penalty is 10 × 0.05 = 0.5 → modest penalty, justified
        - If routed to Reason with low confidence (prob=0.4),
          penalty is 10 × 0.6 = 6.0 → heavy penalty, unjustified expense
        - If routed to Stream (cost=1) with any confidence,
          penalty is ≤ 1.0 → minimal, it's cheap anyway
        """
        if not block_info:
            return torch.tensor(0.0)
        
        device = block_info[0]["routing_weights"].device
        pathway_costs = torch.tensor(self._pathway_costs_values, device=device)
        total_eff_loss = torch.tensor(0.0, device=device)
        
        for info in block_info:
            routing_weights = info["routing_weights"]  # (B, L, 3)
            logits = info["router_info"]["logits"]  # (B, L, 3)
            
            # Routing confidence: max probability after softmax
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values  # (B, L)
            
            # Cost of selected pathway
            selected_costs = (routing_weights * pathway_costs).sum(dim=-1)  # (B, L)
            
            # Normalize costs
            if self.cost_normalization == "mean":
                selected_costs = selected_costs / pathway_costs.mean()
            else:
                selected_costs = selected_costs / pathway_costs.max()
            
            # Efficiency penalty: high cost + low confidence = bad
            uncertainty = 1.0 - confidence
            eff_penalty = selected_costs * uncertainty
            
            total_eff_loss = total_eff_loss + eff_penalty.mean()
        
        return total_eff_loss / len(block_info)
    
    def _compute_consistency_loss(
        self, block_info: list
    ) -> torch.Tensor:
        """
        Consistency loss: neighboring layers should have similar routing.
        
        Uses KL divergence between routing distributions of adjacent layers.
        This prevents chaotic routing patterns and encourages smooth
        computational depth transitions.
        """
        if len(block_info) < 2:
            return torch.tensor(0.0)
        
        device = block_info[0]["routing_weights"].device
        total_consistency = torch.tensor(0.0, device=device)
        
        for i in range(len(block_info) - 1):
            logits_curr = block_info[i]["router_info"]["logits"]
            logits_next = block_info[i + 1]["router_info"]["logits"]
            
            probs_curr = F.softmax(logits_curr, dim=-1)
            probs_next = F.softmax(logits_next, dim=-1)
            
            # Symmetric KL divergence
            kl_forward = F.kl_div(
                probs_curr.log(), probs_next, reduction="batchmean"
            )
            kl_backward = F.kl_div(
                probs_next.log(), probs_curr, reduction="batchmean"
            )
            
            total_consistency = total_consistency + (kl_forward + kl_backward) / 2
        
        return total_consistency / (len(block_info) - 1)


class RoutingDiversityRegularizer(nn.Module):
    """
    Additional regularizer that encourages diverse routing across the sequence.
    
    Without this, the model might learn to route ALL tokens in a batch to
    the same pathway. This regularizer ensures that within a single sequence,
    multiple pathways are utilized.
    
    Implementation: Entropy maximization on per-sequence routing distribution.
    """
    
    def __init__(self, target_entropy: float = 0.8):
        super().__init__()
        self.target_entropy = target_entropy  # Fraction of max entropy
        self.max_entropy = torch.log(torch.tensor(3.0))  # log(3) for 3 pathways
    
    def forward(self, block_info: list) -> torch.Tensor:
        """Compute diversity regularization loss."""
        if not block_info:
            return torch.tensor(0.0)
        
        device = block_info[0]["routing_weights"].device
        total_div_loss = torch.tensor(0.0, device=device)
        
        for info in block_info:
            routing_weights = info["routing_weights"]  # (B, L, 3)
            
            # Per-sequence routing distribution
            seq_distribution = routing_weights.mean(dim=1)  # (B, 3)
            
            # Entropy of the distribution
            entropy = -(seq_distribution * (seq_distribution + 1e-8).log()).sum(dim=-1)
            
            # Penalize low entropy (lack of diversity)
            target = self.target_entropy * self.max_entropy.to(device)
            div_penalty = F.relu(target - entropy).mean()
            
            total_div_loss = total_div_loss + div_penalty
        
        return total_div_loss / len(block_info)
