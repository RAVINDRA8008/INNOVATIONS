"""
HYDRA Full Model — Complete Architecture Assembly

Assembles N HydraBlocks into a full language model with:
    - Token + positional embeddings
    - Stacked HYDRA blocks
    - Language model head (for next-token prediction)
    - Aggregated routing statistics and losses

Model Variants:
    - HydraModel.small():  ~25M params — for debugging and fast iteration
    - HydraModel.base():   ~125M params — default research configuration
    - HydraModel.large():  ~350M params — for scaling experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .config import HydraConfig
from .hydra_block import HydraBlock


class HydraModel(nn.Module):
    """
    HYDRA: Hybrid Dynamic Routing Architecture
    
    A transformer-class language model that dynamically routes each token
    through one of three computational pathways based on learned complexity
    estimation.
    """
    
    def __init__(self, config: HydraConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Learnable positional embeddings (for base position awareness)
        # Note: RoPE in attention pathways provides relative position info
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Stack of HYDRA blocks
        self.blocks = nn.ModuleList([
            HydraBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Language model head (tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Compute model statistics
        self._n_params = sum(p.numel() for p in self.parameters())
        self._n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using scaled initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the HYDRA model.
        
        Args:
            input_ids: (batch, seq_len) — token indices
            targets: Optional (batch, seq_len) — target token indices for loss
            mask: Optional attention/padding mask
            
        Returns:
            dict containing:
                - "logits": (batch, seq_len, vocab_size) — prediction logits
                - "loss": scalar loss (if targets provided)
                - "aux_loss": aggregated auxiliary routing losses
                - "routing_stats": per-layer routing statistics
                - "total_loss": loss + aux_loss
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.emb_dropout(x)
        
        # Pass through HYDRA blocks
        all_block_info = []
        total_aux_loss = torch.tensor(0.0, device=device)
        
        for block in self.blocks:
            x, block_info = block(x, mask)
            all_block_info.append(block_info)
            total_aux_loss = total_aux_loss + block_info["router_info"]["total_aux_loss"]
        
        # Final norm
        x = self.final_norm(x)
        
        # Language model head
        logits = self.lm_head(x)  # (B, L, vocab_size)
        
        # Compute loss if targets provided
        result = {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "routing_stats": self._aggregate_routing_stats(all_block_info),
            "block_info": all_block_info,
        }
        
        if targets is not None:
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
            result["total_loss"] = loss + total_aux_loss
        
        return result
    
    def _aggregate_routing_stats(
        self, all_block_info: List[Dict]
    ) -> Dict[str, object]:
        """Aggregate routing statistics across all layers.
        
        Returns 0-dim tensors (not floats) to avoid CUDA synchronization.
        Call .item() on the values only when you need to log/print them.
        """
        n_layers = len(all_block_info)
        
        # Vectorized aggregation — single GPU operation, NO .item() calls
        all_fracs = torch.stack([
            info["router_info"]["soft_fractions"] for info in all_block_info
        ])  # (n_layers, 3)
        avg_fracs = all_fracs.mean(dim=0)  # (3,)
        
        all_costs = torch.stack([
            info["router_info"]["total_compute_cost"] for info in all_block_info
        ])
        avg_cost = all_costs.mean()
        
        all_aux = torch.stack([
            info["router_info"]["total_aux_loss"] for info in all_block_info
        ])
        total_aux = all_aux.sum()
        
        stats = {
            "avg_stream_frac": avg_fracs[0],   # 0-dim tensor (no CUDA sync)
            "avg_focus_frac": avg_fracs[1],
            "avg_reason_frac": avg_fracs[2],
            "avg_compute_cost": avg_cost,
            "total_aux_loss": total_aux,
            "per_layer": [],  # Populated on demand (eval/visualization only)
        }
        
        return stats
    
    def get_routing_map(
        self, input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a complete routing map for visualization.
        
        Returns per-layer, per-token routing decisions — useful for
        understanding what the model considers complex.
        """
        self.eval()
        with torch.no_grad():
            B, L = input_ids.shape
            device = input_ids.device
            
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            
            routing_map = torch.zeros(self.config.n_layers, B, L, 3, device=device)
            complexity_map = torch.zeros(self.config.n_layers, B, L, device=device)
            
            for i, block in enumerate(self.blocks):
                x, block_info = block(x)
                routing_map[i] = block_info["routing_weights"]
                complexity_map[i] = block_info["router_info"]["complexity_scores"]
            
            return {
                "routing_map": routing_map,     # (n_layers, B, L, 3)
                "complexity_map": complexity_map,  # (n_layers, B, L)
                "decisions": routing_map.argmax(dim=-1),  # (n_layers, B, L)
            }
    
    def count_parameters(self) -> Dict[str, int]:
        """Detailed parameter count by component."""
        counts = {
            "total": self._n_params,
            "trainable": self._n_trainable,
            "embeddings": sum(
                p.numel() for p in self.token_embedding.parameters()
            ) + sum(p.numel() for p in self.position_embedding.parameters()),
            "per_block": {},
        }
        
        for i, block in enumerate(self.blocks):
            counts["per_block"][f"block_{i}"] = {
                "router": sum(p.numel() for p in block.router.parameters()),
                "stream": sum(p.numel() for p in block.stream.parameters()),
                "focus": sum(p.numel() for p in block.focus.parameters()),
                "reason": sum(p.numel() for p in block.reason.parameters()),
                "mixer": sum(p.numel() for p in block.cross_mixer.parameters())
                if block.cross_mixer else 0,
            }
        
        return counts
    
    @classmethod
    def from_config(cls, config_name: str) -> "HydraModel":
        """Create model from a named configuration."""
        configs = {
            "small": HydraConfig.small,
            "base": HydraConfig.base,
            "large": HydraConfig.large,
        }
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
        return cls(configs[config_name]())
    
    def __repr__(self):
        return (
            f"HydraModel(\n"
            f"  d_model={self.config.d_model}, n_layers={self.config.n_layers},\n"
            f"  n_heads={self.config.n_heads}, d_ff={self.config.d_ff},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  params={self._n_params:,} ({self._n_params/1e6:.1f}M)\n"
            f")"
        )
