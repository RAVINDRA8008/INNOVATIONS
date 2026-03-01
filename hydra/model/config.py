"""
HYDRA Model Configuration

Defines all hyperparameters for the Hydra architecture, including
pathway-specific settings and routing parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class HydraConfig:
    """Configuration for the HYDRA architecture.
    
    Design Philosophy:
        The config separates concerns across three axes:
        1. Global model dimensions (shared across pathways)
        2. Pathway-specific parameters (SSM state dim, window size, etc.)
        3. Routing parameters (temperature, budget constraints)
    """
    
    # === Global Model Parameters ===
    vocab_size: int = 32000
    d_model: int = 512           # Hidden dimension
    n_layers: int = 12           # Number of Hydra blocks
    n_heads: int = 8             # Attention heads (for Focus and Reason pathways)
    d_ff: int = 2048             # Feed-forward intermediate dimension
    max_seq_len: int = 2048      # Maximum sequence length
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # === SSM Pathway (Stream) Parameters ===
    ssm_state_dim: int = 64           # State dimension for SSM
    ssm_conv_width: int = 4           # Convolutional kernel width
    ssm_expansion_factor: int = 2     # Inner dimension expansion
    ssm_dt_rank: str = "auto"         # Rank for discretization parameter
    ssm_use_fast_path: bool = True    # Use fused kernels when available
    
    # === Windowed Attention (Focus) Parameters ===
    window_size: int = 256            # Local attention window size
    window_overlap: int = 64          # Overlap between windows
    
    # === Global Attention (Reason) Parameters ===
    reason_use_flash: bool = True     # Use FlashAttention when available
    reason_use_rotary: bool = True    # Use Rotary Position Embeddings
    
    # === Router Parameters ===
    router_hidden_dim: int = 128      # Router MLP hidden dimension
    router_temperature: float = 1.0   # Gumbel-Softmax temperature (annealed during training)
    router_min_temperature: float = 0.1
    router_aux_loss_weight: float = 0.01   # Load balancing auxiliary loss weight
    router_budget_loss_weight: float = 0.05 # Compute budget regularization weight
    target_stream_ratio: float = 0.6   # Target fraction of tokens routed to Stream
    target_focus_ratio: float = 0.25   # Target fraction routed to Focus
    target_reason_ratio: float = 0.15  # Target fraction routed to Reason
    
    # === Cross-Pathway Mixing ===
    cross_pathway_heads: int = 4       # Heads for cross-pathway attention
    cross_pathway_dim: int = 256       # Dimension for cross-pathway mixing
    use_cross_pathway_mixing: bool = True
    mixer_type: str = "gated_linear"   # "cross_attention" (old, expensive) or "gated_linear" (new, O(n))
    mixer_frequency: int = 3           # Apply mixer every N layers (1=every layer, 3=every 3rd)
    
    # === Training Memory Optimization ===
    checkpoint_pathways: str = "non_dominant"  # "none", "non_dominant" (2 of 3), "all"
    # "non_dominant": Gradient-checkpoint 2 of 3 pathways (the ones with lowest routing weight).
    #   Saves ~60-70% of pathway activation memory. Costs ~30-40% extra training time (recompute).
    # "all": Checkpoint all 3 pathways. Saves ~90% of pathway activation memory.
    #   Costs ~2× training time per step (all pathways recomputed during backward).
    # "none": Full activations stored for all pathways (3.7× memory vs dense).
    
    # === Training Parameters ===
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_clip: float = 1.0
    
    # === Curriculum Parameters ===
    curriculum_enabled: bool = True
    curriculum_warmup_steps: int = 5000   # Steps before curriculum kicks in
    curriculum_max_complexity: float = 1.0
    curriculum_schedule: str = "cosine"   # "linear", "cosine", "step"
    
    def __post_init__(self):
        """Derive computed parameters."""
        if self.ssm_dt_rank == "auto":
            self.ssm_dt_rank_value = max(self.d_model // 16, 1)
        else:
            self.ssm_dt_rank_value = int(self.ssm_dt_rank)
        
        self.d_head = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
    
    @classmethod
    def from_yaml(cls, path: str) -> "HydraConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def small(cls) -> "HydraConfig":
        """Small config for debugging / fast experiments."""
        return cls(
            d_model=256, n_layers=6, n_heads=4, d_ff=1024,
            max_seq_len=512, ssm_state_dim=32,
            router_hidden_dim=64, cross_pathway_dim=128,
            mixer_type="gated_linear", mixer_frequency=3,
            checkpoint_pathways="non_dominant",
        )
    
    @classmethod
    def base(cls) -> "HydraConfig":
        """Base config — the default research configuration."""
        return cls(
            mixer_type="gated_linear", mixer_frequency=3,
            checkpoint_pathways="non_dominant",
        )
    
    @classmethod
    def large(cls) -> "HydraConfig":
        """Large config for scaling experiments."""
        return cls(
            d_model=1024, n_layers=24, n_heads=16, d_ff=4096,
            max_seq_len=4096, ssm_state_dim=128,
            router_hidden_dim=256, cross_pathway_dim=512,
            mixer_type="gated_linear", mixer_frequency=4,
            checkpoint_pathways="non_dominant",
        )
