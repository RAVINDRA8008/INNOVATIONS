"""
SSM Pathway (Stream) — The Efficient Backbone

Implements a selective state-space model inspired by Mamba/S4, designed as the
"fast lane" for tokens that don't require expensive attention computation.

Key Design Decisions:
    1. Selective scan mechanism — input-dependent state transitions allow the SSM
       to selectively propagate or forget information (unlike classic S4).
    2. 1D convolution front-end — captures local patterns before state-space
       processing (similar to Mamba's architecture).
    3. Gated output — multiplicative gating prevents information dilution.

Computational Cost: O(n) per token — the cheapest pathway.

Research Note:
    The SSM pathway is intentionally simpler than full Mamba to serve as a 
    "baseline computator." The hypothesis is that most tokens in natural language
    can be adequately processed with this linear-cost pathway, freeing compute
    budget for the critical tokens that truly need attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .config import HydraConfig


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core.
    
    Implements the continuous-time SSM:
        x'(t) = A·x(t) + B·u(t)
        y(t)  = C·x(t) + D·u(t)
    
    With input-dependent discretization (selective mechanism):
        - Δ (step size) is a function of the input
        - B is a function of the input
        - This allows the model to selectively remember or forget
    """
    
    def __init__(self, d_inner: int, d_state: int, dt_rank: int, d_model: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        
        # A is initialized as a structured matrix (HiPPO-inspired)
        # We parameterize log(A) for numerical stability
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Input-dependent projections for selectivity
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        # Initialize dt bias to ensure proper initial step sizes
        dt_init_std = dt_rank ** -0.5 * 1.0  # scale factor
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Inverse softplus for dt bias initialization
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_inner)
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        
        # Compute input-dependent parameters (selectivity)
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        delta, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Compute step size
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Recover A from log parameterization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        # Using ZOH (zero-order hold) discretization
        y = self._selective_scan(x, delta, A, B, C)
        
        # Skip connection via D parameter
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y
    
    def _selective_scan(
        self,
        u: torch.Tensor,    # (B, L, D)
        delta: torch.Tensor, # (B, L, D)
        A: torch.Tensor,     # (D, N)
        B: torch.Tensor,     # (B, L, N)
        C: torch.Tensor,     # (B, L, N)
    ) -> torch.Tensor:
        """
        Sequential selective scan implementation.
        
        For production, this would be replaced with a CUDA kernel (like Mamba's).
        This pure-PyTorch version is for research clarity.
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        # deltaA: (B, L, D, N) = exp(delta[:,:,:,None] * A[None,None,:,:])
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB_u: (B, L, D, N) = delta[:,:,:,None] * B[:,:,None,:] * u[:,:,:,None]
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        # Sequential scan
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (x * C[:, i].unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y)
        
        return torch.stack(ys, dim=1)  # (B, L, D)


class SSMPathway(nn.Module):
    """
    Complete SSM Pathway (Stream) for the HYDRA architecture.
    
    Architecture:
        Input → LayerNorm → Linear (expand) → Conv1D → SiLU → SelectiveSSM 
              → Gated Output → Linear (project) → Dropout → Output
    
    The expansion factor creates an inner dimension larger than d_model,
    giving the SSM more room to work, then projects back down.
    """
    
    def __init__(self, config: HydraConfig):
        super().__init__()
        self.config = config
        
        d_inner = config.d_model * config.ssm_expansion_factor
        
        # Pre-normalization
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Input projection (with gating branch)
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)
        
        # 1D Convolution for local pattern capture
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=config.ssm_conv_width,
            padding=config.ssm_conv_width - 1,
            groups=d_inner,  # Depthwise convolution
            bias=True,
        )
        
        # Core SSM
        self.ssm = SelectiveSSM(
            d_inner=d_inner,
            d_state=config.ssm_state_dim,
            dt_rank=config.ssm_dt_rank_value,
            d_model=config.d_model,
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        # Track compute cost for routing optimization
        self.compute_cost = 1.0  # Normalized: SSM is the cheapest pathway
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional token-level routing mask (batch, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Expand to inner dimension with gating branch
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # 1D convolution (causal)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :x.shape[-1]]  # Truncate for causal
        x = rearrange(x, "b d l -> b l d")
        
        # Activation
        x = F.silu(x)
        
        # Selective SSM
        x = self.ssm(x)
        
        # Gated output
        x = x * F.silu(z)
        
        # Project back to model dimension
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x + residual
    
    def get_compute_profile(self) -> dict:
        """Return computational profile for this pathway."""
        return {
            "name": "Stream (SSM)",
            "complexity": "O(n)",
            "normalized_cost": self.compute_cost,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
