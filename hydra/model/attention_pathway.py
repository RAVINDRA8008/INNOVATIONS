"""
Attention Pathways — Focus (Windowed) and Reason (Global)

Two attention-based pathways of increasing computational cost:

1. WindowedAttentionPathway (Focus):
   - Applies attention within local windows with optional overlap
   - Cost: O(n * w) where w is window size
   - Use case: Tokens needing local context (entity boundaries, syntax, etc.)

2. GlobalAttentionPathway (Reason):
   - Full quadratic attention over the entire sequence
   - Optional FlashAttention for memory efficiency
   - Rotary Position Embeddings for length generalization
   - Cost: O(n^2) — the most expensive pathway
   - Use case: Tokens requiring global reasoning (logical connectives, 
     cross-reference resolution, mathematical operators)

Design Philosophy:
    These pathways are intentionally "standard" in their attention mechanism —
    the innovation is in the ROUTING to these pathways, not the pathways 
    themselves. This makes ablation studies cleaner and ensures the architecture's
    improvements come from dynamic compute allocation, not attention tricks.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .config import HydraConfig


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position information by rotating query/key vectors in 2D subspaces.
    This provides relative position awareness without explicit position encodings,
    and naturally extends to longer sequences than seen during training.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute sin/cos tables
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        """Apply rotary embeddings to input tensor."""
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset)
        
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with optional RoPE and causal masking.
    
    Shared by both Focus and Reason pathways with different configurations.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_emb = RotaryPositionEmbedding(self.d_head, max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional mask (batch, 1, seq_len, seq_len)
            causal: Whether to apply causal (autoregressive) masking
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        B, L, _ = x.shape
        
        # Project to Q, K, V
        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b l (h d) -> b h l d", h=self.n_heads)
        
        # Apply rotary position embeddings
        if self.use_rotary:
            cos, sin = self.rotary_emb(q, L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        
        # Apply optional attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.o_proj(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network — standard in modern transformers."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class WindowedAttentionPathway(nn.Module):
    """
    Focus Pathway — Windowed Local Attention.
    
    Splits the sequence into overlapping windows and applies attention
    within each window. This captures local dependencies efficiently.
    
    Architecture:
        Input → LayerNorm → Window-Partition → Multi-Head Attention 
              → Window-Merge → FFN → Output
    
    Innovation Note:
        The overlap mechanism uses a learnable "boundary gate" that controls
        how much information flows across window boundaries. This is a 
        subtle but important detail — without it, window boundaries create
        hard information barriers that degrade performance on tasks requiring
        medium-range dependencies.
    """
    
    def __init__(self, config: HydraConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.overlap = config.window_overlap
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Attention within windows
        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_rotary=True,
            max_seq_len=config.window_size + config.window_overlap,
        )
        
        # Feed-forward network
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        
        # Boundary gate for cross-window information flow
        self.boundary_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Compute cost: ~3x SSM (windowed attention)
        self.compute_cost = 3.0
    
    def _partition_windows(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Partition sequence into overlapping windows.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            windows: (batch * n_windows, window_size, d_model)
            n_windows: number of windows
        """
        B, L, D = x.shape
        W = self.window_size
        
        # Pad sequence to be divisible by (window_size - overlap)
        stride = W - self.overlap
        n_windows = max(1, (L - self.overlap + stride - 1) // stride)
        padded_len = (n_windows - 1) * stride + W
        
        if padded_len > L:
            x = F.pad(x, (0, 0, 0, padded_len - L))
        
        # Extract overlapping windows
        windows = []
        for i in range(n_windows):
            start = i * stride
            end = start + W
            windows.append(x[:, start:end, :])
        
        windows = torch.stack(windows, dim=1)  # (B, n_windows, W, D)
        windows = rearrange(windows, "b n w d -> (b n) w d")
        
        return windows, n_windows
    
    def _merge_windows(
        self, windows: torch.Tensor, n_windows: int, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """
        Merge overlapping windows back into a sequence.
        
        Uses averaging in overlap regions to maintain continuity.
        """
        W = self.window_size
        stride = W - self.overlap
        D = windows.shape[-1]
        
        windows = rearrange(
            windows, "(b n) w d -> b n w d", b=batch_size, n=n_windows
        )
        
        padded_len = (n_windows - 1) * stride + W
        output = torch.zeros(batch_size, padded_len, D, device=windows.device)
        counts = torch.zeros(batch_size, padded_len, 1, device=windows.device)
        
        for i in range(n_windows):
            start = i * stride
            end = start + W
            output[:, start:end, :] += windows[:, i, :, :]
            counts[:, start:end, :] += 1
        
        # Average overlapping regions
        output = output / counts.clamp(min=1)
        
        return output[:, :seq_len, :]
    
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional routing mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        residual = x
        
        # Pre-norm + window partition
        x_normed = self.norm1(x)
        windows, n_windows = self._partition_windows(x_normed)
        
        # Attention within each window
        attn_out, _ = self.attention(windows, causal=True)
        
        # Merge windows
        x_attn = self._merge_windows(attn_out, n_windows, B, L)
        
        # Boundary gating — blend overlap region information
        if self.overlap > 0 and L > self.window_size:
            # Gate controls information flow at boundaries
            gate_input = torch.cat([x_attn, x_normed], dim=-1)
            gate = self.boundary_gate(gate_input)
            x_attn = gate * x_attn + (1 - gate) * x_normed
        
        x = residual + self.dropout(x_attn)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x
    
    def get_compute_profile(self) -> dict:
        return {
            "name": "Focus (Windowed Attention)",
            "complexity": f"O(n * {self.window_size})",
            "normalized_cost": self.compute_cost,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class GlobalAttentionPathway(nn.Module):
    """
    Reason Pathway — Full Global Attention.
    
    The most expensive but most powerful pathway. Applied only to tokens
    that the router identifies as requiring global context for proper
    processing (logical operators, cross-references, mathematical reasoning).
    
    Architecture:
        Input → LayerNorm → Full Multi-Head Attention (with RoPE)
              → FFN (SwiGLU) → Output
    
    Key: This is standard transformer attention — the innovation is that
    only ~15% of tokens should be routed here, making the amortized cost
    much lower than applying it to all tokens.
    """
    
    def __init__(self, config: HydraConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Full global attention
        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_rotary=config.reason_use_rotary,
            max_seq_len=config.max_seq_len,
        )
        
        # Feed-forward network
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Compute cost: ~10x SSM (full quadratic attention)
        self.compute_cost = 10.0
    
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        
        # Pre-norm + full attention
        x_normed = self.norm1(x)
        attn_out, attn_weights = self.attention(x_normed, causal=True)
        x = residual + self.dropout(attn_out)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        # Store attention weights for analysis (detached)
        self._last_attn_weights = attn_weights.detach()
        
        return x
    
    def get_compute_profile(self) -> dict:
        return {
            "name": "Reason (Global Attention)",
            "complexity": "O(n^2)",
            "normalized_cost": self.compute_cost,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
    
    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights for visualization."""
        return getattr(self, '_last_attn_weights', None)
