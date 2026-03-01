"""
FLOPs Counter — Exact Operation Counts for Every Component

This is NOT a wrapper around fvcore/DeepSpeed profilers. Those tools count
FLOPs for the whole model but can't decompose "router overhead" vs "useful compute."

We manually count FLOPs using analytical formulas for each operation:
    - Linear(in, out): 2 * in * out FLOPs per token (multiply-add)
    - Conv1d(in, out, k): 2 * in * out * k FLOPs per position
    - LayerNorm(d): 5 * d FLOPs per token (mean, var, normalize, scale, shift)
    - Softmax(d): ~5 * d FLOPs per token
    - MatMul(m,k) @ (k,n): 2 * m * k * n FLOPs
    - GELU/SiLU: ~8 FLOPs per element (exp + mul + add + ...)
    - Gumbel-Softmax: ~10 * d FLOPs (sample + softmax)

Convention: 1 FLOP = 1 multiply-add = 2 floating-point operations.
    We report MACs (multiply-accumulate) and convert to FLOPs where needed.
    All counts are per-token unless stated otherwise.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hydra.model.config import HydraConfig


@dataclass
class ComponentFLOPs:
    """FLOPs breakdown for a single component."""
    name: str
    macs: int  # Multiply-accumulate operations (= FLOPs / 2 in some conventions)
    detail: Dict[str, int] = field(default_factory=dict)
    
    @property
    def flops(self) -> int:
        """Total FLOPs (2 * MACs for matmuls, 1:1 for activations)."""
        return self.macs * 2
    
    @property
    def gflops(self) -> float:
        return self.flops / 1e9
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.flops:,} FLOPs ({self.gflops:.4f} GFLOPs)"


class FLOPsCounter:
    """
    Analytical FLOPs counter for HYDRA components.
    
    Decomposes total FLOPs into:
        1. Router FLOPs (the overhead we're measuring)
        2. SSM Pathway FLOPs
        3. Windowed Attention Pathway FLOPs
        4. Global Attention Pathway FLOPs
        5. Cross-Pathway Mixer FLOPs
        6. Embeddings + LM Head FLOPs
        
    Then computes:
        - Router overhead ratio = router_flops / total_flops
        - Effective savings = 1 - (actual_flops / dense_baseline_flops)
        - Break-even analysis: at what routing distribution does overhead = savings?
    """
    
    def __init__(self, config: HydraConfig):
        self.config = config
        self.d = config.d_model
        self.h = config.n_heads
        self.d_h = config.d_head
        self.d_ff = config.d_ff
        self.n_layers = config.n_layers
        self.vocab = config.vocab_size
        self.d_state = config.ssm_state_dim
        self.ssm_expand = config.ssm_expansion_factor
        self.window_size = config.window_size
        self.router_hidden = config.router_hidden_dim
        self.cross_dim = config.cross_pathway_dim
        self.cross_heads = config.cross_pathway_heads
    
    # ── Elementary operation costs (per token) ────────────────────────
    
    @staticmethod
    def _linear_macs(in_features: int, out_features: int) -> int:
        """MACs for a single Linear layer: in * out per token."""
        return in_features * out_features
    
    @staticmethod
    def _conv1d_macs(in_ch: int, out_ch: int, kernel_size: int, groups: int = 1) -> int:
        """MACs for Conv1d per position."""
        return (in_ch // groups) * out_ch * kernel_size
    
    @staticmethod
    def _layernorm_macs(dim: int) -> int:
        """MACs for LayerNorm: ~5 ops per element (mean, var, norm, γ, β)."""
        return 5 * dim
    
    @staticmethod
    def _activation_macs(dim: int, kind: str = "gelu") -> int:
        """MACs for activation function per element."""
        # GELU ≈ x * σ(1.702x): ~8 ops; SiLU ≈ x * σ(x): ~5 ops
        costs = {"gelu": 8, "silu": 5, "sigmoid": 4, "softmax": 5}
        return costs.get(kind, 8) * dim
    
    @staticmethod
    def _attention_macs(seq_len: int, d_head: int, n_heads: int) -> int:
        """MACs for scaled dot-product attention (Q@K^T + softmax + @V)."""
        # Q@K^T: seq_len * seq_len * d_head per head
        qk = seq_len * seq_len * d_head * n_heads
        # softmax: ~5 * seq_len * seq_len * n_heads  
        soft = 5 * seq_len * seq_len * n_heads
        # attn@V: seq_len * seq_len * d_head per head
        av = seq_len * seq_len * d_head * n_heads
        return qk + soft + av
    
    # ── Component-level FLOPs ─────────────────────────────────────────
    
    def router_flops_per_token(self, seq_len: int) -> ComponentFLOPs:
        """
        EXACT FLOPs for the Router (ComplexityEstimator + routing head).
        
        This is THE number that determines if routing overhead kills the benefit.
        """
        d = self.d
        rh = self.router_hidden
        detail = {}
        
        # ─── ComplexityEstimator ───
        
        # 1. Content MLP: Linear(d, rh) + GELU + Linear(rh, rh) + GELU
        detail["content_linear1"] = self._linear_macs(d, rh)
        detail["content_gelu1"] = self._activation_macs(rh, "gelu")
        detail["content_linear2"] = self._linear_macs(rh, rh)
        detail["content_gelu2"] = self._activation_macs(rh, "gelu")
        
        # 2. Context Conv1D: Conv1d(d, rh//2, k=5) + GELU + Conv1d(rh//2, rh, k=3) + GELU
        detail["context_conv1"] = self._conv1d_macs(d, rh // 2, kernel_size=5)
        detail["context_gelu1"] = self._activation_macs(rh // 2, "gelu")
        detail["context_conv2"] = self._conv1d_macs(rh // 2, rh, kernel_size=3)
        detail["context_gelu2"] = self._activation_macs(rh, "gelu")
        
        # 3. Entropy approximation: unfold variance + Linear(d, rh) + GELU
        # Variance over window of 7: ~2*d*7 per token (sum + sq diff)
        detail["entropy_variance"] = 2 * d * min(7, seq_len)
        detail["entropy_linear"] = self._linear_macs(d, rh)
        detail["entropy_gelu"] = self._activation_macs(rh, "gelu")
        
        # 4. Fusion: Linear(3*rh, rh) + GELU + Linear(rh, rh)
        detail["fusion_linear1"] = self._linear_macs(3 * rh, rh)
        detail["fusion_gelu"] = self._activation_macs(rh, "gelu")
        detail["fusion_linear2"] = self._linear_macs(rh, rh)
        
        # 5. LayerNorm
        detail["fusion_layernorm"] = self._layernorm_macs(rh)
        
        # ─── Routing Head ───
        
        # Linear(rh, rh) + GELU + Linear(rh, 3)
        detail["head_linear1"] = self._linear_macs(rh, rh)
        detail["head_gelu"] = self._activation_macs(rh, "gelu")
        detail["head_linear2"] = self._linear_macs(rh, 3)
        
        # ─── Gumbel-Softmax ───
        # Sample from Gumbel: ~3 ops per logit (uniform → log → log)
        # Softmax: ~5 * 3
        detail["gumbel_softmax"] = 3 * 3 + 5 * 3  # = 24 per token
        
        total = sum(detail.values())
        return ComponentFLOPs(name="Router", macs=total, detail=detail)
    
    def ssm_pathway_flops_per_token(self, seq_len: int) -> ComponentFLOPs:
        """FLOPs for the SSM (Stream) pathway per token."""
        d = self.d
        d_inner = d * self.ssm_expand
        d_state = self.d_state
        dt_rank = max(d // 16, 1)
        detail = {}
        
        # LayerNorm
        detail["norm"] = self._layernorm_macs(d)
        
        # in_proj: Linear(d, 2*d_inner)
        detail["in_proj"] = self._linear_macs(d, 2 * d_inner)
        
        # Depthwise Conv1d(d_inner, d_inner, k=4, groups=d_inner)
        detail["conv1d"] = self._conv1d_macs(d_inner, d_inner, kernel_size=4, groups=d_inner)
        
        # SiLU activation
        detail["silu_conv"] = self._activation_macs(d_inner, "silu")
        
        # SelectiveSSM:
        # x_proj: Linear(d_inner, dt_rank + 2*d_state)
        detail["ssm_x_proj"] = self._linear_macs(d_inner, dt_rank + 2 * d_state)
        # dt_proj: Linear(dt_rank, d_inner)
        detail["ssm_dt_proj"] = self._linear_macs(dt_rank, d_inner)
        # softplus on dt: ~5 ops per element
        detail["ssm_softplus"] = 5 * d_inner
        # Discretization: exp(delta*A) → d_inner * d_state muls + exp
        detail["ssm_discretize"] = d_inner * d_state * 10  # mul + exp
        # State update per step: deltaA*x + deltaB_u → d_inner*d_state*2
        detail["ssm_scan_update"] = d_inner * d_state * 2
        # Output projection: (x * C).sum(-1) → d_inner*d_state
        detail["ssm_scan_output"] = d_inner * d_state
        # D skip connection: d_inner muls
        detail["ssm_skip"] = d_inner
        
        # Gated output: x * SiLU(z)
        detail["silu_gate"] = self._activation_macs(d_inner, "silu")
        detail["gate_mul"] = d_inner
        
        # out_proj: Linear(d_inner, d)
        detail["out_proj"] = self._linear_macs(d_inner, d)
        
        total = sum(detail.values())
        return ComponentFLOPs(name="SSM Pathway (Stream)", macs=total, detail=detail)
    
    def windowed_attention_flops_per_token(self, seq_len: int) -> ComponentFLOPs:
        """FLOPs for the Windowed Attention (Focus) pathway per token."""
        d = self.d
        w = min(self.window_size, seq_len)  # Effective window size
        detail = {}
        
        # LayerNorm x2
        detail["norm1"] = self._layernorm_macs(d)
        detail["norm2"] = self._layernorm_macs(d)
        
        # QKV projections: 3 * Linear(d, d) — amortized per token
        detail["qkv_proj"] = 3 * self._linear_macs(d, d)
        
        # RoPE: ~4 * d_head * n_heads per token (sin, cos, mul, add)
        detail["rope"] = 4 * self.d_h * self.h
        
        # Windowed attention: each token attends to w tokens
        # Q@K^T: w * d_head per head per token
        detail["attn_qk"] = w * self.d_h * self.h
        # Softmax: ~5 * w * n_heads
        detail["attn_softmax"] = 5 * w * self.h
        # Attn@V: w * d_head per head per token
        detail["attn_av"] = w * self.d_h * self.h
        
        # Output projection: Linear(d, d)
        detail["attn_o_proj"] = self._linear_macs(d, d)
        
        # Boundary gate (if applicable): Linear(2*d, d) + Sigmoid
        detail["boundary_gate"] = self._linear_macs(2 * d, d) + self._activation_macs(d, "sigmoid")
        
        # SwiGLU FFN: Linear(d, d_ff) + SiLU + Linear(d, d_ff) + mul + Linear(d_ff, d)
        detail["ffn_w1"] = self._linear_macs(d, self.d_ff)
        detail["ffn_silu"] = self._activation_macs(self.d_ff, "silu")
        detail["ffn_w3"] = self._linear_macs(d, self.d_ff)
        detail["ffn_gate_mul"] = self.d_ff  # element-wise multiply
        detail["ffn_w2"] = self._linear_macs(self.d_ff, d)
        
        total = sum(detail.values())
        return ComponentFLOPs(name="Windowed Attention (Focus)", macs=total, detail=detail)
    
    def global_attention_flops_per_token(self, seq_len: int) -> ComponentFLOPs:
        """FLOPs for the Global Attention (Reason) pathway per token."""
        d = self.d
        detail = {}
        
        # LayerNorm x2
        detail["norm1"] = self._layernorm_macs(d)
        detail["norm2"] = self._layernorm_macs(d)
        
        # QKV projections
        detail["qkv_proj"] = 3 * self._linear_macs(d, d)
        
        # RoPE
        detail["rope"] = 4 * self.d_h * self.h
        
        # FULL attention: each token attends to ALL seq_len tokens
        detail["attn_qk"] = seq_len * self.d_h * self.h
        detail["attn_softmax"] = 5 * seq_len * self.h
        detail["attn_av"] = seq_len * self.d_h * self.h
        
        # Output projection
        detail["attn_o_proj"] = self._linear_macs(d, d)
        
        # SwiGLU FFN
        detail["ffn_w1"] = self._linear_macs(d, self.d_ff)
        detail["ffn_silu"] = self._activation_macs(self.d_ff, "silu")
        detail["ffn_w3"] = self._linear_macs(d, self.d_ff)
        detail["ffn_gate_mul"] = self.d_ff
        detail["ffn_w2"] = self._linear_macs(self.d_ff, d)
        
        total = sum(detail.values())
        return ComponentFLOPs(name="Global Attention (Reason)", macs=total, detail=detail)
    
    def cross_mixer_flops_per_token(self, seq_len: int) -> ComponentFLOPs:
        """FLOPs for the Cross-Pathway Mixer (dispatches by mixer_type)."""
        mixer_type = getattr(self.config, 'mixer_type', 'gated_linear')
        if mixer_type == "cross_attention":
            return self._cross_attention_mixer_flops(seq_len)
        else:
            return self._gated_linear_mixer_flops(seq_len)
    
    def _cross_attention_mixer_flops(self, seq_len: int) -> ComponentFLOPs:
        """[LEGACY] FLOPs for the O(n²) cross-attention mixer."""
        d = self.d
        cd = self.cross_dim
        ch = self.cross_heads
        d_ch = cd // ch
        detail = {}
        
        detail["norm"] = self._layernorm_macs(d)
        detail["q_proj"] = self._linear_macs(d, cd)
        detail["k_proj"] = 3 * self._linear_macs(d, cd)
        detail["v_proj"] = 3 * self._linear_macs(d, cd)
        detail["cross_attn_qk"] = 3 * seq_len * d_ch * ch
        detail["cross_attn_softmax"] = 5 * 3 * seq_len * ch
        detail["cross_attn_av"] = 3 * seq_len * d_ch * ch
        detail["o_proj"] = self._linear_macs(cd, d)
        
        total = sum(detail.values())
        return ComponentFLOPs(name="Cross-Pathway Mixer (cross-attn)", macs=total, detail=detail)
    
    def _gated_linear_mixer_flops(self, seq_len: int) -> ComponentFLOPs:
        """FLOPs for the O(n) gated linear mixer — the new default."""
        d = self.d
        detail = {}
        
        # LayerNorm on concatenated pathway outputs (dim = 3*d)
        detail["norm"] = self._layernorm_macs(3 * d)
        
        # fuse_proj: Linear(3*d, d, bias=False) — the main projection
        detail["fuse_proj"] = self._linear_macs(3 * d, d)
        
        # gate_proj: Linear(2*d, d, bias=True) — gating mechanism
        detail["gate_proj"] = self._linear_macs(2 * d, d) + d  # +d for bias add
        
        # Sigmoid activation: ~4 ops per element
        detail["sigmoid"] = 4 * d
        
        # Element-wise: gate*fused + (1-gate)*combined + residual
        # 3 element-wise multiplies + 2 adds = 5*d
        detail["elementwise_blend"] = 5 * d
        
        total = sum(detail.values())
        return ComponentFLOPs(name="Cross-Pathway Mixer (gated-linear)", macs=total, detail=detail)
    
    def embedding_flops(self, seq_len: int) -> ComponentFLOPs:
        """FLOPs for embedding lookup + LM head."""
        detail = {}
        # Embedding lookup: essentially free (gather)
        detail["token_embed"] = 0  # Lookup, not matmul
        detail["position_embed"] = 0
        # LM Head: Linear(d, vocab) — this is actually significant
        detail["lm_head"] = self._linear_macs(self.d, self.vocab)
        # Final LayerNorm
        detail["final_norm"] = self._layernorm_macs(self.d)
        
        total = sum(detail.values())
        return ComponentFLOPs(name="Embeddings + LM Head", macs=total, detail=detail)
    
    # ── Full Model Analysis ───────────────────────────────────────────
    
    def full_analysis(
        self,
        seq_len: int,
        stream_frac: float = 0.6,
        focus_frac: float = 0.25,
        reason_frac: float = 0.15,
    ) -> Dict:
        """
        Complete FLOPs analysis for HYDRA vs a dense baseline.
        
        Args:
            seq_len: Sequence length to analyze
            stream_frac: Fraction of tokens routed to Stream (SSM)
            focus_frac: Fraction of tokens routed to Focus (Windowed)
            reason_frac: Fraction of tokens routed to Reason (Global)
            
        Returns:
            Comprehensive analysis dict with honest overhead assessment.
        """
        assert abs(stream_frac + focus_frac + reason_frac - 1.0) < 1e-6, \
            "Routing fractions must sum to 1.0"
        
        # Per-token FLOPs for each component
        router = self.router_flops_per_token(seq_len)
        ssm = self.ssm_pathway_flops_per_token(seq_len)
        windowed = self.windowed_attention_flops_per_token(seq_len)
        global_attn = self.global_attention_flops_per_token(seq_len)
        mixer = self.cross_mixer_flops_per_token(seq_len)
        embed = self.embedding_flops(seq_len)
        
        # ─── Mixer Frequency ───
        # Mixer is NOT applied every layer — only on layers where
        # layer_idx % mixer_frequency == mixer_frequency - 1 OR final layer.
        mixer_frequency = getattr(self.config, 'mixer_frequency', 1)
        n_mixer_layers = 0
        for i in range(self.n_layers):
            if i % mixer_frequency == mixer_frequency - 1 or i == self.n_layers - 1:
                n_mixer_layers += 1
        mixer_freq_fraction = n_mixer_layers / self.n_layers if self.n_layers > 0 else 1.0
        
        # Amortized mixer cost per layer
        amortized_mixer_macs = mixer.macs * mixer_freq_fraction
        
        # ─── TRAINING MODE (soft dispatch: all pathways for all tokens) ───
        # This is the WORST case — no compute savings during training
        train_per_token_per_layer = (
            router.macs           # Router overhead (always paid)
            + ssm.macs            # ALL tokens through SSM
            + windowed.macs       # ALL tokens through windowed
            + global_attn.macs    # ALL tokens through global
            + amortized_mixer_macs  # Cross-pathway mixing (amortized by frequency)
        )
        train_total = (
            train_per_token_per_layer * self.n_layers * seq_len
            + embed.macs * seq_len
        )
        
        # ─── INFERENCE MODE (hard routing: only selected pathway) ───
        # This is where savings should appear
        infer_per_token_per_layer = (
            router.macs           # Router overhead (ALWAYS paid)
            + stream_frac * ssm.macs
            + focus_frac * windowed.macs
            + reason_frac * global_attn.macs
            + amortized_mixer_macs  # Mixer amortized by frequency
        )
        infer_total = (
            infer_per_token_per_layer * self.n_layers * seq_len
            + embed.macs * seq_len
        )
        
        # ─── DENSE BASELINE (Global Attention for all tokens, no router) ───
        # Standard transformer: global attention + FFN for every token
        dense_per_token_per_layer = global_attn.macs  # No router, no routing
        dense_total = (
            dense_per_token_per_layer * self.n_layers * seq_len
            + embed.macs * seq_len
        )
        
        # ─── SSM-ONLY BASELINE (if you just used SSM for everything) ───
        ssm_only_total = (
            ssm.macs * self.n_layers * seq_len
            + embed.macs * seq_len
        )
        
        # ─── OVERHEAD ANALYSIS ───
        router_total_macs = router.macs * self.n_layers * seq_len
        mixer_total_macs = int(amortized_mixer_macs * self.n_layers * seq_len)
        routing_overhead_macs = router_total_macs + mixer_total_macs  # Both are "routing tax"
        
        # Compute the "useful" FLOPs (pathway compute that does actual work)
        useful_infer_macs = infer_total - routing_overhead_macs - embed.macs * seq_len
        
        # Overhead ratios
        router_overhead_ratio = router_total_macs / infer_total if infer_total > 0 else 0
        total_overhead_ratio = routing_overhead_macs / infer_total if infer_total > 0 else 0
        
        # Savings vs dense
        infer_savings_vs_dense = 1.0 - (infer_total / dense_total) if dense_total > 0 else 0
        train_overhead_vs_dense = (train_total / dense_total) - 1.0 if dense_total > 0 else 0
        
        # ─── BREAK-EVEN ANALYSIS ───
        # At what routing distribution does overhead = savings?
        # Find the max reason_frac where HYDRA is still cheaper than dense
        break_even = self._compute_break_even(seq_len, router, ssm, windowed, global_attn, mixer, mixer_freq_fraction)
        
        return {
            "seq_len": seq_len,
            "routing_distribution": {
                "stream": stream_frac,
                "focus": focus_frac,
                "reason": reason_frac,
            },
            "mixer_info": {
                "mixer_type": getattr(self.config, 'mixer_type', 'gated_linear'),
                "mixer_frequency": mixer_frequency,
                "n_mixer_layers": n_mixer_layers,
                "mixer_freq_fraction": mixer_freq_fraction,
                "raw_mixer_macs_per_token": mixer.macs,
                "amortized_mixer_macs_per_token": amortized_mixer_macs,
            },
            "per_token_per_layer_macs": {
                "router": router.macs,
                "ssm_pathway": ssm.macs,
                "windowed_pathway": windowed.macs,
                "global_pathway": global_attn.macs,
                "cross_mixer": mixer.macs,
                "cross_mixer_amortized": amortized_mixer_macs,
            },
            "per_token_per_layer_detail": {
                "router": router.detail,
                "ssm_pathway": ssm.detail,
                "windowed_pathway": windowed.detail,
                "global_pathway": global_attn.detail,
                "cross_mixer": mixer.detail,
            },
            "total_macs": {
                "training_mode": train_total,
                "inference_mode": infer_total,
                "dense_baseline": dense_total,
                "ssm_only_baseline": ssm_only_total,
            },
            "total_flops": {
                "training_mode": train_total * 2,
                "inference_mode": infer_total * 2,
                "dense_baseline": dense_total * 2,
                "ssm_only_baseline": ssm_only_total * 2,
            },
            "overhead": {
                "router_macs": router_total_macs,
                "mixer_macs": mixer_total_macs,
                "total_routing_overhead_macs": routing_overhead_macs,
                "router_overhead_ratio": router_overhead_ratio,
                "total_overhead_ratio": total_overhead_ratio,
            },
            "savings": {
                "inference_vs_dense": infer_savings_vs_dense,
                "training_overhead_vs_dense": train_overhead_vs_dense,
            },
            "break_even": break_even,
            "component_flops": {
                "router": router,
                "ssm": ssm,
                "windowed": windowed,
                "global_attn": global_attn,
                "mixer": mixer,
                "embed": embed,
            },
            "verdict": self._verdict(
                infer_savings_vs_dense, train_overhead_vs_dense,
                router_overhead_ratio, total_overhead_ratio
            ),
        }
    
    def _compute_break_even(self, seq_len, router, ssm, windowed, global_attn, mixer, mixer_freq_fraction=1.0):
        """Find the routing distribution break-even point vs dense baseline."""
        dense = global_attn.macs  # Dense baseline per-token-per-layer
        overhead = router.macs + mixer.macs * mixer_freq_fraction  # Fixed routing overhead per-token-per-layer
        
        # HYDRA cost = overhead + s*SSM + f*Windowed + r*Global
        # Dense cost = Global
        # Break-even: overhead + s*SSM + f*Windowed + r*Global = Global
        # With s+f+r=1: overhead + s*SSM + f*Windowed + (1-s-f)*Global = Global
        # → overhead + s*(SSM-Global) + f*(Windowed-Global) = 0
        # → s*(Global-SSM) + f*(Global-Windowed) = overhead
        
        ssm_saving = dense - ssm.macs  # Saving per token routed to SSM
        wind_saving = dense - windowed.macs  # Saving per token routed to Windowed
        
        # If ALL tokens go to SSM (s=1, f=0, r=0):
        max_saving_per_token = ssm_saving
        
        # Break-even with pure SSM routing: overhead = s * ssm_saving
        if ssm_saving > 0:
            min_stream_frac_break_even = overhead / ssm_saving
        else:
            min_stream_frac_break_even = float('inf')  # SSM more expensive than dense!
        
        # Max reason fraction where HYDRA still wins (with remaining split 70/30 stream/focus)
        # overhead + s*SSM + f*Windowed + r*Global ≤ Global
        # With f = 0: overhead + (1-r)*SSM + r*Global ≤ Global → overhead ≤ (1-r)*(Global-SSM)
        # → r ≤ 1 - overhead/(Global-SSM)
        if ssm_saving > 0:
            max_reason_frac = 1.0 - overhead / ssm_saving
            max_reason_frac = max(0.0, min(1.0, max_reason_frac))
        else:
            max_reason_frac = 0.0
        
        return {
            "overhead_per_token_per_layer": overhead,
            "ssm_saving_per_token_vs_dense": ssm_saving,
            "windowed_saving_per_token_vs_dense": wind_saving,
            "min_stream_frac_to_break_even": min(1.0, min_stream_frac_break_even),
            "max_reason_frac_with_ssm_rest": max_reason_frac,
            "overhead_as_pct_of_dense": (overhead / dense * 100) if dense > 0 else float('inf'),
        }
    
    def _verdict(self, infer_savings, train_overhead, router_ratio, total_ratio):
        """Honest assessment of whether the routing overhead is acceptable."""
        issues = []
        strengths = []
        
        # Router overhead assessment
        if router_ratio > 0.15:
            issues.append(
                f"CRITICAL: Router alone consumes {router_ratio:.1%} of inference FLOPs. "
                f"Consider reducing router_hidden_dim or simplifying ComplexityEstimator."
            )
        elif router_ratio > 0.08:
            issues.append(
                f"WARNING: Router consumes {router_ratio:.1%} of inference FLOPs. "
                f"Acceptable but worth optimizing."
            )
        else:
            strengths.append(
                f"Router overhead is low ({router_ratio:.1%} of inference FLOPs)."
            )
        
        # Total overhead (router + mixer)
        if total_ratio > 0.25:
            issues.append(
                f"CRITICAL: Total routing overhead (router+mixer) is {total_ratio:.1%} "
                f"of inference FLOPs. This significantly erodes savings."
            )
        elif total_ratio > 0.15:
            issues.append(
                f"WARNING: Total routing overhead is {total_ratio:.1%}. "
                f"Consider disabling cross-pathway mixer or reducing its dimension."
            )
        
        # Inference savings
        if infer_savings < 0:
            issues.append(
                f"FATAL: HYDRA is {-infer_savings:.1%} MORE expensive than dense baseline "
                f"at inference! Routing overhead exceeds savings."
            )
        elif infer_savings < 0.2:
            issues.append(
                f"WARNING: Only {infer_savings:.1%} savings vs dense. "
                f"Not enough to justify routing complexity. Target: 40-60%."
            )
        elif infer_savings < 0.4:
            strengths.append(
                f"Moderate savings of {infer_savings:.1%} vs dense baseline."
            )
        else:
            strengths.append(
                f"Strong savings of {infer_savings:.1%} vs dense baseline."
            )
        
        # Training cost
        if train_overhead > 2.0:
            issues.append(
                f"CRITICAL: Training is {train_overhead:.1%} more expensive than dense. "
                f"Consider conditional computation during training."
            )
        elif train_overhead > 0.5:
            issues.append(
                f"NOTE: Training is {train_overhead:.1%} more expensive than dense "
                f"(expected due to soft dispatch through all pathways)."
            )
        
        overall = "PASS" if infer_savings >= 0.3 and total_ratio < 0.25 else (
            "MARGINAL" if infer_savings >= 0.1 else "FAIL"
        )
        
        return {
            "overall": overall,
            "issues": issues,
            "strengths": strengths,
        }
    
    def scaling_analysis(
        self,
        seq_lengths: List[int] = None,
        stream_frac: float = 0.6,
        focus_frac: float = 0.25,
        reason_frac: float = 0.15,
    ) -> List[Dict]:
        """Analyze how overhead scales with sequence length."""
        if seq_lengths is None:
            seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        
        results = []
        for sl in seq_lengths:
            analysis = self.full_analysis(sl, stream_frac, focus_frac, reason_frac)
            results.append({
                "seq_len": sl,
                "inference_gflops": analysis["total_flops"]["inference_mode"] / 1e9,
                "dense_gflops": analysis["total_flops"]["dense_baseline"] / 1e9,
                "savings_pct": analysis["savings"]["inference_vs_dense"] * 100,
                "router_overhead_pct": analysis["overhead"]["router_overhead_ratio"] * 100,
                "total_overhead_pct": analysis["overhead"]["total_overhead_ratio"] * 100,
            })
        
        return results
