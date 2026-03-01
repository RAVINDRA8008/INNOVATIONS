"""
Memory Bandwidth Profiler — The Hidden Cost Reviewers Will Attack

FLOPs alone are a lie. Modern hardware is memory-bandwidth-bound, not compute-bound.
A router that adds 5% FLOPs but 30% memory traffic is a net negative.

What we measure:
    1. Memory Footprint — peak activation memory for each component
    2. Bandwidth Cost — bytes read/written per operation
    3. Arithmetic Intensity — FLOPs / byte transferred (higher = better)
    4. Soft Dispatch Penalty — the 3x memory multiplier from running all pathways

Key insight: In soft dispatch mode (training), HYDRA stores activations for
ALL THREE pathways for EVERY token, even though each token only "uses" one.
This is a 3x activation memory multiplier that must be reported honestly.

At inference with hard routing, the memory story is better — but we still
materialize the full routing tensor (B, L, 3) and pay for router activations.
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
class MemoryBreakdown:
    """Memory cost for a component in bytes."""
    name: str
    parameters_bytes: int       # Model parameters (weights)
    activation_bytes: int       # Forward pass activations
    gradient_bytes: int         # Backward pass gradient storage
    workspace_bytes: int        # Temporary buffers (e.g., attention scores)
    
    @property
    def total_bytes(self) -> int:
        return self.parameters_bytes + self.activation_bytes + self.gradient_bytes + self.workspace_bytes
    
    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "parameters_MB": self.parameters_bytes / 1e6,
            "activations_MB": self.activation_bytes / 1e6,
            "gradients_MB": self.gradient_bytes / 1e6,
            "workspace_MB": self.workspace_bytes / 1e6,
            "total_MB": self.total_bytes / 1e6,
        }


@dataclass
class BandwidthCost:
    """Memory bandwidth cost (bytes read + written)."""
    name: str
    bytes_read: int
    bytes_written: int
    flops: int  # For arithmetic intensity calculation
    
    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written
    
    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte transferred. Higher = more compute-bound (good)."""
        return self.flops / self.total_bytes if self.total_bytes > 0 else float('inf')
    
    @property
    def is_bandwidth_bound(self) -> bool:
        """
        Rule of thumb: if AI < 10 on modern GPUs, you're bandwidth bound.
        A100: ~312 TFLOPS, ~2 TB/s → breakeven AI ≈ 156
        H100: ~989 TFLOPS, ~3.35 TB/s → breakeven AI ≈ 295
        """
        return self.arithmetic_intensity < 50  # Conservative threshold


class MemoryBandwidthProfiler:
    """
    Profiles memory footprint and bandwidth costs for HYDRA components.
    
    Critical measurements:
        1. Parameter memory per component
        2. Activation memory per component (training vs inference)
        3. Bytes transferred per operation (bandwidth)
        4. Arithmetic intensity (compute-to-memory ratio)
        5. Soft dispatch memory penalty
    """
    
    def __init__(self, config: HydraConfig, dtype: torch.dtype = torch.float32):
        self.config = config
        self.bytes_per_element = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(dtype, 4)
        self.dtype = dtype
        self.bpe = self.bytes_per_element
    
    def _param_bytes(self, in_feat: int, out_feat: int, bias: bool = False) -> int:
        """Bytes for a Linear layer's parameters."""
        return (in_feat * out_feat + (out_feat if bias else 0)) * self.bpe
    
    def _conv1d_param_bytes(self, in_ch: int, out_ch: int, kernel: int, groups: int = 1, bias: bool = True) -> int:
        """Bytes for Conv1d parameters."""
        return ((in_ch // groups) * out_ch * kernel + (out_ch if bias else 0)) * self.bpe
    
    def _activation_bytes(self, *dims) -> int:
        """Bytes for an activation tensor of given dimensions."""
        total_elements = 1
        for d in dims:
            total_elements *= d
        return total_elements * self.bpe
    
    # ── Component Memory Profiles ─────────────────────────────────────
    
    def router_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """Memory profile for the Router (ComplexityEstimator + routing head)."""
        d = self.config.d_model
        rh = self.config.router_hidden_dim
        
        # Parameters
        params = 0
        # Content MLP: Linear(d, rh) + Linear(rh, rh)
        params += self._param_bytes(d, rh) + self._param_bytes(rh, rh)
        # Context Conv: Conv1d(d, rh//2, 5) + Conv1d(rh//2, rh, 3)
        params += self._conv1d_param_bytes(d, rh // 2, 5) + self._conv1d_param_bytes(rh // 2, rh, 3)
        # Entropy proj: Linear(d, rh)
        params += self._param_bytes(d, rh)
        # Fusion: Linear(3*rh, rh) + Linear(rh, rh)
        params += self._param_bytes(3 * rh, rh) + self._param_bytes(rh, rh)
        # LayerNorm: 2*rh
        params += 2 * rh * self.bpe
        # Routing head: Linear(rh, rh) + Linear(rh, 3)
        params += self._param_bytes(rh, rh) + self._param_bytes(rh, 3)
        
        # Activations (forward pass intermediates)
        acts = 0
        # Input: (B, L, d)
        acts += self._activation_bytes(batch, seq_len, d)
        # Content features: (B, L, rh)
        acts += self._activation_bytes(batch, seq_len, rh)
        # Context features (transposed): (B, d, L) + (B, L, rh)
        acts += self._activation_bytes(batch, d, seq_len) + self._activation_bytes(batch, seq_len, rh)
        # Entropy: local variance (B, L, d) + projected (B, L, rh)
        acts += self._activation_bytes(batch, seq_len, d) + self._activation_bytes(batch, seq_len, rh)
        # Fused: (B, L, 3*rh) + (B, L, rh)
        acts += self._activation_bytes(batch, seq_len, 3 * rh) + self._activation_bytes(batch, seq_len, rh)
        # Logits: (B, L, 3)
        acts += self._activation_bytes(batch, seq_len, 3)
        # Routing weights: (B, L, 3)
        acts += self._activation_bytes(batch, seq_len, 3)
        
        # Gradients (training only, same size as activations for backprop)
        grads = acts  # Rough approximation: gradient same size as activations
        
        # Workspace: unfold buffer for entropy estimation
        window = min(7, seq_len)
        workspace = self._activation_bytes(batch, seq_len, d, window)
        
        return MemoryBreakdown(
            name="Router",
            parameters_bytes=params,
            activation_bytes=acts,
            gradient_bytes=grads,
            workspace_bytes=workspace,
        )
    
    def ssm_pathway_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """Memory profile for SSM (Stream) pathway."""
        d = self.config.d_model
        d_inner = d * self.config.ssm_expansion_factor
        d_state = self.config.ssm_state_dim
        dt_rank = max(d // 16, 1)
        
        # Parameters
        params = 0
        params += 2 * d * self.bpe  # LayerNorm
        params += self._param_bytes(d, 2 * d_inner, bias=False)  # in_proj
        params += self._conv1d_param_bytes(d_inner, d_inner, 4, groups=d_inner)  # conv1d
        params += d_inner * d_state * self.bpe  # A_log
        params += d_inner * self.bpe  # D
        params += self._param_bytes(d_inner, dt_rank + 2 * d_state, bias=False)  # x_proj
        params += self._param_bytes(dt_rank, d_inner, bias=True)  # dt_proj
        params += self._param_bytes(d_inner, d, bias=False)  # out_proj
        
        # Activations
        acts = 0
        acts += self._activation_bytes(batch, seq_len, d)  # input
        acts += self._activation_bytes(batch, seq_len, 2 * d_inner)  # after in_proj (x, z)
        acts += self._activation_bytes(batch, seq_len, d_inner)  # after conv1d
        acts += self._activation_bytes(batch, seq_len, d_inner)  # after SiLU
        acts += self._activation_bytes(batch, seq_len, dt_rank + 2 * d_state)  # x_dbl
        acts += self._activation_bytes(batch, seq_len, d_inner)  # delta
        # SSM state: (B, d_inner, d_state) — kept across sequence
        acts += self._activation_bytes(batch, d_inner, d_state)
        # SSM output: (B, L, d_inner)
        acts += self._activation_bytes(batch, seq_len, d_inner)
        # Gated output: (B, L, d_inner)
        acts += self._activation_bytes(batch, seq_len, d_inner)
        # Final output: (B, L, d)
        acts += self._activation_bytes(batch, seq_len, d)
        
        # SSM scan: sequential, stores all intermediate states for backprop
        # (B, L, d_inner, d_state) — THIS IS THE BIG ONE
        scan_workspace = self._activation_bytes(batch, seq_len, d_inner, d_state)
        
        return MemoryBreakdown(
            name="SSM Pathway (Stream)",
            parameters_bytes=params,
            activation_bytes=acts,
            gradient_bytes=acts,
            workspace_bytes=scan_workspace,
        )
    
    def windowed_attention_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """Memory profile for Windowed Attention (Focus) pathway."""
        d = self.config.d_model
        h = self.config.n_heads
        d_h = self.config.d_head
        w = min(self.config.window_size, seq_len)
        d_ff = self.config.d_ff
        overlap = min(self.config.window_overlap, w - 1)  # Overlap can't exceed window
        stride = max(1, w - overlap)  # Ensure stride >= 1
        n_windows = max(1, (seq_len - overlap + stride - 1) // stride)
        
        # Parameters
        params = 0
        params += 2 * (2 * d * self.bpe)  # 2x LayerNorm
        params += 4 * self._param_bytes(d, d, bias=False)  # Q,K,V,O projections
        params += self._param_bytes(2 * d, d)  # boundary gate
        params += self._param_bytes(d, d_ff, bias=False) * 2  # FFN w1, w3
        params += self._param_bytes(d_ff, d, bias=False)  # FFN w2
        # RoPE buffers (not parameters, but stored)
        params += 2 * (w * d_h) * self.bpe  # sin/cos cached
        
        # Activations
        acts = 0
        acts += self._activation_bytes(batch, seq_len, d)  # input
        # After windowing: (B*n_windows, w, d)
        acts += self._activation_bytes(batch * n_windows, w, d)
        # Q, K, V: each (B*n_windows, h, w, d_h)
        acts += 3 * self._activation_bytes(batch * n_windows, h, w, d_h)
        # Attention scores: (B*n_windows, h, w, w)
        attn_workspace = self._activation_bytes(batch * n_windows, h, w, w)
        acts += attn_workspace
        # Attention output: (B*n_windows, w, d)
        acts += self._activation_bytes(batch * n_windows, w, d)
        # After merge: (B, L, d)
        acts += self._activation_bytes(batch, seq_len, d)
        # FFN intermediates: (B, L, d_ff) x2 (w1 and w3)
        acts += 2 * self._activation_bytes(batch, seq_len, d_ff)
        
        return MemoryBreakdown(
            name="Windowed Attention (Focus)",
            parameters_bytes=params,
            activation_bytes=acts,
            gradient_bytes=acts,
            workspace_bytes=attn_workspace,  # Attention scores dominate workspace
        )
    
    def global_attention_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """Memory profile for Global Attention (Reason) pathway."""
        d = self.config.d_model
        h = self.config.n_heads
        d_h = self.config.d_head
        d_ff = self.config.d_ff
        
        # Parameters (same structure as windowed but different cached size)
        params = 0
        params += 2 * (2 * d * self.bpe)  # 2x LayerNorm
        params += 4 * self._param_bytes(d, d, bias=False)  # Q,K,V,O
        params += self._param_bytes(d, d_ff, bias=False) * 2  # FFN w1, w3
        params += self._param_bytes(d_ff, d, bias=False)  # FFN w2
        params += 2 * (seq_len * d_h) * self.bpe  # RoPE sin/cos for full seq
        
        # Activations
        acts = 0
        acts += self._activation_bytes(batch, seq_len, d)  # input
        # Q, K, V: (B, h, L, d_h)
        acts += 3 * self._activation_bytes(batch, h, seq_len, d_h)
        # Attention scores: (B, h, L, L) — THIS IS THE QUADRATIC COST
        attn_workspace = self._activation_bytes(batch, h, seq_len, seq_len)
        acts += attn_workspace
        # Attention output: (B, L, d)
        acts += self._activation_bytes(batch, seq_len, d)
        # FFN intermediates
        acts += 2 * self._activation_bytes(batch, seq_len, d_ff)
        
        return MemoryBreakdown(
            name="Global Attention (Reason)",
            parameters_bytes=params,
            activation_bytes=acts,
            gradient_bytes=acts,
            workspace_bytes=attn_workspace,
        )
    
    def cross_mixer_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """Memory profile for Cross-Pathway Mixer (dispatches by mixer_type)."""
        mixer_type = getattr(self.config, 'mixer_type', 'gated_linear')
        if mixer_type == "cross_attention":
            return self._cross_attention_mixer_memory(batch, seq_len)
        else:
            return self._gated_linear_mixer_memory(batch, seq_len)
    
    def _cross_attention_mixer_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """[LEGACY] Memory for O(n²) cross-attention mixer."""
        d = self.config.d_model
        cd = self.config.cross_pathway_dim
        ch = self.config.cross_pathway_heads
        d_ch = cd // ch
        
        params = 0
        params += 2 * d * self.bpe  # LayerNorm
        params += self._param_bytes(d, cd, bias=False)  # Q proj
        params += self._param_bytes(d, cd, bias=False)  # K proj
        params += self._param_bytes(d, cd, bias=False)  # V proj
        params += self._param_bytes(cd, d, bias=False)  # O proj
        
        acts = 0
        acts += self._activation_bytes(batch, seq_len, d)  # combined input
        acts += 3 * self._activation_bytes(batch, seq_len, d)  # 3 pathway outputs
        # Q: (B, ch, L, d_ch), K/V: (B, ch, 3L, d_ch)
        acts += self._activation_bytes(batch, ch, seq_len, d_ch)
        acts += 2 * self._activation_bytes(batch, ch, 3 * seq_len, d_ch)
        # Cross-attention scores: (B, ch, L, 3L)
        attn_workspace = self._activation_bytes(batch, ch, seq_len, 3 * seq_len)
        acts += attn_workspace
        
        return MemoryBreakdown(
            name="Cross-Pathway Mixer",
            parameters_bytes=params,
            activation_bytes=acts,
            gradient_bytes=acts,
            workspace_bytes=attn_workspace,
        )
    
    def _gated_linear_mixer_memory(self, batch: int, seq_len: int) -> MemoryBreakdown:
        """Memory for O(n) gated linear mixer — the new default."""
        d = self.config.d_model
        
        params = 0
        params += 2 * 3 * d * self.bpe  # LayerNorm(3*d): weight + bias
        params += self._param_bytes(3 * d, d, bias=False)  # fuse_proj
        params += self._param_bytes(2 * d, d, bias=True)   # gate_proj (with bias)
        
        acts = 0
        acts += self._activation_bytes(batch, seq_len, d)      # combined input
        acts += self._activation_bytes(batch, seq_len, 3 * d)  # pathway_concat
        acts += self._activation_bytes(batch, seq_len, d)      # fused output
        acts += self._activation_bytes(batch, seq_len, 2 * d)  # gate_input concat
        acts += self._activation_bytes(batch, seq_len, d)      # gate values
        acts += self._activation_bytes(batch, seq_len, d)      # blended output
        
        # No attention scores! No quadratic workspace!
        workspace = 0
        
        return MemoryBreakdown(
            name="Cross-Pathway Mixer",
            parameters_bytes=params,
            activation_bytes=acts,
            gradient_bytes=acts,
            workspace_bytes=workspace,
        )
    
    # ── Bandwidth Analysis ────────────────────────────────────────────
    
    def router_bandwidth(self, batch: int, seq_len: int, flops: int) -> BandwidthCost:
        """Bandwidth cost for router operations."""
        d = self.config.d_model
        rh = self.config.router_hidden_dim
        
        # Reads: input tensor + all weight matrices
        reads = 0
        reads += self._activation_bytes(batch, seq_len, d)  # Read input
        # Read weight matrices (amortized if cached)
        reads += self._param_bytes(d, rh)  # content linear1
        reads += self._param_bytes(rh, rh)  # content linear2
        reads += self._param_bytes(d, rh)  # entropy proj
        reads += self._param_bytes(3 * rh, rh) + self._param_bytes(rh, rh)  # fusion
        reads += self._param_bytes(rh, rh) + self._param_bytes(rh, 3)  # head
        # Read Conv1d weights
        reads += self._conv1d_param_bytes(d, rh // 2, 5) + self._conv1d_param_bytes(rh // 2, rh, 3)
        # Read intermediate activations for entropy (unfold)
        reads += self._activation_bytes(batch, seq_len, d)
        
        # Writes: all intermediate activations + routing weights
        writes = 0
        writes += self._activation_bytes(batch, seq_len, rh) * 3  # 3 features
        writes += self._activation_bytes(batch, seq_len, 3 * rh)  # concat
        writes += self._activation_bytes(batch, seq_len, rh)  # fused
        writes += self._activation_bytes(batch, seq_len, 3)  # logits
        writes += self._activation_bytes(batch, seq_len, 3)  # routing weights
        
        return BandwidthCost(
            name="Router",
            bytes_read=reads,
            bytes_written=writes,
            flops=flops,
        )
    
    def soft_dispatch_bandwidth(self, batch: int, seq_len: int) -> BandwidthCost:
        """
        The HIDDEN bandwidth cost of soft dispatch.
        
        During training, we compute ALL pathways and weight-combine them.
        This means:
            - Read 3 pathway outputs: 3 * (B, L, d)
            - Read routing weights: (B, L, 3)
            - Element-wise multiply + sum: 3 reads + 3 writes per token
            
        This is memory traffic that a dense model doesn't pay.
        """
        d = self.config.d_model
        
        # Read 3 pathway outputs
        reads = 3 * self._activation_bytes(batch, seq_len, d)
        # Read routing weights
        reads += self._activation_bytes(batch, seq_len, 3)
        
        # Write weighted outputs (3 intermediates + 1 final)
        writes = 3 * self._activation_bytes(batch, seq_len, d)  # weighted intermediates
        writes += self._activation_bytes(batch, seq_len, d)  # combined output
        
        return BandwidthCost(
            name="Soft Dispatch Overhead",
            bytes_read=reads,
            bytes_written=writes,
            flops=3 * batch * seq_len * d * 2,  # 3 element-wise mul + add
        )
    
    # ── Full Analysis ─────────────────────────────────────────────────
    
    def full_analysis(
        self,
        batch: int,
        seq_len: int,
        training: bool = True,
    ) -> Dict:
        """
        Complete memory and bandwidth analysis.
        
        Returns honest assessment of memory overhead from routing.
        """
        # Memory breakdowns
        router_mem = self.router_memory(batch, seq_len)
        ssm_mem = self.ssm_pathway_memory(batch, seq_len)
        windowed_mem = self.windowed_attention_memory(batch, seq_len)
        global_mem = self.global_attention_memory(batch, seq_len)
        mixer_mem = self.cross_mixer_memory(batch, seq_len)
        
        # Total memory
        all_components = [router_mem, ssm_mem, windowed_mem, global_mem, mixer_mem]
        
        # Routing-specific memory (the overhead)
        routing_memory = router_mem.total_bytes + mixer_mem.total_bytes
        
        # Soft dispatch penalty: during training, all 3 pathway activations stored
        if training:
            pathway_memory = ssm_mem.total_bytes + windowed_mem.total_bytes + global_mem.total_bytes
            dense_baseline_memory = global_mem.total_bytes  # Dense = just global attention
            soft_dispatch_penalty = pathway_memory - dense_baseline_memory
        else:
            # At inference, only the routed pathway's activations are needed
            # But in practice, we still compute all (implementation limitation)
            soft_dispatch_penalty = 0
            pathway_memory = max(ssm_mem.total_bytes, windowed_mem.total_bytes, global_mem.total_bytes)
            dense_baseline_memory = global_mem.total_bytes
        
        total_hydra_memory = sum(c.total_bytes for c in all_components)
        
        # ─── Gradient Checkpointing Savings ───
        # When checkpoint_pathways is enabled, intermediate activations of
        # checkpointed pathways are freed during forward pass. Only output
        # tensors (B, L, d) are retained. Activations recomputed during backward.
        #
        # What checkpointing saves per pathway:
        #   - activation_bytes: intermediate activations (Q/K/V, FFN, etc.) — freed
        #   - gradient_bytes: activation gradients don't persist between fwd/bwd
        #   - workspace_bytes: attention scores, temp buffers — freed
        #   - MINUS output_tensor_bytes: the pathway output (B,L,d) must be retained
        #
        # The dominant pathway (highest routing weight, typically SSM at 60%)
        # keeps full activations. The 2 expensive pathways (windowed + global)
        # get checkpointed — these are where the big savings come from.
        checkpoint_mode = getattr(self.config, 'checkpoint_pathways', 'none')
        output_tensor_bytes = self._activation_bytes(batch, seq_len, self.config.d_model)
        
        pathway_mems = [ssm_mem, windowed_mem, global_mem]
        checkpoint_savings = 0
        
        if training and checkpoint_mode != 'none':
            # Per-pathway savings = activations + gradients + workspace freed,
            # minus retained output tensor
            per_pathway_savings = [
                max(0, p.activation_bytes + p.gradient_bytes + p.workspace_bytes - output_tensor_bytes)
                for p in pathway_mems
            ]
            
            if checkpoint_mode == 'all':
                # All 3 pathways checkpointed
                checkpoint_savings = sum(per_pathway_savings)
            else:
                # "non_dominant": 2 of 3 checkpointed.
                # In practice, the cheapest pathway (SSM, ~60% routing weight) is
                # dominant and keeps full activations. The 2 MORE expensive pathways
                # (windowed + global) are checkpointed — yielding larger savings.
                sorted_savings = sorted(per_pathway_savings, reverse=True)
                checkpoint_savings = sum(sorted_savings[:2])  # 2 largest savings
        
        effective_hydra_memory = total_hydra_memory - checkpoint_savings
        
        # Bandwidth analysis
        dispatch_bw = self.soft_dispatch_bandwidth(batch, seq_len)
        
        # Dense baseline: just global attention per layer
        dense_total_memory = global_mem.total_bytes * self.config.n_layers
        hydra_total_memory_model = total_hydra_memory * self.config.n_layers
        effective_hydra_memory_model = effective_hydra_memory * self.config.n_layers
        
        raw_overhead_ratio = (hydra_total_memory_model / dense_total_memory - 1) if dense_total_memory > 0 else 0
        effective_overhead_ratio = (effective_hydra_memory_model / dense_total_memory - 1) if dense_total_memory > 0 else 0
        
        return {
            "mode": "training" if training else "inference",
            "dtype": str(self.dtype),
            "bytes_per_element": self.bpe,
            "per_layer": {
                comp.name: comp.to_dict() for comp in all_components
            },
            "per_layer_total_MB": total_hydra_memory / 1e6,
            "per_layer_effective_MB": effective_hydra_memory / 1e6,
            "model_total_MB": hydra_total_memory_model / 1e6,
            "model_effective_MB": effective_hydra_memory_model / 1e6,
            "dense_baseline_total_MB": dense_total_memory / 1e6,
            "routing_overhead": {
                "router_memory_MB": router_mem.total_bytes / 1e6,
                "mixer_memory_MB": mixer_mem.total_bytes / 1e6,
                "soft_dispatch_penalty_MB": soft_dispatch_penalty / 1e6 if training else 0,
                "checkpoint_mode": checkpoint_mode if training else "n/a",
                "checkpoint_savings_MB": checkpoint_savings / 1e6 if training else 0,
                "effective_penalty_MB": max(0, (soft_dispatch_penalty - checkpoint_savings)) / 1e6 if training else 0,
                "total_routing_memory_MB": (routing_memory + (soft_dispatch_penalty if training else 0)) / 1e6,
                "memory_overhead_vs_dense": effective_overhead_ratio,
                "raw_overhead_vs_dense": raw_overhead_ratio,
            },
            "bandwidth": {
                "soft_dispatch": {
                    "bytes_read_MB": dispatch_bw.bytes_read / 1e6,
                    "bytes_written_MB": dispatch_bw.bytes_written / 1e6,
                    "total_MB": dispatch_bw.total_bytes / 1e6,
                    "arithmetic_intensity": dispatch_bw.arithmetic_intensity,
                    "is_bandwidth_bound": dispatch_bw.is_bandwidth_bound,
                },
            },
            "activation_memory_breakdown": {
                comp.name: {
                    "activations_MB": comp.activation_bytes / 1e6,
                    "workspace_MB": comp.workspace_bytes / 1e6,
                } for comp in all_components
            },
        }
