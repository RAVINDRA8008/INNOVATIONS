"""
Cache Efficiency Analyzer — The Third Attack Vector

Even if FLOPs and bandwidth look good on paper, cache inefficiency kills
real-world throughput. This module measures:

1. Data Locality Penalty
   - Dense models process one pathway per layer → data stays in cache
   - HYDRA runs 3 pathways → each evicts the others from L1/L2 cache
   - The cache thrashing from pathway switching is a real cost

2. Working Set Size
   - How much data must be "hot" (in cache) at any moment?
   - HYDRA's working set = router params + selected pathway params + mixer params
   - Dense working set = just attention + FFN params
   - If HYDRA's working set exceeds L2 cache, it's game over

3. Memory Access Pattern Analysis
   - Router: strided access (Conv1d) + random access (unfold) 
   - SSM: sequential scan (good locality)
   - Attention: random access (attention scores are data-dependent)
   - Soft dispatch: 3 independent tensor reads → interleaved (bad locality)

4. Actual Runtime Profiling (optional, requires CUDA)
   - Use torch.cuda.Event for real timing
   - Compare HYDRA block vs dense transformer block wall-clock time
   - This is the ground truth that overrides all analytical estimates
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hydra.model.config import HydraConfig


# Typical cache sizes for modern hardware (bytes)
CACHE_SPECS = {
    "cpu_laptop": {
        "L1": 64 * 1024,        # 64 KB
        "L2": 512 * 1024,       # 512 KB
        "L3": 8 * 1024 * 1024,  # 8 MB
    },
    "cpu_server": {
        "L1": 64 * 1024,
        "L2": 1024 * 1024,      # 1 MB
        "L3": 32 * 1024 * 1024, # 32 MB
    },
    "gpu_a100": {
        "L1": 192 * 1024,       # 192 KB per SM
        "L2": 40 * 1024 * 1024, # 40 MB total
        "HBM": 80 * 1024 * 1024 * 1024,  # 80 GB HBM2e
        "SRAM": 20 * 1024 * 1024,  # 20 MB total SRAM
    },
    "gpu_h100": {
        "L1": 256 * 1024,       # 256 KB per SM
        "L2": 50 * 1024 * 1024, # 50 MB total
        "HBM": 80 * 1024 * 1024 * 1024,  # 80 GB HBM3
        "SRAM": 50 * 1024 * 1024,  # 50 MB total SRAM
    },
}


@dataclass
class WorkingSetAnalysis:
    """Working set size for a computation phase."""
    phase_name: str
    parameter_bytes: int      # Weights that must be resident
    activation_bytes: int     # Active input/output tensors
    workspace_bytes: int      # Temporary computation buffers
    
    @property
    def total_bytes(self) -> int:
        return self.parameter_bytes + self.activation_bytes + self.workspace_bytes
    
    @property
    def total_kb(self) -> float:
        return self.total_bytes / 1024
    
    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)
    
    def fits_in_cache(self, cache_size_bytes: int) -> bool:
        return self.total_bytes <= cache_size_bytes


@dataclass
class CacheAnalysisResult:
    """Complete cache efficiency analysis."""
    working_sets: List[WorkingSetAnalysis]
    total_working_set_bytes: int
    cache_fits: Dict[str, Dict[str, bool]]  # hardware → cache_level → fits
    locality_score: float  # 0.0 = terrible, 1.0 = perfect
    pathway_switching_penalty: float  # Estimated cache miss rate increase
    runtime_measurements: Optional[Dict] = None


class CacheEfficiencyAnalyzer:
    """
    Analyzes cache efficiency and data locality for HYDRA vs dense models.
    
    The key insight: HYDRA must load parameters for 3 different pathways
    per layer, while a dense model loads parameters for just 1 pathway.
    This means 3x the parameter working set, which may exceed cache.
    """
    
    def __init__(self, config: HydraConfig, dtype: torch.dtype = torch.float32):
        self.config = config
        self.bpe = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    
    def _param_bytes(self, *dims) -> int:
        result = self.bpe
        for d in dims:
            result *= d
        return result
    
    # ── Working Set Analysis ──────────────────────────────────────────
    
    def router_working_set(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """Working set for executing the router."""
        d = self.config.d_model
        rh = self.config.router_hidden_dim
        
        # Parameters that must be in cache for router execution
        params = 0
        params += self._param_bytes(d, rh)          # content_linear1
        params += self._param_bytes(rh, rh)          # content_linear2
        params += self._param_bytes(d * 5, rh // 2)  # context_conv1 (approx)
        params += self._param_bytes(rh // 2 * 3, rh) # context_conv2 (approx)
        params += self._param_bytes(d, rh)            # entropy_proj
        params += self._param_bytes(3 * rh, rh)       # fusion1
        params += self._param_bytes(rh, rh)            # fusion2
        params += self._param_bytes(rh, rh)            # head1
        params += self._param_bytes(rh, 3)             # head2
        
        # Active tensors: input + intermediates for current position
        # In streaming mode, we process one position at a time
        acts = self._param_bytes(batch, seq_len, d)  # input
        acts += self._param_bytes(batch, seq_len, rh)  # current output
        
        # Workspace: entropy unfold buffer
        workspace = self._param_bytes(batch, min(7, seq_len), d)
        
        return WorkingSetAnalysis("Router", params, acts, workspace)
    
    def ssm_working_set(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """Working set for SSM pathway."""
        d = self.config.d_model
        d_inner = d * self.config.ssm_expansion_factor
        d_state = self.config.ssm_state_dim
        dt_rank = max(d // 16, 1)
        
        params = 0
        params += self._param_bytes(d, 2 * d_inner)       # in_proj
        params += self._param_bytes(d_inner, 4)            # conv1d (depthwise)
        params += self._param_bytes(d_inner, d_state)      # A_log
        params += self._param_bytes(d_inner)               # D
        params += self._param_bytes(d_inner, dt_rank + 2 * d_state)  # x_proj
        params += self._param_bytes(dt_rank, d_inner)      # dt_proj
        params += self._param_bytes(d_inner, d)            # out_proj
        
        # SSM has GOOD locality: sequential scan, state is persistent
        acts = self._param_bytes(batch, d_inner)  # current input position
        acts += self._param_bytes(batch, d_inner, d_state)  # state (persistent)
        
        workspace = self._param_bytes(batch, d_inner)  # temp for gating
        
        return WorkingSetAnalysis("SSM Pathway", params, acts, workspace)
    
    def windowed_attention_working_set(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """Working set for windowed attention pathway."""
        d = self.config.d_model
        w = min(self.config.window_size, seq_len)
        h = self.config.n_heads
        d_h = self.config.d_head
        d_ff = self.config.d_ff
        
        params = 0
        params += 4 * self._param_bytes(d, d)  # Q, K, V, O projections
        params += self._param_bytes(d, d_ff) * 2 + self._param_bytes(d_ff, d)  # FFN
        params += self._param_bytes(2 * d, d)  # boundary gate
        params += 4 * self._param_bytes(d)  # LayerNorms
        
        # Windowed attention has MODERATE locality:
        # Attention score matrix is w*w, manageable
        acts = self._param_bytes(batch, w, d)  # one window input
        acts += 3 * self._param_bytes(batch, h, w, d_h)  # Q, K, V for window
        acts += self._param_bytes(batch, h, w, w)  # attention scores
        
        workspace = self._param_bytes(batch, w, d_ff)  # FFN intermediate
        
        return WorkingSetAnalysis("Windowed Attention", params, acts, workspace)
    
    def global_attention_working_set(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """Working set for global attention pathway."""
        d = self.config.d_model
        h = self.config.n_heads
        d_h = self.config.d_head
        d_ff = self.config.d_ff
        
        params = 0
        params += 4 * self._param_bytes(d, d)  # Q, K, V, O
        params += self._param_bytes(d, d_ff) * 2 + self._param_bytes(d_ff, d)  # FFN
        params += 4 * self._param_bytes(d)  # LayerNorms
        
        # Global attention has POOR locality for long sequences:
        # Attention score matrix is L*L — grows quadratically
        acts = self._param_bytes(batch, seq_len, d)  # full input
        acts += 3 * self._param_bytes(batch, h, seq_len, d_h)  # Q, K, V
        acts += self._param_bytes(batch, h, seq_len, seq_len)  # FULL attention matrix
        
        workspace = self._param_bytes(batch, seq_len, d_ff)  # FFN intermediate
        
        return WorkingSetAnalysis("Global Attention", params, acts, workspace)
    
    def mixer_working_set(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """Working set for cross-pathway mixer (dispatches by mixer_type)."""
        mixer_type = getattr(self.config, 'mixer_type', 'gated_linear')
        if mixer_type == "cross_attention":
            return self._cross_attention_mixer_ws(batch, seq_len)
        else:
            return self._gated_linear_mixer_ws(batch, seq_len)
    
    def _cross_attention_mixer_ws(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """[LEGACY] Working set for O(n²) cross-attention mixer."""
        d = self.config.d_model
        cd = self.config.cross_pathway_dim
        ch = self.config.cross_pathway_heads
        d_ch = cd // ch
        
        params = 0
        params += self._param_bytes(d, cd) * 3  # Q, K, V projections
        params += self._param_bytes(cd, d)  # O projection
        params += 2 * self._param_bytes(d)  # LayerNorm
        
        # Cross-attention: Q from L tokens, K/V from 3L tokens
        acts = self._param_bytes(batch, seq_len, d) * 4  # combined + 3 pathway outputs
        acts += self._param_bytes(batch, ch, seq_len, 3 * seq_len)  # cross-attention scores
        
        workspace = self._param_bytes(batch, seq_len, cd)  # output intermediate
        
        return WorkingSetAnalysis("Cross-Pathway Mixer", params, acts, workspace)
    
    def _gated_linear_mixer_ws(self, batch: int, seq_len: int) -> WorkingSetAnalysis:
        """Working set for O(n) gated linear mixer — the new default."""
        d = self.config.d_model
        
        params = 0
        params += 2 * self._param_bytes(3 * d)    # LayerNorm(3*d)
        params += self._param_bytes(3 * d, d)       # fuse_proj
        params += self._param_bytes(2 * d, d) + self._param_bytes(d)  # gate_proj + bias
        
        # Active tensors: pathway concat + fused + gate_input + gate + output
        acts = self._param_bytes(batch, seq_len, 3 * d)  # pathway_concat
        acts += self._param_bytes(batch, seq_len, d)      # fused
        acts += self._param_bytes(batch, seq_len, 2 * d)  # gate_input
        acts += self._param_bytes(batch, seq_len, d)      # gate
        
        workspace = self._param_bytes(batch, seq_len, d)  # blended output
        
        return WorkingSetAnalysis("Cross-Pathway Mixer", params, acts, workspace)
    
    # ── Cache Analysis ────────────────────────────────────────────────
    
    def analyze_cache_efficiency(
        self,
        batch: int,
        seq_len: int,
        hardware: str = "gpu_a100",
    ) -> CacheAnalysisResult:
        """
        Full cache efficiency analysis.
        
        Compares working set sizes against cache hierarchy levels.
        """
        # Compute working sets
        working_sets = [
            self.router_working_set(batch, seq_len),
            self.ssm_working_set(batch, seq_len),
            self.windowed_attention_working_set(batch, seq_len),
            self.global_attention_working_set(batch, seq_len),
            self.mixer_working_set(batch, seq_len),
        ]
        
        # Dense baseline: just global attention
        dense_ws = self.global_attention_working_set(batch, seq_len)
        
        # Total working set for one HydraBlock pass
        # In soft dispatch: ALL working sets are needed
        total_ws = sum(ws.total_bytes for ws in working_sets)
        
        # Check against cache hierarchy
        caches = CACHE_SPECS.get(hardware, CACHE_SPECS["gpu_a100"])
        cache_fits = {}
        for level, size in caches.items():
            cache_fits[level] = {}
            for ws in working_sets:
                cache_fits[level][ws.phase_name] = ws.fits_in_cache(size)
            cache_fits[level]["Total HYDRA Block"] = total_ws <= size
            cache_fits[level]["Dense Baseline"] = dense_ws.total_bytes <= size
        
        # ─── Data Locality Score ───
        # Score based on how many components fit in L2 cache
        l2_size = caches.get("L2", 40 * 1024 * 1024)
        components_in_l2 = sum(1 for ws in working_sets if ws.fits_in_cache(l2_size))
        locality_base = components_in_l2 / len(working_sets)
        
        # ─── Pathway Switching Penalty ───
        # When we switch between pathways, the new pathway's parameters
        # must be loaded into cache, evicting the previous pathway's data.
        # In soft dispatch: 3 switches per token per layer.
        # Penalty = (bytes that must be reloaded / total compute bytes)
        
        ssm_params = working_sets[1].parameter_bytes
        wind_params = working_sets[2].parameter_bytes
        glob_params = working_sets[3].parameter_bytes
        
        # If pathways don't all fit in L2, each switch causes cache misses
        total_pathway_params = ssm_params + wind_params + glob_params
        if total_pathway_params > l2_size:
            # Estimate: fraction of params that must be reloaded per switch
            overflow = total_pathway_params - l2_size
            switching_penalty = overflow / total_pathway_params
        else:
            switching_penalty = 0.0
        
        # Adjust locality score for switching
        locality_score = locality_base * (1.0 - switching_penalty * 0.5)
        
        return CacheAnalysisResult(
            working_sets=working_sets,
            total_working_set_bytes=total_ws,
            cache_fits=cache_fits,
            locality_score=locality_score,
            pathway_switching_penalty=switching_penalty,
        )
    
    # ── Runtime Profiling ─────────────────────────────────────────────
    
    def runtime_comparison(
        self,
        model: nn.Module,
        batch: int = 4,
        seq_len: int = 512,
        n_warmup: int = 5,
        n_measure: int = 20,
        device: str = "cpu",
    ) -> Dict:
        """
        Actual wall-clock comparison: HYDRA block vs dense transformer equivalent.
        
        This is the GROUND TRUTH. If HYDRA is slower wall-clock, no amount of
        FLOPs analysis matters.
        """
        model = model.to(device)
        model.eval()
        
        results = {}
        
        # Create dummy input
        input_ids = torch.randint(0, self.config.vocab_size, (batch, seq_len), device=device)
        
        # ─── Measure full HYDRA forward pass ───
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(input_ids)
        
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
            
            for i in range(n_measure):
                start_events[i].record()
                with torch.no_grad():
                    _ = model(input_ids)
                end_events[i].record()
            
            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        else:
            # CPU timing
            times_ms = []
            for _ in range(n_measure):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_ids)
                elapsed = (time.perf_counter() - start) * 1000
                times_ms.append(elapsed)
        
        hydra_mean = sum(times_ms) / len(times_ms)
        hydra_std = (sum((t - hydra_mean) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
        
        # ─── Measure individual component times ───
        component_times = {}
        with torch.no_grad():
            # Prepare hidden states
            B, L = input_ids.shape
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            x = model.token_embedding(input_ids) + model.position_embedding(positions)
            
            block = model.blocks[0]
            
            # Time router
            router_times = self._measure_component(
                lambda: block.router(x, training=False),
                n_warmup, n_measure, device
            )
            component_times["router"] = router_times
            
            # Time SSM pathway
            ssm_times = self._measure_component(
                lambda: block.stream(x),
                n_warmup, n_measure, device
            )
            component_times["ssm_pathway"] = ssm_times
            
            # Time windowed attention
            focus_times = self._measure_component(
                lambda: block.focus(x),
                n_warmup, n_measure, device
            )
            component_times["windowed_attention"] = focus_times
            
            # Time global attention
            reason_times = self._measure_component(
                lambda: block.reason(x),
                n_warmup, n_measure, device
            )
            component_times["global_attention"] = reason_times
            
            # Time mixer
            if block.cross_mixer is not None:
                stream_out = block.stream(x)
                focus_out = block.focus(x)
                reason_out = block.reason(x)
                combined = stream_out  # simplified
                mixer_times = self._measure_component(
                    lambda: block.cross_mixer(combined, [stream_out, focus_out, reason_out]),
                    n_warmup, n_measure, device
                )
                component_times["cross_mixer"] = mixer_times
        
        # ─── Compute overhead ratios ───
        total_pathway = sum(
            component_times[k]["mean_ms"]
            for k in ["ssm_pathway", "windowed_attention", "global_attention"]
        )
        router_overhead = component_times["router"]["mean_ms"]
        mixer_overhead = component_times.get("cross_mixer", {}).get("mean_ms", 0)
        routing_overhead = router_overhead + mixer_overhead
        
        # Dense baseline estimate: global_attention time only
        dense_estimate = component_times["global_attention"]["mean_ms"]
        
        results["full_model"] = {
            "mean_ms": hydra_mean,
            "std_ms": hydra_std,
            "tokens_per_second": batch * seq_len / (hydra_mean / 1000),
        }
        results["per_component"] = component_times
        results["overhead_analysis"] = {
            "router_time_ms": router_overhead,
            "mixer_time_ms": mixer_overhead,
            "total_routing_overhead_ms": routing_overhead,
            "total_pathway_time_ms": total_pathway,
            "router_pct_of_block": router_overhead / (total_pathway + routing_overhead) * 100 if (total_pathway + routing_overhead) > 0 else 0,
            "routing_overhead_pct_of_block": routing_overhead / (total_pathway + routing_overhead) * 100 if (total_pathway + routing_overhead) > 0 else 0,
            "dense_baseline_estimate_ms": dense_estimate,
            "hydra_vs_dense_ratio": (total_pathway + routing_overhead) / dense_estimate if dense_estimate > 0 else float('inf'),
        }
        
        return results
    
    def _measure_component(
        self, fn, n_warmup: int, n_measure: int, device: str
    ) -> Dict:
        """Measure execution time of a component."""
        for _ in range(n_warmup):
            fn()
        
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
            
            for i in range(n_measure):
                start_events[i].record()
                fn()
                end_events[i].record()
            
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        else:
            times = []
            for _ in range(n_measure):
                start = time.perf_counter()
                fn()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        
        return {
            "mean_ms": mean,
            "std_ms": std,
            "min_ms": min(times),
            "max_ms": max(times),
        }
    
    def format_cache_report(self, result: CacheAnalysisResult, hardware: str = "gpu_a100") -> str:
        """Format cache analysis as a human-readable report."""
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"  CACHE EFFICIENCY ANALYSIS — {hardware.upper()}")
        lines.append(f"{'='*70}")
        lines.append("")
        
        lines.append("Working Set Sizes:")
        for ws in result.working_sets:
            lines.append(f"  {ws.phase_name:30s} {ws.total_mb:10.2f} MB")
            lines.append(f"    {'Parameters:':28s} {ws.parameter_bytes/1e6:10.2f} MB")
            lines.append(f"    {'Activations:':28s} {ws.activation_bytes/1e6:10.2f} MB")
            lines.append(f"    {'Workspace:':28s} {ws.workspace_bytes/1e6:10.2f} MB")
        
        lines.append(f"\n  {'TOTAL HYDRA BLOCK:':30s} {result.total_working_set_bytes/1e6:10.2f} MB")
        
        lines.append(f"\nCache Fit Analysis ({hardware}):")
        for level, fits in result.cache_fits.items():
            cache_size = CACHE_SPECS[hardware][level]
            lines.append(f"\n  {level} ({cache_size/1024:.0f} KB / {cache_size/1e6:.1f} MB):")
            for component, fit in fits.items():
                status = "YES" if fit else " NO"
                lines.append(f"    [{status}] {component}")
        
        lines.append(f"\nLocality Score: {result.locality_score:.3f} (1.0 = perfect)")
        lines.append(f"Pathway Switching Penalty: {result.pathway_switching_penalty:.3f}")
        
        if result.pathway_switching_penalty > 0.3:
            lines.append(f"\n  WARNING: High switching penalty ({result.pathway_switching_penalty:.1%}).")
            lines.append(f"  Pathway parameters exceed L2 cache, causing cache thrashing")
            lines.append(f"  on every pathway switch.")
        elif result.pathway_switching_penalty > 0.1:
            lines.append(f"\n  MODERATE: Some cache pressure from pathway switching.")
        else:
            lines.append(f"\n  GOOD: Pathway parameters fit in L2, minimal switching cost.")
        
        return "\n".join(lines)
