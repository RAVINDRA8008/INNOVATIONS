"""
Overhead Report — The Complete, Honest Assessment

Combines FLOPs, memory, and cache analysis into one report that a
reviewer would actually find convincing. No hand-waving. No cherry-picking.

Produces:
    1. Executive Summary — PASS / MARGINAL / FAIL verdict
    2. FLOPs Breakdown — Router overhead as % of total compute
    3. Memory Report — Activation memory overhead from soft dispatch
    4. Cache Analysis — Working set vs hardware cache hierarchy
    5. Scaling Curves — How overhead changes with sequence length
    6. Recommendations — Concrete actions to reduce overhead
    7. Wall-Clock Timing — Actual runtime if model is provided
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hydra.model.config import HydraConfig
from hydra.profiling.flops_counter import FLOPsCounter
from hydra.profiling.memory_profiler import MemoryBandwidthProfiler
from hydra.profiling.cache_analyzer import CacheEfficiencyAnalyzer


@dataclass
class OverheadReport:
    """Complete overhead analysis report."""
    timestamp: str
    config_name: str
    seq_len: int
    batch_size: int
    flops_analysis: Dict
    memory_analysis: Dict
    cache_analysis: Dict
    scaling_analysis: List[Dict]
    runtime_analysis: Optional[Dict]
    verdict: Dict
    recommendations: List[str]
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON."""
        data = {
            "timestamp": self.timestamp,
            "config": self.config_name,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "verdict": self.verdict,
            "recommendations": self.recommendations,
            "flops": {k: v for k, v in self.flops_analysis.items()
                      if k != "component_flops"},  # Skip non-serializable
            "memory": self.memory_analysis,
            "cache": {
                "total_working_set_MB": self.cache_analysis.get("total_working_set_MB", 0),
                "locality_score": self.cache_analysis.get("locality_score", 0),
                "switching_penalty": self.cache_analysis.get("switching_penalty", 0),
                "cache_fits": self.cache_analysis.get("cache_fits", {}),
            },
            "scaling": self.scaling_analysis,
        }
        if self.runtime_analysis:
            data["runtime"] = self.runtime_analysis
        
        result = json.dumps(data, indent=2, default=str)
        if path:
            with open(path, 'w') as f:
                f.write(result)
        return result


class OverheadProfiler:
    """
    Complete overhead profiler combining all analysis dimensions.
    
    Usage:
        config = HydraConfig.small()
        profiler = OverheadProfiler(config)
        report = profiler.full_report(batch_size=8, seq_len=512)
        print(report.format_text())
    """
    
    def __init__(
        self,
        config: HydraConfig,
        model: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.model = model
        self.dtype = dtype
        
        self.flops_counter = FLOPsCounter(config)
        self.memory_profiler = MemoryBandwidthProfiler(config, dtype)
        self.cache_analyzer = CacheEfficiencyAnalyzer(config, dtype)
    
    def full_report(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        stream_frac: float = 0.6,
        focus_frac: float = 0.25,
        reason_frac: float = 0.15,
        hardware: str = "gpu_a100",
        measure_runtime: bool = False,
        device: str = "cpu",
    ) -> OverheadReport:
        """
        Generate the complete overhead analysis report.
        
        Args:
            batch_size: Batch size for memory analysis
            seq_len: Sequence length for analysis
            stream_frac: Expected fraction of tokens routed to Stream (SSM)
            focus_frac: Expected fraction routed to Focus (Windowed)
            reason_frac: Expected fraction routed to Reason (Global)
            hardware: Hardware profile for cache analysis
            measure_runtime: Whether to measure actual wall-clock time
            device: Device for runtime measurement
        """
        # 1. FLOPs Analysis
        flops = self.flops_counter.full_analysis(
            seq_len, stream_frac, focus_frac, reason_frac
        )
        
        # 2. Memory Analysis
        memory_train = self.memory_profiler.full_analysis(batch_size, seq_len, training=True)
        memory_infer = self.memory_profiler.full_analysis(batch_size, seq_len, training=False)
        memory = {
            "training": memory_train,
            "inference": memory_infer,
        }
        
        # 3. Cache Analysis
        cache_result = self.cache_analyzer.analyze_cache_efficiency(
            batch_size, seq_len, hardware
        )
        cache = {
            "total_working_set_MB": cache_result.total_working_set_bytes / 1e6,
            "locality_score": cache_result.locality_score,
            "switching_penalty": cache_result.pathway_switching_penalty,
            "cache_fits": cache_result.cache_fits,
            "working_sets": [
                {"name": ws.phase_name, "MB": ws.total_mb}
                for ws in cache_result.working_sets
            ],
        }
        
        # 4. Scaling Analysis
        scaling = self.flops_counter.scaling_analysis(
            stream_frac=stream_frac,
            focus_frac=focus_frac,
            reason_frac=reason_frac,
        )
        
        # 5. Runtime (optional)
        runtime = None
        if measure_runtime and self.model is not None:
            runtime = self.cache_analyzer.runtime_comparison(
                self.model, batch_size, seq_len,
                n_warmup=3, n_measure=10, device=device,
            )
        
        # 6. Combined Verdict
        verdict = self._combined_verdict(flops, memory, cache, runtime)
        
        # 7. Recommendations
        recommendations = self._generate_recommendations(flops, memory, cache, runtime)
        
        config_name = "small" if self.config.d_model <= 256 else (
            "base" if self.config.d_model <= 512 else "large"
        )
        
        return OverheadReport(
            timestamp=datetime.now().isoformat(),
            config_name=config_name,
            seq_len=seq_len,
            batch_size=batch_size,
            flops_analysis=flops,
            memory_analysis=memory,
            cache_analysis=cache,
            scaling_analysis=scaling,
            runtime_analysis=runtime,
            verdict=verdict,
            recommendations=recommendations,
        )
    
    def _combined_verdict(self, flops, memory, cache, runtime) -> Dict:
        """Combine all analysis dimensions into one verdict."""
        issues = []
        strengths = []
        
        # From FLOPs
        flops_verdict = flops["verdict"]
        issues.extend(flops_verdict["issues"])
        strengths.extend(flops_verdict["strengths"])
        
        # From Memory
        train_mem = memory["training"]
        mem_overhead = train_mem["routing_overhead"]["memory_overhead_vs_dense"]
        checkpoint_mode = train_mem["routing_overhead"].get("checkpoint_mode", "none")
        raw_overhead = train_mem["routing_overhead"].get("raw_overhead_vs_dense", mem_overhead)
        if mem_overhead > 2.0:
            issues.append(
                f"CRITICAL: Training memory is {mem_overhead+1:.1f}× dense baseline"
                f"{f' (with {checkpoint_mode} checkpointing)' if checkpoint_mode != 'none' else ''}. "
                f"Soft dispatch stores activations for all 3 pathways."
            )
        elif mem_overhead > 1.0:
            issues.append(
                f"WARNING: Training memory is {mem_overhead+1:.1f}× dense baseline"
                f"{f' (with {checkpoint_mode} checkpointing; {raw_overhead+1:.1f}× without)' if checkpoint_mode != 'none' else ''}. "
            )
        elif checkpoint_mode != 'none':
            strengths.append(
                f"Training memory is {mem_overhead+1:.1f}× dense baseline "
                f"(with {checkpoint_mode} checkpointing; {raw_overhead+1:.1f}× without)."
            )
        
        # From Cache
        if cache["switching_penalty"] > 0.3:
            issues.append(
                f"CRITICAL: Cache switching penalty is {cache['switching_penalty']:.1%}. "
                f"Pathway parameters exceed L2 cache."
            )
        elif cache["switching_penalty"] > 0.1:
            issues.append(
                f"WARNING: Cache switching penalty is {cache['switching_penalty']:.1%}."
            )
        else:
            strengths.append(
                f"Cache efficiency is good (switching penalty: {cache['switching_penalty']:.1%})."
            )
        
        if cache["locality_score"] < 0.5:
            issues.append(
                f"WARNING: Data locality score is {cache['locality_score']:.2f}/1.0. "
                f"Multi-pathway dispatch hurts cache utilization."
            )
        
        # From Runtime (if available)
        if runtime:
            overhead_pct = runtime["overhead_analysis"]["routing_overhead_pct_of_block"]
            if overhead_pct > 25:
                issues.append(
                    f"CRITICAL: Router + mixer consume {overhead_pct:.1f}% of wall-clock time."
                )
            elif overhead_pct > 15:
                issues.append(
                    f"WARNING: Router + mixer consume {overhead_pct:.1f}% of wall-clock time."
                )
            else:
                strengths.append(
                    f"Runtime routing overhead is acceptable ({overhead_pct:.1f}% of block time)."
                )
        
        # Overall verdict
        n_critical = sum(1 for i in issues if i.startswith("CRITICAL"))
        n_warning = sum(1 for i in issues if i.startswith("WARNING"))
        
        infer_savings = flops["savings"]["inference_vs_dense"]
        
        if n_critical > 0 or infer_savings < 0.1:
            overall = "FAIL"
        elif n_warning > 2 or infer_savings < 0.3:
            overall = "MARGINAL"
        else:
            overall = "PASS"
        
        return {
            "overall": overall,
            "n_critical": n_critical,
            "n_warnings": n_warning,
            "n_strengths": len(strengths),
            "issues": issues,
            "strengths": strengths,
            "inference_savings_vs_dense": infer_savings,
        }
    
    def _generate_recommendations(self, flops, memory, cache, runtime) -> List[str]:
        """Generate concrete, actionable recommendations."""
        recs = []
        
        # Router overhead
        router_ratio = flops["overhead"]["router_overhead_ratio"]
        if router_ratio > 0.1:
            recs.append(
                f"REDUCE ROUTER COST: Router uses {router_ratio:.1%} of inference FLOPs. "
                f"Options: (1) Reduce router_hidden_dim from {self.config.router_hidden_dim} to "
                f"{self.config.router_hidden_dim // 2}, (2) Remove context conv from "
                f"ComplexityEstimator (saves ~30% router FLOPs), (3) Share router across "
                f"adjacent layers."
            )
        
        # Mixer overhead
        total_ratio = flops["overhead"]["total_overhead_ratio"]
        mixer_ratio = total_ratio - router_ratio
        if mixer_ratio > 0.1:
            recs.append(
                f"REDUCE MIXER COST: Cross-pathway mixer adds {mixer_ratio:.1%} overhead. "
                f"Options: (1) Reduce cross_pathway_dim from {self.config.cross_pathway_dim}, "
                f"(2) Apply mixer every 2-3 layers instead of every layer, "
                f"(3) Replace cross-attention with cheaper linear mixing."
            )
        
        # Training cost
        train_overhead = flops["savings"]["training_overhead_vs_dense"]
        if train_overhead > 1.0:
            recs.append(
                f"REDUCE TRAINING COST: Training is {train_overhead:.0%} more expensive than "
                f"dense. Implement conditional computation: group tokens by routing decision, "
                f"only compute the selected pathway (requires custom CUDA kernel or torch.compile "
                f"optimization)."
            )
        
        # Memory
        train_mem = memory["training"]
        mem_overhead = train_mem["routing_overhead"]["memory_overhead_vs_dense"]
        checkpoint_mode = train_mem["routing_overhead"].get("checkpoint_mode", "none")
        if mem_overhead > 1.5 and checkpoint_mode == 'none':
            recs.append(
                f"REDUCE MEMORY: {mem_overhead+1:.1f}× memory vs dense during training. "
                f"Enable gradient checkpointing: set checkpoint_pathways='non_dominant' "
                f"(saves ~60-70% pathway activations, costs ~30% slower training) or "
                f"'all' (saves ~90%, costs ~2× slower)."
            )
        elif mem_overhead > 1.5:
            recs.append(
                f"REDUCE MEMORY FURTHER: {mem_overhead+1:.1f}× memory vs dense even with "
                f"{checkpoint_mode} checkpointing. Consider: (1) upgrade to 'all' checkpointing, "
                f"(2) implement activation-sparse soft dispatch (only materialize selected "
                f"pathway activations), (3) use mixed-precision activations (fp16/bf16)."
            )
        elif mem_overhead > 0.5 and checkpoint_mode != 'none':
            recs.append(
                f"MEMORY ACCEPTABLE: {mem_overhead+1:.1f}× dense with {checkpoint_mode} "
                f"checkpointing. For further reduction, try 'all' checkpointing mode."
            )
        
        # Cache
        if cache["switching_penalty"] > 0.1:
            recs.append(
                f"IMPROVE CACHE EFFICIENCY: Pathway switching penalty is "
                f"{cache['switching_penalty']:.1%}. Options: (1) Process tokens in "
                f"pathway-grouped batches (sort by routing decision before dispatch), "
                f"(2) Fuse router + cheap pathway (SSM) into single kernel."
            )
        
        # Break-even analysis
        break_even = flops["break_even"]
        max_reason = break_even["max_reason_frac_with_ssm_rest"]
        if max_reason < 0.15:
            recs.append(
                f"ROUTING BUDGET TIGHT: Max reason fraction before overhead exceeds savings "
                f"is {max_reason:.1%}. This leaves very little room for global attention. "
                f"Consider: (1) Make router cheaper, (2) Remove mixer for small models, "
                f"(3) Accept SSM-dominant routing as the practical regime."
            )
        
        # If everything looks good
        if not recs:
            recs.append(
                "No critical issues found. Focus on validating that learned routing "
                "actually improves accuracy vs compute-matched dense baselines."
            )
        
        return recs
    
    def format_text_report(self, report: OverheadReport) -> str:
        """Format the report as human-readable text."""
        lines = []
        lines.append("")
        lines.append("=" * 78)
        lines.append("  HYDRA ROUTING OVERHEAD ANALYSIS — HONEST ASSESSMENT")
        lines.append(f"  Config: {report.config_name} | Seq: {report.seq_len} | Batch: {report.batch_size}")
        lines.append(f"  Generated: {report.timestamp}")
        lines.append("=" * 78)
        
        # ── Verdict ──
        v = report.verdict
        lines.append("")
        lines.append(f"  VERDICT: {v['overall']}")
        lines.append(f"  Inference savings vs dense: {v['inference_savings_vs_dense']:.1%}")
        lines.append(f"  Issues: {v['n_critical']} critical, {v['n_warnings']} warnings")
        lines.append(f"  Strengths: {v['n_strengths']}")
        
        # ── FLOPs Breakdown ──
        lines.append("")
        lines.append("-" * 78)
        lines.append("  1. FLOPs ANALYSIS")
        lines.append("-" * 78)
        
        flops = report.flops_analysis
        ptl = flops["per_token_per_layer_macs"]
        lines.append(f"\n  Per-token per-layer MACs:")
        for name, macs in ptl.items():
            lines.append(f"    {name:30s} {macs:>15,} MACs ({macs*2/1e6:.3f} MFLOPs)")
        
        lines.append(f"\n  Total model FLOPs:")
        for mode, val in flops["total_flops"].items():
            lines.append(f"    {mode:30s} {val/1e9:>10.3f} GFLOPs")
        
        oh = flops["overhead"]
        lines.append(f"\n  Routing Overhead:")
        lines.append(f"    Router FLOPs:             {oh['router_macs']*2/1e6:.3f} MFLOPs "
                     f"({oh['router_overhead_ratio']:.2%} of inference)")
        lines.append(f"    Mixer FLOPs:              {oh['mixer_macs']*2/1e6:.3f} MFLOPs")
        lines.append(f"    Total Overhead:           {oh['total_routing_overhead_macs']*2/1e6:.3f} MFLOPs "
                     f"({oh['total_overhead_ratio']:.2%} of inference)")
        
        sv = flops["savings"]
        lines.append(f"\n  Savings:")
        lines.append(f"    Inference vs Dense:       {sv['inference_vs_dense']:>+.1%}")
        lines.append(f"    Training overhead:        {sv['training_overhead_vs_dense']:>+.1%}")
        
        be = flops["break_even"]
        lines.append(f"\n  Break-Even Analysis:")
        lines.append(f"    Router overhead as % of dense per-token cost: {be['overhead_as_pct_of_dense']:.1f}%")
        lines.append(f"    Min Stream fraction to break even:            {be['min_stream_frac_to_break_even']:.1%}")
        lines.append(f"    Max Reason fraction (rest=Stream):            {be['max_reason_frac_with_ssm_rest']:.1%}")
        
        # ── Memory ──
        lines.append("")
        lines.append("-" * 78)
        lines.append("  2. MEMORY & BANDWIDTH ANALYSIS")
        lines.append("-" * 78)
        
        for mode_name, mem in report.memory_analysis.items():
            lines.append(f"\n  [{mode_name.upper()}]")
            lines.append(f"    Total per-layer (raw):    {mem['per_layer_total_MB']:.2f} MB")
            effective = mem.get('per_layer_effective_MB', mem['per_layer_total_MB'])
            if effective != mem['per_layer_total_MB']:
                lines.append(f"    Total per-layer (eff):    {effective:.2f} MB")
            lines.append(f"    Total model:              {mem.get('model_effective_MB', mem['model_total_MB']):.2f} MB")
            lines.append(f"    Dense baseline:           {mem['dense_baseline_total_MB']:.2f} MB")
            ro = mem["routing_overhead"]
            lines.append(f"    Router memory:            {ro['router_memory_MB']:.2f} MB")
            lines.append(f"    Mixer memory:             {ro['mixer_memory_MB']:.2f} MB")
            lines.append(f"    Soft dispatch penalty:    {ro['soft_dispatch_penalty_MB']:.2f} MB")
            ckpt_mode = ro.get('checkpoint_mode', 'none')
            if ckpt_mode != 'none' and ckpt_mode != 'n/a':
                lines.append(f"    Checkpoint savings:       {ro.get('checkpoint_savings_MB', 0):.2f} MB ({ckpt_mode})")
                lines.append(f"    Effective penalty:        {ro.get('effective_penalty_MB', 0):.2f} MB")
            lines.append(f"    Memory overhead vs dense: {ro['memory_overhead_vs_dense']:+.1%}")
            raw_overhead = ro.get('raw_overhead_vs_dense')
            if raw_overhead is not None and raw_overhead != ro['memory_overhead_vs_dense']:
                lines.append(f"    (Without checkpointing:  {raw_overhead:+.1%})")
        
        # ── Cache ──
        lines.append("")
        lines.append("-" * 78)
        lines.append("  3. CACHE EFFICIENCY")
        lines.append("-" * 78)
        
        ca = report.cache_analysis
        lines.append(f"\n  Total working set:          {ca['total_working_set_MB']:.2f} MB")
        lines.append(f"  Locality score:             {ca['locality_score']:.3f} / 1.0")
        lines.append(f"  Switching penalty:          {ca['switching_penalty']:.3f}")
        
        lines.append(f"\n  Working set per component:")
        for ws in ca["working_sets"]:
            lines.append(f"    {ws['name']:30s} {ws['MB']:>8.2f} MB")
        
        # ── Scaling ──
        lines.append("")
        lines.append("-" * 78)
        lines.append("  4. SCALING WITH SEQUENCE LENGTH")
        lines.append("-" * 78)
        
        lines.append(f"\n  {'SeqLen':>8s} {'HYDRA GFLOPs':>14s} {'Dense GFLOPs':>14s} "
                     f"{'Savings':>10s} {'Router OH':>10s} {'Total OH':>10s}")
        lines.append(f"  {'─'*8} {'─'*14} {'─'*14} {'─'*10} {'─'*10} {'─'*10}")
        for s in report.scaling_analysis:
            lines.append(f"  {s['seq_len']:>8d} {s['inference_gflops']:>14.3f} "
                        f"{s['dense_gflops']:>14.3f} {s['savings_pct']:>+9.1f}% "
                        f"{s['router_overhead_pct']:>9.1f}% {s['total_overhead_pct']:>9.1f}%")
        
        # ── Runtime (if available) ──
        if report.runtime_analysis:
            lines.append("")
            lines.append("-" * 78)
            lines.append("  5. WALL-CLOCK RUNTIME")
            lines.append("-" * 78)
            
            rt = report.runtime_analysis
            fm = rt["full_model"]
            lines.append(f"\n  Full model forward pass:    {fm['mean_ms']:.2f} ms "
                        f"(±{fm['std_ms']:.2f})")
            lines.append(f"  Throughput:                 {fm['tokens_per_second']:.0f} tokens/s")
            
            lines.append(f"\n  Per-component times:")
            for name, times in rt["per_component"].items():
                lines.append(f"    {name:30s} {times['mean_ms']:>8.2f} ms (±{times['std_ms']:.2f})")
            
            oa = rt["overhead_analysis"]
            lines.append(f"\n  Router % of block time:     {oa['router_pct_of_block']:.1f}%")
            lines.append(f"  Total overhead % of block:  {oa['routing_overhead_pct_of_block']:.1f}%")
            lines.append(f"  HYDRA:Dense ratio:          {oa['hydra_vs_dense_ratio']:.2f}x")
        
        # ── Issues ──
        lines.append("")
        lines.append("-" * 78)
        lines.append("  ISSUES")
        lines.append("-" * 78)
        for i, issue in enumerate(report.verdict["issues"], 1):
            lines.append(f"\n  {i}. {issue}")
        
        if report.verdict["strengths"]:
            lines.append("")
            lines.append("  STRENGTHS:")
            for s in report.verdict["strengths"]:
                lines.append(f"    + {s}")
        
        # ── Recommendations ──
        lines.append("")
        lines.append("-" * 78)
        lines.append("  RECOMMENDATIONS")
        lines.append("-" * 78)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"\n  {i}. {rec}")
        
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"  END OF REPORT")
        lines.append("=" * 78)
        lines.append("")
        
        return "\n".join(lines)
