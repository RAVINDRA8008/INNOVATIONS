"""
HYDRA Routing Overhead Profiler

The single most important validation for HYDRA's core hypothesis:
    "Learned routing can match transformer accuracy at 40-60% compute."

If router_overhead + switching_cost > saved_compute, the entire claim collapses.

This module provides HONEST measurement of:
    1. Router FLOPs — exact operation counts for ComplexityEstimator + routing head
    2. Memory Bandwidth Cost — bytes moved for routing decisions + soft dispatch
    3. Cache Inefficiency — data locality penalty from multi-pathway dispatch
    4. Net Compute Budget — the real savings (or deficit) after all overhead

Usage:
    from hydra.profiling import OverheadProfiler
    profiler = OverheadProfiler(model, config)
    report = profiler.full_report(batch_size=8, seq_len=512)
    print(report)
"""

from .flops_counter import FLOPsCounter
from .memory_profiler import MemoryBandwidthProfiler
from .cache_analyzer import CacheEfficiencyAnalyzer
from .overhead_report import OverheadProfiler, OverheadReport

__all__ = [
    "FLOPsCounter",
    "MemoryBandwidthProfiler",
    "CacheEfficiencyAnalyzer",
    "OverheadProfiler",
    "OverheadReport",
]
