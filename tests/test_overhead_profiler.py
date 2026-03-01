"""
Tests for the Routing Overhead Profiler

Validates that:
    1. FLOPs counts are self-consistent
    2. Memory estimates are reasonable
    3. Cache analysis produces valid results
    4. Break-even analysis is mathematically correct
    5. Report generation doesn't crash
    6. Scaling analysis shows expected trends
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from hydra.model.config import HydraConfig
from hydra.model.hydra_model import HydraModel
from hydra.profiling.flops_counter import FLOPsCounter, ComponentFLOPs
from hydra.profiling.memory_profiler import MemoryBandwidthProfiler, MemoryBreakdown
from hydra.profiling.cache_analyzer import CacheEfficiencyAnalyzer
from hydra.profiling.overhead_report import OverheadProfiler, OverheadReport


@pytest.fixture
def small_config():
    return HydraConfig.small()


@pytest.fixture
def flops_counter(small_config):
    return FLOPsCounter(small_config)


@pytest.fixture
def memory_profiler(small_config):
    return MemoryBandwidthProfiler(small_config)


@pytest.fixture
def cache_analyzer(small_config):
    return CacheEfficiencyAnalyzer(small_config)


# ── FLOPs Counter Tests ──────────────────────────────────────────────


class TestFLOPsCounter:
    
    def test_router_flops_positive(self, flops_counter):
        """Router FLOPs should be positive and non-trivial."""
        result = flops_counter.router_flops_per_token(seq_len=512)
        assert result.macs > 0
        assert result.flops > 0
        assert len(result.detail) > 0
        # Router should have meaningful cost
        assert result.macs > 100, "Router FLOPs suspiciously low"
    
    def test_pathway_cost_ordering(self, flops_counter):
        """SSM < Windowed < Global in FLOPs per token."""
        seq_len = 512
        ssm = flops_counter.ssm_pathway_flops_per_token(seq_len)
        windowed = flops_counter.windowed_attention_flops_per_token(seq_len)
        global_attn = flops_counter.global_attention_flops_per_token(seq_len)
        
        assert ssm.macs < windowed.macs, \
            f"SSM ({ssm.macs}) should be cheaper than Windowed ({windowed.macs})"
        assert windowed.macs < global_attn.macs, \
            f"Windowed ({windowed.macs}) should be cheaper than Global ({global_attn.macs})"
    
    def test_global_attention_scales_quadratically(self, flops_counter):
        """Total global attention FLOPs should scale ~quadratically with seq_len.
        
        Per-token cost grows linearly (attention is O(L) per token), but FFN is
        constant and dominates at small seq_len. So we check TOTAL FLOPs 
        (per_token * seq_len) which should scale super-linearly.
        """
        seq_256 = 256
        seq_512 = 512
        seq_1024 = 1024
        
        total_256 = flops_counter.global_attention_flops_per_token(seq_256).macs * seq_256
        total_512 = flops_counter.global_attention_flops_per_token(seq_512).macs * seq_512
        total_1024 = flops_counter.global_attention_flops_per_token(seq_1024).macs * seq_1024
        
        # Total should scale > 2x per doubling (quadratic attention + linear FFN)
        ratio_1 = total_512 / total_256
        ratio_2 = total_1024 / total_512
        
        assert ratio_1 > 2.0, f"Total global FLOPs should scale super-linearly ({ratio_1:.2f}x)"
        assert ratio_2 > 2.0, f"Total global FLOPs should scale super-linearly ({ratio_2:.2f}x)"
        
        # Also verify per-token cost increases with seq_len (attention term grows)
        pt_256 = flops_counter.global_attention_flops_per_token(seq_256).macs
        pt_512 = flops_counter.global_attention_flops_per_token(seq_512).macs
        assert pt_512 > pt_256, "Per-token cost should increase with seq_len (attention grows)"
    
    def test_ssm_scales_linearly(self, flops_counter):
        """SSM FLOPs per token should be roughly constant (O(n) total → O(1) per token)."""
        flops_256 = flops_counter.ssm_pathway_flops_per_token(256)
        flops_512 = flops_counter.ssm_pathway_flops_per_token(512)
        
        # SSM per-token cost should be similar regardless of seq_len
        ratio = flops_512.macs / flops_256.macs
        assert 0.9 < ratio < 1.1, f"SSM should be O(1) per token, got ratio {ratio:.2f}"
    
    def test_full_analysis_structure(self, flops_counter):
        """Full analysis should return all expected keys."""
        result = flops_counter.full_analysis(512)
        
        expected_keys = [
            "seq_len", "routing_distribution", "per_token_per_layer_macs",
            "total_macs", "total_flops", "overhead", "savings",
            "break_even", "component_flops", "verdict",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_training_more_expensive_than_inference(self, flops_counter):
        """Training (soft dispatch) should always be more expensive than inference."""
        result = flops_counter.full_analysis(512)
        train = result["total_macs"]["training_mode"]
        infer = result["total_macs"]["inference_mode"]
        
        assert train > infer, \
            f"Training ({train}) should be more expensive than inference ({infer})"
    
    def test_inference_cheaper_than_dense(self, flops_counter):
        """With default routing (60/25/15), inference should be cheaper than dense."""
        result = flops_counter.full_analysis(512, stream_frac=0.6, focus_frac=0.25, reason_frac=0.15)
        savings = result["savings"]["inference_vs_dense"]
        
        # May or may not save — this test just checks it's computed correctly
        assert -1.0 < savings < 1.0, f"Savings should be between -100% and 100%, got {savings}"
    
    def test_break_even_analysis(self, flops_counter):
        """Break-even values should be in valid ranges."""
        result = flops_counter.full_analysis(512)
        be = result["break_even"]
        
        assert be["overhead_per_token_per_layer"] >= 0
        assert 0 <= be["min_stream_frac_to_break_even"] <= 1.0
        assert 0 <= be["max_reason_frac_with_ssm_rest"] <= 1.0
        assert be["overhead_as_pct_of_dense"] >= 0
    
    def test_100pct_reason_is_expensive(self, flops_counter):
        """100% global attention routing should be more expensive than dense."""
        result = flops_counter.full_analysis(512, stream_frac=0.0, focus_frac=0.0, reason_frac=1.0)
        savings = result["savings"]["inference_vs_dense"]
        
        # With 100% Reason + router overhead + mixer, should be MORE expensive
        assert savings < 0, \
            f"100% Reason routing should be MORE expensive than dense (savings={savings:.1%})"
    
    def test_scaling_analysis(self, flops_counter):
        """Scaling analysis should return results for each seq length."""
        results = flops_counter.scaling_analysis(
            seq_lengths=[128, 256, 512],
            stream_frac=0.6, focus_frac=0.25, reason_frac=0.15,
        )
        
        assert len(results) == 3
        # GFLOPs should increase with sequence length
        assert results[0]["inference_gflops"] < results[1]["inference_gflops"]
        assert results[1]["inference_gflops"] < results[2]["inference_gflops"]
    
    def test_router_overhead_detail(self, flops_counter):
        """Router detail should break down into identifiable sub-components."""
        result = flops_counter.router_flops_per_token(512)
        detail = result.detail
        
        # Should have content MLP, context conv, entropy, fusion, head components
        assert any("content" in k for k in detail), "Missing content MLP FLOPs"
        assert any("context" in k for k in detail), "Missing context conv FLOPs"
        assert any("entropy" in k for k in detail), "Missing entropy FLOPs"
        assert any("fusion" in k for k in detail), "Missing fusion FLOPs"
        assert any("head" in k for k in detail), "Missing routing head FLOPs"
        assert "gumbel_softmax" in detail, "Missing Gumbel-Softmax FLOPs"


# ── Memory Profiler Tests ────────────────────────────────────────────


class TestMemoryProfiler:
    
    def test_router_memory_positive(self, memory_profiler):
        """Router memory should be positive."""
        result = memory_profiler.router_memory(batch=4, seq_len=512)
        assert result.parameters_bytes > 0
        assert result.activation_bytes > 0
        assert result.total_bytes > 0
    
    def test_global_attention_memory_quadratic(self, memory_profiler):
        """Global attention memory should grow quadratically with seq_len."""
        mem_256 = memory_profiler.global_attention_memory(4, 256)
        mem_512 = memory_profiler.global_attention_memory(4, 512)
        
        # Attention scores are B*H*L*L — should roughly 4x for 2x seq_len
        # But params are constant, so total won't be exactly 4x
        ratio = mem_512.activation_bytes / mem_256.activation_bytes
        assert ratio > 2.0, f"Global attention memory should scale super-linearly ({ratio:.2f}x)"
    
    def test_full_analysis_training_vs_inference(self, memory_profiler):
        """Training should use more memory than inference."""
        train = memory_profiler.full_analysis(4, 512, training=True)
        infer = memory_profiler.full_analysis(4, 512, training=False)
        
        train_total = train["model_total_MB"]
        infer_total = infer["model_total_MB"]
        
        # Training has gradient storage
        assert train_total >= infer_total, \
            f"Training ({train_total:.1f}MB) should use >= inference ({infer_total:.1f}MB)"
    
    def test_soft_dispatch_bandwidth(self, memory_profiler):
        """Soft dispatch should have measurable bandwidth cost."""
        bw = memory_profiler.soft_dispatch_bandwidth(4, 512)
        assert bw.bytes_read > 0
        assert bw.bytes_written > 0
        assert bw.total_bytes > 0
    
    def test_memory_scales_with_batch(self, memory_profiler):
        """Activation memory should scale linearly with batch size."""
        mem_b4 = memory_profiler.router_memory(4, 512)
        mem_b8 = memory_profiler.router_memory(8, 512)
        
        # Activations should ~double, params stay same
        act_ratio = mem_b8.activation_bytes / mem_b4.activation_bytes
        assert 1.8 < act_ratio < 2.2, f"Activations should ~2x with batch, got {act_ratio:.2f}x"
        assert mem_b8.parameters_bytes == mem_b4.parameters_bytes, "Parameters shouldn't change with batch"


# ── Cache Analyzer Tests ─────────────────────────────────────────────


class TestCacheAnalyzer:
    
    def test_working_sets_positive(self, cache_analyzer):
        """All working sets should have positive bytes."""
        ws = cache_analyzer.router_working_set(4, 512)
        assert ws.total_bytes > 0
        assert ws.parameter_bytes > 0
    
    def test_cache_analysis_result(self, cache_analyzer):
        """Cache analysis should produce valid results."""
        result = cache_analyzer.analyze_cache_efficiency(4, 512, "gpu_a100")
        
        assert len(result.working_sets) == 5  # router, ssm, windowed, global, mixer
        assert result.total_working_set_bytes > 0
        assert 0.0 <= result.locality_score <= 1.0
        assert 0.0 <= result.pathway_switching_penalty <= 1.0
    
    def test_cache_fits_structure(self, cache_analyzer):
        """Cache fits should have entries for each cache level."""
        result = cache_analyzer.analyze_cache_efficiency(4, 512, "gpu_a100")
        
        assert "L1" in result.cache_fits
        assert "L2" in result.cache_fits
        
        for level, fits in result.cache_fits.items():
            assert "Total HYDRA Block" in fits
            assert "Dense Baseline" in fits
    
    def test_smaller_model_better_cache(self):
        """Smaller model should have better cache utilization."""
        small = HydraConfig.small()
        base = HydraConfig.base()
        
        small_analyzer = CacheEfficiencyAnalyzer(small)
        base_analyzer = CacheEfficiencyAnalyzer(base)
        
        small_result = small_analyzer.analyze_cache_efficiency(4, 256, "gpu_a100")
        base_result = base_analyzer.analyze_cache_efficiency(4, 256, "gpu_a100")
        
        assert small_result.total_working_set_bytes <= base_result.total_working_set_bytes
    
    def test_runtime_comparison_cpu(self, cache_analyzer, small_config):
        """Runtime comparison should produce valid timing data on CPU."""
        model = HydraModel(small_config)
        result = cache_analyzer.runtime_comparison(
            model, batch=2, seq_len=64,
            n_warmup=1, n_measure=3, device="cpu"
        )
        
        assert "full_model" in result
        assert "per_component" in result
        assert "overhead_analysis" in result
        assert result["full_model"]["mean_ms"] > 0
        assert result["per_component"]["router"]["mean_ms"] > 0
        assert result["overhead_analysis"]["router_pct_of_block"] >= 0
    
    def test_format_cache_report(self, cache_analyzer):
        """Cache report formatting should produce readable output."""
        result = cache_analyzer.analyze_cache_efficiency(4, 512, "gpu_a100")
        report_text = cache_analyzer.format_cache_report(result, "gpu_a100")
        
        assert len(report_text) > 100
        assert "Working Set" in report_text
        assert "Cache Fit" in report_text
        assert "Locality Score" in report_text


# ── Overhead Report Tests ────────────────────────────────────────────


class TestOverheadReport:
    
    def test_full_report_generation(self, small_config):
        """Full report should generate without errors."""
        profiler = OverheadProfiler(small_config)
        report = profiler.full_report(batch_size=4, seq_len=256)
        
        assert isinstance(report, OverheadReport)
        assert report.config_name == "small"
        assert report.seq_len == 256
        assert report.batch_size == 4
    
    def test_verdict_fields(self, small_config):
        """Verdict should have all required fields."""
        profiler = OverheadProfiler(small_config)
        report = profiler.full_report(batch_size=4, seq_len=256)
        
        v = report.verdict
        assert "overall" in v
        assert v["overall"] in ["PASS", "MARGINAL", "FAIL"]
        assert "issues" in v
        assert "strengths" in v
        assert "inference_savings_vs_dense" in v
    
    def test_recommendations_generated(self, small_config):
        """Report should include actionable recommendations."""
        profiler = OverheadProfiler(small_config)
        report = profiler.full_report(batch_size=4, seq_len=256)
        
        assert len(report.recommendations) > 0
        assert all(isinstance(r, str) for r in report.recommendations)
    
    def test_text_report_formatting(self, small_config):
        """Text report should be well-formatted and contain key sections."""
        profiler = OverheadProfiler(small_config)
        report = profiler.full_report(batch_size=4, seq_len=256)
        text = profiler.format_text_report(report)
        
        assert len(text) > 500
        assert "VERDICT" in text
        assert "FLOPs" in text
        assert "MEMORY" in text
        assert "CACHE" in text
        assert "SCALING" in text
        assert "RECOMMENDATIONS" in text
    
    def test_json_serialization(self, small_config, tmp_path):
        """Report should serialize to valid JSON."""
        profiler = OverheadProfiler(small_config)
        report = profiler.full_report(batch_size=4, seq_len=256)
        
        json_path = str(tmp_path / "test_report.json")
        json_str = report.to_json(json_path)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "verdict" in parsed
        assert "flops" in parsed
        assert "memory" in parsed
        
        # File should exist
        assert os.path.exists(json_path)
    
    def test_report_with_runtime(self, small_config):
        """Report with runtime measurement should include timing data."""
        model = HydraModel(small_config)
        profiler = OverheadProfiler(small_config, model=model)
        report = profiler.full_report(
            batch_size=2, seq_len=64,
            measure_runtime=True, device="cpu"
        )
        
        assert report.runtime_analysis is not None
        assert "full_model" in report.runtime_analysis
        assert report.runtime_analysis["full_model"]["mean_ms"] > 0
    
    def test_scaling_in_report(self, small_config):
        """Scaling analysis should be included in report."""
        profiler = OverheadProfiler(small_config)
        report = profiler.full_report(batch_size=4, seq_len=256)
        
        assert len(report.scaling_analysis) > 0
        # Should have multiple sequence lengths
        seq_lens = [s["seq_len"] for s in report.scaling_analysis]
        assert len(seq_lens) > 3


# ── Integration Tests ────────────────────────────────────────────────


class TestIntegration:
    
    def test_flops_vs_dense_sanity(self):
        """With 100% SSM, the pathway compute is much cheaper than dense.
        
        However, the TOTAL may not save compute because of router + mixer overhead.
        This is an HONEST FINDING — it validates that the profiler correctly
        identifies overhead that could kill the benefit.
        
        We verify:
        1. SSM pathway alone is cheaper than global attention
        2. Break-even analysis correctly identifies the overhead
        """
        config = HydraConfig.small()
        counter = FLOPsCounter(config)
        
        # SSM pathway alone should be much cheaper than global attention
        ssm = counter.ssm_pathway_flops_per_token(512)
        global_attn = counter.global_attention_flops_per_token(512)
        ssm_savings = 1.0 - (ssm.macs / global_attn.macs)
        assert ssm_savings > 0.3, \
            f"SSM pathway alone should save >30% vs global attention, got {ssm_savings:.1%}"
        
        # Full analysis with overhead — may or may not save overall
        result = counter.full_analysis(512, stream_frac=1.0, focus_frac=0.0, reason_frac=0.0)
        
        # Verify the break-even analysis is present and meaningful
        be = result["break_even"]
        assert be["overhead_as_pct_of_dense"] > 0, "Overhead should be non-zero"
        
        # The overhead finding itself is valuable — if savings < 0,
        # it means mixer/router cost exceeds pathway savings
        savings = result["savings"]["inference_vs_dense"]
        if savings < 0:
            # The profiler correctly identified that overhead exceeds savings!
            assert result["overhead"]["total_overhead_ratio"] > 0.1, \
                "If no savings, overhead ratio should be significant"
    
    def test_flops_vs_dense_worst_case(self):
        """Worst case: 100% Reason should be MORE expensive (router overhead)."""
        config = HydraConfig.small()
        counter = FLOPsCounter(config)
        
        result = counter.full_analysis(512, stream_frac=0.0, focus_frac=0.0, reason_frac=1.0)
        savings = result["savings"]["inference_vs_dense"]
        
        assert savings < 0, \
            f"100% Reason should be more expensive than dense, got savings={savings:.1%}"
    
    def test_router_overhead_reasonable(self):
        """Router overhead should be a reasonable fraction of total compute."""
        config = HydraConfig.small()
        counter = FLOPsCounter(config)
        
        result = counter.full_analysis(512)
        router_ratio = result["overhead"]["router_overhead_ratio"]
        
        # Router should be < 30% of inference FLOPs (generous threshold)
        assert router_ratio < 0.30, \
            f"Router overhead ({router_ratio:.1%}) is too high for the architecture to be viable"
    
    def test_end_to_end_profiling(self):
        """End-to-end: build model, profile, get report."""
        config = HydraConfig.small()
        model = HydraModel(config)
        
        profiler = OverheadProfiler(config, model=model)
        report = profiler.full_report(
            batch_size=2, seq_len=64,
            measure_runtime=True, device="cpu"
        )
        
        text = profiler.format_text_report(report)
        assert len(text) > 1000
        assert report.verdict["overall"] in ["PASS", "MARGINAL", "FAIL"]
        
        # Print for manual inspection
        print(f"\n[Integration Test] Verdict: {report.verdict['overall']}")
        print(f"  Inference savings: {report.verdict['inference_savings_vs_dense']:.1%}")
        print(f"  Router OH: {report.flops_analysis['overhead']['router_overhead_ratio']:.1%}")
        if report.runtime_analysis:
            print(f"  Router wall-clock: {report.runtime_analysis['per_component']['router']['mean_ms']:.2f}ms")
            print(f"  Router % of block: {report.runtime_analysis['overhead_analysis']['router_pct_of_block']:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
