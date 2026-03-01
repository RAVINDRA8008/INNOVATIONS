"""
Tests for HYDRA Model Components

Validates:
    1. Model instantiation and forward pass
    2. Pathway modules produce correct shapes
    3. Router produces valid routing weights
    4. Losses are computable and finite
    5. Benchmark generates valid tasks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from hydra.model.config import HydraConfig
from hydra.model.hydra_model import HydraModel
from hydra.model.hydra_block import HydraBlock
from hydra.model.router import PathwayRouter
from hydra.model.ssm_pathway import SSMPathway
from hydra.model.attention_pathway import (
    WindowedAttentionPathway,
    GlobalAttentionPathway,
)
from hydra.training.adaptive_loss import AdaptiveComputeBudgetLoss
from hydra.training.curriculum import CurriculumScheduler
from hydra.benchmark.cera import CERABenchmark
from hydra.benchmark.metrics import ComputeEfficiencyMetrics, TaskResult


# Use small config for fast tests
@pytest.fixture
def config():
    return HydraConfig.small()


@pytest.fixture
def batch(config):
    """Create a test batch."""
    B, L = 2, 64
    return torch.randn(B, L, config.d_model)


@pytest.fixture
def input_ids(config):
    """Create test input IDs."""
    B, L = 2, 64
    return torch.randint(0, config.vocab_size, (B, L))


class TestSSMPathway:
    def test_forward_shape(self, config, batch):
        ssm = SSMPathway(config)
        output = ssm(batch)
        assert output.shape == batch.shape, f"Expected {batch.shape}, got {output.shape}"
    
    def test_compute_profile(self, config):
        ssm = SSMPathway(config)
        profile = ssm.get_compute_profile()
        assert profile["normalized_cost"] == 1.0
        assert "SSM" in profile["name"]


class TestWindowedAttention:
    def test_forward_shape(self, config, batch):
        focus = WindowedAttentionPathway(config)
        output = focus(batch)
        assert output.shape == batch.shape
    
    def test_short_sequence(self, config):
        """Test with sequence shorter than window size."""
        focus = WindowedAttentionPathway(config)
        short_batch = torch.randn(2, 16, config.d_model)
        output = focus(short_batch)
        assert output.shape == short_batch.shape


class TestGlobalAttention:
    def test_forward_shape(self, config, batch):
        reason = GlobalAttentionPathway(config)
        output = reason(batch)
        assert output.shape == batch.shape


class TestRouter:
    def test_forward_shape(self, config, batch):
        router = PathwayRouter(config)
        weights, info = router(batch, training=True)
        assert weights.shape == (batch.shape[0], batch.shape[1], 3)
    
    def test_routing_weights_valid(self, config, batch):
        router = PathwayRouter(config)
        weights, info = router(batch, training=True)
        # Weights should sum to ~1 per token (one-hot from Gumbel-Softmax)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_aux_losses_finite(self, config, batch):
        router = PathwayRouter(config)
        _, info = router(batch, training=True)
        assert torch.isfinite(info["load_balance_loss"])
        assert torch.isfinite(info["compute_budget_loss"])
        assert torch.isfinite(info["total_aux_loss"])
    
    def test_inference_mode(self, config, batch):
        router = PathwayRouter(config)
        router.eval()
        weights, info = router(batch, training=False)
        # Inference should produce hard one-hot routing
        assert weights.sum(dim=-1).allclose(torch.ones(batch.shape[0], batch.shape[1]))


class TestHydraBlock:
    def test_forward_shape(self, config, batch):
        block = HydraBlock(config, layer_idx=0)
        output, info = block(batch)
        assert output.shape == batch.shape
    
    def test_block_info_contents(self, config, batch):
        block = HydraBlock(config, layer_idx=0)
        _, info = block(batch)
        assert "layer_idx" in info
        assert "router_info" in info
        assert "routing_weights" in info


class TestHydraModel:
    def test_forward_no_targets(self, config, input_ids):
        model = HydraModel(config)
        output = model(input_ids)
        B, L = input_ids.shape
        assert output["logits"].shape == (B, L, config.vocab_size)
    
    def test_forward_with_targets(self, config, input_ids):
        model = HydraModel(config)
        targets = torch.randint(0, config.vocab_size, input_ids.shape)
        output = model(input_ids, targets=targets)
        assert "loss" in output
        assert "total_loss" in output
        assert torch.isfinite(output["loss"])
    
    def test_routing_stats(self, config, input_ids):
        model = HydraModel(config)
        output = model(input_ids)
        stats = output["routing_stats"]
        assert "avg_stream_frac" in stats
        assert "avg_focus_frac" in stats
        assert "avg_reason_frac" in stats
    
    def test_routing_map(self, config, input_ids):
        model = HydraModel(config)
        rmap = model.get_routing_map(input_ids)
        assert "routing_map" in rmap
        assert "decisions" in rmap
        assert rmap["decisions"].shape[0] == config.n_layers
    
    def test_parameter_count(self, config):
        model = HydraModel(config)
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["trainable"] > 0
    
    def test_backward_pass(self, config, input_ids):
        model = HydraModel(config)
        targets = torch.randint(0, config.vocab_size, input_ids.shape)
        output = model(input_ids, targets=targets)
        output["total_loss"].backward()
        # Check that most parameters have gradients (some like boundary_gate
        # only activate for long sequences exceeding window_size)
        grads_found = sum(
            1 for _, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        )
        total_params = sum(
            1 for _, p in model.named_parameters() if p.requires_grad
        )
        assert grads_found / total_params > 0.95, (
            f"Only {grads_found}/{total_params} parameters received gradients"
        )


class TestAdaptiveLoss:
    def test_loss_computation(self, config, input_ids):
        model = HydraModel(config)
        loss_fn = AdaptiveComputeBudgetLoss()
        
        targets = torch.randint(0, config.vocab_size, input_ids.shape)
        output = model(input_ids, targets=targets)
        
        loss_info = loss_fn(output)
        assert torch.isfinite(loss_info["total_loss"])
        assert torch.isfinite(loss_info["efficiency_loss"])
        assert torch.isfinite(loss_info["consistency_loss"])


class TestCurriculum:
    def test_temperature_annealing(self, config):
        scheduler = CurriculumScheduler(config)
        model = HydraModel(config)
        
        # Step through some iterations
        temps = []
        for _ in range(100):
            info = scheduler.step(model)
            temps.append(info["temperature"])
        
        # Temperature should generally decrease
        assert temps[-1] <= temps[0]
    
    def test_phase_progression(self, config):
        # Use small max_steps so we hit multiple phases
        config.max_steps = 100
        scheduler = CurriculumScheduler(config)
        model = HydraModel(config)
        
        phases_seen = set()
        for _ in range(100):
            info = scheduler.step(model)
            phases_seen.add(info["phase"])
        
        # Should see at least 2 different phases
        assert len(phases_seen) >= 2


class TestCERABenchmark:
    def test_generation(self):
        benchmark = CERABenchmark(tasks_per_category_per_level=5)
        tasks = benchmark.generate()
        
        # 4 categories × 5 levels × 5 tasks = 100
        assert len(tasks) == 100
    
    def test_task_fields(self):
        benchmark = CERABenchmark(tasks_per_category_per_level=2)
        tasks = benchmark.generate()
        
        for task in tasks:
            assert task.task_id
            assert task.category in ["arithmetic", "logical", "pattern", "compositional"]
            assert 1 <= task.difficulty <= 5
            assert task.input_text
            assert task.target_text
            assert task.reasoning_steps > 0
    
    def test_difficulty_filtering(self):
        benchmark = CERABenchmark(tasks_per_category_per_level=5)
        benchmark.generate()
        
        level3 = benchmark.get_tasks_by_difficulty(3)
        assert all(t.difficulty == 3 for t in level3)


class TestCERAMetrics:
    def test_basic_metrics(self):
        metrics = ComputeEfficiencyMetrics()
        
        # Add some mock results
        for i in range(20):
            metrics.add_result(TaskResult(
                task_id=f"test_{i}",
                category="arithmetic",
                difficulty=(i % 5) + 1,
                correct=i % 3 != 0,  # ~67% accuracy
                predicted=str(i),
                target=str(i if i % 3 != 0 else i + 1),
                routing_decisions=[i % 3] * 10,
                compute_cost=float((i % 3) * 3 + 1),
                target_compute=float((i % 5) * 2 + 1),
            ))
        
        result = metrics.compute_all()
        
        assert "cera_score" in result
        assert "accuracy" in result
        assert "routing_entropy" in result
        assert 0 <= result["cera_score"]["score"] <= 1
    
    def test_report_generation(self):
        metrics = ComputeEfficiencyMetrics()
        
        for i in range(10):
            metrics.add_result(TaskResult(
                task_id=f"test_{i}",
                category="logical",
                difficulty=3,
                correct=True,
                predicted="yes",
                target="yes",
                routing_decisions=[1] * 5,
                compute_cost=3.0,
                target_compute=3.0,
            ))
        
        report = metrics.generate_report()
        assert "CERA" in report
        assert len(report) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
