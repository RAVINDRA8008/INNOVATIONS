"""Quick smoke test for the HYDRA system."""
import torch
from hydra.model.config import HydraConfig
from hydra.model.hydra_model import HydraModel

# Instantiate small model
config = HydraConfig.small()
model = HydraModel(config)
print(model)
print()

# Quick forward pass
x = torch.randint(0, 32000, (2, 64))
t = torch.randint(0, 32000, (2, 64))
out = model(x, targets=t)

print(f"Loss: {out['loss'].item():.4f}")
print(f"Aux Loss: {out['aux_loss'].item():.6f}")
stats = out['routing_stats']
print(f"Routing: Stream={stats['avg_stream_frac']:.3f} Focus={stats['avg_focus_frac']:.3f} Reason={stats['avg_reason_frac']:.3f}")
print(f"Compute Cost: {stats['avg_compute_cost']:.2f}")
print()

# Quick backward pass
out['total_loss'].backward()
print("Backward pass: OK")
print()

# Parameter count
counts = model.count_parameters()
print(f"Total parameters: {counts['total']:,}")
print(f"Trainable parameters: {counts['trainable']:,}")

# CERA benchmark test
from hydra.benchmark.cera import CERABenchmark
bench = CERABenchmark(tasks_per_category_per_level=3)
tasks = bench.generate()
print(f"\nCERA Benchmark: {len(tasks)} tasks generated")
print(f"Sample task: {tasks[0].input_text[:80]}...")
print(f"  Answer: {tasks[0].target_text}")
print(f"  Difficulty: {tasks[0].difficulty}, Steps: {tasks[0].reasoning_steps}")
print(f"\nAll systems operational!")
