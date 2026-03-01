#!/usr/bin/env python3
"""
HYDRA Routing Overhead Profiler — CLI Script

Runs the complete overhead analysis and produces an honest report.
Use this BEFORE claiming any compute savings in a paper.

Usage:
    python experiments/profile_overhead.py --config small --seq-len 512
    python experiments/profile_overhead.py --config base --seq-len 1024 --runtime
    python experiments/profile_overhead.py --config small --sweep  # All seq lengths
    python experiments/profile_overhead.py --config small --json results/overhead.json
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from hydra.model.config import HydraConfig
from hydra.model.hydra_model import HydraModel
from hydra.profiling import OverheadProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="HYDRA Routing Overhead Profiler")
    parser.add_argument("--config", type=str, default="small",
                       choices=["small", "base", "large"],
                       help="Model configuration")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Sequence length for analysis")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for memory analysis")
    parser.add_argument("--stream-frac", type=float, default=0.6,
                       help="Expected Stream (SSM) routing fraction")
    parser.add_argument("--focus-frac", type=float, default=0.25,
                       help="Expected Focus (Windowed) routing fraction")
    parser.add_argument("--reason-frac", type=float, default=0.15,
                       help="Expected Reason (Global) routing fraction")
    parser.add_argument("--hardware", type=str, default="gpu_a100",
                       choices=["cpu_laptop", "cpu_server", "gpu_a100", "gpu_h100"],
                       help="Hardware profile for cache analysis")
    parser.add_argument("--runtime", action="store_true",
                       help="Run actual wall-clock timing (slower)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for runtime timing (cpu/cuda)")
    parser.add_argument("--sweep", action="store_true",
                       help="Run analysis for multiple sequence lengths")
    parser.add_argument("--json", type=str, default=None,
                       help="Save report as JSON to this path")
    parser.add_argument("--sensitivity", action="store_true",
                       help="Run routing distribution sensitivity analysis")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="Data type for memory analysis")
    return parser.parse_args()


def run_sensitivity_analysis(profiler: OverheadProfiler, seq_len: int):
    """How does overhead change with different routing distributions?"""
    print("\n" + "=" * 78)
    print("  ROUTING DISTRIBUTION SENSITIVITY ANALYSIS")
    print("=" * 78)
    print(f"\n  Seq Length: {seq_len}")
    
    distributions = [
        (1.0, 0.0, 0.0, "100% Stream (SSM only)"),
        (0.8, 0.15, 0.05, "80/15/5 — Heavy SSM"),
        (0.6, 0.25, 0.15, "60/25/15 — Target distribution"),
        (0.4, 0.35, 0.25, "40/35/25 — Moderate reasoning"),
        (0.2, 0.4, 0.4, "20/40/40 — Heavy reasoning"),
        (0.0, 0.0, 1.0, "100% Reason (global only)"),
        (0.33, 0.34, 0.33, "Uniform — worst case for overhead"),
    ]
    
    print(f"\n  {'Distribution':45s} {'Inf GFLOPs':>12s} {'Savings':>10s} "
          f"{'Router OH':>10s} {'Verdict':>10s}")
    print(f"  {'─'*45} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    
    for s, f, r, label in distributions:
        analysis = profiler.flops_counter.full_analysis(seq_len, s, f, r)
        inf_gflops = analysis["total_flops"]["inference_mode"] / 1e9
        savings = analysis["savings"]["inference_vs_dense"]
        router_oh = analysis["overhead"]["router_overhead_ratio"]
        verdict = analysis["verdict"]["overall"]
        
        print(f"  {label:45s} {inf_gflops:>12.3f} {savings:>+9.1%} "
              f"{router_oh:>9.1%} {verdict:>10s}")
    
    print(f"\n  Key Insight: At target distribution (60/25/15), router overhead")
    print(f"  must be < savings from SSM routing to justify the architecture.")


def main():
    args = parse_args()
    
    # Build config
    config_builders = {
        "small": HydraConfig.small,
        "base": HydraConfig.base,
        "large": HydraConfig.large,
    }
    config = config_builders[args.config]()
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Build model if runtime profiling requested
    model = None
    if args.runtime:
        print(f"Building {args.config} model for runtime profiling...")
        model = HydraModel(config)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    profiler = OverheadProfiler(config, model=model, dtype=dtype)
    
    # Generate report
    print(f"\nRunning overhead analysis for {args.config} config...")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Routing: Stream={args.stream_frac:.0%} / Focus={args.focus_frac:.0%} / "
          f"Reason={args.reason_frac:.0%}")
    print(f"  Hardware: {args.hardware}")
    print(f"  dtype: {args.dtype}")
    
    report = profiler.full_report(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stream_frac=args.stream_frac,
        focus_frac=args.focus_frac,
        reason_frac=args.reason_frac,
        hardware=args.hardware,
        measure_runtime=args.runtime,
        device=args.device,
    )
    
    # Print text report
    text = profiler.format_text_report(report)
    print(text)
    
    # Sensitivity analysis
    if args.sensitivity:
        run_sensitivity_analysis(profiler, args.seq_len)
    
    # Sweep across sequence lengths
    if args.sweep:
        print("\n" + "=" * 78)
        print("  SEQUENCE LENGTH SWEEP")
        print("=" * 78)
        
        seq_lengths = [64, 128, 256, 512, 1024, 2048]
        if config.max_seq_len >= 4096:
            seq_lengths.append(4096)
        
        for sl in seq_lengths:
            if sl > config.max_seq_len:
                continue
            
            sweep_report = profiler.full_report(
                batch_size=args.batch_size,
                seq_len=sl,
                stream_frac=args.stream_frac,
                focus_frac=args.focus_frac,
                reason_frac=args.reason_frac,
                hardware=args.hardware,
                measure_runtime=args.runtime,
                device=args.device,
            )
            
            v = sweep_report.verdict
            flops = sweep_report.flops_analysis
            print(f"\n  SeqLen={sl:5d}: {v['overall']:8s} | "
                  f"Savings: {v['inference_savings_vs_dense']:+.1%} | "
                  f"Router OH: {flops['overhead']['router_overhead_ratio']:.1%} | "
                  f"Issues: {v['n_critical']}C/{v['n_warnings']}W")
    
    # Save JSON
    if args.json:
        os.makedirs(os.path.dirname(args.json) if os.path.dirname(args.json) else ".", exist_ok=True)
        report.to_json(args.json)
        print(f"\nReport saved to {args.json}")
    
    # Exit code based on verdict
    if report.verdict["overall"] == "FAIL":
        print("\n  *** FAIL: Routing overhead exceeds savings. See recommendations above. ***")
        sys.exit(1)
    elif report.verdict["overall"] == "MARGINAL":
        print("\n  *** MARGINAL: Overhead is concerning. Review recommendations. ***")
        sys.exit(0)
    else:
        print("\n  *** PASS: Overhead is within acceptable bounds. ***")
        sys.exit(0)


if __name__ == "__main__":
    main()
