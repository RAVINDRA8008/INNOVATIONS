"""
Evaluate HYDRA with CERA Benchmark

Usage:
    python experiments/evaluate.py --checkpoint path/to/checkpoint.pt
    python experiments/evaluate.py --config small  # Evaluate random init (for testing)

Runs the full CERA evaluation and outputs a detailed report including:
    - CERA Score (Accuracy × Efficiency)
    - Per-category and per-difficulty breakdowns
    - Routing distribution analysis
    - Difficulty calibration assessment
"""

import argparse
import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hydra.model.config import HydraConfig
from hydra.model.hydra_model import HydraModel
from hydra.benchmark.evaluator import CERAEvaluator

logger = logging.getLogger("hydra.evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HYDRA with CERA benchmark")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        help="Config name (used when no checkpoint is provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device",
    )
    parser.add_argument(
        "--tasks_per_level",
        type=int,
        default=20,
        help="Number of tasks per category per difficulty level",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Categories to evaluate (default: all)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        type=int,
        default=None,
        help="Difficulty levels to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/cera_results.json",
        help="Output file for results",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = HydraConfig(**{
            k: v for k, v in checkpoint["config"].items()
            if k in HydraConfig.__dataclass_fields__
        })
        model = HydraModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        named_configs = {"small": HydraConfig.small, "base": HydraConfig.base}
        config = named_configs.get(args.config, HydraConfig.small)()
        model = HydraModel(config)
        logger.info(f"Using random-initialized model ({args.config})")
    
    logger.info(f"Model: {model}")
    
    # Create evaluator
    evaluator = CERAEvaluator(
        model=model,
        device=args.device,
        tasks_per_category_per_level=args.tasks_per_level,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        categories=args.categories,
        difficulties=args.difficulties,
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    
    logger.info(f"Results saved to: {args.output}")
    
    # Print summary
    cera = results.get("cera_score", {})
    print(f"\n{'='*50}")
    print(f"CERA Score: {cera.get('score', 'N/A'):.4f}")
    print(f"  Accuracy:   {cera.get('accuracy_component', 'N/A'):.4f}")
    print(f"  Efficiency: {cera.get('efficiency_component', 'N/A'):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
