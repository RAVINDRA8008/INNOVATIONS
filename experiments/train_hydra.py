"""
Train HYDRA — Main Training Script

Usage:
    python experiments/train_hydra.py --config small    # Quick debug run
    python experiments/train_hydra.py --config base     # Full training
    python experiments/train_hydra.py --config path/to/config.yaml  # Custom config

This script orchestrates:
    1. Config loading
    2. Model instantiation  
    3. Training with curriculum scheduling
    4. Periodic evaluation on CERA benchmark
    5. Checkpoint management
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hydra.model.config import HydraConfig
from hydra.model.hydra_model import HydraModel
from hydra.training.trainer import HydraTrainer

logger = logging.getLogger("hydra.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HYDRA model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        help="Config name ('small', 'base', 'large') or path to YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device ('auto', 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--no_curriculum",
        action="store_true",
        help="Disable curriculum learning",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    return parser.parse_args()


def load_config(config_arg: str) -> HydraConfig:
    """Load configuration from name or file path."""
    named_configs = {
        "small": HydraConfig.small,
        "base": HydraConfig.base,
        "large": HydraConfig.large,
    }
    
    if config_arg in named_configs:
        config = named_configs[config_arg]()
        logger.info(f"Using named config: {config_arg}")
    elif os.path.exists(config_arg):
        config = HydraConfig.from_yaml(config_arg)
        logger.info(f"Loaded config from: {config_arg}")
    else:
        raise ValueError(f"Unknown config: {config_arg}")
    
    return config


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_curriculum:
        config.curriculum_enabled = False
    
    # Create model
    model = HydraModel(config)
    
    logger.info(f"Model created: {model}")
    logger.info(f"Parameters: {model._n_params:,}")
    
    # Create trainer
    trainer = HydraTrainer(
        config=config,
        model=model,
        output_dir=args.output_dir,
        device=args.device,
        use_amp=not args.no_amp,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    summary = trainer.train(max_steps=args.max_steps)
    
    logger.info("Training complete!")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
