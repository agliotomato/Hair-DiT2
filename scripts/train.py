"""
Training entry point.

Usage:
  # Phase 1 (unbraid pretraining):
  python scripts/train.py --config configs/phase1_unbraid.yaml

  # Phase 2 (braid fine-tuning):
  python scripts/train.py --config configs/phase2_braid.yaml

  # Multi-GPU:
  accelerate launch scripts/train.py --config configs/phase1_unbraid.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base config if specified."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Merge with base config
    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        # Deep merge: cfg overrides base
        merged = deep_merge(base_cfg, cfg)
        return merged

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config loaded: phase={cfg['training']['phase']}, dataset={cfg['training']['dataset']}")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
