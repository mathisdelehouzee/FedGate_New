#!/usr/bin/env python3
"""Run a paper-aligned federated benchmark in local simulation mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fedgate_final.training.federated import run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Scenario YAML config.")
    parser.add_argument("--method", required=True, choices=["fedavg", "fedgate"])
    parser.add_argument("--output-root", default="", help="Override results root.")
    parser.add_argument("--device", default="", help="Override device.")
    parser.add_argument("--rounds", type=int, default=0, help="Override number of federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=0, help="Override local epochs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override dataloader workers.")
    parser.add_argument("--max-samples-per-client", type=int, default=0, help="Cap train/val samples per client for smoke tests.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print without training.")
    return parser.parse_args()


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths_cfg = dict(cfg.get("paths", {}))
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else _resolve_path(config_path.parent, str(paths_cfg.get("results_root", "../results")))
    )

    print(f"[federated] config={config_path}")
    print(f"[federated] method={args.method}")
    print(f"[federated] output_root={output_root}")
    print(
        "[federated] overrides="
        f"rounds={args.rounds or 'config'} local_epochs={args.local_epochs or 'config'} "
        f"batch_size={args.batch_size or 'config'} num_workers={args.num_workers if args.num_workers >= 0 else 'config'} "
        f"device={args.device or 'config'} seeds={args.seeds or 'config'} "
        f"max_samples_per_client={args.max_samples_per_client or 'config'}"
    )
    if args.dry_run:
        return

    result = run_experiment(
        method=args.method,
        config_path=config_path,
        output_root=output_root,
        seeds=args.seeds,
        device_override=args.device,
        rounds_override=args.rounds,
        local_epochs_override=args.local_epochs,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        max_samples_per_client=args.max_samples_per_client,
    )
    print(f"[federated] wrote {result['experiment_dir']}")


if __name__ == "__main__":
    main()
