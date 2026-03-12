#!/usr/bin/env python3
"""Run the pooled multimodal centralized baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fedgate_final.training.centralized import run_centralized_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "centralized_multimodal.yaml"),
        help="Path to the centralized YAML config.",
    )
    parser.add_argument("--output-root", default="", help="Override results root.")
    parser.add_argument("--device", default="", help="Override device.")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size.")
    parser.add_argument("--folds", type=int, default=0, help="Override number of folds.")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override dataloader workers.")
    parser.add_argument("--limit-rows", type=int, default=-1, help="Limit rows for smoke tests.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds.")
    parser.add_argument("--no-resume", action="store_true", help="Force retraining even if fold checkpoints exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config only.")
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
    print(f"[centralized] config={config_path}")
    print(f"[centralized] output_root={output_root}")
    print(
        "[centralized] overrides="
        f"epochs={args.epochs or 'config'} batch_size={args.batch_size or 'config'} "
        f"folds={args.folds or 'config'} num_workers={args.num_workers if args.num_workers >= 0 else 'config'} "
        f"device={args.device or 'config'} seeds={args.seeds or 'config'} "
        f"limit_rows={args.limit_rows if args.limit_rows >= 0 else 'config'} "
        f"resume={'no' if args.no_resume else 'yes'}"
    )
    if args.dry_run:
        return

    result = run_centralized_experiment(
        config_path=config_path,
        output_root=output_root,
        seeds_override=args.seeds,
        device_override=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        folds_override=args.folds,
        num_workers_override=args.num_workers,
        limit_rows_override=args.limit_rows,
        resume_existing=not args.no_resume,
    )
    print(f"[centralized] wrote {result['experiment_dir']}")


if __name__ == "__main__":
    main()
