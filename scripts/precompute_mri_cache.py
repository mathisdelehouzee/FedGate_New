#!/usr/bin/env python3
"""Precompute resized MRI tensors on disk for faster training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fedgate_final.data.cbms import build_mri_transforms, cache_mri_volume, load_cbms_csv


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config with data and mri_cache settings.")
    parser.add_argument("--limit-rows", type=int, default=0, help="Optional row cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = dict(cfg.get("data", {}))
    if "mri_cache" not in data_cfg or "dir" not in dict(data_cfg.get("mri_cache", {})):
        raise ValueError(f"Missing data.mri_cache.dir in {config_path}")

    csv_path = _resolve_path(config_path.parent, str(data_cfg["csv"]))
    data_root = _resolve_path(config_path.parent, str(data_cfg["data_root"]))
    cache_cfg = dict(data_cfg.get("mri_cache", {}))
    cache_dir = _resolve_path(config_path.parent, str(cache_cfg["dir"]))
    mri_shape = tuple(int(v) for v in data_cfg.get("mri_shape", [128, 128, 128]))
    transforms = build_mri_transforms(mri_shape)
    rows = load_cbms_csv(csv_path, data_root=data_root)
    if args.limit_rows > 0:
        rows = rows[: args.limit_rows]

    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    print(f"[precompute_mri_cache] config={config_path}")
    print(f"[precompute_mri_cache] cache_dir={cache_dir}")
    print(f"[precompute_mri_cache] rows={total}")
    for index, row in enumerate(rows, start=1):
        cache_path = cache_mri_volume(
            row.scan_path,
            transforms,
            mri_shape,
            cache_dir=cache_dir,
            cache_enabled=False,
            cache_max_items=0,
        )
        if index == 1 or index % 100 == 0 or index == total:
            print(f"[precompute_mri_cache] {index}/{total} -> {cache_path.name}", flush=True)


if __name__ == "__main__":
    main()
