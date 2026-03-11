#!/usr/bin/env python3
"""Precompute resized MRI tensors on disk for faster training runs."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fedgate_final.data.cbms import (
    build_mri_transforms,
    cache_mri_volume,
    load_cbms_csv,
    mri_cache_path,
)

_WORKER_TRANSFORMS: Any = None
_WORKER_SHAPE: tuple[int, int, int] | None = None
_WORKER_CACHE_DIR: Path | None = None


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for MRI preprocessing. 0/1 keeps sequential mode.",
    )
    return parser.parse_args()


def _init_worker(mri_shape: tuple[int, int, int], cache_dir: str) -> None:
    global _WORKER_TRANSFORMS, _WORKER_SHAPE, _WORKER_CACHE_DIR
    _WORKER_SHAPE = mri_shape
    _WORKER_CACHE_DIR = Path(cache_dir)
    _WORKER_TRANSFORMS = build_mri_transforms(mri_shape)


def _precompute_single(scan_path_str: str) -> tuple[str, bool]:
    if _WORKER_TRANSFORMS is None or _WORKER_SHAPE is None or _WORKER_CACHE_DIR is None:
        raise RuntimeError("Precompute worker was not initialized.")
    scan_path = Path(scan_path_str)
    cache_path = mri_cache_path(scan_path, _WORKER_SHAPE, _WORKER_CACHE_DIR)
    already_exists = cache_path.exists()
    cache_mri_volume(
        scan_path,
        _WORKER_TRANSFORMS,
        _WORKER_SHAPE,
        cache_dir=_WORKER_CACHE_DIR,
        cache_enabled=False,
        cache_max_items=0,
    )
    return cache_path.name, already_exists


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
    requested_workers = int(args.num_workers)
    num_workers = max(1, requested_workers) if requested_workers > 0 else 1
    pending_rows = [
        row
        for row in rows
        if not mri_cache_path(row.scan_path, mri_shape, cache_dir).exists()
    ]
    print(f"[precompute_mri_cache] config={config_path}")
    print(f"[precompute_mri_cache] cache_dir={cache_dir}")
    print(f"[precompute_mri_cache] rows={total}")
    print(f"[precompute_mri_cache] pending={len(pending_rows)}")
    print(f"[precompute_mri_cache] workers={num_workers}")
    if not pending_rows:
        print("[precompute_mri_cache] cache already complete", flush=True)
        return

    if num_workers == 1:
        for index, row in enumerate(pending_rows, start=1):
            cache_path = cache_mri_volume(
                row.scan_path,
                transforms,
                mri_shape,
                cache_dir=cache_dir,
                cache_enabled=False,
                cache_max_items=0,
            )
            if index == 1 or index % 100 == 0 or index == len(pending_rows):
                print(
                    f"[precompute_mri_cache] {index}/{len(pending_rows)} -> {cache_path.name}",
                    flush=True,
                )
        return

    max_workers = min(num_workers, len(pending_rows), os.cpu_count() or num_workers)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(mri_shape, str(cache_dir)),
    ) as executor:
        futures = [executor.submit(_precompute_single, str(row.scan_path)) for row in pending_rows]
        for future in as_completed(futures):
            cache_name, _ = future.result()
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(pending_rows):
                print(
                    f"[precompute_mri_cache] {completed}/{len(pending_rows)} -> {cache_name}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
