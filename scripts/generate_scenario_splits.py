#!/usr/bin/env python3
"""Generate scenario split manifests from federated YAML configs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fedgate_final.data.cbms import FEATURES, load_cbms_csv


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        required=True,
        help="Scenario YAML config. Can be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing splits_manifest.json and splits_report.json.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _labels_by_index(rows: list[Any]) -> dict[int, int]:
    return {int(row.index): int(row.label) for row in rows}


def _subject_index(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(row.index),
            "subject_id": str(row.subject_id),
            "label": int(row.label),
            "source": str(row.source),
        }
        for row in rows
    ]


def _compute_size_gap_tolerance(total_count: int, partition_cfg: dict[str, Any]) -> int:
    percent = float(partition_cfg.get("max_size_gap_percent", 0.0))
    sample_limit = int(partition_cfg.get("max_size_gap_samples", 0))
    percent_limit = int(total_count * percent / 100.0)
    if sample_limit <= 0:
        return percent_limit
    if percent_limit <= 0:
        return sample_limit
    return min(percent_limit, sample_limit)


def _group_indices_by_label(indices: list[int], labels: dict[int, int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        grouped[int(labels[int(index)])].append(int(index))
    return grouped


def _positive_ratio(indices: list[int], labels: dict[int, int]) -> float:
    if not indices:
        return 0.0
    positives = sum(1 for index in indices if int(labels[int(index)]) == 1)
    return float(positives / len(indices))


def _stratified_split(
    indices: list[int],
    labels: dict[int, int],
    ratio: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    pool = [int(index) for index in indices]
    if not pool or ratio <= 0.0:
        return pool, []
    if len(pool) == 1:
        return pool, []

    grouped = _group_indices_by_label(pool, labels)
    selected: list[int] = []
    remaining: list[int] = []
    for bucket in grouped.values():
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        take = int(round(len(bucket_copy) * ratio))
        take = max(0, min(take, len(bucket_copy)))
        selected.extend(bucket_copy[:take])
        remaining.extend(bucket_copy[take:])

    target = int(round(len(pool) * ratio))
    if ratio > 0.0:
        target = max(1, target)
    target = min(target, len(pool) - 1)

    rng.shuffle(selected)
    rng.shuffle(remaining)
    while len(selected) < target and remaining:
        selected.append(remaining.pop())
    while len(selected) > target:
        remaining.append(selected.pop())

    rng.shuffle(selected)
    rng.shuffle(remaining)
    return remaining, selected


def _balanced_partition(
    indices: list[int],
    labels: dict[int, int],
    num_clients: int,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    grouped = _group_indices_by_label(indices, labels)
    client_indices: dict[int, list[int]] = {client_id: [] for client_id in range(num_clients)}
    for bucket in grouped.values():
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        for client_id, part in enumerate(np.array_split(np.asarray(bucket_copy, dtype=np.int64), num_clients)):
            client_indices[client_id].extend(int(index) for index in part.tolist())
    for client_id in client_indices:
        rng.shuffle(client_indices[client_id])
    return client_indices


def _dirichlet(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    safe_alpha = max(float(alpha), 1e-3)
    return rng.dirichlet(np.full(size, safe_alpha, dtype=np.float64))


def _bounded_multinomial(
    total: int,
    probs: np.ndarray,
    capacities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.zeros_like(capacities, dtype=np.int64)
    remaining = int(total)
    caps = capacities.astype(np.int64, copy=True)
    weights = probs.astype(np.float64, copy=True)
    while remaining > 0:
        eligible = caps > 0
        if not np.any(eligible):
            raise RuntimeError("Insufficient remaining capacity while allocating class counts.")
        active_weights = np.where(eligible, weights, 0.0)
        if active_weights.sum() <= 0.0:
            active_weights = eligible.astype(np.float64)
        active_weights = active_weights / active_weights.sum()
        draw = rng.choice(len(caps), p=active_weights)
        counts[draw] += 1
        caps[draw] -= 1
        remaining -= 1
    return counts


def _sample_client_totals(
    total_count: int,
    num_clients: int,
    min_client_samples: int,
    quantity_alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if total_count < num_clients * min_client_samples:
        raise RuntimeError(
            f"Cannot satisfy min_client_samples={min_client_samples} with total_count={total_count} "
            f"and num_clients={num_clients}."
        )
    quantity_weights = _dirichlet(quantity_alpha, num_clients, rng)
    residual = total_count - num_clients * min_client_samples
    extras = rng.multinomial(residual, quantity_weights) if residual > 0 else np.zeros(num_clients, dtype=np.int64)
    totals = extras.astype(np.int64) + int(min_client_samples)
    return totals, quantity_weights


def _label_quantity_partition(
    indices: list[int],
    labels: dict[int, int],
    partition_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    num_clients = int(partition_cfg["num_clients"])
    quantity_alpha = float(partition_cfg.get("quantity_skew_alpha", 1.0))
    label_alpha = float(partition_cfg.get("label_skew_alpha", 1.0))
    min_client_samples = int(partition_cfg.get("min_client_samples", 1))
    max_attempts = int(partition_cfg.get("max_partition_attempts", 128))

    grouped = _group_indices_by_label(indices, labels)
    for bucket in grouped.values():
        rng.shuffle(bucket)
    ordered_labels = sorted(grouped.keys())

    for _ in range(max_attempts):
        desired_totals, quantity_weights = _sample_client_totals(
            total_count=len(indices),
            num_clients=num_clients,
            min_client_samples=min_client_samples,
            quantity_alpha=quantity_alpha,
            rng=rng,
        )
        remaining_capacity = desired_totals.copy()
        class_counts: dict[int, np.ndarray] = {}
        for label in ordered_labels[:-1]:
            bucket = grouped[label]
            label_weights = _dirichlet(label_alpha, num_clients, rng)
            probs = quantity_weights * label_weights
            probs = probs / probs.sum()
            counts = _bounded_multinomial(len(bucket), probs, remaining_capacity, rng)
            class_counts[int(label)] = counts
            remaining_capacity = remaining_capacity - counts
        last_label = ordered_labels[-1]
        class_counts[int(last_label)] = remaining_capacity.copy()

        client_indices: dict[int, list[int]] = {client_id: [] for client_id in range(num_clients)}
        for label, bucket in grouped.items():
            cursor = 0
            for client_id, take in enumerate(class_counts[int(label)].tolist()):
                if take:
                    client_indices[client_id].extend(int(index) for index in bucket[cursor : cursor + take])
                cursor += take
        for client_id in client_indices:
            rng.shuffle(client_indices[client_id])
        return client_indices

    raise RuntimeError(
        "Unable to sample a label_quantity_skew partition that satisfies min_client_samples. "
        f"Try increasing quantity_skew_alpha or reducing min_client_samples (attempts={max_attempts})."
    )


def _partition_indices(
    indices: list[int],
    labels: dict[int, int],
    partition_cfg: dict[str, Any],
    seed: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    strategy = str(partition_cfg.get("strategy", "balanced")).strip()
    if strategy == "balanced":
        return _balanced_partition(indices, labels, int(partition_cfg["num_clients"]), rng)
    if strategy == "label_quantity_skew":
        return _label_quantity_partition(indices, labels, partition_cfg, rng)
    raise ValueError(f"Unsupported partition strategy: {strategy}")


def _seed_payload(
    train_pool_indices: list[int],
    labels: dict[int, int],
    split_cfg: dict[str, Any],
    partition_cfg: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client_partitions = _partition_indices(train_pool_indices, labels, partition_cfg, seed)
    client_splits: dict[str, dict[str, list[int]]] = {}
    global_train_indices: list[int] = []
    global_val_indices: list[int] = []
    client_pos_ratios: dict[str, float] = {}
    val_ratio = float(split_cfg.get("client_val_ratio", 0.1))

    for client_id, indices in client_partitions.items():
        split_rng = np.random.default_rng(seed * 1000 + client_id)
        train_indices, val_indices = _stratified_split(indices, labels, val_ratio, split_rng)
        client_splits[str(client_id)] = {
            "train_indices": [int(index) for index in train_indices],
            "val_indices": [int(index) for index in val_indices],
        }
        global_train_indices.extend(train_indices)
        global_val_indices.extend(val_indices)
        client_pos_ratios[str(client_id)] = _positive_ratio(indices, labels)

    partition_sizes = [len(indices) for indices in client_partitions.values()]
    size_gap = max(partition_sizes) - min(partition_sizes) if partition_sizes else 0
    size_gap_tolerance = _compute_size_gap_tolerance(len(train_pool_indices), partition_cfg)
    global_pos_ratio = _positive_ratio(train_pool_indices, labels)
    max_ratio_deviation = (
        max(abs(float(ratio) - global_pos_ratio) for ratio in client_pos_ratios.values())
        if client_pos_ratios
        else 0.0
    )
    diagnostics = {
        "size_gap": int(size_gap),
        "size_gap_tolerance": int(size_gap_tolerance),
        "global_pos_ratio": float(global_pos_ratio),
        "client_pos_ratios": client_pos_ratios,
        "max_ratio_deviation": float(max_ratio_deviation),
        "label_tolerance": float(partition_cfg.get("label_tolerance", 0.0)),
        "partition_strategy": str(partition_cfg.get("strategy", "balanced")),
    }

    payload = {
        "seed": int(seed),
        "client_partitions": {str(cid): [int(index) for index in indices] for cid, indices in client_partitions.items()},
        "client_splits": client_splits,
        "global_train_indices": [int(index) for index in global_train_indices],
        "global_val_indices": [int(index) for index in global_val_indices],
        "diagnostics": diagnostics,
    }
    report = {
        "num_clients": int(partition_cfg["num_clients"]),
        "size_min": int(min(partition_sizes)) if partition_sizes else 0,
        "size_max": int(max(partition_sizes)) if partition_sizes else 0,
        "size_gap": int(size_gap),
        "size_gap_tolerance": int(size_gap_tolerance),
        "max_label_ratio_deviation": float(max_ratio_deviation),
        "label_tolerance": float(partition_cfg.get("label_tolerance", 0.0)),
        "num_global_train": int(len(global_train_indices)),
        "num_global_val": int(len(global_val_indices)),
        "num_global_test": 0,
    }
    return payload, report


def _global_split_payload(
    all_indices: list[int],
    labels: dict[int, int],
    split_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[int], list[int]]:
    seed = int(split_cfg.get("global_test_seed", 2026))
    ratio = float(split_cfg.get("global_test_ratio", 0.2))
    rng = np.random.default_rng(seed)
    train_pool_indices, test_indices = _stratified_split(all_indices, labels, ratio, rng)
    train_pool_counts = {
        "negative": int(sum(1 for index in train_pool_indices if int(labels[int(index)]) == 0)),
        "positive": int(sum(1 for index in train_pool_indices if int(labels[int(index)]) == 1)),
    }
    test_counts = {
        "negative": int(sum(1 for index in test_indices if int(labels[int(index)]) == 0)),
        "positive": int(sum(1 for index in test_indices if int(labels[int(index)]) == 1)),
    }
    payload = {
        "global_test_seed": int(seed),
        "global_test_ratio": float(ratio),
        "train_pool_indices": [int(index) for index in train_pool_indices],
        "test_indices": [int(index) for index in test_indices],
        "train_pool_counts": train_pool_counts,
        "test_counts": test_counts,
    }
    return payload, train_pool_indices, test_indices


def _validate_seed_report(
    seed_report: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    seed: int,
) -> None:
    if int(seed_report["size_gap"]) > int(seed_report["size_gap_tolerance"]):
        errors.append(
            f"seed={seed}: size_gap={seed_report['size_gap']} exceeds tolerance={seed_report['size_gap_tolerance']}"
        )
    if float(seed_report["max_label_ratio_deviation"]) > float(seed_report["label_tolerance"]):
        errors.append(
            "seed="
            f"{seed}: max_label_ratio_deviation={seed_report['max_label_ratio_deviation']:.6f} "
            f"exceeds tolerance={seed_report['label_tolerance']:.6f}"
        )
    if int(seed_report["num_global_train"]) <= 0:
        errors.append(f"seed={seed}: empty global_train split")
    if int(seed_report["num_global_val"]) <= 0:
        warnings.append(f"seed={seed}: empty global_val split")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _generate_for_config(config_path: Path, overwrite: bool) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML mapping: {config_path}")

    artifacts_root = _resolve_path(config_path.parent, str(cfg["paths"]["artifacts_root"]))
    manifest_path = artifacts_root / "splits" / "splits_manifest.json"
    report_path = artifacts_root / "splits" / "splits_report.json"
    if manifest_path.exists() and report_path.exists() and not overwrite:
        print(f"[generate_scenario_splits] skip {config_path} (already exists)")
        return

    data_cfg = dict(cfg.get("data", {}))
    split_cfg = dict(cfg.get("split", {}))
    partition_cfg = dict(cfg.get("partition", {}))
    scenario_cfg = dict(cfg.get("scenario", {}))
    seeds = [int(seed) for seed in cfg.get("seeds", [])]
    if not seeds:
        raise ValueError(f"Missing seeds in {config_path}")
    if "num_clients" not in partition_cfg:
        raise ValueError(f"Missing partition.num_clients in {config_path}")

    csv_path = _resolve_path(config_path.parent, str(data_cfg["csv"]))
    data_root = _resolve_path(config_path.parent, str(data_cfg["data_root"]))
    rows = load_cbms_csv(csv_path, data_root=data_root, features=data_cfg.get("feature_names", FEATURES))
    labels = _labels_by_index(rows)
    all_indices = [int(row.index) for row in rows]

    global_split, train_pool_indices, test_indices = _global_split_payload(all_indices, labels, split_cfg)
    size_gap_tolerance = _compute_size_gap_tolerance(len(train_pool_indices), partition_cfg)

    manifest = {
        "experiment": str(cfg.get("experiment", config_path.stem)),
        "created_at_utc": _utc_now(),
        "config_path": str(config_path),
        "data": {
            "csv": str(csv_path),
            "data_root": str(data_root),
            "num_samples": int(len(rows)),
            "num_features": int(len(data_cfg.get("feature_names", FEATURES))),
        },
        "constraints": {
            "num_clients": int(partition_cfg["num_clients"]),
            "partition_strategy": str(partition_cfg.get("strategy", "balanced")),
            "label_tolerance": float(partition_cfg.get("label_tolerance", 0.0)),
            "size_gap_tolerance": int(size_gap_tolerance),
            "size_gap_rule": {
                "max_size_gap_percent": float(partition_cfg.get("max_size_gap_percent", 0.0)),
                "max_size_gap_samples": int(partition_cfg.get("max_size_gap_samples", 0)),
            },
            "partition": partition_cfg,
        },
        "global_split": global_split,
        "seeds": {},
        "scenario": scenario_cfg,
        "subject_index": _subject_index(rows),
    }

    errors: list[str] = []
    warnings: list[str] = []
    seed_reports: dict[str, Any] = {}
    for seed in seeds:
        payload, seed_report = _seed_payload(train_pool_indices, labels, split_cfg, partition_cfg, seed)
        seed_report["num_global_test"] = int(len(test_indices))
        manifest["seeds"][str(seed)] = payload
        seed_reports[str(seed)] = seed_report
        _validate_seed_report(seed_report, errors, warnings, seed)

    _write_json(manifest_path, manifest)
    _write_json(
        report_path,
        {
            "checked_at_utc": _utc_now(),
            "manifest": str(manifest_path),
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "seed_reports": seed_reports,
        },
    )
    print(
        "[generate_scenario_splits] wrote "
        f"{manifest_path} and {report_path} (ok={not errors}, warnings={len(warnings)})"
    )


def main() -> None:
    args = _parse_args()
    for raw_config in args.configs:
        config_path = Path(raw_config).expanduser().resolve()
        _generate_for_config(config_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
