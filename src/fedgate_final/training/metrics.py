"""Metrics and reporting helpers for benchmark runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

METRIC_NAMES = ("loss", "acc", "f1", "auroc", "auprc", "sens", "spec")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def tune_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1",
    min_value: float = 0.05,
    max_value: float = 0.95,
    num_points: int = 181,
) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or y_prob.size == 0:
        return 0.5
    metric_name = str(metric).strip().lower()
    thresholds = np.linspace(float(min_value), float(max_value), max(2, int(num_points)))
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric_name == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric_name in {"acc", "accuracy"}:
            score = accuracy_score(y_true, y_pred)
        elif metric_name in {"sens", "recall", "tpr"}:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            score = tp / max(1, tp + fn)
        elif metric_name in {"spec", "specificity", "tnr"}:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            score = tn / max(1, tn + fp)
        else:
            raise ValueError(f"Unsupported threshold tuning metric: {metric}")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loss: float,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= float(threshold)).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float("nan")
    return {
        "loss": float(loss),
        "acc": float(acc),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "sens": float(sens),
        "spec": float(spec),
        "threshold": float(threshold),
    }


def weighted_average_metrics(items: Iterable[tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    values = list(items)
    if not values:
        return {name: float("nan") for name in METRIC_NAMES}
    total_weight = sum(int(weight) for weight, _ in values)
    total_weight = max(1, total_weight)
    out: Dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        weighted_sum = 0.0
        for weight, metrics in values:
            weighted_sum += int(weight) * safe_float(metrics.get(metric_name))
        out[metric_name] = float(weighted_sum / total_weight)
    return out


def summarize_client_metrics(client_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not client_metrics:
        return summary
    for metric_name in METRIC_NAMES:
        values = np.asarray([safe_float(metrics.get(metric_name)) for metrics in client_metrics.values()], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[metric_name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "fairness_gap": float("nan"),
            }
            continue
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "fairness_gap": float(values.max() - values.min()),
        }
    return summary


def mean_std_across_runs(rows: List[Dict[str, Any]], metric_keys: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray([safe_float(row.get(key)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_round_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in history:
        row = {
            "round": int(record["round"]),
            "num_clients": int(record.get("num_clients", 0)),
        }
        for prefix in ("train", "val"):
            payload = record.get(prefix, {})
            for metric_name in METRIC_NAMES:
                row[f"{prefix}_{metric_name}"] = safe_float(payload.get(metric_name))
        rows.append(row)
    return rows


def flatten_client_metrics(client_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client_id, metrics in sorted(client_metrics.items(), key=lambda item: item[0]):
        row = {"client_id": client_id}
        for key, value in metrics.items():
            if isinstance(value, (str, int, float)):
                row[key] = value
        rows.append(row)
    return rows
