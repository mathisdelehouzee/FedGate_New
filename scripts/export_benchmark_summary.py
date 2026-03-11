#!/usr/bin/env python3
"""Export benchmark summary tables for centralized, FedAvg, and FedGate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "benchmark_summary"))
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_centralized(results_root: Path) -> List[Dict[str, Any]]:
    metrics_path = results_root / "centralized_multimodal_paper_like" / "metrics.json"
    if not metrics_path.exists():
        return []
    payload = _load_json(metrics_path)
    aggregate = payload.get("aggregate", {}).get("ad_traj", {})
    return [
        {
            "method": "centralized",
            "scenario": "pooled",
            "acc_mean": aggregate.get("acc", {}).get("mean"),
            "acc_std": aggregate.get("acc", {}).get("std"),
            "f1_mean": aggregate.get("f1", {}).get("mean"),
            "f1_std": aggregate.get("f1", {}).get("std"),
            "auroc_mean": aggregate.get("auroc", {}).get("mean"),
            "auroc_std": aggregate.get("auroc", {}).get("std"),
            "auprc_mean": aggregate.get("auprc", {}).get("mean"),
            "auprc_std": aggregate.get("auprc", {}).get("std"),
            "loss_mean": aggregate.get("loss", {}).get("mean"),
            "loss_std": aggregate.get("loss", {}).get("std"),
        }
    ]


def _collect_federated(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for aggregate_path in sorted(results_root.glob("*_fedavg_*/aggregate_mean_std.json")) + sorted(
        results_root.glob("*_fedgate_*/aggregate_mean_std.json")
    ):
        payload = _load_json(aggregate_path)
        summary = payload.get("summary", {})
        rows.append(
            {
                "method": payload.get("method"),
                "scenario": payload.get("scenario_name"),
                "acc_mean": summary.get("acc", {}).get("mean"),
                "acc_std": summary.get("acc", {}).get("std"),
                "f1_mean": summary.get("f1", {}).get("mean"),
                "f1_std": summary.get("f1", {}).get("std"),
                "auroc_mean": summary.get("auroc", {}).get("mean"),
                "auroc_std": summary.get("auroc", {}).get("std"),
                "auprc_mean": summary.get("auprc", {}).get("mean"),
                "auprc_std": summary.get("auprc", {}).get("std"),
                "loss_mean": summary.get("loss", {}).get("mean"),
                "loss_std": summary.get("loss", {}).get("std"),
                "client_acc_gap_mean": summary.get("client_acc_gap", {}).get("mean"),
                "client_auroc_gap_mean": summary.get("client_auroc_gap", {}).get("mean"),
                "client_auprc_gap_mean": summary.get("client_auprc_gap", {}).get("mean"),
            }
        )
    return rows


def _collect_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scenario_method = {(str(row["scenario"]), str(row["method"])): row for row in rows}
    delta_rows: List[Dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in rows if str(row["scenario"]) != "pooled"})
    for scenario in scenarios:
        fedavg = by_scenario_method.get((scenario, "fedavg"))
        fedgate = by_scenario_method.get((scenario, "fedgate"))
        if not fedavg or not fedgate:
            continue
        delta_rows.append(
            {
                "scenario": scenario,
                "delta_acc_fg_minus_fa": (fedgate.get("acc_mean") or 0.0) - (fedavg.get("acc_mean") or 0.0),
                "delta_f1_fg_minus_fa": (fedgate.get("f1_mean") or 0.0) - (fedavg.get("f1_mean") or 0.0),
                "delta_auroc_fg_minus_fa": (fedgate.get("auroc_mean") or 0.0) - (fedavg.get("auroc_mean") or 0.0),
                "delta_auprc_fg_minus_fa": (fedgate.get("auprc_mean") or 0.0) - (fedavg.get("auprc_mean") or 0.0),
            }
        )
    return delta_rows


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = _collect_centralized(results_root) + _collect_federated(results_root)
    deltas = _collect_deltas(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(output_dir / "summary.csv", rows)
    (output_dir / "delta_fg_minus_fa.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    _write_csv(output_dir / "delta_fg_minus_fa.csv", deltas)
    print(f"Saved {output_dir}")


if __name__ == "__main__":
    main()
