#!/usr/bin/env python3
"""Export paper-ready summary tables and comparison figures from fedgate_full results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("S0", "FR_S0_congruent_iid", "Congruent + IID"),
    ("S1", "FR_S1_congruent_non_iid", "Congruent + non-IID"),
    ("S2", "FR_S2_non_congruent_iid", "Non-congruent + IID"),
    ("S3", "FR_S3_non_congruent_non_iid", "Non-congruent + non-IID"),
)
METHODS: tuple[str, ...] = ("centralized", "fedavg", "fedgate")
METHOD_LABELS = {
    "centralized": "Centralized",
    "fedavg": "FedAvg",
    "fedgate": "FedGate",
}
PLOT_METHODS: tuple[str, ...] = ("fedavg", "fedgate")
PLOT_COLORS = {
    "fedavg": "#2b6cb0",
    "fedgate": "#c05621",
}
METRICS: tuple[str, ...] = ("auprc", "auroc", "f1", "acc", "loss")
METRIC_LABELS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "f1": "F1",
    "acc": "Accuracy",
    "loss": "Loss",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt_pm(mean_value: Any, std_value: Any) -> str:
    if not (_is_finite(mean_value) and _is_finite(std_value)):
        return "NA"
    return f"{float(mean_value):.3f} +/- {float(std_value):.3f}"


def _fmt_num(value: Any) -> str:
    if not _is_finite(value):
        return "NA"
    return f"{float(value):.3f}"


def _scenario_meta() -> dict[str, tuple[str, str]]:
    return {scenario_name: (short_key, display) for short_key, scenario_name, display in SCENARIOS}


def load_aggregates(results_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    meta = _scenario_meta()
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for agg_path in sorted(results_root.glob("*/aggregate_mean_std.json")):
        payload = _read_json(agg_path)
        scenario_name = str(payload.get("scenario", "")).strip()
        method = str(payload.get("method", "")).strip().lower()
        if scenario_name not in meta or method not in METHODS:
            continue
        out.setdefault(scenario_name, {})[method] = payload
    return out


def load_seed_runs(results_root: Path) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    meta = _scenario_meta()
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for metrics_path in sorted(results_root.glob("*/seed_*/metrics.json")):
        payload = _read_json(metrics_path)
        method = str(payload.get("method", "")).strip().lower()
        if method not in METHODS:
            continue
        scenario_meta = payload.get("scenario", {})
        if isinstance(scenario_meta, dict):
            scenario_name = str(scenario_meta.get("name", "")).strip()
        else:
            scenario_name = str(scenario_meta).strip()
        if scenario_name not in meta:
            continue
        seed = int(payload.get("seed"))
        out.setdefault(scenario_name, {}).setdefault(method, {})[seed] = payload
    return out


def _table_rows(
    aggregates: dict[str, dict[str, dict[str, Any]]],
    *,
    checkpoint: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for short_key, scenario_name, display in SCENARIOS:
        for method in METHODS:
            payload = aggregates.get(scenario_name, {}).get(method)
            if payload is None:
                continue
            block = payload.get(checkpoint, {})
            row = {
                "scenario_key": short_key,
                "scenario": display,
                "method": METHOD_LABELS[method],
            }
            for metric in METRICS:
                stat = block.get(metric, {})
                row[f"{metric}_mean"] = _fmt_num(stat.get("mean"))
                row[f"{metric}_std"] = _fmt_num(stat.get("std"))
                row[f"{metric}_pm"] = _fmt_pm(stat.get("mean"), stat.get("std"))
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, str]], *, metrics: tuple[str, ...]) -> None:
    headers = ["Scenario", "Method"] + [METRIC_LABELS[metric] for metric in metrics]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [row["scenario"], row["method"]] + [row[f"{metric}_pm"] for metric in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metrics_tex(path: Path, rows: list[dict[str, str]], *, metrics: tuple[str, ...], caption: str, label: str) -> None:
    headers = ["Scenario", "Method"] + [f"{METRIC_LABELS[metric]} $\\uparrow$" for metric in metrics]
    align = "ll" + ("c" * len(metrics))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["scenario"], row["method"]] + [row[f"{metric}_pm"] for metric in metrics]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delta_rows(
    aggregates: dict[str, dict[str, dict[str, Any]]],
    *,
    checkpoint: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for short_key, scenario_name, display in SCENARIOS:
        fedavg = aggregates.get(scenario_name, {}).get("fedavg")
        fedgate = aggregates.get(scenario_name, {}).get("fedgate")
        if fedavg is None or fedgate is None:
            continue
        row = {
            "scenario_key": short_key,
            "scenario": display,
        }
        for metric in METRICS:
            x = fedavg.get(checkpoint, {}).get(metric, {}).get("mean")
            y = fedgate.get(checkpoint, {}).get(metric, {}).get("mean")
            if _is_finite(x) and _is_finite(y):
                row[f"delta_{metric}"] = f"{float(y) - float(x):+.3f}"
            else:
                row[f"delta_{metric}"] = "NA"
        rows.append(row)
    return rows


def _write_delta_md(path: Path, rows: list[dict[str, str]]) -> None:
    metrics = ("auprc", "auroc", "f1", "acc", "loss")
    headers = ["Scenario"] + [f"Delta {METRIC_LABELS[metric]} (FG-FA)" for metric in metrics]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [row["scenario"]] + [row[f"delta_{metric}"] for metric in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_delta_tex(path: Path, rows: list[dict[str, str]], *, caption: str, label: str) -> None:
    metrics = ("auprc", "auroc", "f1", "acc", "loss")
    headers = ["Scenario"] + [f"Delta {METRIC_LABELS[metric]} (FG-FA)" for metric in metrics]
    align = "l" + ("c" * len(metrics))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["scenario"]] + [row[f"delta_{metric}"] for metric in metrics]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_final_metrics(aggregates: dict[str, dict[str, dict[str, Any]]], out_path: Path) -> None:
    metrics = ("auprc", "auroc", "f1")
    labels = [short_key for short_key, _, _ in SCENARIOS]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 4.6), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])

    for ax, metric in zip(axes, metrics):
        for idx, method in enumerate(PLOT_METHODS):
            offsets = x + (idx - 0.5) * width
            means = []
            stds = []
            for _, scenario_name, _ in SCENARIOS:
                payload = aggregates.get(scenario_name, {}).get(method)
                stat = (payload or {}).get("final_test", {}).get(metric, {})
                means.append(float(stat.get("mean", float("nan"))))
                stds.append(float(stat.get("std", 0.0)) if _is_finite(stat.get("std")) else 0.0)
            ax.bar(
                offsets,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                color=PLOT_COLORS[method],
                label=METHOD_LABELS[method],
                alpha=0.92,
            )
        ax.set_title(f"Final {METRIC_LABELS[metric]}")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        if metric != "loss":
            ax.set_ylim(0.0, 1.0)

    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("FedAvg vs FedGate across scenarios", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_metrics(aggregates: dict[str, dict[str, dict[str, Any]]], out_path: Path) -> None:
    metrics = ("auprc", "auroc", "f1")
    labels = [short_key for short_key, _, _ in SCENARIOS]
    x = np.arange(len(labels), dtype=np.float64)

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 4.3), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])

    for ax, metric in zip(axes, metrics):
        deltas = []
        for _, scenario_name, _ in SCENARIOS:
            fedavg = aggregates.get(scenario_name, {}).get("fedavg")
            fedgate = aggregates.get(scenario_name, {}).get("fedgate")
            x_val = (fedavg or {}).get("final_test", {}).get(metric, {}).get("mean")
            y_val = (fedgate or {}).get("final_test", {}).get(metric, {}).get("mean")
            if _is_finite(x_val) and _is_finite(y_val):
                deltas.append(float(y_val) - float(x_val))
            else:
                deltas.append(float("nan"))
        colors = ["#2f855a" if _is_finite(v) and v >= 0.0 else "#c53030" for v in deltas]
        ax.bar(x, deltas, color=colors, alpha=0.9)
        ax.axhline(0.0, color="#1a202c", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Delta {METRIC_LABELS[metric]} (FG-FA)")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("FedGate gain over FedAvg (final metrics)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_s1_seed_pairs(seed_runs: dict[str, dict[str, dict[int, dict[str, Any]]]], out_path: Path) -> None:
    scenario_name = "FR_S1_congruent_non_iid"
    scenario_runs = seed_runs.get(scenario_name, {})
    fedavg = scenario_runs.get("fedavg", {})
    fedgate = scenario_runs.get("fedgate", {})
    paired_seeds = sorted(set(fedavg) & set(fedgate))
    if not paired_seeds:
        return

    metrics = ("auprc", "f1")
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 4.2), sharey=False)
    if len(metrics) == 1:
        axes = np.asarray([axes])

    x_pos = np.asarray([0.0, 1.0])
    x_labels = ["FedAvg", "FedGate"]
    palette = ["#2b6cb0", "#c05621", "#2f855a"]

    for ax, metric in zip(axes, metrics):
        for idx, seed in enumerate(paired_seeds):
            x_val = float(fedavg[seed]["final_test"][metric])
            y_val = float(fedgate[seed]["final_test"][metric])
            ax.plot(
                x_pos,
                [x_val, y_val],
                marker="o",
                linewidth=2.0,
                color=palette[idx % len(palette)],
                label=f"seed {seed}",
                alpha=0.9,
            )
        ax.set_xticks(x_pos, x_labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"S1 final {METRIC_LABELS[metric]}")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("S1 stability across seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready assets for fedgate_full")
    parser.add_argument(
        "--results-root",
        default="fedgate_full/results",
        help="Root directory containing result folders",
    )
    parser.add_argument(
        "--out-dir",
        default="fedgate_full/results/paper_assets",
        help="Directory where tables and figures will be written",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates = load_aggregates(results_root)
    seed_runs = load_seed_runs(results_root)

    final_rows = _table_rows(aggregates, checkpoint="final_test")
    best_rows = _table_rows(aggregates, checkpoint="best_test")
    final_delta = _delta_rows(aggregates, checkpoint="final_test")
    best_delta = _delta_rows(aggregates, checkpoint="best_test")

    _write_csv(out_dir / "table_final_metrics.csv", final_rows)
    _write_md(out_dir / "table_final_metrics.md", final_rows, metrics=("auprc", "auroc", "f1", "acc"))
    _write_metrics_tex(
        out_dir / "table_final_metrics.tex",
        final_rows,
        metrics=("auprc", "auroc", "f1", "acc"),
        caption="Final metrics across scenarios.",
        label="tab:paper_final_metrics",
    )
    _write_csv(out_dir / "table_best_metrics.csv", best_rows)
    _write_md(out_dir / "table_best_metrics.md", best_rows, metrics=("auprc", "auroc", "f1", "acc"))
    _write_metrics_tex(
        out_dir / "table_best_metrics.tex",
        best_rows,
        metrics=("auprc", "auroc", "f1", "acc"),
        caption="Best metrics across scenarios.",
        label="tab:paper_best_metrics",
    )
    _write_csv(out_dir / "table_delta_final_fg_minus_fa.csv", final_delta)
    _write_delta_md(out_dir / "table_delta_final_fg_minus_fa.md", final_delta)
    _write_delta_tex(
        out_dir / "table_delta_final_fg_minus_fa.tex",
        final_delta,
        caption="Final metric deltas (FedGate minus FedAvg).",
        label="tab:paper_delta_final",
    )
    _write_csv(out_dir / "table_delta_best_fg_minus_fa.csv", best_delta)
    _write_delta_md(out_dir / "table_delta_best_fg_minus_fa.md", best_delta)
    _write_delta_tex(
        out_dir / "table_delta_best_fg_minus_fa.tex",
        best_delta,
        caption="Best metric deltas (FedGate minus FedAvg).",
        label="tab:paper_delta_best",
    )

    _plot_final_metrics(aggregates, out_dir / "figure_final_metrics_comparison.png")
    _plot_delta_metrics(aggregates, out_dir / "figure_delta_fg_minus_fa.png")
    _plot_s1_seed_pairs(seed_runs, out_dir / "figure_s1_seed_stability.png")

    print(f"[export_paper_assets] out_dir={out_dir}")


if __name__ == "__main__":
    main()
