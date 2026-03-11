#!/usr/bin/env python3
"""Generate per-scenario split allocation figures for slide use.

Each figure shows:
- quantity allocation per client (mean across seeds, with std),
- realized label skew via positive-label ratio per client,
- modality pairing/client-type counts from the scenario config.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "fedgate_full" / "plots" / "scenario_allocations"

SCENARIOS = [
    {
        "slug": "s0",
        "config": ROOT / "fedgate_full" / "configs" / "s0_congruent_iid.yaml",
        "manifest": ROOT / "fedgate_full" / "artifacts_s0_iid" / "splits" / "splits_manifest.json",
        "report": ROOT / "fedgate_full" / "artifacts_s0_iid" / "splits" / "splits_report.json",
    },
    {
        "slug": "s1",
        "config": ROOT / "fedgate_full" / "configs" / "s1_congruent_non_iid.yaml",
        "manifest": ROOT / "fedgate_full" / "artifacts_s1_non_iid" / "splits" / "splits_manifest.json",
        "report": ROOT / "fedgate_full" / "artifacts_s1_non_iid" / "splits" / "splits_report.json",
    },
    {
        "slug": "s2",
        "config": ROOT / "fedgate_full" / "configs" / "s2_non_congruent_iid.yaml",
        "manifest": ROOT / "fedgate_full" / "artifacts_s2_non_congruent_iid" / "splits" / "splits_manifest.json",
        "report": ROOT / "fedgate_full" / "artifacts_s2_non_congruent_iid" / "splits" / "splits_report.json",
    },
    {
        "slug": "s3",
        "config": ROOT / "fedgate_full" / "configs" / "s3_non_congruent_non_iid.yaml",
        "manifest": ROOT / "fedgate_full" / "artifacts_s3_non_congruent_non_iid" / "splits" / "splits_manifest.json",
        "report": ROOT / "fedgate_full" / "artifacts_s3_non_congruent_non_iid" / "splits" / "splits_report.json",
    },
]

CLIENT_TYPE_ORDER = ["both_aligned", "both_partial", "mri_only", "tab_only"]
CLIENT_TYPE_LABELS = {
    "both_aligned": "Both aligned",
    "both_partial": "Both partial",
    "mri_only": "MRI only",
    "tab_only": "Tab only",
}
CLIENT_TYPE_COLORS = {
    "both_aligned": "#2A9D8F",
    "both_partial": "#E9C46A",
    "mri_only": "#E76F51",
    "tab_only": "#457B9D",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def _label_skew_text(cfg: dict[str, Any], manifest: dict[str, Any]) -> str:
    strategy = str(cfg.get("partition", {}).get("strategy", "unknown")).strip()
    diagnostics = next(iter(manifest["seeds"].values()))["diagnostics"]
    global_pos_ratio = float(diagnostics["global_pos_ratio"])
    if strategy == "balanced":
        return f"Label skew: none (balanced)\nGlobal positive ratio: {global_pos_ratio:.3f}"

    quantity_alpha = cfg.get("partition", {}).get("quantity_skew_alpha")
    label_alpha = cfg.get("partition", {}).get("label_skew_alpha")
    parts = [f"Label skew: {strategy}"]
    if label_alpha is not None:
        parts.append(f"alpha_l={float(label_alpha):.2f}")
    if quantity_alpha is not None:
        parts.append(f"alpha_q={float(quantity_alpha):.2f}")
    parts.append(f"global pos={global_pos_ratio:.3f}")
    return "\n".join([parts[0], ", ".join(parts[1:])])


def _collect_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    seed_payloads = list(manifest["seeds"].values())
    client_ids = sorted(int(cid) for cid in seed_payloads[0]["client_partitions"].keys())

    quantity_mean: list[float] = []
    quantity_std: list[float] = []
    train_mean: list[float] = []
    val_mean: list[float] = []
    pos_ratio_mean: list[float] = []
    pos_ratio_std: list[float] = []

    first_diag = seed_payloads[0]["diagnostics"]
    global_pos_ratio = float(first_diag["global_pos_ratio"])

    for cid in client_ids:
        total_counts: list[float] = []
        train_counts: list[float] = []
        val_counts: list[float] = []
        pos_ratios: list[float] = []
        key = str(cid)
        for seed_payload in seed_payloads:
            total_counts.append(float(len(seed_payload["client_partitions"][key])))
            train_counts.append(float(len(seed_payload["client_splits"][key]["train_indices"])))
            val_counts.append(float(len(seed_payload["client_splits"][key]["val_indices"])))
            pos_ratios.append(float(seed_payload["diagnostics"]["client_pos_ratios"][key]))

        q_mean, q_std = _mean_std(total_counts)
        t_mean, _ = _mean_std(train_counts)
        v_mean, _ = _mean_std(val_counts)
        p_mean, p_std = _mean_std(pos_ratios)
        quantity_mean.append(q_mean)
        quantity_std.append(q_std)
        train_mean.append(t_mean)
        val_mean.append(v_mean)
        pos_ratio_mean.append(p_mean)
        pos_ratio_std.append(p_std)

    return {
        "client_ids": client_ids,
        "quantity_mean": quantity_mean,
        "quantity_std": quantity_std,
        "train_mean": train_mean,
        "val_mean": val_mean,
        "pos_ratio_mean": pos_ratio_mean,
        "pos_ratio_std": pos_ratio_std,
        "global_pos_ratio": global_pos_ratio,
    }


def _plot_single(spec: dict[str, Path | str]) -> Path:
    cfg = _load_yaml(Path(spec["config"]))
    manifest = _load_json(Path(spec["manifest"]))
    report = _load_json(Path(spec["report"]))
    stats = _collect_stats(manifest)

    scenario_cfg = dict(cfg.get("scenario", {}))
    scenario_name = str(scenario_cfg.get("name", "unknown"))
    scenario_desc = str(scenario_cfg.get("description", ""))
    seed_reports = report["seed_reports"]
    size_min = min(float(seed_reports[s]["size_min"]) for s in seed_reports)
    size_max = max(float(seed_reports[s]["size_max"]) for s in seed_reports)
    max_ratio_dev = max(float(seed_reports[s]["max_label_ratio_deviation"]) for s in seed_reports)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    x = list(range(len(stats["client_ids"])))
    ax = axes[0]
    ax.bar(x, stats["train_mean"], color="#4E79A7", label="Train", width=0.75)
    ax.bar(x, stats["val_mean"], bottom=stats["train_mean"], color="#A0CBE8", label="Val", width=0.75)
    ax.errorbar(
        x,
        stats["quantity_mean"],
        yerr=stats["quantity_std"],
        fmt="none",
        ecolor="#1F1F1F",
        elinewidth=1.0,
        capsize=3,
        label="Across-seed std",
    )
    ax.set_title("Quantity per client")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels([str(cid) for cid in stats["client_ids"]], fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(
        0.02,
        0.98,
        f"size range: {size_min:.0f}-{size_max:.0f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.25"},
    )

    ax = axes[1]
    ax.axhline(stats["global_pos_ratio"], color="#D62728", linestyle="--", linewidth=1.3, label="Global positive ratio")
    ax.errorbar(
        x,
        stats["pos_ratio_mean"],
        yerr=stats["pos_ratio_std"],
        fmt="o",
        color="#2A9D8F",
        ecolor="#2A9D8F",
        capsize=3,
        markersize=5,
        label="Client positive ratio",
    )
    ax.set_title("Label skew per client")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Positive-label ratio")
    ax.set_xticks(x)
    ax.set_xticklabels([str(cid) for cid in stats["client_ids"]], fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(
        0.02,
        0.98,
        f"max deviation: {max_ratio_dev:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.25"},
    )

    ax = axes[2]
    counts = dict(scenario_cfg.get("client_type_counts", {}))
    y_labels: list[str] = []
    widths: list[int] = []
    colors: list[str] = []
    for key in CLIENT_TYPE_ORDER:
        y_labels.append(CLIENT_TYPE_LABELS[key])
        widths.append(int(counts.get(key, 0)))
        colors.append(CLIENT_TYPE_COLORS[key])
    ypos = list(range(len(y_labels)))
    ax.barh(ypos, widths, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Number of clients")
    ax.set_title("Modality pairing")
    ax.invert_yaxis()
    for idx, width in enumerate(widths):
        ax.text(width + 0.08, idx, str(width), va="center", fontsize=9)
    ax.set_xlim(0, max(1, max(widths) + 1))
    partial_rate = float(scenario_cfg.get("partial_pairing_rate", 1.0))
    text = (
        f"{_label_skew_text(cfg, manifest)}\n"
        f"Partial pairing rate: {partial_rate:.2f}\n"
        f"Seeds: {', '.join(sorted(seed_reports.keys(), key=int))}"
    )
    ax.text(
        1.02,
        0.5,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.35"},
    )

    fig.suptitle(f"{scenario_name} | {scenario_desc}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = OUT_DIR / f"{spec['slug']}_allocation_overview.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for spec in SCENARIOS:
        output_paths.append(_plot_single(spec))
    print("Saved:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
