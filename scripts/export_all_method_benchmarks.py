#!/usr/bin/env python3
"""Export strict benchmark tables and plots for all FedGate baselines."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

METHODS: tuple[str, ...] = (
    "fedavg_concat",
    "fedavg_mean",
    "fedgate",
    "ditto_concat_global",
    "ditto_concat_personal",
)
DISPLAY_NAMES = {
    "fedavg_concat": "FedAvg Concat",
    "fedavg_mean": "FedAvg Mean",
    "fedgate": "FedGate",
    "ditto_concat_global": "Ditto Global",
    "ditto_concat_personal": "Ditto Personal",
}
COLORS = {
    "fedavg_concat": "#2b6cb0",
    "fedavg_mean": "#5a7ea6",
    "fedgate": "#c05621",
    "ditto_concat_global": "#2f855a",
    "ditto_concat_personal": "#c53030",
}
METRICS = ("auroc", "auprc", "acc", "f1", "loss")


@dataclass(frozen=True)
class ScenarioConfig:
    config_path: Path
    scenario_name: str
    seeds: tuple[int, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="fedgate_full/results",
        help="Root directory containing result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="fedgate_full/results/paper_assets_all_methods",
        help="Output directory for aggregate tables and plots.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "fedgate_full/configs/s0_congruent_iid.yaml",
            "fedgate_full/configs/s1_congruent_non_iid.yaml",
            "fedgate_full/configs/s2_non_congruent_iid.yaml",
            "fedgate_full/configs/s3_non_congruent_non_iid.yaml",
        ],
        help="Scenario configs used to determine expected scenario names and seeds.",
    )
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _load_scenario_config(path: Path) -> ScenarioConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    scenario = payload.get("scenario")
    if not isinstance(scenario, dict) or not scenario.get("name"):
        raise ValueError(f"Missing scenario.name in {path}")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Missing seeds in {path}")
    return ScenarioConfig(
        config_path=path,
        scenario_name=str(scenario["name"]),
        seeds=tuple(int(seed) for seed in seeds),
    )


def _find_experiment_dir(results_root: Path, method: str, scenario_name: str) -> Path:
    matches = sorted(results_root.glob(f"*_{method}_{scenario_name}"))
    if not matches:
        raise FileNotFoundError(
            f"Missing result directory for method={method} scenario={scenario_name} under {results_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected a single result directory for method={method} scenario={scenario_name}, found {len(matches)}"
        )
    return matches[0]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON at {path}: {exc}") from exc


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def _validate_seed_artifacts(seed_dir: Path, method: str, scenario_name: str, seed: int) -> dict[str, Any]:
    metrics_path = seed_dir / "metrics.json"
    final_json = seed_dir / "final_metrics.json"
    final_csv = seed_dir / "final_metrics.csv"
    rounds_json = seed_dir / "per_round_metrics.json"
    rounds_csv = seed_dir / "per_round_metrics.csv"
    for path in (metrics_path, final_json, final_csv, rounds_json, rounds_csv):
        _require_file(path)

    payload = _load_json(metrics_path)
    if str(payload.get("method", "")).strip().lower() != method:
        raise ValueError(f"Unexpected method in {metrics_path}: {payload.get('method')!r} != {method!r}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Unexpected seed in {metrics_path}: {payload.get('seed')!r} != {seed!r}")
    scenario = payload.get("scenario")
    if not isinstance(scenario, dict) or str(scenario.get("name", "")).strip() != scenario_name:
        raise ValueError(
            f"Unexpected scenario in {metrics_path}: {scenario!r} != {scenario_name!r}"
        )
    if not isinstance(payload.get("history"), list) or not payload["history"]:
        raise ValueError(f"Missing or empty history in {metrics_path}")
    final_test = payload.get("final_test")
    if not isinstance(final_test, dict):
        raise ValueError(f"Missing final_test in {metrics_path}")
    for metric in METRICS:
        if metric not in final_test:
            raise ValueError(f"Missing final_test.{metric} in {metrics_path}")
    return payload


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_runs(results_root: Path, scenarios: list[ScenarioConfig]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    histories: dict[tuple[str, str], list[dict[str, Any]]] = {}
    client_rows: list[dict[str, Any]] = []

    for scenario_cfg in scenarios:
        for method in METHODS:
            exp_dir = _find_experiment_dir(results_root, method, scenario_cfg.scenario_name)
            for seed in scenario_cfg.seeds:
                seed_dir = exp_dir / f"seed_{seed}"
                if not seed_dir.exists():
                    raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
                payload = _validate_seed_artifacts(seed_dir, method, scenario_cfg.scenario_name, seed)
                final_test = dict(payload["final_test"])
                row = {
                    "method": method,
                    "scenario": scenario_cfg.scenario_name,
                    "seed": int(seed),
                    "auroc": float(final_test["auroc"]),
                    "auprc": float(final_test["auprc"]),
                    "acc": float(final_test["acc"]),
                    "f1": float(final_test["f1"]),
                    "loss": float(final_test["loss"]),
                    "is_personalized": bool(payload.get("is_personalized", False)),
                    "variant": str(payload.get("variant", "")),
                }
                final_rows.append(row)
                histories.setdefault((scenario_cfg.scenario_name, method), []).append(payload)

                client_path = seed_dir / "client_test_metrics.json"
                if client_path.exists():
                    client_payload = _load_json(client_path)
                    for client_key, metrics in sorted(client_payload.items(), key=lambda item: item[0]):
                        if not isinstance(metrics, dict):
                            raise ValueError(f"Malformed client metrics at {client_path}: {client_key!r}")
                        client_rows.append(
                            {
                                "method": method,
                                "scenario": scenario_cfg.scenario_name,
                                "seed": int(seed),
                                "client_id": int(str(client_key).replace("client_", "")),
                                "auprc": float(metrics["auprc"]),
                                "auroc": float(metrics["auroc"]),
                                "f1": float(metrics["f1"]),
                                "acc": float(metrics["acc"]),
                            }
                        )
    return final_rows, histories, client_rows


def _build_mean_std_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in final_rows:
        grouped.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)

    out: list[dict[str, Any]] = []
    for scenario_name, method in sorted(grouped):
        rows = grouped[(scenario_name, method)]
        item: dict[str, Any] = {"scenario": scenario_name, "method": method}
        for metric in METRICS:
            mean_value, std_value = _mean_std([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = mean_value
            item[f"{metric}_std"] = std_value
        out.append(item)
    return out


def _build_delta_rows(mean_std_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {(str(row["scenario"]), str(row["method"])): row for row in mean_std_rows}
    out: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in mean_std_rows})
    for scenario_name in scenarios:
        baseline = grouped.get((scenario_name, "fedavg_concat"))
        if baseline is None:
            raise ValueError(f"Missing fedavg_concat aggregate for scenario={scenario_name}")
        for method in METHODS:
            row = grouped.get((scenario_name, method))
            if row is None:
                raise ValueError(f"Missing aggregate for scenario={scenario_name} method={method}")
            item: dict[str, Any] = {"scenario": scenario_name, "method": method}
            for metric in METRICS:
                item[f"{metric}_delta_mean"] = float(row[f"{metric}_mean"]) - float(baseline[f"{metric}_mean"])
                item[f"{metric}_delta_std"] = float(row[f"{metric}_std"])
            out.append(item)
    return out


def _build_ranking_rows(mean_std_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in mean_std_rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    out: list[dict[str, Any]] = []
    for scenario_name, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["auprc_mean"]),
                -float(row["auroc_mean"]),
                -float(row["f1_mean"]),
                float(row["loss_mean"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            out.append(
                {
                    "scenario": scenario_name,
                    "rank": rank,
                    "method": row["method"],
                    "auprc_mean": row["auprc_mean"],
                    "auroc_mean": row["auroc_mean"],
                    "f1_mean": row["f1_mean"],
                    "loss_mean": row["loss_mean"],
                }
            )
    return out


def _scenario_short_name(name: str) -> str:
    if "_S0_" in name or name.startswith("FR_S0"):
        return "S0"
    if "_S1_" in name or name.startswith("FR_S1"):
        return "S1"
    if "_S2_" in name or name.startswith("FR_S2"):
        return "S2"
    if "_S3_" in name or name.startswith("FR_S3"):
        return "S3"
    return name


def _plot_final_comparison(mean_std_rows: list[dict[str, Any]], out_path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in mean_std_rows})
    grouped = {(str(row["scenario"]), str(row["method"])): row for row in mean_std_rows}
    metrics = ("auroc", "auprc")
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5), sharey=False)
    x = np.arange(len(scenarios))
    width = 0.15

    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for idx, method in enumerate(METHODS):
            means = [float(grouped[(scenario, method)][f"{metric}_mean"]) for scenario in scenarios]
            stds = [float(grouped[(scenario, method)][f"{metric}_std"]) for scenario in scenarios]
            ax.bar(
                x + (idx - (len(METHODS) - 1) / 2.0) * width,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                color=COLORS[method],
                label=DISPLAY_NAMES[method],
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_scenario_short_name(name) for name in scenarios])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Final Performance Across Methods and Scenarios", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_delta(delta_rows: list[dict[str, Any]], out_path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in delta_rows})
    grouped = {(str(row["scenario"]), str(row["method"])): row for row in delta_rows}
    metrics = ("auroc", "auprc")
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=False)
    x = np.arange(len(scenarios))

    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for method in METHODS:
            values = [float(grouped[(scenario, method)][f"{metric}_delta_mean"]) for scenario in scenarios]
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=DISPLAY_NAMES[method],
            )
        ax.axhline(0.0, color="#2d3748", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([_scenario_short_name(name) for name in scenarios])
        ax.set_title(f"{metric.upper()} delta vs FedAvg Concat")
        ax.grid(alpha=0.25)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curves(histories: dict[tuple[str, str], list[dict[str, Any]]], out_dir: Path) -> None:
    scenario_names = sorted({scenario for scenario, _ in histories})
    for scenario_name in scenario_names:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHODS:
            payloads = histories.get((scenario_name, method), [])
            if not payloads:
                raise ValueError(f"Missing histories for scenario={scenario_name} method={method}")
            round_to_values: dict[int, list[float]] = {}
            for payload in payloads:
                for row in payload["history"]:
                    round_idx = int(row.get("round", row.get("epoch", 0)))
                    round_to_values.setdefault(round_idx, []).append(float(row["test"]["auprc"]))
            xs = sorted(round_to_values)
            means = [float(np.mean(round_to_values[x])) for x in xs]
            stds = [float(np.std(round_to_values[x], ddof=0)) for x in xs]
            ax.plot(xs, means, linewidth=2, color=COLORS[method], label=DISPLAY_NAMES[method])
            ax.fill_between(xs, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), color=COLORS[method], alpha=0.15)
        ax.set_title(f"AUPRC Learning Curves - {_scenario_short_name(scenario_name)}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"learning_curves_{_scenario_short_name(scenario_name).lower()}.png", dpi=180)
        plt.close(fig)


def _plot_client_violin(client_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not client_rows:
        return
    scenario_names = sorted({str(row["scenario"]) for row in client_rows})
    fig, axes = plt.subplots(1, len(scenario_names), figsize=(5 * len(scenario_names), 5), sharey=True)
    axes_arr = np.atleast_1d(axes)

    for ax, scenario_name in zip(axes_arr, scenario_names):
        data = [row for row in client_rows if str(row["scenario"]) == scenario_name]
        methods = [method for method in METHODS if any(str(row["method"]) == method for row in data)]
        if not methods:
            continue
        values = [[float(row["auprc"]) for row in data if str(row["method"]) == method] for method in methods]
        vp = ax.violinplot(values, showmeans=True, showextrema=False)
        for body, method in zip(vp["bodies"], methods):
            body.set_facecolor(COLORS[method])
            body.set_alpha(0.65)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels([DISPLAY_NAMES[method] for method in methods], rotation=25, ha="right")
        ax.set_title(_scenario_short_name(scenario_name))
        ax.grid(axis="y", alpha=0.25)

    axes_arr[0].set_ylabel("Client AUPRC")
    fig.suptitle("Client-Level AUPRC by Method", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results_root = _resolve_path(args.results_root)
    output_dir = _resolve_path(args.output_dir)
    config_paths = [_resolve_path(path) for path in args.configs]
    scenario_cfgs = [_load_scenario_config(path) for path in config_paths]

    final_rows, histories, client_rows = _collect_runs(results_root, scenario_cfgs)
    mean_std_rows = _build_mean_std_rows(final_rows)
    delta_rows = _build_delta_rows(mean_std_rows)
    ranking_rows = _build_ranking_rows(mean_std_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "final_metrics_all_methods.csv", final_rows, fieldnames=list(final_rows[0].keys()))
    _write_csv(output_dir / "final_metrics_mean_std_all_methods.csv", mean_std_rows, fieldnames=list(mean_std_rows[0].keys()))
    _write_csv(output_dir / "delta_vs_fedavg_concat.csv", delta_rows, fieldnames=list(delta_rows[0].keys()))
    _write_csv(output_dir / "ranking_by_scenario.csv", ranking_rows, fieldnames=list(ranking_rows[0].keys()))

    _plot_final_comparison(mean_std_rows, output_dir / "final_performance_comparison.png")
    _plot_delta(delta_rows, output_dir / "delta_vs_fedavg_concat.png")
    _plot_learning_curves(histories, output_dir)
    _plot_client_violin(client_rows, output_dir / "client_auprc_violin.png")

    print(f"[export_all_method_benchmarks] output_dir={output_dir}")
    print(f"[export_all_method_benchmarks] rows={len(final_rows)}")
    print(f"[export_all_method_benchmarks] client_rows={len(client_rows)}")


if __name__ == "__main__":
    main()
