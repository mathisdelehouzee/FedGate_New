#!/usr/bin/env python3
"""Export IEEE-ready LaTeX tables from fedgate_full results."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METHODS = ("fedavg", "fedgate")
SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("S0", "FR_S0_congruent_iid", "S0 (Congruent IID)"),
    ("S1", "FR_S1_congruent_non_iid", "S1 (Congruent non-IID)"),
    ("S2", "FR_S2_non_congruent_iid", "S2 (Non-congruent IID)"),
    ("S3", "FR_S3_non_congruent_non_iid", "S3 (Non-congruent non-IID)"),
)
CHECKPOINTS = ("best_test", "final_test")
METRIC_ORDER = ("auprc", "auroc", "f1", "acc")
METRIC_LABELS = {
    "auprc": "AUPRC $\\uparrow$",
    "auroc": "AUROC $\\uparrow$",
    "f1": "F1 $\\uparrow$",
    "acc": "Acc. $\\uparrow$",
}
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedgate": "FedGate",
}


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt_mean_std(values: list[float] | None) -> str:
    if not values:
        return "NA"
    mu = mean(values)
    sigma = pstdev(values) if len(values) > 1 else 0.0
    return f"${mu:.3f} \\pm {sigma:.3f}$"


def _rank_abs(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        current = abs(indexed[pos][1])
        while end < len(indexed) and abs(abs(indexed[end][1]) - current) <= 1e-12:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx, _ in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _wilcoxon_two_sided_exact(xs: list[float], ys: list[float]) -> float | None:
    paired = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite(x) and _is_finite(y)]
    diffs = [x - y for x, y in paired]
    diffs = [d for d in diffs if abs(d) > 1e-12]
    if not diffs:
        return None

    ranks = _rank_abs(diffs)
    rank_sum = sum(ranks)
    obs_pos = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0.0)
    obs_t = min(obs_pos, rank_sum - obs_pos)

    total = 0
    extreme = 0
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        total += 1
        pos_sum = 0.0
        for sign, rank in zip(signs, ranks):
            if sign:
                pos_sum += rank
        t_val = min(pos_sum, rank_sum - pos_sum)
        if t_val <= obs_t + 1e-12:
            extreme += 1
    return extreme / total if total > 0 else None


def _latex_header() -> list[str]:
    return [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Scenario & Method & AUPRC $\\uparrow$ & AUROC $\\uparrow$ & F1 $\\uparrow$ & Acc. $\\uparrow$ \\\\",
        "\\midrule",
    ]


def _latex_footer() -> list[str]:
    return [
        "\\bottomrule",
        "\\end{tabular}",
    ]


def _scenario_key_to_meta() -> dict[str, tuple[str, str]]:
    return {scenario_name: (short_key, display) for short_key, scenario_name, display in SCENARIOS}


def load_runs(results_root: Path) -> dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]:
    runs: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]] = {
        method: {} for method in METHODS
    }
    for path in sorted(results_root.glob("*/seed_*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = str(data.get("method", "")).strip().lower()
        if method not in METHODS:
            continue
        seed = int(data.get("seed"))
        scenario_meta = data.get("scenario", {})
        if isinstance(scenario_meta, dict):
            scenario_name = str(scenario_meta.get("name", "")).strip()
        else:
            scenario_name = str(scenario_meta).strip()
        if not scenario_name:
            continue
        runs.setdefault(method, {}).setdefault(scenario_name, {})[seed] = {
            checkpoint: {
                metric: float(data.get(checkpoint, {}).get(metric, float("nan")))
                for metric in METRIC_ORDER
            }
            for checkpoint in CHECKPOINTS
        }
    return runs


def aggregate_runs(
    runs: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]],
    seeds: list[int],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for method in METHODS:
        out[method] = {}
        for _, scenario_name, _ in SCENARIOS:
            seed_map = runs.get(method, {}).get(scenario_name, {})
            available = sorted(seed for seed in seeds if seed in seed_map)
            checkpoint_payload: dict[str, dict[str, Any]] = {}
            for checkpoint in CHECKPOINTS:
                metrics_payload: dict[str, Any] = {
                    "available_seeds": available,
                    "complete": len(available) == len(seeds),
                    "missing_seeds": [seed for seed in seeds if seed not in seed_map],
                    "metrics": {},
                }
                for metric in METRIC_ORDER:
                    values = [
                        float(seed_map[seed][checkpoint][metric])
                        for seed in seeds
                        if seed in seed_map and _is_finite(seed_map[seed][checkpoint][metric])
                    ]
                    metrics_payload["metrics"][metric] = {
                        "values": values,
                        "mean": mean(values) if values else None,
                        "std": pstdev(values) if values else None,
                    }
                checkpoint_payload[checkpoint] = metrics_payload
            out[method][scenario_name] = checkpoint_payload
    return out


def build_main_table(
    aggregates: dict[str, dict[str, dict[str, dict[str, Any]]]],
    checkpoint: str,
) -> str:
    lines = _latex_header()
    for idx, (_, scenario_name, display) in enumerate(SCENARIOS):
        best_method = None
        best_auprc = None
        for method in METHODS:
            block = aggregates[method][scenario_name][checkpoint]
            if not block["complete"]:
                continue
            value = block["metrics"]["auprc"]["mean"]
            if value is None:
                continue
            if best_auprc is None or float(value) > float(best_auprc):
                best_auprc = float(value)
                best_method = method
        for method in METHODS:
            block = aggregates[method][scenario_name][checkpoint]
            cells: list[str] = []
            for metric in METRIC_ORDER:
                values = block["metrics"][metric]["values"] if block["complete"] else None
                cell = _fmt_mean_std(values)
                if metric == "auprc" and method == best_method and cell != "NA":
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(
                f"{display} & {METHOD_LABELS[method]} & "
                + " & ".join(cells)
                + " \\\\"
            )
        if idx < len(SCENARIOS) - 1:
            lines.append("\\midrule")
    lines.extend(_latex_footer())
    return "\n".join(lines) + "\n"


def build_robustness_table(
    aggregates: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> str:
    s0 = SCENARIOS[0][1]
    s3 = SCENARIOS[-1][1]
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Best S0 & Best S3 & $\\Delta_{\\mathrm{best}}$ & Final S0 & Final S3 & $\\Delta_{\\mathrm{final}}$ \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        best_s0 = (
            aggregates[method][s0]["best_test"]["metrics"]["auprc"]["mean"]
            if aggregates[method][s0]["best_test"]["complete"]
            else None
        )
        best_s3 = (
            aggregates[method][s3]["best_test"]["metrics"]["auprc"]["mean"]
            if aggregates[method][s3]["best_test"]["complete"]
            else None
        )
        final_s0 = (
            aggregates[method][s0]["final_test"]["metrics"]["auprc"]["mean"]
            if aggregates[method][s0]["final_test"]["complete"]
            else None
        )
        final_s3 = (
            aggregates[method][s3]["final_test"]["metrics"]["auprc"]["mean"]
            if aggregates[method][s3]["final_test"]["complete"]
            else None
        )

        def fmt(v: float | None) -> str:
            return "NA" if v is None else f"${v:.3f}$"

        def fmt_delta(a: float | None, b: float | None) -> str:
            if a is None or b is None:
                return "NA"
            return f"${(b - a):+.3f}$"

        lines.append(
            f"{METHOD_LABELS[method]} & {fmt(best_s0)} & {fmt(best_s3)} & {fmt_delta(best_s0, best_s3)} & "
            f"{fmt(final_s0)} & {fmt(final_s3)} & {fmt_delta(final_s0, final_s3)} \\\\"
        )
    lines.extend(_latex_footer())
    return "\n".join(lines) + "\n"


def build_wilcoxon_table(
    runs: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]],
    seeds: list[int],
) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Scenario & Paired seeds & $p_{\\mathrm{best}}$ (AUPRC) & $p_{\\mathrm{final}}$ (AUPRC) \\\\",
        "\\midrule",
    ]
    for _, scenario_name, display in SCENARIOS:
        fedavg_seed_map = runs.get("fedavg", {}).get(scenario_name, {})
        fedgate_seed_map = runs.get("fedgate", {}).get(scenario_name, {})
        paired_seeds = [seed for seed in seeds if seed in fedavg_seed_map and seed in fedgate_seed_map]

        best_x = [fedavg_seed_map[seed]["best_test"]["auprc"] for seed in paired_seeds]
        best_y = [fedgate_seed_map[seed]["best_test"]["auprc"] for seed in paired_seeds]
        final_x = [fedavg_seed_map[seed]["final_test"]["auprc"] for seed in paired_seeds]
        final_y = [fedgate_seed_map[seed]["final_test"]["auprc"] for seed in paired_seeds]

        paired_n = len(paired_seeds)
        pb = _wilcoxon_two_sided_exact(best_y, best_x) if paired_n > 0 else None
        pf = _wilcoxon_two_sided_exact(final_y, final_x) if paired_n > 0 else None

        def fmt_p(v: float | None) -> str:
            return "NA" if v is None else f"${v:.3f}$"

        lines.append(f"{display} & {paired_n} & {fmt_p(pb)} & {fmt_p(pf)} \\\\")
    lines.extend(_latex_footer())
    return "\n".join(lines) + "\n"


def raw_summary(
    runs: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]],
) -> dict[str, Any]:
    scenario_meta = _scenario_key_to_meta()
    payload: dict[str, Any] = {
        "methods": {},
    }
    for method in METHODS:
        payload["methods"][method] = {}
        for scenario_name, per_seed in sorted(runs.get(method, {}).items()):
            short_key, display = scenario_meta.get(scenario_name, (scenario_name, scenario_name))
            payload["methods"][method][short_key] = {
                "scenario_name": scenario_name,
                "scenario_label": display,
                "seeds": {
                    str(seed): per_seed[seed]
                    for seed in sorted(per_seed)
                },
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IEEE LaTeX tables for fedgate_full")
    parser.add_argument(
        "--results-root",
        default="fedgate_full/results",
        help="Root directory containing fedgate_full result folders",
    )
    parser.add_argument(
        "--out-dir",
        default="fedgate_full/results/ieee_tables",
        help="Directory where JSON and LaTeX outputs will be written",
    )
    parser.add_argument(
        "--seeds",
        default="7,11,17",
        help="Comma-separated seed list to require for aggregation",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]

    runs = load_runs(results_root)
    aggregates = aggregate_runs(runs, seeds=seeds)

    (out_dir / "raw_summary.json").write_text(
        json.dumps(raw_summary(runs), indent=2),
        encoding="utf-8",
    )
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps({"required_seeds": seeds, "aggregates": aggregates}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "table_best.tex").write_text(
        build_main_table(aggregates, checkpoint="best_test"),
        encoding="utf-8",
    )
    (out_dir / "table_final.tex").write_text(
        build_main_table(aggregates, checkpoint="final_test"),
        encoding="utf-8",
    )
    (out_dir / "table_robustness_delta_s0_s3.tex").write_text(
        build_robustness_table(aggregates),
        encoding="utf-8",
    )
    (out_dir / "table_wilcoxon.tex").write_text(
        build_wilcoxon_table(runs, seeds=seeds),
        encoding="utf-8",
    )

    print(f"[export_ieee_tables] out_dir={out_dir}")
    print(f"[export_ieee_tables] required_seeds={seeds}")


if __name__ == "__main__":
    main()
