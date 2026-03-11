#!/usr/bin/env python3
"""Generate additional plots for FedGate paper analysis."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configuration
SCENARIOS = {
    "S0": ("FR_S0_congruent_iid", "Congruent + IID"),
    "S1": ("FR_S1_congruent_non_iid", "Congruent + non-IID"),
    "S2": ("FR_S2_non_congruent_iid", "Non-congruent + IID"),
    "S3": ("FR_S3_non_congruent_non_iid", "Non-congruent + non-IID"),
}

COLORS = {
    "fedavg": "#2b6cb0",
    "fedgate": "#c05621",
    "centralized": "#2d3748",
}

def load_metrics_file(path: Path) -> dict:
    """Load a metrics.json file."""
    with open(path, 'r') as f:
        return json.load(f)

def extract_learning_curves(results_root: Path, scenario_key: str, method: str, seeds: list[int]):
    """Extract test AUPRC and AUROC over rounds for a given scenario and method."""
    scenario_name = SCENARIOS[scenario_key][0]
    all_curves = {seed: {"rounds": [], "auprc": [], "auroc": []} for seed in seeds}
    
    for seed in seeds:
        # Try to find the metrics file
        pattern = f"*{method}*{scenario_name}/seed_{seed}/metrics.json"
        matches = list(results_root.glob(pattern))
        
        if not matches:
            print(f"Warning: No metrics file found for {method}, {scenario_key}, seed {seed}")
            continue
            
        metrics = load_metrics_file(matches[0])
        
        if "history" not in metrics:
            continue
            
        for entry in metrics["history"]:
            round_num = entry.get("round", 0)
            test_metrics = entry.get("test", {})
            
            auprc = test_metrics.get("auprc")
            auroc = test_metrics.get("auroc")
            
            if auprc is not None and not (isinstance(auprc, float) and np.isnan(auprc)):
                all_curves[seed]["rounds"].append(round_num)
                all_curves[seed]["auprc"].append(auprc)
                all_curves[seed]["auroc"].append(auroc if auroc is not None else 0)
    
    return all_curves

def plot_learning_curves(results_root: Path, output_dir: Path):
    """Generate learning curves comparing FedAvg and FedGate for S1."""
    print("Generating learning curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seeds = [7, 11, 17]
    
    for method_idx, method in enumerate(["fedavg", "fedgate"]):
        curves = extract_learning_curves(results_root, "S1", method, seeds)
        
        # Aggregate across seeds
        max_rounds = max([len(c["rounds"]) for c in curves.values() if c["rounds"]])
        
        for metric_idx, metric in enumerate(["auprc", "auroc"]):
            ax = axes[metric_idx]
            
            # Collect all values per round
            rounds_data = {}
            for seed, data in curves.items():
                for i, r in enumerate(data["rounds"]):
                    if r not in rounds_data:
                        rounds_data[r] = []
                    rounds_data[r].append(data[metric][i])
            
            if not rounds_data:
                continue
                
            # Calculate mean and std
            rounds = sorted(rounds_data.keys())
            means = [np.mean(rounds_data[r]) for r in rounds]
            stds = [np.std(rounds_data[r]) for r in rounds]
            
            label = "FedAvg" if method == "fedavg" else "FedGate"
            color = COLORS[method]
            
            ax.plot(rounds, means, label=label, color=color, linewidth=2)
            ax.fill_between(rounds, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2, color=color)
            
            ax.set_xlabel("Communication Round", fontsize=11)
            ax.set_ylabel("AUPRC" if metric == "auprc" else "AUROC", fontsize=11)
            ax.set_title(f"{'AUPRC' if metric == 'auprc' else 'AUROC'} Evolution (S1: Congruent + non-IID)", 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "figure_learning_curves_s1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_correlation_heatmap(results_root: Path, output_dir: Path):
    """Generate correlation heatmap between gates and performance."""
    print("Generating correlation heatmap...")
    
    # Load client analysis data
    client_csv = results_root / "fedgate_client_analysis" / "client_level_summary.csv"
    if not client_csv.exists():
        print(f"Warning: {client_csv} not found, skipping heatmap")
        return
    
    df = pd.read_csv(client_csv)
    
    # Select relevant columns
    columns_of_interest = [
        'mean_g_mri', 'mean_g_tab', 
        'mean_local_auprc', 'mean_local_auroc', 'mean_local_f1', 'mean_local_acc'
    ]
    
    df_subset = df[columns_of_interest].copy()
    df_subset.columns = ['Gate MRI', 'Gate Tab', 'AUPRC', 'AUROC', 'F1', 'Accuracy']
    
    # Compute correlation matrix
    corr_matrix = df_subset.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Correlation: Gates vs. Performance Metrics (All Clients)', 
                fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = output_dir / "figure_correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_robustness_analysis(results_root: Path, output_dir: Path):
    """Generate robustness plot showing S0 vs S3 performance."""
    print("Generating robustness analysis...")
    
    # Load aggregate data
    data = {}
    for scenario_key in ["S0", "S3"]:
        scenario_name = SCENARIOS[scenario_key][0]
        for method in ["fedavg", "fedgate"]:
            pattern = f"*{method}*{scenario_name}/aggregate_mean_std.json"
            matches = list(results_root.glob(pattern))
            
            if matches:
                with open(matches[0], 'r') as f:
                    agg = json.load(f)
                    data[(scenario_key, method)] = agg["final_test"]
    
    # Prepare data for plotting
    metrics = ["auprc", "auroc", "f1", "acc"]
    x = np.arange(len(metrics))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_s0 = {"fedavg": "#6B9BD1", "fedgate": "#E88B6F"}
    colors_s3 = {"fedavg": "#2b6cb0", "fedgate": "#c05621"}
    
    for i, method in enumerate(["fedavg", "fedgate"]):
        # S0 data
        if ("S0", method) in data:
            means_s0 = [data[("S0", method)][m]["mean"] for m in metrics]
            stds_s0 = [data[("S0", method)][m]["std"] for m in metrics]
            
            offset = -width*1.5 + i*width
            bars_s0 = ax.bar(x + offset, means_s0, width, 
                            label=f'{method.upper() if method == "fedavg" else "FedGate"} - S0 (easy)',
                            color=colors_s0[method], yerr=stds_s0, capsize=3, alpha=0.8)
        
        # S3 data
        if ("S3", method) in data:
            means_s3 = [data[("S3", method)][m]["mean"] for m in metrics]
            stds_s3 = [data[("S3", method)][m]["std"] for m in metrics]
            
            offset = width*0.5 + i*width
            bars_s3 = ax.bar(x + offset, means_s3, width,
                            label=f'{method.upper() if method == "fedavg" else "FedGate"} - S3 (hard)',
                            color=colors_s3[method], yerr=stds_s3, capsize=3, alpha=0.8,
                            hatch='//')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('Robustness Analysis: Performance Degradation from S0 to S3', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['AUPRC', 'AUROC', 'F1', 'Accuracy'])
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "figure_robustness_s0_vs_s3.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_multibar_comparison(results_root: Path, output_dir: Path):
    """Generate comprehensive multi-metric comparison across all scenarios."""
    print("Generating multi-metric comparison...")
    
    # Load data for all scenarios
    data = {}
    for scenario_key in SCENARIOS.keys():
        scenario_name = SCENARIOS[scenario_key][0]
        for method in ["fedavg", "fedgate"]:
            pattern = f"*{method}*{scenario_name}/aggregate_mean_std.json"
            matches = list(results_root.glob(pattern))
            
            if matches:
                with open(matches[0], 'r') as f:
                    agg = json.load(f)
                    data[(scenario_key, method)] = agg["final_test"]
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ["auprc", "auroc", "f1", "acc"]
    metric_labels = ["AUPRC", "AUROC", "F1 Score", "Accuracy"]
    
    scenarios = ["S0", "S1", "S2", "S3"]
    x = np.arange(len(scenarios))
    width = 0.35
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        means_fa = []
        stds_fa = []
        means_fg = []
        stds_fg = []
        
        for scenario in scenarios:
            if (scenario, "fedavg") in data:
                means_fa.append(data[(scenario, "fedavg")][metric]["mean"])
                stds_fa.append(data[(scenario, "fedavg")][metric]["std"])
            else:
                means_fa.append(0)
                stds_fa.append(0)
                
            if (scenario, "fedgate") in data:
                means_fg.append(data[(scenario, "fedgate")][metric]["mean"])
                stds_fg.append(data[(scenario, "fedgate")][metric]["std"])
            else:
                means_fg.append(0)
                stds_fg.append(0)
        
        bars1 = ax.bar(x - width/2, means_fa, width, label='FedAvg', 
                      color=COLORS["fedavg"], yerr=stds_fa, capsize=4)
        bars2 = ax.bar(x + width/2, means_fg, width, label='FedGate',
                      color=COLORS["fedgate"], yerr=stds_fg, capsize=4)
        
        ax.set_ylabel(metric_labels[idx], fontsize=11)
        ax.set_xlabel('Scenario', fontsize=11)
        ax.set_title(f'{metric_labels[idx]} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / "figure_multibar_all_scenarios.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate additional plots for FedGate paper")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Path to results directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/paper_assets)"
    )
    
    args = parser.parse_args()
    
    results_root = args.results_root
    output_dir = args.output or (results_root / "paper_assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results root: {results_root}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate all plots
    plot_learning_curves(results_root, output_dir)
    plot_correlation_heatmap(results_root, output_dir)
    plot_robustness_analysis(results_root, output_dir)
    plot_multibar_comparison(results_root, output_dir)
    
    print("\n✅ All additional plots generated successfully!")

if __name__ == "__main__":
    main()
