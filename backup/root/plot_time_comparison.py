#!/usr/bin/env python3
"""
Plot time efficiency comparison between foundation models and graph/multivariate models.

Usage:
    python plot_time_comparison.py [dataset]
    
    Examples:
        python plot_time_comparison.py sd/2019/short
        python plot_time_comparison.py  # defaults to sd/2019/short
"""

import sys
import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError as e:
    print("Error: Missing required dependencies.")
    print("Please install: pip install pandas matplotlib numpy")
    print(f"Or use: uv pip install pandas matplotlib numpy")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "time_per_node_all_traffic.csv")

# Selected representative models for comparison
SELECTED_FOUNDATION = [
    "chronos_bolt_base",
    "timesfm_2_0_500m",
    "FlowState-9.1M",
    "Moirai2",
    "moirai_small",
    "feedforward",
]

SELECTED_GRAPH = [
    "agcrn",
    "dcrnn",
    "d2stgnn",
    "gwnet",
    "lstm",
    "hl",
]

# Model name display mapping (shorter names for plot)
DISPLAY_NAMES = {
    "chronos_bolt_base": "Chronos-Bolt",
    "timesfm_2_0_500m": "TimesFM-500M",
    "FlowState-9.1M": "FlowState-9.1M",
    "Moirai2": "Moirai2",
    "moirai_small": "Moirai-Small",
    "feedforward": "FeedForward",
    "agcrn": "AGCRN",
    "dcrnn": "DCRNN",
    "d2stgnn": "D2STGNN",
    "gwnet": "GWNet",
    "lstm": "LSTM",
    "hl": "HL",
}


def load_and_filter_data(dataset: str = "sd/2019/short"):
    """Load CSV and filter to selected models and dataset."""
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}\n"
                                f"Run compare_time_efficiency.py first to generate it.")
    
    df = pd.read_csv(CSV_PATH)
    
    # Filter to target dataset
    df = df[df["dataset"] == dataset].copy()
    
    if df.empty:
        raise ValueError(f"No data found for dataset: {dataset}")
    
    # Filter to selected models
    selected = SELECTED_FOUNDATION + SELECTED_GRAPH
    df = df[df["model_name"].isin(selected)].copy()
    
    if df.empty:
        raise ValueError(f"No selected models found for dataset: {dataset}")
    
    # Add display names
    df["display_name"] = df["model_name"].map(DISPLAY_NAMES)
    df.loc[df["display_name"].isna(), "display_name"] = df.loc[df["display_name"].isna(), "model_name"]
    
    # Sort: foundation first, then graph, then by time_per_node_ms
    df["sort_key"] = df["family"].map({"foundation_benchmark": 0, "graph_experiment": 1})
    df = df.sort_values(["sort_key", "time_per_node_ms"]).reset_index(drop=True)
    
    return df


def plot_comparison(df: pd.DataFrame, dataset: str, output_dir: str = None):
    """Create comparison plots."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    foundation_df = df[df["family"] == "foundation_benchmark"].copy()
    graph_df = df[df["family"] == "graph_experiment"].copy()
    
    # Colors
    color_foundation = "#2E86AB"  # Blue
    color_graph = "#A23B72"       # Purple/Pink
    
    # Create figure with two subplots: linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # === Plot 1: Linear scale ===
    x_pos_foundation = np.arange(len(foundation_df))
    x_pos_graph = np.arange(len(foundation_df), len(foundation_df) + len(graph_df))
    
    bars1_f = ax1.bar(x_pos_foundation, foundation_df["time_per_node_ms"], 
                       color=color_foundation, alpha=0.7, label="Foundation Models")
    bars1_g = ax1.bar(x_pos_graph, graph_df["time_per_node_ms"], 
                       color=color_graph, alpha=0.7, label="Graph/Multivariate Models")
    
    # Add value labels on bars
    for bars in [bars1_f, bars1_g]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Combine labels
    all_labels = list(foundation_df["display_name"]) + list(graph_df["display_name"])
    all_x_pos = list(x_pos_foundation) + list(x_pos_graph)
    
    ax1.set_xticks(all_x_pos)
    ax1.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel("Time per Node (ms)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Time Efficiency Comparison: {dataset}\n(Linear Scale)", 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    
    # === Plot 2: Log scale ===
    bars2_f = ax2.bar(x_pos_foundation, foundation_df["time_per_node_ms"], 
                       color=color_foundation, alpha=0.7, label="Foundation Models")
    bars2_g = ax2.bar(x_pos_graph, graph_df["time_per_node_ms"], 
                       color=color_graph, alpha=0.7, label="Graph/Multivariate Models")
    
    ax2.set_yscale('log')
    ax2.set_xticks(all_x_pos)
    ax2.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel("Time per Node (ms, log scale)", fontsize=12, fontweight='bold')
    ax2.set_title(f"Time Efficiency Comparison: {dataset}\n(Log Scale)", 
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    safe_ds_name = dataset.replace("/", "_")
    output_path = os.path.join(output_dir, f"time_comparison_{safe_ds_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {os.path.abspath(output_path)}")
    
    # Also save a summary table
    summary_df = df[["display_name", "family", "time_per_node_ms", "nodes_per_second", 
                      "time_full_graph_step_s"]].copy()
    summary_df.columns = ["Model", "Family", "Time/Node (ms)", "Nodes/Second", "Full Graph Time (s)"]
    summary_df["Family"] = summary_df["Family"].map({
        "foundation_benchmark": "Foundation",
        "graph_experiment": "Graph/Multivariate"
    })
    summary_csv = os.path.join(output_dir, f"time_summary_{safe_ds_name}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary table to: {os.path.abspath(summary_csv)}")
    
    plt.close()


def plot_full_graph_time(df: pd.DataFrame, dataset: str, output_dir: str = None):
    """Plot full graph prediction time comparison."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    foundation_df = df[df["family"] == "foundation_benchmark"].copy()
    graph_df = df[df["family"] == "graph_experiment"].copy()
    
    color_foundation = "#2E86AB"
    color_graph = "#A23B72"
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x_pos_foundation = np.arange(len(foundation_df))
    x_pos_graph = np.arange(len(foundation_df), len(foundation_df) + len(graph_df))
    
    bars_f = ax.bar(x_pos_foundation, foundation_df["time_full_graph_step_s"], 
                     color=color_foundation, alpha=0.7, label="Foundation Models")
    bars_g = ax.bar(x_pos_graph, graph_df["time_full_graph_step_s"], 
                     color=color_graph, alpha=0.7, label="Graph/Multivariate Models")
    
    # Add value labels
    for bars in [bars_f, bars_g]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    all_labels = list(foundation_df["display_name"]) + list(graph_df["display_name"])
    all_x_pos = list(x_pos_foundation) + list(x_pos_graph)
    
    ax.set_xticks(all_x_pos)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Time for Full Graph Prediction (seconds)", fontsize=12, fontweight='bold')
    ax.set_title(f"Full Graph Prediction Time: {dataset}", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    
    safe_ds_name = dataset.replace("/", "_")
    output_path = os.path.join(output_dir, f"full_graph_time_{safe_ds_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved full graph time plot to: {os.path.abspath(output_path)}")
    
    plt.close()


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "sd/2019/short"
    
    print(f"Loading data for dataset: {dataset}")
    df = load_and_filter_data(dataset)
    
    print(f"\nFound {len(df)} models:")
    print(f"  Foundation models: {len(df[df['family'] == 'foundation_benchmark'])}")
    print(f"  Graph/Multivariate models: {len(df[df['family'] == 'graph_experiment'])}")
    
    print("\nGenerating plots...")
    plot_comparison(df, dataset)
    plot_full_graph_time(df, dataset)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
