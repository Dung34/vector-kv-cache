"""
Visualization Utilities for Project III.
Generates Single-Needle Heatmaps, Multi-Needle Recall Charts, and VRAM Comparison Charts.
"""

from typing import Dict, List, Optional
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError:
    plt = None
    sns = None
    np = None


def plot_niah_heatmap(
    matrix: Dict[int, Dict[float, int]],
    context_lengths: List[int],
    depth_percents: List[float],
    title: str = "Needle-In-A-Haystack Retrieval Accuracy",
    output_path: Optional[str] = None,
):
    """
    Renders and optionally saves a 2D Heatmap of Single-Needle NIAH retrieval accuracy.
    X-axis: Context Length (tokens)
    Y-axis: Document Depth (%)
    """
    if plt is None or sns is None or np is None:
        print("[Warning] Matplotlib/Seaborn/Numpy not available. Skipping heatmap generation.")
        return

    grid = np.zeros((len(depth_percents), len(context_lengths)), dtype=float)
    for col_idx, length in enumerate(context_lengths):
        for row_idx, depth in enumerate(depth_percents):
            grid[row_idx, col_idx] = matrix.get(length, {}).get(depth, 0.0)

    plt.figure(figsize=(9, 6))
    depth_labels = [f"{int(d * 100)}%" for d in depth_percents]
    length_labels = [f"{l}" for l in context_lengths]

    cmap = sns.color_palette(["#e74c3c", "#2ecc71"])

    sns.heatmap(
        grid,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar=False,
        xticklabels=length_labels,
        yticklabels=depth_labels,
        linewidths=1.0,
        linecolor="#ffffff",
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Context Length (Tokens)", fontsize=12, labelpad=8)
    plt.ylabel("Document Depth (%)", fontsize=12, labelpad=8)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Heatmap saved to: {output_path}")

    plt.close()


def plot_multi_needle_comparison(
    comparison_data: Dict[str, Dict[int, float]],
    context_length: int = 4096,
    title: str = "Multi-Needle Retrieval Recall (3 vs 5 Needles)",
    output_path: Optional[str] = None,
):
    """
    Renders grouped bar chart comparing Multi-Needle Recall (%) across configurations.
    comparison_data format: {
        "Full-FP16": {3: 1.0, 5: 1.0},
        "H2O (p_c=50%)": {3: 0.67, 5: 0.60},
        "H2O (p_c=75%)": {3: 0.33, 5: 0.20},
    }
    """
    if plt is None or np is None:
        print("[Warning] Matplotlib not available. Skipping chart generation.")
        return

    configs = list(comparison_data.keys())
    needle_counts = [3, 5]

    x = np.arange(len(configs))
    width = 0.35

    recall_3 = [comparison_data[cfg].get(3, 0.0) * 100 for cfg in configs]
    recall_5 = [comparison_data[cfg].get(5, 0.0) * 100 for cfg in configs]

    plt.figure(figsize=(9, 5))
    rects1 = plt.bar(x - width/2, recall_3, width, label='3 Needles', color='#3498db')
    rects2 = plt.bar(x + width/2, recall_5, width, label='5 Needles', color='#e74c3c')

    plt.title(f"{title} @ {context_length} Tokens", fontsize=13, fontweight='bold', pad=10)
    plt.ylabel('Recall (%)', fontsize=11)
    plt.xticks(x, configs, fontsize=10)
    plt.ylim(0, 115)
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            plt.annotate(f'{h:.0f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Chart saved to: {output_path}")

    plt.close()


def plot_vram_comparison(
    experiments_data: Dict[str, float],
    title: str = "Peak VRAM Consumption Comparison",
    output_path: Optional[str] = None,
):
    """
    Renders bar chart comparing Peak VRAM (MB) across different cache configurations.
    """
    if plt is None:
        print("[Warning] Matplotlib not available. Skipping chart generation.")
        return

    labels = list(experiments_data.keys())
    values = list(experiments_data.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#3498db", "#9b59b6", "#e67e22", "#e74c3c"])

    plt.title(title, fontsize=13, fontweight="bold", pad=10)
    plt.ylabel("Peak VRAM (MB)", fontsize=11)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.01,
            f"{height:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Chart saved to: {output_path}")

    plt.close()
