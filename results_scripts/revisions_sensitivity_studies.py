"""
Sensitivity Analysis Plots for IEEE TMI Revision
Creates plots showing how Macro ACE and DSC vary with:
1. Bin number (5, 10, 20, 50, 100)
2. Loss weighting (0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)

For both Hard L1-ACE and Soft L1-ACE variants on ACDC17 dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path("/workspaces/Average-Calibration-Losses")
BUNDLES_DIR = BASE_DIR / "bundles"
OUTPUT_DIR = BASE_DIR / "results_plots" / "revision_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEED = "seed_12345"
BIN_NUMBERS = [5, 10, 20, 50, 100]
LOSS_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]

# Style configuration - Arial font size 8 for all text
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.titlesize"] = 8
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.dpi"] = 300


def load_metric(bundle_dir, metric_file):
    """
    Load a metric CSV file and extract mean and std from the 'mean' row.

    Args:
        bundle_dir: Path to the bundle directory
        metric_file: Name of the metric file (e.g., 'mean_dice_summary.csv')

    Returns:
        tuple: (mean_value, std_value) or (None, None) if file doesn't exist
    """
    file_path = bundle_dir / SEED / "inference_results" / metric_file

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None, None

    try:
        df = pd.read_csv(file_path)
        # Get the 'mean' row which contains aggregate statistics
        mean_row = df[df["class"] == "mean"]

        if mean_row.empty:
            print(f"Warning: No 'mean' row found in {file_path}")
            return None, None

        mean_value = mean_row["mean"].values[0]
        std_value = mean_row["std"].values[0]

        return mean_value, std_value
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None


def collect_bin_sensitivity_data():
    """
    Collect data for bin number sensitivity analysis.

    Returns:
        dict: Nested dictionary with structure:
              {loss_type: {metric: {bin_num: (mean, std)}}}
    """
    data = {"hardl1ace": {"dice": {}, "ace": {}}, "softl1ace": {"dice": {}, "ace": {}}}

    for loss_type in ["hardl1ace", "softl1ace"]:
        for bin_num in BIN_NUMBERS:
            bundle_name = f"acdc17_{loss_type}_dice_ce_2_bin_{bin_num}"
            bundle_dir = BUNDLES_DIR / bundle_name

            if not bundle_dir.exists():
                print(f"Warning: Bundle directory not found: {bundle_dir}")
                continue

            # Load Dice score
            dice_mean, dice_std = load_metric(bundle_dir, "mean_dice_summary.csv")
            if dice_mean is not None:
                data[loss_type]["dice"][bin_num] = (dice_mean, dice_std)

            # Load Macro ACE
            ace_mean, ace_std = load_metric(bundle_dir, "macro_ace_summary.csv")
            if ace_mean is not None:
                data[loss_type]["ace"][bin_num] = (ace_mean, ace_std)

    return data


def collect_weight_sensitivity_data():
    """
    Collect data for loss weight sensitivity analysis.

    Returns:
        dict: Nested dictionary with structure:
              {loss_type: {metric: {weight: (mean, std)}}}
    """
    data = {"hardl1ace": {"dice": {}, "ace": {}}, "softl1ace": {"dice": {}, "ace": {}}}

    for loss_type in ["hardl1ace", "softl1ace"]:
        for weight in LOSS_WEIGHTS:
            # Format weight for directory name (e.g., 0.1 -> 0_10, 1.0 -> 1_00)
            # Special case: 10.0 is formatted as 10_0 (not 10_00)
            if weight == 10.0:
                weight_str = "10_0"
            else:
                weight_str = f"{weight:.2f}".replace(".", "_")
            bundle_name = f"acdc17_{loss_type}_dice_ce_2_ace_{weight_str}"
            bundle_dir = BUNDLES_DIR / bundle_name

            if not bundle_dir.exists():
                print(f"Warning: Bundle directory not found: {bundle_dir}")
                continue

            # Load Dice score
            dice_mean, dice_std = load_metric(bundle_dir, "mean_dice_summary.csv")
            if dice_mean is not None:
                data[loss_type]["dice"][weight] = (dice_mean, dice_std)

            # Load Macro ACE
            ace_mean, ace_std = load_metric(bundle_dir, "macro_ace_summary.csv")
            if ace_mean is not None:
                data[loss_type]["ace"][weight] = (ace_mean, ace_std)

    return data


def plot_bin_sensitivity(data):
    """
    Create plots showing sensitivity to bin number.

    Args:
        data: Dictionary with bin sensitivity data
    """
    # Full page width: 181.5mm = 7.14 inches
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 3.0))

    # Colors and markers - orange for Hard L1-ACE, green for Soft L1-ACE
    colors = {"hardl1ace": "#ff7f0e", "softl1ace": "#2ca02c"}
    markers = {"hardl1ace": "o", "softl1ace": "s"}
    labels = {"hardl1ace": "hL1-ACE", "softl1ace": "sL1-ACE"}

    # Plot Dice scores (left panel)
    ax_dice = axes[0]
    for loss_type in ["hardl1ace", "softl1ace"]:
        dice_data = data[loss_type]["dice"]
        if dice_data:
            bins = sorted(dice_data.keys())
            means = [dice_data[b][0] for b in bins]
            stds = [dice_data[b][1] for b in bins]

            ax_dice.errorbar(
                bins,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
            )

    ax_dice.set_xlabel("Number of Bins")
    ax_dice.set_ylabel("Dice Score")
    ax_dice.grid(True, alpha=0.3)
    ax_dice.legend(loc="best")
    ax_dice.set_xscale("log")
    ax_dice.set_xticks(BIN_NUMBERS)
    ax_dice.set_xticklabels(BIN_NUMBERS)

    # Plot Macro ACE (right panel)
    ax_ace = axes[1]
    for loss_type in ["hardl1ace", "softl1ace"]:
        ace_data = data[loss_type]["ace"]
        if ace_data:
            bins = sorted(ace_data.keys())
            means = [ace_data[b][0] for b in bins]
            stds = [ace_data[b][1] for b in bins]

            ax_ace.errorbar(
                bins,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
            )

    ax_ace.set_xlabel("Number of Bins")
    ax_ace.set_ylabel("Macro ACE")
    ax_ace.grid(True, alpha=0.3)
    ax_ace.legend(loc="best")
    ax_ace.set_xscale("log")
    ax_ace.set_xticks(BIN_NUMBERS)
    ax_ace.set_xticklabels(BIN_NUMBERS)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sensitivity_bin_number.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved bin sensitivity plot to {output_file}")
    plt.close()


def plot_weight_sensitivity(data):
    """
    Create plots showing sensitivity to loss weighting.

    Args:
        data: Dictionary with weight sensitivity data
    """
    # Full page width: 181.5mm = 7.14 inches
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 3.0))

    # Colors and markers - orange for Hard L1-ACE, green for Soft L1-ACE
    colors = {"hardl1ace": "#ff7f0e", "softl1ace": "#2ca02c"}
    markers = {"hardl1ace": "o", "softl1ace": "s"}
    labels = {"hardl1ace": "hL1-ACE", "softl1ace": "sL1-ACE"}

    # Plot Dice scores (left panel)
    ax_dice = axes[0]
    for loss_type in ["hardl1ace", "softl1ace"]:
        dice_data = data[loss_type]["dice"]
        if dice_data:
            weights = sorted(dice_data.keys())
            means = [dice_data[w][0] for w in weights]
            stds = [dice_data[w][1] for w in weights]

            ax_dice.errorbar(
                weights,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
            )

    ax_dice.set_xlabel("Loss Weight (λ)")
    ax_dice.set_ylabel("Dice Score")
    ax_dice.grid(True, alpha=0.3)
    ax_dice.legend(loc="best")
    ax_dice.set_xscale("log")
    # Set custom x-ticks for better readability
    ax_dice.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    ax_dice.set_xticklabels(["0.1", "0.25", "0.5", "1.0", "2.0", "4.0", "10.0"])

    # Plot Macro ACE (right panel)
    ax_ace = axes[1]
    for loss_type in ["hardl1ace", "softl1ace"]:
        ace_data = data[loss_type]["ace"]
        if ace_data:
            weights = sorted(ace_data.keys())
            means = [ace_data[w][0] for w in weights]
            stds = [ace_data[w][1] for w in weights]

            ax_ace.errorbar(
                weights,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
            )

    ax_ace.set_xlabel("Loss Weight (λ)")
    ax_ace.set_ylabel("Macro ACE")
    ax_ace.grid(True, alpha=0.3)
    ax_ace.legend(loc="best")
    ax_ace.set_xscale("log")
    # Set custom x-ticks for better readability
    ax_ace.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    ax_ace.set_xticklabels(["0.1", "0.25", "0.5", "1.0", "2.0", "4.0", "10.0"])

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sensitivity_loss_weight.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved loss weight sensitivity plot to {output_file}")
    plt.close()


def print_summary_statistics(bin_data, weight_data):
    """
    Print summary statistics for both sensitivity analyses.

    Args:
        bin_data: Dictionary with bin sensitivity data
        weight_data: Dictionary with weight sensitivity data
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\n--- BIN NUMBER SENSITIVITY ---")
    for loss_type in ["hardl1ace", "softl1ace"]:
        print(f"\n{loss_type.upper()}:")

        dice_data = bin_data[loss_type]["dice"]
        if dice_data:
            print("\n  Dice Scores:")
            for bin_num in sorted(dice_data.keys()):
                mean, std = dice_data[bin_num]
                print(f"    Bins={bin_num:3d}: {mean:.4f} ± {std:.4f}")

        ace_data = bin_data[loss_type]["ace"]
        if ace_data:
            print("\n  Macro ACE:")
            for bin_num in sorted(ace_data.keys()):
                mean, std = ace_data[bin_num]
                print(f"    Bins={bin_num:3d}: {mean:.4f} ± {std:.4f}")

    print("\n--- LOSS WEIGHT SENSITIVITY ---")
    for loss_type in ["hardl1ace", "softl1ace"]:
        print(f"\n{loss_type.upper()}:")

        dice_data = weight_data[loss_type]["dice"]
        if dice_data:
            print("\n  Dice Scores:")
            for weight in sorted(dice_data.keys()):
                mean, std = dice_data[weight]
                print(f"    λ={weight:5.2f}: {mean:.4f} ± {std:.4f}")

        ace_data = weight_data[loss_type]["ace"]
        if ace_data:
            print("\n  Macro ACE:")
            for weight in sorted(ace_data.keys()):
                mean, std = ace_data[weight]
                print(f"    λ={weight:5.2f}: {mean:.4f} ± {std:.4f}")

    print("\n" + "=" * 80)


def save_summary_tables(bin_data, weight_data):
    """
    Save summary statistics as markdown tables.

    Args:
        bin_data: Dictionary with bin sensitivity data
        weight_data: Dictionary with weight sensitivity data
    """
    output_file = OUTPUT_DIR / "sensitivity_analysis_tables.md"

    with open(output_file, "w") as f:
        f.write("# Sensitivity Analysis Results\n\n")
        f.write("## Bin Number Sensitivity\n\n")

        # Bin number table for Dice
        f.write("### Dice Score vs. Bin Number\n\n")
        f.write("| Bins | Hard L1-ACE | Soft L1-ACE |\n")
        f.write("|------|-------------|-------------|\n")

        all_bins = sorted(
            set(bin_data["hardl1ace"]["dice"].keys())
            | set(bin_data["softl1ace"]["dice"].keys())
        )
        for bin_num in all_bins:
            hard_dice = bin_data["hardl1ace"]["dice"].get(bin_num)
            soft_dice = bin_data["softl1ace"]["dice"].get(bin_num)

            hard_str = (
                f"{hard_dice[0]:.4f} ± {hard_dice[1]:.4f}" if hard_dice else "N/A"
            )
            soft_str = (
                f"{soft_dice[0]:.4f} ± {soft_dice[1]:.4f}" if soft_dice else "N/A"
            )

            f.write(f"| {bin_num} | {hard_str} | {soft_str} |\n")

        # Bin number table for ACE
        f.write("\n### Macro ACE vs. Bin Number\n\n")
        f.write("| Bins | Hard L1-ACE | Soft L1-ACE |\n")
        f.write("|------|-------------|-------------|\n")

        for bin_num in all_bins:
            hard_ace = bin_data["hardl1ace"]["ace"].get(bin_num)
            soft_ace = bin_data["softl1ace"]["ace"].get(bin_num)

            hard_str = f"{hard_ace[0]:.4f} ± {hard_ace[1]:.4f}" if hard_ace else "N/A"
            soft_str = f"{soft_ace[0]:.4f} ± {soft_ace[1]:.4f}" if soft_ace else "N/A"

            f.write(f"| {bin_num} | {hard_str} | {soft_str} |\n")

        # Loss weight sensitivity
        f.write("\n## Loss Weight Sensitivity\n\n")

        # Loss weight table for Dice
        f.write("### Dice Score vs. Loss Weight (λ)\n\n")
        f.write("| λ | Hard L1-ACE | Soft L1-ACE |\n")
        f.write("|---|-------------|-------------|\n")

        all_weights = sorted(
            set(weight_data["hardl1ace"]["dice"].keys())
            | set(weight_data["softl1ace"]["dice"].keys())
        )
        for weight in all_weights:
            hard_dice = weight_data["hardl1ace"]["dice"].get(weight)
            soft_dice = weight_data["softl1ace"]["dice"].get(weight)

            hard_str = (
                f"{hard_dice[0]:.4f} ± {hard_dice[1]:.4f}" if hard_dice else "N/A"
            )
            soft_str = (
                f"{soft_dice[0]:.4f} ± {soft_dice[1]:.4f}" if soft_dice else "N/A"
            )

            f.write(f"| {weight:.2f} | {hard_str} | {soft_str} |\n")

        # Loss weight table for ACE
        f.write("\n### Macro ACE vs. Loss Weight (λ)\n\n")
        f.write("| λ | Hard L1-ACE | Soft L1-ACE |\n")
        f.write("|---|-------------|-------------|\n")

        for weight in all_weights:
            hard_ace = weight_data["hardl1ace"]["ace"].get(weight)
            soft_ace = weight_data["softl1ace"]["ace"].get(weight)

            hard_str = f"{hard_ace[0]:.4f} ± {hard_ace[1]:.4f}" if hard_ace else "N/A"
            soft_str = f"{soft_ace[0]:.4f} ± {soft_ace[1]:.4f}" if soft_ace else "N/A"

            f.write(f"| {weight:.2f} | {hard_str} | {soft_str} |\n")

    print(f"\n✓ Summary tables saved to: {output_file}")


def plot_combined_sensitivity(bin_data, weight_data):
    """
    Create combined plots with both Dice and ACE on dual y-axes.
    Left plot: Bin sensitivity, Right plot: Loss weight sensitivity.

    Args:
        bin_data: Dictionary with bin sensitivity data
        weight_data: Dictionary with weight sensitivity data
    """
    # Full page width: 181.5mm = 7.14 inches
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 3.0))

    # Colors - orange for Hard L1-ACE, green for Soft L1-ACE
    colors = {"hardl1ace": "#ff7f0e", "softl1ace": "#2ca02c"}
    # Markers - circle for Hard, square for Soft
    markers = {"hardl1ace": "o", "softl1ace": "s"}
    # Line styles - solid for Dice, dashed for ACE
    linestyles = {"dice": "-", "ace": "--"}
    labels = {"hardl1ace": "hL1-ACE", "softl1ace": "sL1-ACE"}

    # ===== LEFT PANEL: Bin Number Sensitivity =====
    ax_dice_bin = axes[0]
    ax_ace_bin = ax_dice_bin.twinx()

    for loss_type in ["hardl1ace", "softl1ace"]:
        # Plot Dice scores (solid line, left y-axis)
        dice_data = bin_data[loss_type]["dice"]
        if dice_data:
            bins = sorted(dice_data.keys())
            means = [dice_data[b][0] for b in bins]
            stds = [dice_data[b][1] for b in bins]

            ax_dice_bin.errorbar(
                bins,
                means,
                yerr=stds,
                label=f"{labels[loss_type]} (Dice)",
                color=colors[loss_type],
                marker=markers[loss_type],
                linestyle=linestyles["dice"],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
            )

        # Plot ACE (dashed line, right y-axis)
        ace_data = bin_data[loss_type]["ace"]
        if ace_data:
            bins = sorted(ace_data.keys())
            means = [ace_data[b][0] for b in bins]
            stds = [ace_data[b][1] for b in bins]

            ax_ace_bin.errorbar(
                bins,
                means,
                yerr=stds,
                label=f"{labels[loss_type]} (ACE)",
                color=colors[loss_type],
                marker=markers[loss_type],
                linestyle=linestyles["ace"],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
                fillstyle="none",
            )

    ax_dice_bin.set_xlabel("Number of Bins")
    ax_dice_bin.set_ylabel("Dice Score")
    ax_ace_bin.set_ylabel("Macro ACE")
    ax_dice_bin.grid(True, alpha=0.3)
    ax_dice_bin.set_xscale("log")
    ax_dice_bin.set_xticks(BIN_NUMBERS)
    ax_dice_bin.set_xticklabels(BIN_NUMBERS)

    # Combined legend
    lines1, labels1 = ax_dice_bin.get_legend_handles_labels()
    lines2, labels2 = ax_ace_bin.get_legend_handles_labels()
    ax_dice_bin.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=6)

    ax_dice_bin.set_title("(a) Bin Number Sensitivity")

    # ===== RIGHT PANEL: Loss Weight Sensitivity =====
    ax_dice_weight = axes[1]
    ax_ace_weight = ax_dice_weight.twinx()

    for loss_type in ["hardl1ace", "softl1ace"]:
        # Plot Dice scores (solid line, left y-axis)
        dice_data = weight_data[loss_type]["dice"]
        if dice_data:
            weights = sorted(dice_data.keys())
            # Filter out 0.0 for log scale
            weights = [w for w in weights if w > 0]
            means = [dice_data[w][0] for w in weights]
            stds = [dice_data[w][1] for w in weights]

            ax_dice_weight.errorbar(
                weights,
                means,
                yerr=stds,
                label=f"{labels[loss_type]} (Dice)",
                color=colors[loss_type],
                marker=markers[loss_type],
                linestyle=linestyles["dice"],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
            )

        # Plot ACE (dashed line, right y-axis)
        ace_data = weight_data[loss_type]["ace"]
        if ace_data:
            weights = sorted(ace_data.keys())
            # Filter out 0.0 for log scale
            weights = [w for w in weights if w > 0]
            means = [ace_data[w][0] for w in weights]
            stds = [ace_data[w][1] for w in weights]

            ax_ace_weight.errorbar(
                weights,
                means,
                yerr=stds,
                label=f"{labels[loss_type]} (ACE)",
                color=colors[loss_type],
                marker=markers[loss_type],
                linestyle=linestyles["ace"],
                markersize=4,
                linewidth=1.0,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
                fillstyle="none",
            )

    ax_dice_weight.set_xlabel("Loss Weight (λ)")
    ax_dice_weight.set_ylabel("Dice Score")
    ax_ace_weight.set_ylabel("Macro ACE")
    ax_dice_weight.grid(True, alpha=0.3)
    ax_dice_weight.set_xscale("log")
    ax_dice_weight.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    ax_dice_weight.set_xticklabels(["0.1", "0.25", "0.5", "1.0", "2.0", "4.0", "10.0"])

    # Combined legend
    lines1, labels1 = ax_dice_weight.get_legend_handles_labels()
    lines2, labels2 = ax_ace_weight.get_legend_handles_labels()
    ax_dice_weight.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=6)

    ax_dice_weight.set_title("(b) Loss Weight Sensitivity")

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sensitivity_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved combined sensitivity plot to {output_file}")
    plt.close()


def plot_sensitivity_4x1(bin_data, weight_data):
    """
    Create a 4-panel plot (1 row, 4 columns) showing all sensitivity analyses.
    Panel (a): Bin sensitivity - Dice Score
    Panel (b): Bin sensitivity - Macro ACE
    Panel (c): Loss weight sensitivity - Dice Score
    Panel (d): Loss weight sensitivity - Macro ACE

    Args:
        bin_data: Dictionary with bin sensitivity data
        weight_data: Dictionary with weight sensitivity data
    """
    # Full page width: 181.5mm = 7.14 inches, shorter height for 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(7.14, 2.0))

    # Colors and markers - orange for Hard L1-ACE, green for Soft L1-ACE
    colors = {"hardl1ace": "#ff7f0e", "softl1ace": "#2ca02c"}
    markers = {"hardl1ace": "o", "softl1ace": "s"}
    labels = {"hardl1ace": "hL1-ACE", "softl1ace": "sL1-ACE"}

    # ===== Panel (a): Bin Sensitivity - Dice =====
    ax = axes[0]
    for loss_type in ["hardl1ace", "softl1ace"]:
        dice_data = bin_data[loss_type]["dice"]
        if dice_data:
            bins = sorted(dice_data.keys())
            means = [dice_data[b][0] for b in bins]
            stds = [dice_data[b][1] for b in bins]

            ax.errorbar(
                bins,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=3,
                linewidth=1.0,
                capsize=2,
                capthick=0.6,
                elinewidth=0.6,
            )

    ax.set_xlabel("Number of Bins")
    ax.set_ylabel("DSC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=6)
    ax.set_xscale("log")
    ax.set_xticks(BIN_NUMBERS)
    ax.set_xticklabels(BIN_NUMBERS)

    # ===== Panel (b): Bin Sensitivity - ACE =====
    ax = axes[1]
    for loss_type in ["hardl1ace", "softl1ace"]:
        ace_data = bin_data[loss_type]["ace"]
        if ace_data:
            bins = sorted(ace_data.keys())
            means = [ace_data[b][0] for b in bins]
            stds = [ace_data[b][1] for b in bins]

            ax.errorbar(
                bins,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=3,
                linewidth=1.0,
                capsize=2,
                capthick=0.6,
                elinewidth=0.6,
            )

    ax.set_xlabel("Number of Bins")
    ax.set_ylabel("ACE")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(BIN_NUMBERS)
    ax.set_xticklabels(BIN_NUMBERS)

    # ===== Panel (c): Loss Weight Sensitivity - Dice =====
    ax = axes[2]
    for loss_type in ["hardl1ace", "softl1ace"]:
        dice_data = weight_data[loss_type]["dice"]
        if dice_data:
            weights = sorted(dice_data.keys())
            means = [dice_data[w][0] for w in weights]
            stds = [dice_data[w][1] for w in weights]

            ax.errorbar(
                weights,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=3,
                linewidth=1.0,
                capsize=2,
                capthick=0.6,
                elinewidth=0.6,
            )

    ax.set_xlabel("Loss Weight (λ)")
    ax.set_ylabel("DSC")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    ax.set_xticklabels(["0.1", "", "0.5", "1", "2", "4", "10"], fontsize=7)

    # ===== Panel (d): Loss Weight Sensitivity - ACE =====
    ax = axes[3]
    for loss_type in ["hardl1ace", "softl1ace"]:
        ace_data = weight_data[loss_type]["ace"]
        if ace_data:
            weights = sorted(ace_data.keys())
            means = [ace_data[w][0] for w in weights]
            stds = [ace_data[w][1] for w in weights]

            ax.errorbar(
                weights,
                means,
                yerr=stds,
                label=labels[loss_type],
                color=colors[loss_type],
                marker=markers[loss_type],
                markersize=3,
                linewidth=1.0,
                capsize=2,
                capthick=0.6,
                elinewidth=0.6,
            )

    ax.set_xlabel("Loss Weight (λ)")
    ax.set_ylabel("ACE")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
    ax.set_xticklabels(["0.1", "", "0.5", "1", "2", "4", "10"], fontsize=7)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "sensitivity_4x1.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved 4x1 sensitivity plot to {output_file}")
    plt.close()


def main():
    """
    Main function to run the sensitivity analysis and generate plots.
    """
    print("=" * 80)
    print("SENSITIVITY ANALYSIS FOR IEEE TMI REVISION")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Analyzing ACDC17 dataset experiments")
    print(f"Seed: {SEED}")

    # Collect data
    print("\n--- Collecting bin number sensitivity data ---")
    bin_data = collect_bin_sensitivity_data()

    print("\n--- Collecting loss weight sensitivity data ---")
    weight_data = collect_weight_sensitivity_data()

    # Generate plots
    print("\n--- Generating plots ---")
    plot_bin_sensitivity(bin_data)
    plot_weight_sensitivity(weight_data)
    plot_combined_sensitivity(bin_data, weight_data)
    plot_sensitivity_4x1(bin_data, weight_data)

    # Print summary statistics
    print_summary_statistics(bin_data, weight_data)

    # Save summary tables
    print("\n--- Saving summary tables ---")
    save_summary_tables(bin_data, weight_data)

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    print("  - sensitivity_bin_number.png/pdf")
    print("  - sensitivity_loss_weight.png/pdf")
    print("  - sensitivity_combined.png/pdf")
    print("  - sensitivity_4x1.png/pdf")
    print("  - sensitivity_analysis_tables.md")
    print("\nDone!")


if __name__ == "__main__":
    main()
