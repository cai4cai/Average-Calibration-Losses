"""
Script to compile and compare results from different bundle configurations.
Extracts mean ± std from the 'mean' class row of summary CSV files.
"""

import os
import pandas as pd
from pathlib import Path

# Base path
BASE_PATH = Path("/workspaces/Average-Calibration-Losses/bundles")

# Bundle configurations to compare
BUNDLES = [
    "acdc17_baseline_ce_2",
    "acdc17_baseline_dice_ce_2",
    "acdc17_dc_dice_ce",
    "acdc17_hardl1ace_dice_ce_2",
    "acdc17_softl1ace_dice_ce_2",
]

# Metrics to extract
METRICS = [
    "macro_ace_summary",
    "macro_ece_summary",
    "macro_mce_summary",
    "mean_dice_summary",
    "micro_ace_summary",
    "micro_ece_summary",
    "micro_mce_summary",
]

# Seed
SEED = "seed_12345"


def get_mean_std_from_summary(csv_path: Path) -> tuple[float, float] | None:
    """
    Read a summary CSV and extract mean and std from the 'mean' class row.

    Returns:
        Tuple of (mean, std) or None if file doesn't exist or parsing fails.
    """
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        # Get the row where class == 'mean'
        mean_row = df[df["class"] == "mean"]
        if mean_row.empty:
            return None

        mean_val = mean_row["mean"].values[0]
        std_val = mean_row["std"].values[0]
        return (mean_val, std_val)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format mean ± std as a string."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def compile_results() -> pd.DataFrame:
    """
    Compile results from all bundles and metrics.

    Returns:
        DataFrame with bundles as rows and metrics as columns.
    """
    results = {}

    for bundle in BUNDLES:
        results[bundle] = {}
        inference_path = BASE_PATH / bundle / SEED / "inference_results"

        for metric in METRICS:
            csv_path = inference_path / f"{metric}.csv"
            result = get_mean_std_from_summary(csv_path)

            if result is not None:
                mean_val, std_val = result
                results[bundle][metric] = format_mean_std(mean_val, std_val)
            else:
                results[bundle][metric] = "N/A"

    # Create DataFrame
    df = pd.DataFrame(results).T
    df.index.name = "Bundle"

    return df


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate a markdown table from the DataFrame."""
    # Create header
    headers = ["Bundle"] + list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Create rows
    rows = []
    for bundle, row in df.iterrows():
        row_values = [bundle] + [row[col] for col in df.columns]
        rows.append("| " + " | ".join(str(v) for v in row_values) + " |")

    return "\n".join([header_line, separator] + rows)


def main():
    print("Compiling results from bundles...")
    print(f"Bundles: {BUNDLES}")
    print(f"Metrics: {METRICS}")
    print()

    # Compile results
    df = compile_results()

    # Generate and print markdown table
    markdown_table = generate_markdown_table(df)

    print("=" * 80)
    print("RESULTS (mean ± std)")
    print("=" * 80)
    print()
    print(markdown_table)
    print()

    # Also save to a markdown file
    output_path = Path(__file__).parent / "compare_dc_loss_results.md"
    with open(output_path, "w") as f:
        f.write("# DC Loss Comparison Results\n\n")
        f.write(
            "Results showing mean ± std for the mean class across all test samples.\n\n"
        )
        f.write(markdown_table)
        f.write("\n")

    print(f"Results saved to: {output_path}")

    # Print raw DataFrame for reference
    print()
    print("=" * 80)
    print("Raw DataFrame:")
    print("=" * 80)
    print(df.to_string())


if __name__ == "__main__":
    main()
