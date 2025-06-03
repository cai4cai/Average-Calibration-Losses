import os
import pandas as pd
import numpy as np

# Setup raw CSV file names as used in journal_results_main.ipynb
TEST_METRICS_CSVS = {
    "macro_ACE": "macro_ace_raw.csv",
    "macro_ECE": "macro_ece_raw.csv",
    "macro_MCE": "macro_mce_raw.csv",
    "DSC": "mean_dice_raw.csv",
}

SEED = 12345

# Directories as specified in journal_results_main.ipynb
RUNS = {
    "amos": {
        "baseline": "bundles/amos22_baseline_dice_ce_nl",
        "hardl1ace": "bundles/amos22_hardl1ace_dice_ce_nl",
        "softl1ace": "bundles/amos22_softl1ace_dice_ce_nl",
    },
    "acdc": {
        "baseline": "bundles/acdc17_baseline_dice_ce_1",
        "hardl1ace": "bundles/acdc17_hardl1ace_dice_ce_1",
        "softl1ace": "bundles/acdc17_softl1ace_dice_ce_1",
    },
    "kits": {
        "baseline": "bundles/kits23_baseline_dice_ce_nl",
        "hardl1ace": "bundles/kits23_hardl1ace_dice_ce_nl",
        "softl1ace": "bundles/kits23_softl1ace_dice_ce_nl",
    },
    "brats": {
        "baseline": "bundles/brats21_baseline_dice_ce_nl",
        "hardl1ace": "bundles/brats21_hardl1ace_dice_ce_nl",
        "softl1ace": "bundles/brats21_softl1ace_dice_ce_nl",
    },
}


def load_csv(metric, run_dir, inference_dir="inference_results_complete_cases_only"):
    csv_filename = TEST_METRICS_CSVS[metric]
    csv_path = os.path.join(run_dir, f"seed_{SEED}", inference_dir, csv_filename)
    if not os.path.exists(csv_path):
        inference_dir = "inference_results"  # fallback to original directory for ACDC which is complete anyway
        csv_path = os.path.join(run_dir, f"seed_{SEED}", inference_dir, csv_filename)
    return pd.read_csv(csv_path)


def flatten_numeric_values(arr):
    return arr.ravel()


def main():
    # prepare a per-metric aggregator
    agg_pct = {m: {"hard": [], "soft": []} for m in TEST_METRICS_CSVS}

    for dataset, paths in RUNS.items():
        print(f"\n=== Results for dataset: {dataset.upper()} ===")
        for metric in TEST_METRICS_CSVS.keys():
            try:
                df_base = load_csv(metric, paths["baseline"])
                df_hard = load_csv(metric, paths["hardl1ace"])
                df_soft = load_csv(metric, paths["softl1ace"])
            except FileNotFoundError as e:
                print(f"  · {metric}: {e}")
                continue

            base_vals = flatten_numeric_values(df_base["mean"].to_numpy())
            hard_vals = flatten_numeric_values(df_hard["mean"].to_numpy())
            soft_vals = flatten_numeric_values(df_soft["mean"].to_numpy())

            # per-case percentage change
            hard_pct = (hard_vals - base_vals) / base_vals * 100
            soft_pct = (soft_vals - base_vals) / base_vals * 100

            # accumulate in the per-metric buckets
            agg_pct[metric]["hard"].extend(hard_pct.tolist())
            agg_pct[metric]["soft"].extend(soft_pct.tolist())

            # print per‐dataset summary
            print(
                f"  · {metric:9s} | "
                f"Hard: {hard_pct.mean():+6.2f}% ± {hard_pct.std():.2f}%, "
                f"Soft: {soft_pct.mean():+6.2f}% ± {soft_pct.std():.2f}%"
            )

    # now print per‐metric aggregate across datasets
    print("\n=== Aggregate across ALL DATASETS (per metric) ===")
    for metric in TEST_METRICS_CSVS.keys():
        hard_all = np.array(agg_pct[metric]["hard"])
        soft_all = np.array(agg_pct[metric]["soft"])
        print(
            f"{metric:9s} | "
            f"Hard: {hard_all.mean():+6.2f}% ± {hard_all.std():.2f}%, "
            f"Soft: {soft_all.mean():+6.2f}% ± {soft_all.std():.2f}%"
        )


if __name__ == "__main__":
    main()
