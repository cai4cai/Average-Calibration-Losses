# %%
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from tabulate import tabulate

# Setup raw CSV file names as used in journal_results_main.ipynb
TEST_METRICS_CSVS = {
    # "micro_ACE": "micro_ace_raw.csv",  # cannot calculate p-vals for micro metrics as there is only one value
    "macro_ACE": "macro_ace_raw.csv",
    # "micro_ECE": "micro_ece_raw.csv",
    "macro_ECE": "macro_ece_raw.csv",
    # "micro_MCE": "micro_mce_raw.csv",
    "macro_MCE": "macro_mce_raw.csv",
    "DSC": "mean_dice_raw.csv",
}

SEED = 12345

# Directories as specified in journal_results_main.ipynb
RUNS = {
    "amos": {
        "baseline": "../bundles/amos22_baseline_dice_ce_nl",
        "hardl1ace": "../bundles/amos22_hardl1ace_dice_ce_nl",
        "softl1ace": "../bundles/amos22_softl1ace_dice_ce_nl",
    },
    "acdc": {
        "baseline": "../bundles/acdc17_baseline_dice_ce_1",
        "hardl1ace": "../bundles/acdc17_hardl1ace_dice_ce_1",
        "softl1ace": "../bundles/acdc17_softl1ace_dice_ce_1",
    },
    "kits": {
        "baseline": "../bundles/kits23_baseline_dice_ce_nl",
        "hardl1ace": "../bundles/kits23_hardl1ace_dice_ce_nl",
        "softl1ace": "../bundles/kits23_softl1ace_dice_ce_nl",
    },
    "brats": {
        "baseline": "../bundles/brats21_baseline_dice_ce_nl",
        "hardl1ace": "../bundles/brats21_hardl1ace_dice_ce_nl",
        "softl1ace": "../bundles/brats21_softl1ace_dice_ce_nl",
    },
}


def load_csv(metric, run_dir, inference_dir="inference_results"):
    # Build csv file path using the provided run_dir and SEED
    csv_filename = TEST_METRICS_CSVS[metric]
    csv_path = os.path.join(run_dir, f"seed_{SEED}", inference_dir, csv_filename)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"File does not exist: {csv_path}")


def flatten_numeric_values(df):
    # Remove non-numeric columns and flatten all numeric values
    # numeric_df = df.select_dtypes(include=[np.number])
    return df.to_numpy().ravel()


def main():
    for dataset, paths in RUNS.items():
        print(f"\n=== Results for dataset: {dataset.upper()} ===\n")
        results_table = []
        for metric in TEST_METRICS_CSVS.keys():
            try:
                df_base = load_csv(metric, paths["baseline"])
                df_hard = load_csv(metric, paths["hardl1ace"])
                df_soft = load_csv(metric, paths["softl1ace"])
            except FileNotFoundError as e:
                print(e)
                continue

            base_vals = flatten_numeric_values(df_base["mean"])
            hard_vals = flatten_numeric_values(df_hard["mean"])
            soft_vals = flatten_numeric_values(df_soft["mean"])

            try:
                _, p_val_hard = ttest_rel(base_vals, hard_vals)
            except Exception:
                p_val_hard = np.nan

            try:
                _, p_val_soft = ttest_rel(base_vals, soft_vals)
            except Exception:
                p_val_soft = np.nan

            results_table.append([metric, f"{p_val_hard:.3e}", f"{p_val_soft:.3e}"])

        headers = [
            "Metric",
            "Baseline vs HardL1-ACE p-value",
            "Baseline vs SoftL1-ACE p-value",
        ]
        print(tabulate(results_table, headers=headers, tablefmt="pipe"))

        # # Optionally plot histogram for DSC
        # if "DSC" in TEST_METRICS_CSVS:
        #     try:
        #         plt.figure()
        #         base_dsc = flatten_numeric_values(load_csv("DSC", paths["baseline"]))
        #         hard_dsc = flatten_numeric_values(load_csv("DSC", paths["hardl1ace"]))
        #         soft_dsc = flatten_numeric_values(load_csv("DSC", paths["softl1ace"]))
        #         plt.hist(base_dsc, alpha=0.5, label="Baseline")
        #         plt.hist(hard_dsc, alpha=0.5, label="HardL1-ACE")
        #         plt.hist(soft_dsc, alpha=0.5, label="SoftL1-ACE")
        #         plt.xlabel("DSC values")
        #         plt.ylabel("Count")
        #         plt.legend()
        #         plt.title(f"Histogram of DSC values for {dataset.upper()}")
        #         plt.show()
        #         plt.savefig(f"{dataset.upper()}_DSC_hist.png")
        #         plt.close()
        #     except Exception as e:
        #         print(f"Could not plot DSC histogram for {dataset}: {e}")


# %%
if __name__ == "__main__":
    main()
