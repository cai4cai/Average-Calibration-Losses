import os
import csv
from tabulate import tabulate

directories = [
    "bundles/kits23_baseline_dice_ce_8/seed_12345",
    "bundles/kits23_hardl1ace_dice_ce_8/seed_12345",
    "bundles/kits23_softl1ace_dice_ce_8/seed_12345",
]

directories = [
    "bundles/amos22_baseline_dice_ce_nl/seed_12345",
    "bundles/amos22_hardl1ace_dice_ce_nl/seed_12345",
    "bundles/amos22_softl1ace_dice_ce_nl/seed_12345",
    "bundles/acdc17_baseline_dice_ce_1/seed_12345",
    "bundles/acdc17_hardl1ace_dice_ce_1/seed_12345",
    "bundles/acdc17_softl1ace_dice_ce_1/seed_12345",
    "bundles/kits23_baseline_dice_ce_8/seed_12345",
    "bundles/kits23_hardl1ace_dice_ce_8/seed_12345",
    "bundles/kits23_softl1ace_dice_ce_8/seed_12345",
    "bundles/brats21_baseline_dice_ce_nl/seed_12345",
    "bundles/brats21_hardl1ace_dice_ce_nl/seed_12345",
    "bundles/brats21_softl1ace_dice_ce_nl/seed_12345",
]

if __name__ == "__main__":
    output_dir = "epoch_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    for directory in directories:
        run_name = directory.split("/")[1]
        data = []

        for file in os.listdir(directory):
            # Check if file matches format "model_key_metric=VALUE.pt"
            if file.startswith("model_key_metric="):
                dsc_value_str = file.split("=")[-1].split(".pt")[0]
                try:
                    dsc_value = float(dsc_value_str)
                except ValueError:
                    dsc_value = 0.0

                log_path = os.path.join(directory, "log.txt")
                # Parse log file for the matching DSC value
                with open(log_path, "r") as f:
                    for line in f:
                        if (
                            f"val_mean_dice: {dsc_value_str}" in line
                            and "INFO: Epoch" in line
                        ):
                            epoch = line.split("[")[1].split("]")[0]
                            data.append((dsc_value, epoch))

        # Sort in descending order by DSC value
        data.sort(key=lambda x: x[0], reverse=True)

        # Create a table for printing
        table = [[f"{val:.4f}", ep] for (val, ep) in data]
        print(f"Results for {run_name}")
        print(tabulate(table, headers=["DSC Value", "Epoch"], tablefmt="pipe"))
        print()

        # Save the table to the "epoch_checkpoints" directory
        csv_filename = os.path.join(output_dir, f"{run_name}_epochs.csv")
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["DSC Value", "Epoch"])
            for row in table:
                writer.writerow(row)
