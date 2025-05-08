import pandas as pd

acdc_class_names = ["right_ventricle", "myocardium", "left_ventricle"]

amos_class_names = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
    "duodenum",
    "bladder",
    "prostate/uterus",
]

brats_class_names = ["Necrotic Tumour Core", "Peritumoral Edema", "Enhancing Tumour"]

kits_class_names = ["kidney", "tumour", "cyst"]

acdc_missing_classes_csv_path = "bundles/acdc17_baseline_dice_ce_1/seed_12345/inference_results/missing_classes_raw.csv"
amos_missing_classes_csv_path = "bundles/amos22_baseline_dice_ce_nl/seed_12345/inference_results/missing_classes_raw.csv"
brats_missing_classes_csv_path = "bundles/brats21_baseline_dice_ce_nl/seed_12345/inference_results/missing_classes_raw.csv"
kits_missing_classes_csv_path = "bundles/kits23_baseline_dice_ce_nl/seed_12345/inference_results/missing_classes_raw.csv"


def count_missing_classes(csv_path, class_names):
    df = pd.read_csv(csv_path)
    missing_counts = df.iloc[:, 1:].sum(axis=0).tolist()
    incomplete_cases = (df.iloc[:, 1:].sum(axis=1) > 0).sum()
    total_cases = len(df)
    return missing_counts, incomplete_cases, total_cases


def generate_markdown_table(
    dataset_name, class_names, missing_counts, incomplete_cases, total_cases
):
    table = f"### {dataset_name} Dataset\n\n"
    table += "| Class | Missing Count |\n"
    table += "|-------|---------------|\n"
    for class_name, count in zip(class_names, missing_counts):
        table += f"| {class_name} | {count} |\n"
    table += f"\n**Incomplete Cases:** {incomplete_cases}/{total_cases}\n"
    return table


datasets = [
    ("ACDC", acdc_class_names, acdc_missing_classes_csv_path),
    ("AMOS", amos_class_names, amos_missing_classes_csv_path),
    ("BraTS", brats_class_names, brats_missing_classes_csv_path),
    ("KiTS", kits_class_names, kits_missing_classes_csv_path),
]

markdown_report = ""
for dataset_name, class_names, csv_path in datasets:
    missing_counts, incomplete_cases, total_cases = count_missing_classes(
        csv_path, class_names
    )
    markdown_report += generate_markdown_table(
        dataset_name, class_names, missing_counts, incomplete_cases, total_cases
    )
    markdown_report += "\n"

with open("missing_classes_report.md", "w") as f:
    f.write(markdown_report)
