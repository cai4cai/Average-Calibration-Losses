# Script to take TESTING DATASETS!! for AMOS, KiTS and BraTS dataset and copy cases with all classes represented in each case
# In the case of AMOS22 these are the imagesVa and labelsVa, filtering out MRI data

import os
import glob
import pandas as pd
import tqdm


missing_classes_csvs = {
    "amos": "bundles/amos22_baseline_dice_ce_nl/seed_12345/inference_results/missing_classes_raw.csv",
    "brats": "bundles/brats21_baseline_dice_ce_nl/seed_12345/inference_results/missing_classes_raw.csv",
    "kits": "bundles/kits23_baseline_dice_ce_8/seed_12345/inference_results/missing_classes_raw.csv",
}


def get_complete_cases(missing_classes_csvs):
    missing_classes_dataframes = {
        k: pd.read_csv(v) for k, v in missing_classes_csvs.items()
    }
    return {
        k: missing_classes_dataframes[k][missing_classes_dataframes[k]["mean"] == 0][
            "filename"
        ]
        for k in missing_classes_dataframes.keys()
    }


def get_amos_testing_case_filepaths(dataset_dir="../data/amos22"):
    test_images = sorted(
        [
            case
            for case in glob.glob(
                os.path.join(dataset_dir, "imagesVa", "amos_*.nii.gz")
            )
            if int(os.path.basename(case).split("_")[1].split(".")[0]) < 500
        ]
    )
    test_labels = [
        os.path.join(dataset_dir, "labelsVa", os.path.basename(case))
        for case in test_images
    ]
    return test_images, test_labels


def get_brats_testing_case_directories(
    dataset_dir="../data/brats/BraTS2021_TestingData",
):
    # In this case we can just get a list of the directories and transfer those
    directories = [os.path.join(dataset_dir, case) for case in os.listdir(dataset_dir)]
    return directories


def get_kits_testing_case_directories(data_dir="../data/kits23/dataset"):
    directories = [os.path.join(data_dir, case) for case in os.listdir(data_dir)]
    return directories


def copy_amos_complete_files(images, labels, complete_cases):
    os.makedirs("../data/amos22_complete/imagesVa/", exist_ok=True)
    os.makedirs("../data/amos22_complete/labelsVa/", exist_ok=True)

    for img, lbl in tqdm.tqdm(
        zip(images, labels), total=len(images), desc="Copying complete AMOS files"
    ):
        if os.path.basename(img) in complete_cases.to_list():
            os.system(f"cp {img} ../data/amos22_complete/imagesVa/")
            os.system(f"cp {lbl} ../data/amos22_complete/labelsVa/")


def copy_complete_brats_directories(directories, complete_cases):
    os.makedirs("../data/brats/BraTS2021_TestingData_complete/", exist_ok=True)

    for directory in tqdm.tqdm(
        directories, total=len(directories), desc="Copying complete BraTS directories"
    ):
        if os.path.basename(directory) in complete_cases.to_list():
            os.system(
                f"cp -r {directory} ../data/brats/BraTS2021_TestingData_complete/"
            )


def copy_complete_kits_directories(directories, complete_cases):
    os.makedirs("../data/kits23_complete/dataset/", exist_ok=True)

    for directory in tqdm.tqdm(
        directories, total=len(directories), desc="Copying complete KiTS directories"
    ):
        if os.path.basename(directory) in complete_cases.to_list():
            os.system(f"cp -r {directory} ../data/kits23_complete/dataset/")


if __name__ == "__main__":
    complete_cases = get_complete_cases(missing_classes_csvs)

    # AMOS  - DONE
    # amos_images, amos_labels = get_amos_testing_case_filepaths()
    # copy_amos_complete_files(amos_images, amos_labels, complete_cases['amos'])

    # BraTS:  - DONE
    # brats_directories = get_brats_testing_case_directories()
    # copy_complete_brats_directories(brats_directories, complete_cases['brats'])

    # KiTS:
    kits_directories = get_kits_testing_case_directories()
    copy_complete_kits_directories(kits_directories, complete_cases["kits"])
