import os
import shutil
import argparse
import hashlib
import tempfile


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a MONAI bundle based on loss and dataset."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Dataset name, e.g., brats_2021."
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        help="Loss type, e.g., baseline_ce or softl1ace_dice_ce.",
    )
    return parser


def hash_configs(config_paths):
    hasher = hashlib.sha256()  # Using SHA-256 here
    for path in config_paths:
        with open(path, "rb") as f:
            file_content = f.read()
            hasher.update(file_content)
    return hasher.hexdigest()[:8]  # Taking the first 8 characters for brevity


def create_bundle_directory(data, loss, config_hash):
    bundles_dir = "bundles"
    os.makedirs(bundles_dir, exist_ok=True)  # Ensure the bundles directory exists
    bundle_name = f"{data}_{loss}_{config_hash}"
    bundle_path = os.path.join(bundles_dir, bundle_name)
    if not os.path.exists(bundle_path):
        os.makedirs(bundle_path)
    return bundle_path


def copy_configs(data, loss, bundle_path, template_root):
    new_bundle_configs = os.path.join(bundle_path, "configs")
    os.makedirs(new_bundle_configs, exist_ok=True)

    # Common configurations
    common_configs = [
        "common.yaml",
        "train.yaml",
        "validation.yaml",
        "inference_eval.yaml",
        "inference_pred.yaml",
        "logging.conf",
        "metadata.json",
        "debug.yaml",
    ]
    config_paths = []
    for config in common_configs:
        src = os.path.join(template_root, config)
        dest = os.path.join(new_bundle_configs, config)
        shutil.copy(src, dest)
        config_paths.append(src)

    # Loss and Data specific configs with standardized naming
    loss_config_path = os.path.join(template_root, "loss", f"{loss}.yaml")
    data_config_path = os.path.join(template_root, "data", f"{data}.yaml")

    shutil.copy(loss_config_path, os.path.join(new_bundle_configs, "loss.yaml"))
    shutil.copy(data_config_path, os.path.join(new_bundle_configs, "data.yaml"))

    config_paths.extend([loss_config_path, data_config_path])
    return config_paths


def main():
    parser = get_parser()
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdirname:
        config_paths = copy_configs(
            args.data, args.loss, tmpdirname, "bundle_templates/configs"
        )
        config_hash = hash_configs(
            config_paths
        )  # Generate hash based on contents of configs
        bundle_path = create_bundle_directory(
            args.data, args.loss, config_hash
        )  # Use hash in the directory name
        copy_configs(
            args.data, args.loss, bundle_path, "bundle_templates/configs"
        )  # Copy configs to the real path

    print(f"Bundle created at {bundle_path}")


if __name__ == "__main__":
    main()
