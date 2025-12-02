import argparse
import os
from monai.bundle.scripts import run
from monai.utils.misc import set_determinism


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run a MONAI bundle with specified configurations."
    )
    parser.add_argument(
        "--bundle",
        type=str,
        required=True,
        help="Bundle directory name, e.g., brats21_softl1ace_dice_ce_1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train",
            "inference_pred",
            "inference_eval",
            "temp_scale_train",
            "temp_scale_eval",
            "inference_eval_additional",
            "temp_scale_eval_additional",
        ],
        help="Operation mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for deterministic training.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to test train and val."
    )
    return parser


def get_config_files(bundle_root, mode, debug):
    if mode == "train":
        # Config order for training mode
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "train.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(bundle_root, "configs", "train.yaml"),
            os.path.join(bundle_root, "configs", "loss.yaml"),
            os.path.join(bundle_root, "configs", "data.yaml"),
        ]
    elif mode == "inference_pred":
        # Config order for inference prediction mode
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(bundle_root, "configs", "inference_pred.yaml"),
            os.path.join(bundle_root, "configs", "data.yaml"),
        ]
    elif mode == "inference_eval":
        # Config order for inference evaluation mode
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(bundle_root, "configs", "inference_eval.yaml"),
            os.path.join(bundle_root, "configs", "data.yaml"),
        ]
    elif mode == "temp_scale_train":
        # Special config order for temperature scaling: common, validation, data, temp_scale
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(bundle_root, "configs", "data.yaml"),
            os.path.join(bundle_root, "configs", "temp_scale.yaml"),
        ]
    elif mode == "temp_scale_eval":
        # Config order for temperature scaled model evaluation
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(
                bundle_root, "configs", "inference_eval.yaml"
            ),  # Base evaluation config
            os.path.join(bundle_root, "configs", "data.yaml"),
            os.path.join(
                bundle_root, "configs", "temp_scale_eval.yaml"
            ),  # Override with temp scaled model
        ]
    elif mode == "inference_eval_additional":
        # Config order for inference evaluation with additional metrics
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(bundle_root, "configs", "inference_eval_additional.yaml"),
            os.path.join(bundle_root, "configs", "data.yaml"),
        ]
    elif mode == "temp_scale_eval_additional":
        # Config order for temperature scaled model evaluation with additional metrics
        config_files = [
            os.path.join(bundle_root, "configs", "common.yaml"),
            os.path.join(bundle_root, "configs", "validation.yaml"),
            os.path.join(bundle_root, "configs", "inference_eval_additional.yaml"),
            os.path.join(bundle_root, "configs", "data.yaml"),
            os.path.join(
                bundle_root, "configs", "temp_scale_eval_additional.yaml"
            ),  # Override with temp scaled model
        ]
    else:
        # Standard config order for other modes
        raise ValueError(f"Unsupported mode: {mode}")

    if debug:
        config_files.append(os.path.join(bundle_root, "configs", "debug.yaml"))

    return config_files


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set the determinism seed
    set_determinism(seed=args.seed)

    # Prepend the "bundles" directory to the bundle path
    bundle_root = os.path.join("bundles", args.bundle)

    config_files = get_config_files(bundle_root, args.mode, args.debug)

    run(
        bundle_root=bundle_root,
        meta_file=os.path.join(bundle_root, "configs", "metadata.json"),
        config_file=config_files,
        logging_file=os.path.join(bundle_root, "configs", "logging.conf"),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
