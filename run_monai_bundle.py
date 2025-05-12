import argparse
import os
from monai.bundle.scripts import run
from monai.utils import set_determinism


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
        choices=["train", "inference_pred", "inference_eval"],
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
    config_files = [
        os.path.join(bundle_root, "configs", "common.yaml"),
        os.path.join(
            bundle_root, "configs", "validation.yaml"
        ),  # validation is used for training and inference
        os.path.join(bundle_root, "configs", "train.yaml") if mode == "train" else None,
        (
            os.path.join(bundle_root, "configs", "inference_eval.yaml")
            if mode == "inference_eval"
            else None
        ),
        (
            os.path.join(bundle_root, "configs", "inference_pred.yaml")
            if mode == "inference_pred"
            else None
        ),
        os.path.join(bundle_root, "configs", "loss.yaml"),
        os.path.join(bundle_root, "configs", "data.yaml"),
    ]

    if debug:
        config_files.append(os.path.join(bundle_root, "configs", "debug.yaml"))

    return [f for f in config_files if f is not None]


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
