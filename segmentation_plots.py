import os
import argparse
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch
from abc import ABC
from skimage.morphology import remove_small_objects
from scipy.ndimage import center_of_mass

from monai.networks.nets import SegResNetDS
from monai.networks import one_hot
from monai.inferers import SlidingWindowInferer
from monai.data import Dataset, DataLoader, partition_dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Activationsd,
    AsDiscreted,
    ConcatItemsd,
    NormalizeIntensityd,
)
from monai.transforms import AsDiscrete
from monai.utils import ImageMetaKey as Key

from src.transforms import ConvertToKits23Classesd, ConvertToKits23ClassesSoftmaxd
from src.brats_transforms import ConvertToBratsClassesd, ConvertToBratsClassesSoftmaxd

import ipywidgets as widgets
from IPython.display import display, clear_output


class Model(ABC):
    def __init__(
        self,
        data_paths,
        transforms,
        post_transforms,
        in_channels=1,
        out_channels=4,
        ckpt_path=None,
        roi_size=(128, 128, 128),
        case_name_batch_transform=None,
    ):
        """
        Parent Model now expects data_paths and transforms from the child,
        so that dataset and dataloader are built safely.
        """
        self.data_paths = data_paths
        self.transforms = transforms
        self.post_transforms = post_transforms
        self.ckpt_path = ckpt_path
        self.roi_size = roi_size
        self.num_classes = out_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.case_name_batch_transform = case_name_batch_transform

        self.network = SegResNetDS(
            init_filters=32,
            blocks_down=[1, 2, 2, 4, 4],
            norm="INSTANCE_NVFUSER",
            in_channels=in_channels,
            out_channels=out_channels,
            dsdepth=4,
        )
        self.load_model()
        self.dataset = Dataset(data=self.data_paths, transform=self.transforms)
        # print(f"Number of cases found: {len(self.dataset)}")
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.inferer = SlidingWindowInferer(
            roi_size=self.roi_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="gaussian",
            sigma_scale=0.125,
        )

    def load_model(self):
        """
        Load the model checkpoint if provided, then place the network on the device in eval mode.
        """
        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.network.load_state_dict(checkpoint)
            self.network.to(self.device)
        else:
            raise ValueError("Checkpoint path is not provided.")

    # def run_inference(self, out_device="cpu"):  # TODO: this needs updating if is to be used
    #     """
    #     Use the existing DataLoader, retrieve a single batch, and run inference.
    #     """
    #     self.network.eval()
    #     batch_data = next(iter(self.data_loader))
    #     case_name = self.case_name_batch_transform(batch_data)
    #     with torch.no_grad():
    #         image = batch_data["image"].to(self.device)
    #         pred_logits = self.inferer(image, self.network) # shape: [B, C, H, D, W] eg: [1, 4, 472, 387, 316]
    #         pred_probs = torch.softmax(pred_logits, dim=1)

    #     pred_descrete = one_hot(pred_probs.argmax(dim=1).unsqueeze(0), num_classes=self.num_classes)
    #     label = one_hot(batch_data["label"], num_classes=self.num_classes)
    #     out = {
    #         Key.FILENAME_OR_OBJ: case_name,
    #         "image": image.to(out_device).squeeze(0),
    #         "label": label.to(out_device).squeeze(0),
    #         "pred": pred_descrete.to(out_device).squeeze(0),
    #         "pred_probs": pred_probs.to(out_device).squeeze(0)
    #     }
    #     return out

    def run_inference_case(self, idx, out_device="cpu"):
        import torch

        # Use dataset[idx] to get a single case
        case_data = self.dataset[idx]
        case_name = self.case_name_batch_transform(case_data)
        print(f"Running inference for case: {case_name}")
        # Ensure each key is a tensor and add batch dimension
        for key in case_data:
            if not isinstance(case_data[key], torch.Tensor):
                case_data[key] = torch.as_tensor(case_data[key])
            case_data[key] = torch.unsqueeze(case_data[key], 0)
        image = case_data["image"]
        with torch.no_grad():
            pred = self.inferer(image.to(self.device), self.network)
            pred = pred[0]  # returns a tuple so take the first element

        out = {
            Key.FILENAME_OR_OBJ: case_name,
            "image": image.squeeze(0),
            "label": case_data["label"].squeeze(0),
            "pred": pred.squeeze(0),
        }

        out = self.post_transforms(out)

        # # Add discrete and one_hot encoded version of 'pred'
        # pred_discrete = one_hot(out["pred"].unsqueeze(0).argmax(dim=1, keepdim=True), num_classes=self.num_classes).squeeze(0)  # argmax doesn't work for HEC
        # out["pred_discrete"] = pred_discrete.squeeze(0).to(out_device)  # shape: [C, H, D, W]

        pred_discrete = AsDiscrete(threshold=0.5)(out["pred"])
        out["pred_discrete"] = pred_discrete.to(out_device)  # shape: [C, H, D, W]

        # Send tensors to the specified device
        out["image"] = out["image"].to(out_device)  # shape: [C, H, D, W]
        out["label"] = out["label"].to(out_device)  # shape: [C, H, D, W]
        out["pred"] = out["pred"].to(out_device)  # shape: [C, H, D, W]

        return out


class KiTSModel(Model):
    def __init__(self, loss="soft", version="nl", hec=True):
        """
        Initialize a KiTSModel by specifying ....
        """
        in_channels = 1
        out_channels = 4
        self.num_classes = 4
        if hec:
            self.class_names = [
                "background",
                "tumour",
                "kidney_mass",
                "kindey_and_masses",
            ]
        else:
            self.class_names = ["background", "kidney", "tumor", "cyst"]

        test_seed = 42
        roi_size = (256, 256, 256)
        dataset_dir = "../data/kits23/dataset"
        case_name_batch_transform = (
            lambda b: b["image"].meta[Key.FILENAME_OR_OBJ].split("/")[-2]
        )

        ckpt_paths = {
            "baseline": {
                "1": "bundles/kits23_baseline_dice_ce_1/seed_12345/model_key_metric=0.7936.pt",
                "nl": "bundles/kits23_baseline_dice_ce_nl/seed_12345/model_key_metric=0.8083.pt",
            },
            "hard": {
                "1": "bundles/kits23_hardl1ace_dice_ce_1/seed_12345/model_key_metric=0.7822.pt",
                "nl": "bundles/kits23_hardl1ace_dice_ce_nl/seed_12345/model_key_metric=0.8053.pt",
            },
            "soft": {
                "1": "bundles/kits23_softl1ace_dice_ce_1/seed_12345/model_key_metric=0.7709.pt",
                "nl": "bundles/kits23_softl1ace_dice_ce_nl/seed_12345/model_key_metric=0.7774.pt",
            },
        }

        try:
            ckpt_path = ckpt_paths[loss][version]
        except KeyError:
            raise ValueError(f"Unsupported loss '{loss}' or version '{version}'")

        data_paths = self.make_data_paths(dataset_dir, test_seed)

        transforms = self.make_transforms()

        if hec:
            post_transforms = self.make_post_transforms_hec()
        else:
            post_transforms = self.make_post_transforms()

        super().__init__(
            data_paths,
            transforms,
            post_transforms,
            in_channels,
            out_channels,
            ckpt_path,
            roi_size,
            case_name_batch_transform,
        )

    def make_data_paths(self, dataset_dir, test_seed):
        """
        Return the test partition from a list of data dictionaries. Each item contains image and label paths.
        """
        all_cases = sorted(glob.glob(os.path.join(dataset_dir, "case_*")))
        all_images = [os.path.join(case, "imaging.nii.gz") for case in all_cases]
        all_labels = [os.path.join(case, "segmentation.nii.gz") for case in all_cases]
        all_dicts = [
            {"image": img, "label": lbl} for img, lbl in zip(all_images, all_labels)
        ]
        partitions_train_test = partition_dataset(
            all_dicts, (0.8, 0.2), shuffle=True, seed=test_seed
        )
        return partitions_train_test[1]

    def make_transforms(self):
        """
        Return a Compose of data augmentation/preprocessing steps for KiTS dataset images and labels.
        """
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-54, a_max=242, b_min=0, b_max=1.0, clip=True
                ),
            ]
        )

    def make_post_transforms_hec(self):
        """
        Return a Compose of post-processing steps for KiTS dataset segmentation outputs.
        """
        return Compose(
            [
                Activationsd(keys=["pred"], softmax=True),
                ConvertToKits23Classesd(keys=["label"]),
                ConvertToKits23ClassesSoftmaxd(keys=["pred"]),
            ]
        )

    def make_post_transforms(self):
        return Compose(
            [
                Activationsd(keys=["pred"], softmax=True),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
            ]
        )


class BraTSModel(Model):
    def __init__(self, loss="soft", version="nl", hec=True):
        # in_channels from YAML: 4, out_channels: 4, use roi_size from val_windows_size.
        in_channels = 4
        out_channels = 4
        self.num_classes = out_channels
        if hec:
            self.class_names = [
                "Background (BG)",
                "Enhancing Tumour (ET)",
                "Tumour Core (TC)",
                "Whole Tumour (WT)",
            ]  # HEC classes
        else:
            self.class_names = [
                "Background (BG)",
                "Necrotic Tumour Core (NCR)",
                "Peritumoral Edema (ED)",
                "Enhancing Tumour (ET)",
            ]  # original classes

        roi_size = (240, 240, 160)
        # Define dataset_dir based on YAML config.
        dataset_dir = "../data/brats/BraTS2021_TestingData"
        # For BraTS, we use a simple lambda for case name extraction.
        case_name_batch_transform = lambda b: "_".join(
            b["image"].meta[Key.FILENAME_OR_OBJ].split("/")[-1].split("_")[:2]
        )

        try:
            ckpt_paths = {
                "baseline": {
                    "1": "bundles/brats21_baseline_dice_ce_1/seed_12345/model_key_metric=0.9392.pt",
                    "nl": "bundles/brats21_baseline_dice_ce_nl/seed_12345/model_key_metric=0.9389.pt",
                },
                "hard": {
                    "1": "bundles/brats21_hardl1ace_dice_ce_1/seed_12345/model_key_metric=0.9386.pt",
                    "nl": "bundles/brats21_hardl1ace_dice_ce_nl/seed_12345/model_key_metric=0.9396.pt",
                },
                "soft": {
                    "1": "bundles/brats21_softl1ace_dice_ce_1/seed_12345/model_key_metric=0.9205.pt",
                    "nl": "bundles/brats21_softl1ace_dice_ce_nl/seed_12345/model_key_metric=0.9230.pt",
                },
            }
            ckpt_path = ckpt_paths[loss][version]
        except KeyError:
            raise ValueError(f"Unsupported loss '{loss}' or version '{version}'")

        data_paths = self.make_data_paths(dataset_dir)
        transforms = self.make_transforms()
        if hec:
            post_transforms = self.make_post_transforms_hec()
        else:
            post_transforms = self.make_post_transforms()
        super().__init__(
            data_paths,
            transforms,
            post_transforms,
            in_channels,
            out_channels,
            ckpt_path,
            roi_size,
            case_name_batch_transform,
        )

    def make_data_paths(self, dataset_dir):
        # Glob for t1 images as base; derive other modalities by string replacements.
        imgs = sorted(
            glob.glob(os.path.join(dataset_dir, "BraTS2021_*", "BraTS2021_*_t1.nii.gz"))
        )
        data_dicts = []
        for i in imgs:
            data_dicts.append(
                {
                    "t1": i,
                    "t1ce": i.replace("t1", "t1ce"),
                    "t2": i.replace("t1", "t2"),
                    "flair": i.replace("t1", "flair"),
                    "label": i.replace("t1", "seg_c"),
                }
            )
        return data_dicts

    def make_transforms(self):
        # Updated val_transforms pipeline from YAML
        return Compose(
            [
                LoadImaged(
                    keys=["t1", "t1ce", "t2", "flair", "label"], image_only=True
                ),
                EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
                EnsureTyped(keys=["t1", "t1ce", "t2", "flair", "label"]),
                ConcatItemsd(keys=["t1", "t1ce", "t2", "flair"], name="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ]
        )

    def make_post_transforms_hec(self):
        return Compose(
            [
                Activationsd(keys=["pred"], softmax=True),
                ConvertToBratsClassesd(keys=["label"]),
                ConvertToBratsClassesSoftmaxd(keys=["pred"]),
            ]
        )

    def make_post_transforms(self):
        return Compose(
            [
                Activationsd(keys=["pred"], softmax=True),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
            ]
        )


class AMOSModel(Model):
    def __init__(self, loss="soft", version="nl"):
        # in_channels: 1, out_channels: 16, roi_size from YAML val_windows_size.
        in_channels = 1
        out_channels = 16
        self.num_classes = out_channels
        self.class_names = [
            "background",
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
            "prostate or uterus",
        ]
        roi_size = (192, 192, 64)
        dataset_dir = "../data/amos22"
        case_name_batch_transform = lambda b: "_".join(
            os.path.basename(b["image"].meta[Key.FILENAME_OR_OBJ])
            .split(".")[0]
            .split("_")[:2]
        )

        try:
            ckpt_paths = {
                "baseline": {
                    "1": "bundles/amos22_baseline_dice_ce_1/seed_12345/model_key_metric=0.8703.pt",
                    "nl": "bundles/amos22_baseline_dice_ce_nl/seed_12345/model_key_metric=0.8681.pt",
                },
                "hard": {
                    "1": "bundles/amos22_hardl1ace_dice_ce_1/seed_12345/model_key_metric=0.8673.pt",
                    "nl": "bundles/amos22_hardl1ace_dice_ce_nl/seed_12345/model_key_metric=0.8663.pt",
                },
                "soft": {
                    "1": "bundles/amos22_softl1ace_dice_ce_1/seed_12345/model_key_metric=0.8635.pt",
                    "nl": "bundles/amos22_softl1ace_dice_ce_nl/seed_12345/model_key_metric=0.8627.pt",
                },
            }
            ckpt_path = ckpt_paths[loss][version]
        except KeyError:
            raise ValueError(f"Unsupported loss '{loss}' or version '{version}'")

        data_paths = self.make_data_paths(dataset_dir)
        transforms = self.make_transforms()
        post_transforms = self.make_post_transforms()
        super().__init__(
            data_paths,
            transforms,
            post_transforms,
            in_channels,
            out_channels,
            ckpt_path,
            roi_size,
            case_name_batch_transform,
        )

    def make_data_paths(self, dataset_dir):
        # Filter training cases in imagesTr that are numbered <500.
        all_images = sorted(
            [
                os.path.join(dataset_dir, "imagesVa", f)
                for f in os.listdir(os.path.join(dataset_dir, "imagesVa"))
                if f.startswith("amos_") and int(f.split("_")[1].split(".")[0]) < 500
            ]
        )
        all_labels = [
            os.path.join(dataset_dir, "labelsVa", os.path.basename(img))
            for img in all_images
        ]
        return [
            {"image": img, "label": lbl} for img, lbl in zip(all_images, all_labels)
        ]

    def make_transforms(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 5.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-100,
                    a_max=200,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

    def make_post_transforms(self):
        return Compose(
            [
                Activationsd(keys=["pred"], softmax=True),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
            ]
        )


class ACDCSModel(Model):
    def __init__(self, loss="soft", version="1"):
        # in_channels: 1, out_channels: 4, roi_size from YAML val_windows_size for ACDC.
        in_channels = 1
        out_channels = 4
        self.num_classes = out_channels
        self.class_names = [
            "background",
            "right_ventricle",
            "myocardium",
            "left_ventricle",
        ]

        roi_size = (256, 256, 16)
        dataset_dir = "../data/acdc/ACDC/database"
        case_name_batch_transform = lambda b: "_".join(
            os.path.basename(b["image"].meta[Key.FILENAME_OR_OBJ])
            .split(".")[0]
            .split("_")[:2]
        )

        try:
            ckpt_paths = {
                "baseline": {
                    "1": "bundles/acdc17_baseline_dice_ce_1/seed_12345/model_key_metric=0.8940.pt",
                },
                "hard": {
                    "1": "bundles/acdc17_hardl1ace_dice_ce_1/seed_12345/model_key_metric=0.8901.pt",
                },
                "soft": {
                    "1": "bundles/acdc17_softl1ace_dice_ce_1/seed_12345/model_key_metric=0.8809.pt",
                },
            }
            ckpt_path = ckpt_paths[loss][version]
        except KeyError:
            raise ValueError(f"Unsupported loss '{loss}' or version '{version}'")

        data_paths = self.make_data_paths(dataset_dir)
        transforms = self.make_transforms()
        post_transforms = self.make_post_transforms()
        super().__init__(
            data_paths,
            transforms,
            post_transforms,
            in_channels,
            out_channels,
            ckpt_path,
            roi_size,
            case_name_batch_transform,
        )

    def make_data_paths(self, dataset_dir):
        # Grab training cases recursively from the 'testing' directory.
        base_dir = os.path.join(dataset_dir, "testing")
        data_dicts = []
        for patient in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, patient)
            if os.path.isdir(patient_path):
                for frame in os.listdir(patient_path):
                    if (
                        frame.endswith(".nii.gz")
                        and "frame" in frame
                        and "_gt" not in frame
                    ):
                        image_path = os.path.join(patient_path, frame)
                        label_path = os.path.join(
                            patient_path, frame.replace(".nii.gz", "_gt.nii.gz")
                        )
                        data_dicts.append({"image": image_path, "label": label_path})
        return data_dicts

    def make_transforms(self):
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 5.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ]
        )

    def make_post_transforms(self):
        return Compose(
            [
                Activationsd(keys=["pred"], softmax=True),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
            ]
        )


def extract_component(
    arrays,
    component=3,
    keys_to_extract=None,
    exclude_keys=["image", Key.FILENAME_OR_OBJ],
):
    """Extracts a specified component from multiple arrays in a dictionary, with the
       option to provide keys to process and a list of keys to exclude.

    Args:
        arrays (dict): A dictionary containing arrays.
        component (int, optional): The component index to extract. Defaults to 3.
        keys_to_extract (list, optional): A list of keys specifying which arrays to
                                         process. If None, processes all keys.
                                         Defaults to None.
        exclude_keys (list, optional):  A list of keys to exclude from processing.
                                        Defaults to [Key.FILENAME_OR_OBJ].
    Returns:
        dict: The modified dictionary with components extracted.
    """

    if keys_to_extract:
        keys = keys_to_extract  # Use the specified list of keys
    else:
        keys = arrays.keys()  # Process all keys if none specified

    for key in keys:
        if key not in exclude_keys:
            arrays[key] = arrays[key][component]

    return arrays


def find_roi_slice(arrays, key="gt_seg", padding=(10, 10, 10, 10), square=True):
    value = arrays[key]
    slice_areas = np.sum(value, axis=(0, 1))
    largest_slice_idx = np.argmax(slice_areas)
    value2d = value[:, :, largest_slice_idx]

    rows = np.any(value2d, axis=1)
    cols = np.any(value2d, axis=0)

    # If no foreground is found, return ROI covering the full image
    if not (np.any(rows) and np.any(cols)):
        return slice(0, value2d.shape[0]), slice(0, value2d.shape[1]), largest_slice_idx

    r_indices = np.where(rows)[0]
    c_indices = np.where(cols)[0]
    rmin, rmax = r_indices[[0, -1]]
    cmin, cmax = c_indices[[0, -1]]

    top_pad, bottom_pad, left_pad, right_pad = padding
    rmin = max(rmin - top_pad, 0)
    rmax = min(rmax + bottom_pad, value2d.shape[0])
    cmin = max(cmin - left_pad, 0)
    cmax = min(cmax + right_pad, value2d.shape[1])

    if square:
        max_dim = max(rmax - rmin, cmax - cmin)
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2
        half_side = max_dim // 2
        rmin = max(0, center_r - half_side)
        rmax = min(value2d.shape[0], center_r + half_side)
        cmin = max(0, center_c - half_side)
        cmax = min(value2d.shape[1], center_c + half_side)

    return slice(rmin, rmax), slice(cmin, cmax), largest_slice_idx


def crop_to_roi(arrays, roi):
    for key, value in arrays.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            arrays[key] = value[roi]
    return arrays


def calculate_and_plot_contours(
    prob_cropped,
    seg_cropped,
    gt_seg_cropped,
    min_size,
    cmap_prob,
    gamma_correction,
    legend_patches,
    TP_color,
    FP_color,
    FN_color,
    ax,
):

    im = ax.imshow(prob_cropped, cmap=cmap_prob)
    # im = ax.imshow(prob_cropped, cmap=cmap_prob, norm=gamma_correction)

    # Calculate TP, FP, FN, and remove small objects
    TP = np.logical_and(seg_cropped, gt_seg_cropped)
    FP = np.logical_and(seg_cropped, 1 - gt_seg_cropped)
    FN = np.logical_and(1 - seg_cropped, gt_seg_cropped)
    TP = remove_small_objects(TP, min_size=min_size)
    FP = remove_small_objects(FP, min_size=min_size)
    FN = remove_small_objects(FN, min_size=min_size)

    # Plot contours
    ax.contour(TP, colors=TP_color, linewidths=1, levels=[0.5])
    ax.contour(FP, colors=FP_color, linewidths=1, levels=[0.5])
    ax.contour(FN, colors=FN_color, linewidths=1, levels=[0.5])

    # ax.legend(handles=legend_patches, loc="lower left")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return im


def make_plot(
    baseline_out,
    hard_out,
    soft_out,
    out_dir="./seg_plots/seg_plots",
    class_name="",
    min_size=50,
):
    """
    Creates a 4-panel plot:
      - Left: image with ground truth overlay for specified class.
      - Next: baseline, hard, soft predictions with contours.
    Saves the plot as pdf: kits_{case_name}_slice{slice_idx}_class{class_idx}.pdf
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    label = baseline_out["label"]
    image = baseline_out["image"]

    cmap_seg = ListedColormap(
        [
            (0.0, 0.5, 0.0),
        ]
    )
    cmap_prob = plt.get_cmap("PRGn")
    TP_color = "blue"
    FP_color = "yellow"
    FN_color = "red"

    fig, axs = plt.subplots(
        1,
        5,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]},  # including colorbar
    )
    ax0, ax1, ax2, ax3, cax = axs

    gamma_correction = PowerNorm(gamma=0.1)
    legend_patches = [
        Patch(color=TP_color, label="TP"),
        Patch(color=FP_color, label="FP"),
        Patch(color=FN_color, label="FN"),
    ]

    # First panel: image with GT overlay
    masked_label = np.ma.masked_where(label == 0, label)
    ax0.imshow(image, cmap="gray", interpolation="nearest")
    ax0.imshow(masked_label, alpha=0.7, cmap=cmap_seg, interpolation="nearest")
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])

    # Baseline Probability Map
    _ = calculate_and_plot_contours(
        baseline_out["pred"],
        baseline_out["pred_discrete"],
        baseline_out["label"],
        min_size,
        cmap_prob,
        gamma_correction,
        legend_patches,
        TP_color,
        FP_color,
        FN_color,
        ax1,
    )

    # hard-L1ACE Probability Map
    _ = calculate_and_plot_contours(
        hard_out["pred"],
        hard_out["pred_discrete"],
        hard_out["label"],
        min_size,
        cmap_prob,
        gamma_correction,
        legend_patches,
        TP_color,
        FP_color,
        FN_color,
        ax2,
    )

    # soft-L1ACE Probability Map
    im3 = calculate_and_plot_contours(
        soft_out["pred"],
        soft_out["pred_discrete"],
        soft_out["label"],
        min_size,
        cmap_prob,
        gamma_correction,
        legend_patches,
        TP_color,
        FP_color,
        FN_color,
        ax3,
    )

    # ... (Code for adjusting colorbar - Same as before) ...
    # Adjust the colorbar position
    pos = ax3.get_position()
    cax.set_position([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(im3, cax=cax, orientation="vertical")
    # Ensure maximum data value is included
    if np.max(soft_out["pred"]) < 1.0:
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.savefig(
        os.path.join(out_dir, f"{baseline_out[Key.FILENAME_OR_OBJ]}_{class_name}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def make_acdc_plot(acdc_case, baseline_model, hard_model, soft_model, min_size=50):
    border_size_2d = (10, 10, 10, 10)
    baseline_out = baseline_model.run_inference_case(acdc_case)
    hard_out = hard_model.run_inference_case(acdc_case)
    soft_out = soft_model.run_inference_case(acdc_case)
    for component_idx in range(1, baseline_model.num_classes):
        baseline_out_comp = extract_component(
            baseline_out.copy(), component=component_idx
        )
        baseline_out_comp["image"] = baseline_out_comp["image"].squeeze(0)
        hard_out_comp = extract_component(hard_out.copy(), component=component_idx)
        hard_out_comp["image"] = hard_out_comp["image"].squeeze(0)
        soft_out_comp = extract_component(soft_out.copy(), component=component_idx)
        soft_out_comp["image"] = soft_out_comp["image"].squeeze(0)

        roi = find_roi_slice(
            baseline_out_comp, key="label", padding=border_size_2d, square=True
        )
        baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
        hard_out_comp = crop_to_roi(hard_out_comp, roi)
        soft_out_comp = crop_to_roi(soft_out_comp, roi)

        class_name = baseline_model.class_names[component_idx]
        make_plot(
            baseline_out_comp,
            hard_out_comp,
            soft_out_comp,
            out_dir="./seg_plots/seg_plots_acdc_all",
            class_name=class_name,
            min_size=min_size,
        )


def make_acdc_plot_select(
    case_class_dict, baseline_model, hard_model, soft_model, min_size=50
):
    """
    For each class_idx and its list of case indices in the dictionary,
    run inference, extract the specified class component, crop to ROI,
    and plot for ACDC.
    """
    border_size_2d = (10, 10, 10, 10)
    for class_idx, case_list in case_class_dict.items():
        for case_idx in case_list:
            baseline_out = baseline_model.run_inference_case(case_idx)
            hard_out = hard_model.run_inference_case(case_idx)
            soft_out = soft_model.run_inference_case(case_idx)

            baseline_out_comp = extract_component(
                baseline_out.copy(), component=class_idx
            )
            baseline_out_comp["image"] = baseline_out_comp["image"].squeeze(0)
            hard_out_comp = extract_component(hard_out.copy(), component=class_idx)
            hard_out_comp["image"] = hard_out_comp["image"].squeeze(0)
            soft_out_comp = extract_component(soft_out.copy(), component=class_idx)
            soft_out_comp["image"] = soft_out_comp["image"].squeeze(0)

            roi = find_roi_slice(
                baseline_out_comp, key="label", padding=border_size_2d, square=True
            )
            baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
            hard_out_comp = crop_to_roi(hard_out_comp, roi)
            soft_out_comp = crop_to_roi(soft_out_comp, roi)

            class_name = baseline_model.class_names[class_idx]
            make_plot(
                baseline_out_comp,
                hard_out_comp,
                soft_out_comp,
                out_dir="./seg_plots/seg_plots_acdc_select",
                class_name=class_name,
                min_size=min_size,
            )


def make_amos_plot(amos_case, baseline_model, hard_model, soft_model, min_size=50):
    border_size_2d = (10, 10, 10, 10)
    baseline_out = baseline_model.run_inference_case(amos_case)
    hard_out = hard_model.run_inference_case(amos_case)
    soft_out = soft_model.run_inference_case(amos_case)
    for component_idx in range(1, baseline_model.num_classes):
        baseline_out_comp = extract_component(
            baseline_out.copy(), component=component_idx
        )
        baseline_out_comp["image"] = baseline_out_comp["image"].squeeze(0)
        hard_out_comp = extract_component(hard_out.copy(), component=component_idx)
        hard_out_comp["image"] = hard_out_comp["image"].squeeze(0)
        soft_out_comp = extract_component(soft_out.copy(), component=component_idx)
        soft_out_comp["image"] = soft_out_comp["image"].squeeze(0)

        roi = find_roi_slice(
            baseline_out_comp, key="label", padding=border_size_2d, square=True
        )
        baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
        hard_out_comp = crop_to_roi(hard_out_comp, roi)
        soft_out_comp = crop_to_roi(soft_out_comp, roi)

        class_name = baseline_model.class_names[component_idx]
        make_plot(
            baseline_out_comp,
            hard_out_comp,
            soft_out_comp,
            out_dir="./seg_plots/seg_plots_amos_all",
            class_name=class_name,
            min_size=min_size,
        )


def make_amos_plot_select(
    case_class_dict, baseline_model, hard_model, soft_model, min_size=50
):
    border_size_2d = (10, 10, 10, 10)
    for class_idx, case_list in case_class_dict.items():
        for case_idx in case_list:
            baseline_out = baseline_model.run_inference_case(case_idx)
            hard_out = hard_model.run_inference_case(case_idx)
            soft_out = soft_model.run_inference_case(case_idx)

            # Extract only the specified class component for each case
            baseline_out_comp = extract_component(
                baseline_out.copy(), component=class_idx
            )
            baseline_out_comp["image"] = baseline_out_comp["image"].squeeze(0)
            hard_out_comp = extract_component(hard_out.copy(), component=class_idx)
            hard_out_comp["image"] = hard_out_comp["image"].squeeze(0)
            soft_out_comp = extract_component(soft_out.copy(), component=class_idx)
            soft_out_comp["image"] = soft_out_comp["image"].squeeze(0)

            roi = find_roi_slice(
                baseline_out_comp, key="label", padding=border_size_2d, square=True
            )
            baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
            hard_out_comp = crop_to_roi(hard_out_comp, roi)
            soft_out_comp = crop_to_roi(soft_out_comp, roi)

            class_name = baseline_model.class_names[class_idx]
            make_plot(
                baseline_out_comp,
                hard_out_comp,
                soft_out_comp,
                out_dir="./seg_plots/seg_plots_amos_select",
                class_name=class_name,
                min_size=min_size,
            )


def make_brats_plot(
    brats_case, baseline_model, hard_model, soft_model, hec=True, min_size=50
):
    border_size_2d = (10, 10, 10, 10)
    baseline_out = baseline_model.run_inference_case(brats_case)
    hard_out = hard_model.run_inference_case(brats_case)
    soft_out = soft_model.run_inference_case(brats_case)
    for component_idx in range(1, baseline_model.num_classes):
        baseline_out_comp = extract_component(
            baseline_out.copy(), component=component_idx
        )
        baseline_out_comp["image"] = baseline_out_comp["image"][3]
        hard_out_comp = extract_component(hard_out.copy(), component=component_idx)
        hard_out_comp["image"] = hard_out_comp["image"][3]
        soft_out_comp = extract_component(soft_out.copy(), component=component_idx)
        soft_out_comp["image"] = soft_out_comp["image"][3]

        roi = find_roi_slice(
            baseline_out_comp, key="label", padding=border_size_2d, square=True
        )
        baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
        hard_out_comp = crop_to_roi(hard_out_comp, roi)
        soft_out_comp = crop_to_roi(soft_out_comp, roi)

        class_name = baseline_model.class_names[component_idx]
        if hec:
            out_dir = "./seg_plots/seg_plots_brats_hec"
        else:
            out_dir = "./seg_plots/seg_plots_brats_no_hec"
        make_plot(
            baseline_out_comp,
            hard_out_comp,
            soft_out_comp,
            out_dir=out_dir,
            class_name=class_name,
            min_size=min_size,
        )


def make_brats_plot_select(
    case_class_dict, baseline_model, hard_model, soft_model, hec=True, min_size=50
):
    """
    For each class_idx (key) and its list of case indices (values) in the dictionary, run inference,
    extract the specified class component, crop to ROI, and plot for BraTS.
    """
    border_size_2d = (10, 10, 10, 10)
    for class_idx, case_list in case_class_dict.items():
        for case_idx in case_list:
            baseline_out = baseline_model.run_inference_case(case_idx)
            hard_out = hard_model.run_inference_case(case_idx)
            soft_out = soft_model.run_inference_case(case_idx)

            baseline_out_comp = extract_component(
                baseline_out.copy(), component=class_idx
            )
            baseline_out_comp["image"] = baseline_out_comp["image"][3]
            hard_out_comp = extract_component(hard_out.copy(), component=class_idx)
            hard_out_comp["image"] = hard_out_comp["image"][3]
            soft_out_comp = extract_component(soft_out.copy(), component=class_idx)
            soft_out_comp["image"] = soft_out_comp["image"][3]

            roi = find_roi_slice(
                baseline_out_comp, key="label", padding=border_size_2d, square=True
            )
            baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
            hard_out_comp = crop_to_roi(hard_out_comp, roi)
            soft_out_comp = crop_to_roi(soft_out_comp, roi)

            class_name = baseline_model.class_names[class_idx]
            out_dir = (
                "./seg_plots/seg_plots_brats_hec_select"
                if hec
                else "./seg_plots_brats_no_hec_select"
            )

            make_plot(
                baseline_out_comp,
                hard_out_comp,
                soft_out_comp,
                out_dir=out_dir,
                class_name=class_name,
                min_size=min_size,
            )


def make_kits_plot(
    kits_case, baseline_model, hard_model, soft_model, hec=True, min_size=20
):
    border_size_2d = (10, 10, 10, 10)
    baseline_out = baseline_model.run_inference_case(kits_case)
    hard_out = hard_model.run_inference_case(kits_case)
    soft_out = soft_model.run_inference_case(kits_case)
    for component_idx in range(1, baseline_model.num_classes):
        baseline_out_comp = extract_component(
            baseline_out.copy(), component=component_idx
        )
        baseline_out_comp["image"] = baseline_out_comp["image"].squeeze(0)
        hard_out_comp = extract_component(hard_out.copy(), component=component_idx)
        hard_out_comp["image"] = hard_out_comp["image"].squeeze(0)
        soft_out_comp = extract_component(soft_out.copy(), component=component_idx)
        soft_out_comp["image"] = soft_out_comp["image"].squeeze(0)

        roi = find_roi_slice(
            baseline_out_comp, key="label", padding=border_size_2d, square=True
        )
        baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
        hard_out_comp = crop_to_roi(hard_out_comp, roi)
        soft_out_comp = crop_to_roi(soft_out_comp, roi)
        class_name = baseline_model.class_names[component_idx]
        if hec:
            out_dir = "./seg_plots/seg_plots_kits_hec"
        else:
            out_dir = "./seg_plots/seg_plots_kits_no_hec"
        make_plot(
            baseline_out_comp,
            hard_out_comp,
            soft_out_comp,
            out_dir=out_dir,
            class_name=class_name,
            min_size=min_size,
        )


def make_kits_plot_select(
    case_class_dict, baseline_model, hard_model, soft_model, hec=True, min_size=50
):
    """
    For each class_idx and its list of case indices in the dictionary, run inference,
    extract the specified class component, crop to ROI, and plot for KiTS.
    Example input: {1: [95, 19, 5], 2: [5, 31, 59], 3: [3, 78, 84]}
    """
    border_size_2d = (10, 10, 10, 10)
    for class_idx, case_list in case_class_dict.items():
        for case_idx in case_list:
            baseline_out = baseline_model.run_inference_case(case_idx)
            hard_out = hard_model.run_inference_case(case_idx)
            soft_out = soft_model.run_inference_case(case_idx)

            baseline_out_comp = extract_component(
                baseline_out.copy(), component=class_idx
            )
            baseline_out_comp["image"] = baseline_out_comp["image"].squeeze(0)
            hard_out_comp = extract_component(hard_out.copy(), component=class_idx)
            hard_out_comp["image"] = hard_out_comp["image"].squeeze(0)
            soft_out_comp = extract_component(soft_out.copy(), component=class_idx)
            soft_out_comp["image"] = soft_out_comp["image"].squeeze(0)

            roi = find_roi_slice(
                baseline_out_comp, key="label", padding=border_size_2d, square=True
            )
            baseline_out_comp = crop_to_roi(baseline_out_comp, roi)
            hard_out_comp = crop_to_roi(hard_out_comp, roi)
            soft_out_comp = crop_to_roi(soft_out_comp, roi)

            class_name = baseline_model.class_names[class_idx]
            if hec:
                out_dir = "./seg_plots/seg_plots_kits_hec_select"
            else:
                out_dir = "./seg_plots/seg_plots_kits_no_hec_select"
            make_plot(
                baseline_out_comp,
                hard_out_comp,
                soft_out_comp,
                out_dir=out_dir,
                class_name=class_name,
                min_size=min_size,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run segmentation visualizations for different datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["kits", "brats", "amos", "acdc"],
        help="Specify which dataset to visualize.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["all", "select"],
        help="Specify whether to run for all cases or a selection.",
    )
    parser.add_argument(
        "--hec",
        dest="hec",
        action="store_true",
        help="Use Hierarchical Class Evaluation (HEC).",
    )
    parser.add_argument(
        "--no-hec",
        dest="hec",
        action="store_false",
        help="Do not use Hierarchical Class Evaluation (HEC).",
    )
    parser.set_defaults(hec=True)
    args = parser.parse_args()

    if args.dataset == "acdc":
        acdc_baseline = ACDCSModel(loss="baseline", version="1")
        acdc_hard = ACDCSModel(loss="hard", version="1")
        acdc_soft = ACDCSModel(loss="soft", version="1")
        if args.mode == "all":
            for case in range(len(acdc_baseline.data_paths)):  # All cases
                make_acdc_plot(case, acdc_baseline, acdc_hard, acdc_soft, min_size=20)
        elif args.mode == "select":
            # idxs = [21]  # best case visually
            idxs = {1: [95, 19, 5], 2: [5, 31, 59], 3: [3, 78, 84]}
            make_acdc_plot_select(
                idxs, acdc_baseline, acdc_hard, acdc_soft, min_size=20
            )

    elif args.dataset == "amos":
        amos_baseline = AMOSModel(loss="baseline", version="nl")
        amos_hard = AMOSModel(loss="hard", version="nl")
        amos_soft = AMOSModel(loss="soft", version="nl")
        if args.mode == "all":
            for case in range(len(amos_baseline.dataset)):  # all cases
                print("case: ", case)
                make_amos_plot(case, amos_baseline, amos_hard, amos_soft, min_size=5)
        elif args.mode == "select":
            idxs = {
                1: [74, 56, 49],
                2: [49, 18, 6],
                3: [27, 16, 92],
                4: [19, 67, 86],
                5: [65, 97, 71],
                6: [25, 85, 88],
                7: [87, 94, 85],
                8: [31, 37, 55],
                9: [30, 56, 54],
                10: [94, 86, 1],
                11: [73, 50, 98],
                12: [26, 35, 66],
                13: [63, 87, 33],
                14: [44, 6, 37],
                15: [18, 20, 44],
            }
            make_amos_plot_select(idxs, amos_baseline, amos_hard, amos_soft, min_size=5)

    elif args.dataset == "brats":
        brats_baseline = BraTSModel(loss="baseline", version="nl", hec=args.hec)
        brats_hard = BraTSModel(loss="hard", version="nl", hec=args.hec)
        brats_soft = BraTSModel(loss="soft", version="nl", hec=args.hec)
        if args.mode == "all":
            for case in range(len(brats_baseline.dataset)):  # all cases
                make_brats_plot(
                    case, brats_baseline, brats_hard, brats_soft, args.hec, min_size=5
                )
        elif args.mode == "select":
            if args.hec:
                idxs = {1: [176, 172, 179], 2: [176, 179, 113], 3: [86, 113, 160]}
            else:
                idxs = {1: [172, 195, 176], 2: [113, 86, 109], 3: [176, 172, 179]}
            make_brats_plot_select(
                idxs, brats_baseline, brats_hard, brats_soft, args.hec, min_size=5
            )

    elif args.dataset == "kits":
        kits_baseline = KiTSModel(loss="baseline", version="nl", hec=args.hec)
        kits_hard = KiTSModel(loss="hard", version="nl", hec=args.hec)
        kits_soft = KiTSModel(loss="soft", version="nl", hec=args.hec)
        if args.mode == "all":
            for case in range(len(kits_soft.data_paths)):
                make_kits_plot(
                    case, kits_baseline, kits_hard, kits_soft, hec=args.hec, min_size=20
                )
        elif args.mode == "select":
            if args.hec:
                idxs = {1: [90, 31, 95], 2: [31, 90, 19], 3: [19, 47, 31]}
            else:
                idxs = {1: [23, 47, 85], 2: [90, 31, 95], 3: [93, 78, 90]}
            make_kits_plot_select(
                idxs, kits_baseline, kits_hard, kits_soft, hec=args.hec, min_size=20
            )
