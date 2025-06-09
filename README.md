<div align="center">

# Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation

</div>

[Pre-print arXiv paper](https://arxiv.org/abs/2506.03942v1)

---

## Abstract

Deep neural networks for medical image segmentation are often overconfident, compromising both reliability and clinical utility.
In this work, we propose differentiable formulations of marginal L1 Average Calibration Error (mL1-ACE) as an auxiliary loss that can be computed on a per-image basis.
We compare both hard- and soft-binning approaches to directly improve pixel-wise calibration.
Our experiments on four datasets (ACDC, AMOS, KiTS, BraTS) demonstrate that incorporating mL1-ACE significantly reduces calibration errors, particularly Average Calibration Error (ACE) and Maximum Calibration Error (MCE), while largely maintaining high Dice Similarity Coefficients (DSCs).
We find that the soft-binned variant yields the greatest improvements in calibration, but often compromises segmentation performance, with hard-binned mL1-ACE maintaining segmentation performance, albeit with weaker calibration improvement.
To gain further insight into calibration performance and its variability across an imaging dataset, we introduce dataset reliability histograms, an aggregation of per-image reliability diagrams.
The resulting analysis highlights improved alignment between predicted confidences and true accuracies.
Overall, our approach not only enhances the trustworthiness of segmentation predictions but also shows potential for safer integration of deep learning methods into clinical workflows.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Bundle Generation](#bundle-generation)
  - [Training, Inference, and Evaluation](#training-inference-and-evaluation)
  - [Docker Usage](#docker-usage)
- [Results](#results)
- [Datasets](#datasets)
- [Unit Testing](#unit-testing)
- [Development Environment](#development-environment)
- [Model Weights](#model-weights)
- [Functionality Overview](#functionality-overview)
- [References](#references)

---

## Project Overview

This repository provides code and configuration for training, evaluating, and analysing deep learning models for medical image segmentation with a focus on improving calibration using differentiable marginal L1 Average Calibration Error (mL1-ACE) losses. The framework supports multiple datasets and loss functions, and includes tools for generating MONAI bundles, running experiments, and visualising results.

---

## Installation & Setup

This project is designed to run on a system with at least a 24GB VRAM GPU (e.g., RTX 4090) and 96GB RAM.
It is recommended to use Docker or the provided `.devcontainer` for a reproducible environment.

---

## Usage

### Bundle Generation

To generate a MONAI bundle for a specific dataset and loss function, use:

```bash
python generate_bundle.py --data <dataset_name> --loss <loss_name>
```

- `<dataset_name>`: One of:
  - `acdc17`
  - `amos22`
  - `kits23`
  - `brats21`
- `<loss_name>`: One of:
  - `baseline_ce`
  - `baseline_dice`
  - `baseline_dice_ce`
  - `hardl1ace_ce`
  - `hardl1ace_dice`
  - `hardl1ace_dice_ce`
  - `softl1ace_ce`
  - `softl1ace_dice`
  - `softl1ace_dice_ce`

(see `bundle_templates/configs/loss/`) for loss configs

**Note:**
 This repository and the associated paper focus on results for the `*_dice_ce` variants.

This will create a bundle directory under `bundles/` with the appropriate configuration.

**Bundle Naming and Hashing:**
The generated bundle directory will be named as:

```
<dataset_name>_<loss_name>_<hash>
```

where `<hash>` is an 8-character hash computed from the contents of the YAML configuration files (data, loss, and common configs).
This ensures that if you change any configuration, a new unique bundle directory is created, preventing accidental overwriting of previous bundles.
You can safely rename the bundle directory if desired—the hash is only for uniqueness and reproducibility.

### Training, Inference, and Evaluation

The main entry point for running experiments is `run_monai_bundle.py`:

```bash
python run_monai_bundle.py --bundle <bundle_name> --mode <mode> [--seed <seed>] [--debug]
```

- `--bundle`: Name of the generated bundle directory (e.g., `brats21_softl1ace_dice_ce_<hash>`)
- `--mode`: One of `train`, `inference_pred`, `inference_eval`
- `--seed`: (Optional) Set random seed for reproducibility
- `--debug`: (Optional) Enable debug mode

**Example:**

```bash
python run_monai_bundle.py --bundle brats21_softl1ace_dice_ce_1234abcd --mode train
```

### Docker Usage

Docker is the recommended way to ensure a consistent and reproducible environment for training and evaluation.
All Docker-related scripts are located in `scripts/`.

#### 1. Build the Docker Image

Build the image using the provided script:

```bash
./scripts/docker_build.sh
```

- This script builds the image as `${USER}/acl:latest` using your user and group IDs for correct file permissions.
- You can optionally push the image to a registry by uncommenting the `docker push` line in the script.

#### 2. Run Experiments in Docker

Use the `scripts/docker_run.sh` script to launch training, inference, or evaluation inside the container.

**Example usage:**

```bash
./scripts/docker_run.sh --mode train --bundle <bundle_name> [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>] [--seed <seed>] [other run_monai_bundle.py args]
```

**Arguments:**

- `--mode`: Operation mode for `run_monai_bundle.py` (`train`, `inference_pred`, `inference_eval`)
- `--bundle`: Name of the generated bundle directory (e.g., `brats21_softl1ace_dice_ce_<hash>`)
- `--gpu`: GPU index to use (default: 0)
- `--cpus`: CPU range to allocate (default: 0-5)
- `--shm-size`: Shared memory size for the container (default: 32g)
- `--seed`: Random seed (default: 12345)
- Any additional arguments are passed directly to `run_monai_bundle.py`

**Environment and Mounts:**

- The project directory is mounted to `/workspace/project` inside the container.
- The data directory (expected at the same level as the project) is mounted to `/workspace/data`.
- The working directory is set to `/workspace/project`.

**Example:**

```bash
./scripts/docker_run.sh --mode train --bundle acdc17_softl1ace_dice_ce_769fc24c --gpu 0 --cpus 0-11 --shm-size 32g --seed 12345
```

This will run training on GPU 0, using CPUs 0-11, with 64GB shared memory, and seed 42.

**Notes:**

- The container runs in detached mode (`-d`) and is automatically removed after completion (`--rm`).
- You can modify the script to run interactively or attach logs as needed.

---

## Results

All results included in the paper can be found in the following notebooks:

- **Main results:**
  [`journal_results_main.ipynb`](journal_results_main.ipynb)
- **Inference on test cases with all semantic classes:**
  [`journal_results_complete_classes.ipynb`](journal_results_complete_classes.ipynb)
- **Results from the final epoch:**
  [`journal_results_final_epoch.ipynb`](journal_results_final_epoch.ipynb)
- **Results using the old loss function:**
  [`journal_results_old_loss.ipynb`](journal_results_old_loss.ipynb)

---

## Datasets

This project used the following datasets.

**Dataset details:**

- [**ACDC 2017 (Cardiac MRI)**](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
  - *Task*: Cardiac MRI segmentation (Left Ventricle, Myocardium, Right Ventricle + background)
  - *Modality*: MRI
  - *Size*: 150 MRI cases (100 Training, 50 Testing)
  - *Source*: University Hospital of Dijon, France
  - *Config*: [`bundle_templates/configs/data/acdc17.yaml`](bundle_templates/configs/data/acdc17.yaml)

- [**AMOS 2022 (Abdominal CT)**](https://amos22.grand-challenge.org/Instructions/)
  - *Task*: Abdominal organ segmentation (15 organs + background)
  - *Modality*: CT
  - *Size*: 500 CT cases (240 Training, 100 Validation, 160 Testing)
  - *Source*: Longgang District Central Hospital and Longgang District People's Hospital, SZ, China
  - *Config*: [`bundle_templates/configs/data/amos22.yaml`](bundle_templates/configs/data/amos22.yaml)

- [**KiTS23 (Kidney Tumor Segmentation)**](https://kits-challenge.org/kits23/)
  - *Task*: Segmentation of kidney, tumor, and cyst from contrast-enhanced preoperative CT scans
  - *Modality*: CT
  - *Size*: 599 cases (489 Training, 110 Testing)
  - *Source*: M Health Fairview medical center
  - *Config*: [`bundle_templates/configs/data/kits23.yaml`](bundle_templates/configs/data/kits23.yaml)

- [**BraTS21 (Brain Tumor Segmentation)**](http://braintumorsegmentation.org/)
  - *Task*: Segmentation of brain tumors (multiple classes)
  - *Modality*: MRI
  - *Size*: See BraTS 2021 challenge
  - *Config*: [`bundle_templates/configs/data/brats21.yaml`](bundle_templates/configs/data/brats21.yaml)
  - For this dataset the training dataset available from BraTS was split into training and testing
  - The splits are found in ./brats21_train_val.txt and ./brats21_test.txt

---

## Unit Testing

Unit tests are provided using `pytest`. To run all tests:

```bash
pytest
```

---

## Development Environment

For development, a `.devcontainer` setup is provided for use with Visual Studio Code and Docker.
**Requirements:**

- NVIDIA GPU with CUDA support
- `nvidia-container-toolkit`, `nvidia-docker2`
- Docker
- Visual Studio Code with [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

---

## Model Weights

**Pretrained model weights are not included in this repository due to size constraints.**
A download link will be provided here in the future.

<!-- ---

## Dataset Download

*Instructions for downloading and preparing the datasets will be added here.* -->

---

## Functionality Overview

This repository implements:

- Differentiable calibration losses (hard/soft mL1-ACE)
- Calibration Metrics
- Dataset reliability histograms and reliability diagrams
- MONAI bundle-based training, inference, and evaluation pipelines

### Implemented Losses

The following loss functions are implemented in [`src/losses/hardl1ace.py`](src/losses/hardl1ace.py) and [`src/losses/softl1ace.py`](src/losses/softl1ace.py):

#### Hard Binned Calibration Losses (`hardl1ace.py`)

- **HardL1ACELoss**  - Computes the hard-binned marginal L1 Average Calibration Error (ACE) loss.

- **HardL1ACEandCELoss** - Combines HardL1ACELoss and CrossEntropyLoss with configurable weights.

- **HardL1ACEandDiceLoss** - Combines HardL1ACELoss and DiceLoss

- **HardL1ACEandDiceCELoss** - Combines HardL1ACELoss, DiceLoss, and CrossEntropyLoss. This is the hard-binned loss variant explored in this repo and assosciated paper.

#### Soft Binned Calibration Losses (`softl1ace.py`)

- **SoftL1ACELoss**  - Computes the soft-binned marginal L1 Average Calibration Error (ACE) loss.

- **SoftL1ACEandCELoss**  - Combines SoftL1ACELoss and CrossEntropyLoss

- **SoftL1ACEandDiceLoss** - Combines SoftL1ACELoss and DiceLoss

- **SoftL1ACEandDiceCELoss**  - Combines SoftL1ACELoss, DiceLoss, and CrossEntropyLoss. This is the soft-binned loss variant explored in this repo and assosciated paper.

Each loss supports options for background exclusion, one-hot encoding, and ignoring empty classes, and can be configured via the MONAI bundle system.

### Implemented Metrics

The repository provides calibration metrics, implemented in [`src/metrics/calibration.py`](src/metrics/calibration.py).
These metrics inherit from MONAI's `CumulativeIterationMetric` and are fully compatible with MONAI workflows.

- **CalibrationErrorMetric**
  Computes calibration errors between predicted probabilities and ground truth labels for multi-class segmentation. Supports several reduction modes:
  - *Expected Calibration Error (ECE)*: Weighted average of absolute differences between predicted confidence and accuracy.
  - *Average Calibration Error (ACE)*: Simple average of absolute differences across bins.
  - *Maximum Calibration Error (MCE)*: Maximum absolute difference across bins.
  Can exclude background, supports batched and multi-channel data, and is configurable for number of bins and reduction type.

- **ReliabilityDiagramMetric**
  Computes and visualises case reliability diagrams and dataset reliability histograms for model calibration assessment.

*Additional segmentation metrics (e.g., Dice) are available via MONAI and are integrated into the evaluation pipelines.*

### Implemented Handlers

The repository provides several handlers in [`src/handlers/calibration.py`](src/handlers/calibration.py) for attaching metrics and calibration analysis to the PyTorch Ignite engine.
These handlers are wrappers around the MONAI metrics and inherit from `IgniteMetricHandler`, making them easy to integrate into MONAI and Ignite-based training and evaluation loops.

- **CalibrationError**
  A handler for computing and logging calibration errors (ECE, ACE, MCE) during training or evaluation. Supports batch, class, and iteration averaging, and can save detailed results per image.

- **ReliabilityDiagramHandler**
  A handler for generating and saving reliability diagrams and dataset reliability histograms during training or evaluation. Supports per-case and dataset-level diagrams, customisable plotting, and saving to disk.

- **CalibrationErrorHandler**
  Computes and saves both micro and macro calibration errors, tracks missing classes, and writes results to CSV for further analysis. Aggregates binning data and calibration errors over epochs.

- **BinningDataHandler**
  Computes and stores calibration binning data in the Ignite engine state for efficient reuse by other metrics or handlers.

These handlers are designed for seamless integration with MONAI's and Ignite's event-driven workflows, enabling robust and reproducible calibration analysis in segmentation pipelines.

### Visualisation Utilities

The repository includes a visualisation module [`src/visualize/reliability_diagrams.py`](src/visualize/reliability_diagrams.py) for generating publication-quality reliability diagrams and dataset reliability histograms.

- **draw_case_reliability_diagrams**
  Generates and saves reliability diagrams for individual cases, visualising the calibration of predicted probabilities against ground truth for each class. Supports overlaying calibration gaps, ECE/ACE/MCE annotations, and optional histograms of bin counts.

- **draw_dataset_reliability_diagrams**
  Aggregates calibration results across the dataset to produce dataset-level reliability diagrams and reliability histograms. These visualisations help assess overall calibration and the distribution of predictions across bins and classes.

- **Customisation**
  The plotting functions support extensive customisation, including figure size, class/case names, color maps, and saving options for integration into publications.

#### Example Outputs

**Reliability Diagram (per-case):** [Reliability Diagram Example](bundles/brats21_softl1ace_dice_ce_nl/seed_12345/inference_results/reliability_diagrams/BraTS2021_00006_reliability_diagram.pdf)

**Dataset Reliability Histogram:** [Dataset Reliability Histogram Example](bundles/brats21_softl1ace_dice_ce_nl/seed_12345/inference_results/reliability_diagrams/dataset_reliability_diagram.pdf)

---

## References

- [Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation](https://arxiv.org/abs/2403.06759)
- [MONAI Framework](https://monai.io/)

