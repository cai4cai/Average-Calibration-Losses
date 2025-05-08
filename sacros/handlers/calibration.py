from __future__ import annotations
import os
import torch

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..metrics import (
    CalibrationErrorMetric,
    CalibrationReduction,
    ReliabilityDiagramMetric,
    calibration_binning,
)
from monai.utils import MetricReduction
from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.handlers.tensorboard_handlers import TensorBoardHandler
from monai.handlers.utils import write_metrics_reports
from monai.metrics.utils import ignore_background

from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import
from monai.visualize import plot_2d_or_3d_image

from monai.utils.enums import CommonKeys as Keys
from monai.utils import ImageMetaKey
from monai.utils.enums import StrEnum

Events, _ = optional_import(
    "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events"
)

if TYPE_CHECKING:
    from ignite.engine import Engine
    from tensorboardX import SummaryWriter as SummaryWriterX
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import(
        "ignite.engine",
        IgniteInfo.OPT_IMPORT_VERSION,
        min_version,
        "Engine",
        as_type="decorator",
    )
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")
    SummaryWriterX, _ = optional_import("tensorboardX", name="SummaryWriter")

from ..metrics.calibration import _aggregate_binning_data

__all__ = ["CalibrationError", "ReliabilityDiagramHandler", "CalibrationErrorHandler"]


class BinningKeys(StrEnum):
    """
    Keys for storing binning data in the engine's state.
    
    Attributes:
        ITERATION_BINNING: Key for storing binning data after each iteration.
        RUNNING_BINNING_DATA: Key for storing running binning data across iterations.
        AGGREGATED_BINNING_DATA: Key for storing aggregated binning data after each epoch.
    """
    ITERATION_BINNING = "iteration_binning"
    RUNNING_BINNING_DATA = "running_binning_data"
    AGGREGATED_BINNING_DATA = "aggregated_binning_data"


class BinningDataHandler:
    """
    Computes the calibration binning data and stores it in the engine's state.
    The computed binning data can be accessed by other metrics for efficient reuse.

    This handler computes binning data after every iteration and aggregates it after every epoch.
    The binning data includes mean predicted probability per bin, mean ground truth per bin, and bin counts.
    """

    def __init__(self, num_bins: int = 20, 
                 include_background: bool = True, 
                 right: bool = False):
        
        self.num_bins = num_bins
        self.include_background = include_background
        self.right = right
        self.running_binning_data = []
        

    def attach(self, engine: Engine) -> None:
        """
        Attach this handler to an Ignite Engine to compute binning data after every iteration
        and aggregate it after every epoch.

        Args:
            engine: The Ignite Engine to attach to.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.aggregate_binning_data)

    def __call__(self, engine: Engine) -> None:
        """
        This is called after every iteration to compute and store the binning data in `engine.state`.
        
        Args:
            engine: The Ignite Engine to which this handler is attached.
        """
        y_pred, y = engine.state.output[Keys.PRED], engine.state.output[Keys.LABEL]  # TODO: could also define an output_transform to get these values
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
            
        binning_data = calibration_binning(y_pred, y, num_bins=self.num_bins, right=self.right) 
        # Binng data is mean_p_per_bin, mean_gt_per_bin, bin_counts. Each one has shape: [batch_size, num_channels, num_bins]
        binning_data = torch.stack(binning_data, dim=-2)
        
        # Store the binning data in the engine state
        engine.state.output[BinningKeys.ITERATION_BINNING] = binning_data

        # Aggregate binning data
        self.running_binning_data.append(binning_data)

    def aggregate_binning_data(self, engine: Engine) -> None:
        """
        This is called after every epoch to aggregate the binning data.

        Args:
            engine: The Ignite Engine to which this handler is attached.
        """
        # Aggregate the binning data across iterations
        running_binning_data = torch.cat(self.running_binning_data, dim=0)  # shape: [N, C, 3, num_bins]
        engine.state.output[BinningKeys.RUNNING_BINNING_DATA] = running_binning_data

        aggregated_binning_data = _aggregate_binning_data(running_binning_data, aggregate_classes=False)  # shape: [C, 3, num_bins]

        # Store the aggregated binning data in the engine state
        engine.state.output[BinningKeys.AGGREGATED_BINNING_DATA] = aggregated_binning_data

        # Reset the aggregated binning data for the next epoch
        self.running_binning_data = []


class CalibrationError(IgniteMetricHandler):
    """
    Computes Calibration Error and collects the average over batch, class-channels, iterations.
    Can return the expected, average, or maximum calibration error.

    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        calibration_reduction: (
            CalibrationReduction | str
        ) = CalibrationReduction.EXPECTED,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Args:
            num_bins: number of bins to calculate calibration.
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            calibration_reduction (CalibrationReduction | str): Method for calculating calibration error values from binned data.
                Available modes are `"expected"`, `"average"`, and `"maximum"`. Defaults to `"expected"`.
            metric_reduction (MetricReduction | str): Mode of reduction to apply to the metrics. Reduction is only applied to non-NaN values.
                Available reduction modes are `"none"`, `"mean"`, `"sum"`, `"mean_batch"`, `"sum_batch"`, `"mean_channel"`, and `"sum_channel"`.
            Defaults to `"mean"`. If set to `"none"`, no reduction will be performed.
            num_classes: number of input channels (always including the background). When this is None,
                ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
                single-channel class indices and the number of classes is not automatically inferred from data.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean dice of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        """
        metric_fn = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=include_background,
            calibration_reduction=calibration_reduction,
            metric_reduction=metric_reduction,
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class ReliabilityDiagramHandler(IgniteMetricHandler):
    """
    Handler to compute and save reliability diagrams during training/validation.
    This handler extends IgniteMetricHandler and utilizes ReliabilityDiagramMetric
    to generate reliability diagrams, which are visual representations of model calibration.

    Reliability diagrams compare the predicted probabilities of a model with the actual outcomes,
    helping to understand how well the model's predicted probabilities are calibrated.
    """

    def __init__(
        self,
        num_classes: int,
        num_bins: int = 20,
        include_background: bool = True,
        output_dir: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        class_names: list[str] | None = None,
        draw_case_diagrams: bool = False,
        draw_case_histograms: bool = False,
        case_name_transform: Callable = None,
        print_case_ece: bool = True,
        print_case_ace: bool = True,
        print_case_mce: bool = True,
        draw_dataset_diagrams: bool = True,
        draw_dataset_histograms: bool = False,
        draw_dataset_average_over_classes: bool = False,
        dataset_imshow_kwargs: dict[str, Any] = {},
        savefig_kwargs: dict[str, Any] = {},  # Updated to use savefig_kwargs
        rc_params: dict[str, Any] = {},  # Added rc_params
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Initializes the handler with the given parameters for computing and saving reliability diagrams.

        Args:
            num_classes: Number of classes (including background) to compute the reliability diagrams for.
            num_bins: Number of bins to use for the reliability diagrams.
            include_background: Whether to include background class in the reliability diagrams.
            output_dir: Directory where the diagrams will be saved.
            figsize: Size of each diagram.
            class_names: Names of the classes for the diagrams.
            draw_case_diagrams: Whether to draw diagrams for individual cases.
            draw_case_histograms: Whether to draw histograms for individual cases.
            case_name_transform: Function to transform case names.
            print_case_ece: Whether to print Expected Calibration Error for cases.
            print_case_ace: Whether to print Average Calibration Error for cases.
            print_case_mce: Whether to print Maximum Calibration Error for cases.
            draw_dataset_diagrams: Whether to draw diagrams for the entire dataset.
            draw_dataset_histograms: Whether to draw histograms for the entire dataset.
            draw_dataset_average_over_classes: Whether to draw the average diagram over classes for the entire dataset.
            dataset_imshow_kwargs: Additional keyword arguments for imshow function for dataset diagrams.
            savefig_kwargs: Additional keyword arguments for saving figures.
            rc_params: Additional keyword arguments for matplotlib rcParams.
            output_transform: Function to transform the output for metric computation.
            save_details: Whether to save detailed results.
        """

        metric_fn = ReliabilityDiagramMetric(
            num_classes=num_classes,
            num_bins=num_bins,
            include_background=include_background,
            output_dir=output_dir,
            figsize=figsize,
            class_names=class_names,
            draw_case_diagrams=draw_case_diagrams,
            draw_case_histograms=draw_case_histograms,
            case_name_transform=case_name_transform,
            print_case_ece=print_case_ece,
            print_case_ace=print_case_ace,
            print_case_mce=print_case_mce,
            draw_dataset_diagrams=draw_dataset_diagrams,
            draw_dataset_histograms=draw_dataset_histograms,
            dataset_imshow_kwargs=dataset_imshow_kwargs,
            draw_dataset_average_over_classes=draw_dataset_average_over_classes,
            savefig_kwargs=savefig_kwargs,  # Use savefig_kwargs here
            rc_params=rc_params,  # Use rc_params here
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class TensorBoardReliabilityDiagramHandler(TensorBoardHandler):
    """
    Text
    """

    pass


class CalibrationErrorHandler:
    """
    Handler to compute and save micro and macro calibration errors during training/validation.
    This handler calculates the binning data at each iteration and uses it to compute the expected,
    maximum, and average calibration errors. It also aggregates the binning data at the end of each epoch
    to compute the micro calibration errors. The results are saved to a CSV file.
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        output_dir: str = "./",
        summary_ops: str | Sequence[str] | None = None,
        delimiter: str = ",",
        output_type: str = "csv",
        meta_batch_transform: Callable = lambda x: x,
    ):
        self.num_bins = num_bins
        self.include_background = include_background
        self.output_dir = output_dir
        self.summary_ops = summary_ops
        self.delimiter = delimiter
        self.output_type = output_type
        self.meta_batch_transform = meta_batch_transform
        self.macro_errors = []
        self.micro_errors = []
        self.running_binning_data = []
        self.missing_classes = []  # New list to store missing classes information
        self._filenames: list[str] = []

    def attach(self, engine: Engine) -> None:
        """
        Attach this handler to an Ignite Engine to compute calibration errors after every iteration
        and aggregate them after every epoch.

        Args:
            engine: The Ignite Engine to attach to.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.aggregate_and_save)

    def __call__(self, engine: Engine) -> None:
        """
        This is called after every iteration to compute and store the calibration errors in `engine.state`.

        Args:
            engine: The Ignite Engine to which this handler is attached.
        """
        # Check if there are any missing classes in the ground truth label: NOTE: This is a very clunky way of doing it
        missing_classes = self.check_missing_classes(engine)      
        self.missing_classes.append(missing_classes)  # Store missing classes information
        
        y_pred, y = engine.state.output[0]['pred'].unsqueeze(0), engine.state.output[0]['label'].unsqueeze(0)  # add in batch dim for simplicity
        # TODO: could also define an output_transform to get these values
        # TODO: this is hard coded just to work with single batch element provided by engine
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        binning_data = calibration_binning(y_pred, y, num_bins=self.num_bins)
        binning_data = torch.stack(binning_data, dim=-2).squeeze(0)   # remove batch dim. shape: [num_channels, 3, num_bins]
        self.running_binning_data.append(binning_data)

        macro_errors = self.compute_calibration_errors(binning_data)
        self.macro_errors.append(macro_errors)

        meta_data = self.meta_batch_transform(engine.state.batch)
        if isinstance(meta_data, dict):
            meta_data = decollate_batch(meta_data)
        for m in meta_data:
            self._filenames.append(f"{m.get(ImageMetaKey.FILENAME_OR_OBJ)}")
            
    def check_missing_classes(self, engine: Engine) -> torch.Tensor:
        """
        Check if there are any missing classes in the ground truth label.

        Args:
            engine: The Ignite Engine to which this handler is attached.

        Returns:
            torch.Tensor: A tensor of shape [C] with 1s for missing classes and 0s for present classes.
        """
        label = engine.state.batch[0]['label']  # should be shape [1, H, W, D]
        pred = engine.state.output[0]['pred']  # Should be OH [C, H, W, D]
        num_classes = pred.shape[0]

        present_classes = label.unique().tolist()
        missing_classes = torch.tensor([1 if c not in present_classes else 0 for c in range(num_classes)])
        
        if not self.include_background:
            missing_classes = missing_classes[1:]
            
        return missing_classes

    def compute_calibration_errors(self, binning_data):
        """
        Compute calibration errors directly using the binned data.

        Args:
            binning_data: Binned data containing mean predicted values, mean ground truth values, and bin counts.

        Returns:
            dict: A dictionary containing expected, average, and maximum calibration errors.
        """
        mean_p_per_bin, mean_gt_per_bin, bin_counts = binning_data[:, 0, :], binning_data[:, 1, :], binning_data[:, 2, :]

        abs_diff = torch.abs(mean_p_per_bin - mean_gt_per_bin)

        expected_error = torch.nansum(abs_diff * bin_counts, dim=-1) / torch.sum(bin_counts, dim=-1)
        average_error = torch.nanmean(abs_diff, dim=-1)
        abs_diff[torch.isnan(abs_diff)] = 0
        max_error = torch.max(abs_diff, dim=-1).values

        return {
            "ece": expected_error,
            "ace": average_error,
            "mce": max_error,
        }

    def aggregate_and_save(self, engine: Engine) -> None:
        """
        This is called after every epoch to aggregate the binning data and save the calibration errors.

        Args:
            engine: The Ignite Engine to which this handler is attached.
        """
        running_binning_data = torch.stack(self.running_binning_data, dim=0)   #  [N, C, 3, num_bins]
        aggregated_binning_data = _aggregate_binning_data(running_binning_data)  # [C, 3, num_bins]

        micro_errors = self.compute_calibration_errors(aggregated_binning_data)
        self.micro_errors.append(micro_errors)  # TODO: this doesn't need to be a list as it only has one 

        self.save_to_csv(engine)

        self.running_binning_data = []
        self.macro_errors = []
        self.micro_errors = []
        self.missing_classes = []  # Reset missing classes information
        self._filenames = []

    def save_to_csv(self, engine: Engine) -> None:
        """
        Save the calibration errors to a CSV file.

        Args:
            engine: The Ignite Engine to which this handler is attached.
        """
        save_dir = self.output_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        macro_errors = {k: torch.stack([e[k] for e in self.macro_errors]) for k in self.macro_errors[0]}
        micro_errors = {k: torch.stack([e[k] for e in self.micro_errors]) for k in self.micro_errors[0]}
        missing_classes_stacked = torch.stack(self.missing_classes)  # Stack missing classes to shape [N, C]

        metrics = {f"macro_{k}": v.mean().item() for k, v in macro_errors.items()}
        metrics.update({f"micro_{k}": v.mean().item() for k, v in micro_errors.items()})
        metrics.update({"missing_classes": missing_classes_stacked.any(dim=0).tolist()})  # Add missing classes metric

        metric_details = {f"macro_{k}": v.cpu().numpy() for k, v in macro_errors.items()}
        metric_details.update({f"micro_{k}": v.cpu().numpy() for k, v in micro_errors.items()})
        metric_details.update({"missing_classes": missing_classes_stacked.cpu().numpy()})  # Add missing classes details

        write_metrics_reports(
            save_dir=save_dir,
            images=self._filenames,
            metrics=metrics,
            metric_details=metric_details,
            summary_ops=self.summary_ops,
            deli=self.delimiter,
            output_type=self.output_type,
        )

# TODO: options for missing classes is either to try and save summaries with and without missing classes here, or just keep the data as it is, but save another csv called "missing_classes.csv" with the missing classes and then process that from the compile results, but we'd also have to change the way the results are computed and calculate it from the _raw files, but only for the missing class metrics.