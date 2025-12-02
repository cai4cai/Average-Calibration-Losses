from typing import Any, Callable, Literal
import torch
import torch.nn.functional as F
from torch import Tensor

from monai.config.type_definitions import TensorOrList
from monai.metrics.metric import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils.enums import MetricReduction
from monai.utils.enums import StrEnum


__all__ = [
    "BrierScore",
    "CategoricalNLL",
    "AURC",
]


class BrierScore(CumulativeIterationMetric):
    """
    Compute the Brier score for multi-class or multi-label classification.

    The Brier Score measures the mean squared difference between predicted
    probabilities and actual target values. It is used to evaluate the
    accuracy of probabilistic predictions, where a lower score indicates
    better calibration and prediction quality.

    Args:
        is_multilabel (bool): Whether this is a multi-label task. If False, assumes multi-class.
            Defaults to False.
        top_class (bool): If True, computes the Brier score for the top predicted class only.
            Only applicable for multi-class. Defaults to False.
        metric_reduction (MetricReduction | str): Mode of reduction to apply to the metrics.
            Defaults to "mean".
        get_not_nans (bool): Whether to return the count of non-NaN values. Defaults to False.
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        top_class: bool = False,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.is_multilabel = is_multilabel
        self.top_class = top_class
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Compute Brier score for a single tensor pair.

        Args:
            y_pred: Predicted probabilities of shape [batch_size, num_classes, ...spatial_dims]
            y: Ground truth labels of shape [batch_size, num_classes, ...spatial_dims]
               (one-hot for multi-class, binary per channel for multi-label)

        Returns:
            Brier score tensor of shape [batch_size, num_classes]
        """
        if y is None:
            raise ValueError(
                "Ground truth labels y must be provided for Brier score computation"
            )

        # Ensure y is in [B, C, ...] format and matches prediction shape
        if y.shape != y_pred.shape:
            raise ValueError(
                f"Ground truth shape {y.shape} must match prediction shape {y_pred.shape}"
            )

        if self.top_class and not self.is_multilabel:
            # Compute Brier score for top predicted class only (multi-class only)
            probs, indices = y_pred.max(dim=1, keepdim=True)
            target = y.gather(1, indices)
            brier_score = F.mse_loss(probs, target, reduction="none")
            # Expand to [B, C] shape
            batch_size, num_classes = y_pred.shape[0], y_pred.shape[1]
            brier_expanded = torch.zeros(batch_size, num_classes, device=y_pred.device)
            brier_expanded.scatter_(1, indices.squeeze(1), brier_score.squeeze(1))
            brier_score = brier_expanded
        else:
            # Compute Brier score per class: (p_i - y_i)^2
            brier_score = F.mse_loss(
                y_pred, y.float(), reduction="none"
            )  # [B, C, ...spatial_dims]

        # Average over spatial dimensions, keeping [B, C] shape
        spatial_dims = tuple(range(2, len(brier_score.shape)))
        if spatial_dims:
            brier_score = brier_score.mean(dim=spatial_dims)

        return brier_score  # shape [batch_size, num_classes]

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_tensor`.

        Args:
            reduction: define mode of reduction to the metrics.

        Returns:
            Aggregated metric results or tuple of (metric, not_nans) if get_not_nans=True
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # Flatten and remove NaNs
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f


class CategoricalNLL(CumulativeIterationMetric):
    """
    Computes the Negative Log-Likelihood (NLL) metric for classification tasks.

    This metric evaluates the performance of a probabilistic classification model by
    calculating the negative log likelihood of the predicted probabilities.

    Args:
        is_multilabel (bool): Whether this is a multi-label task. If False, assumes multi-class.
            Defaults to False.
        metric_reduction (MetricReduction | str): Mode of reduction to apply to the metrics.
            Defaults to "mean".
        get_not_nans (bool): Whether to return the count of non-NaN values. Defaults to False.
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.is_multilabel = is_multilabel
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Compute NLL for a single tensor pair.

        Args:
            y_pred: Predicted probabilities of shape [batch_size, num_classes, ...spatial_dims]
            y: Ground truth labels of shape [batch_size, num_classes, ...spatial_dims]
               (one-hot for multi-class, binary per channel for multi-label)

        Returns:
            NLL tensor of shape [batch_size, num_classes]
        """
        if y is None:
            raise ValueError(
                "Ground truth labels y must be provided for NLL computation"
            )

        # Ensure y is in [B, C, ...] format and matches prediction shape
        if y.shape != y_pred.shape:
            raise ValueError(
                f"Ground truth shape {y.shape} must match prediction shape {y_pred.shape}"
            )

        # Add small epsilon to prevent log(0)
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)

        if self.is_multilabel:
            # Use Binary Cross Entropy per channel: -y*log(p) - (1-y)*log(1-p)
            nll = F.binary_cross_entropy(
                y_pred, y.float(), reduction="none"
            )  # [B, C, ...spatial_dims]
        else:
            # Standard Categorical NLL (Cross Entropy): -y*log(p)
            nll = -y * torch.log(y_pred)  # [B, C, ...spatial_dims]

        # Average over spatial dimensions, keeping [B, C] shape
        spatial_dims = tuple(range(2, len(nll.shape)))
        if spatial_dims:
            nll = nll.mean(dim=spatial_dims)

        return nll  # shape [batch_size, num_classes]

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_tensor`.

        Args:
            reduction: define mode of reduction to the metrics.

        Returns:
            Aggregated metric results or tuple of (metric, not_nans) if get_not_nans=True
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # Flatten and remove NaNs
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f


class AURC(CumulativeIterationMetric):
    """
    Calculate Area Under the Risk-Coverage curve.

    The Area Under the Risk-Coverage curve (AURC) is the main metric for
    Selective Classification (SC) performance assessment. It evaluates the
    quality of uncertainty estimates by measuring the ability to
    discriminate between correct and incorrect predictions based on their
    rank.

    Args:
        is_multilabel (bool): Whether this is a multi-label task. If False, assumes multi-class.
            Defaults to False.
        metric_reduction (MetricReduction | str): Mode of reduction to apply to the metrics.
            Defaults to "mean".
        get_not_nans (bool): Whether to return the count of non-NaN values. Defaults to False.
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.is_multilabel = is_multilabel
        self.metric_reduction = metric_reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Compute AURC for each class.

        Args:
            y_pred: Predicted probabilities of shape [batch_size, num_classes, ...spatial_dims]
            y: Ground truth labels of shape [batch_size, num_classes, ...spatial_dims]
               (one-hot for multi-class, binary per channel for multi-label)

        Returns:
            AURC tensor of shape [batch_size, num_classes]
        """
        if y is None:
            raise ValueError(
                "Ground truth labels y must be provided for AURC computation"
            )

        # Ensure y is in [B, C, ...] format and matches prediction shape
        if y.shape != y_pred.shape:
            raise ValueError(
                f"Ground truth shape {y.shape} must match prediction shape {y_pred.shape}"
            )

        batch_size, num_classes = y_pred.shape[0], y_pred.shape[1]

        # Compute AURC per class per batch
        aurc_results = torch.zeros(batch_size, num_classes, device=y_pred.device)

        for b in range(batch_size):
            for c in range(num_classes):
                # We only evaluate AURC on pixels where the model PREDICTS class c.
                # This makes it a "Conditional AURC" or "Selective Classification Risk" for that class.

                if self.is_multilabel:
                    # Multi-label: Prediction is p(c) > 0.5
                    pred_mask = y_pred[b, c] > 0.5

                    if pred_mask.sum() == 0:
                        aurc_results[b, c] = float("nan")
                        continue

                    class_probs = y_pred[b, c][pred_mask]
                    target_flat = y[b, c][pred_mask]
                    class_errors = 1.0 - target_flat

                else:
                    # Multi-class: Prediction is argmax == c
                    pred_mask = y_pred[b].argmax(dim=0) == c

                    if pred_mask.sum() == 0:
                        aurc_results[b, c] = float("nan")
                        continue

                    class_probs = y_pred[b, c][pred_mask]
                    target_flat = y[b, c][pred_mask]

                    # Error: Is the ground truth NOT c?
                    class_errors = 1.0 - target_flat

                # Compute AURC for this class
                if len(class_probs) > 1:
                    aurc_val = self._compute_aurc(class_probs, class_errors)
                    aurc_results[b, c] = aurc_val
                else:
                    aurc_results[b, c] = float("nan")

        return aurc_results  # shape [batch_size, num_classes]

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_tensor`.

        Args:
            reduction: define mode of reduction to the metrics.

        Returns:
            Aggregated metric results or tuple of (metric, not_nans) if get_not_nans=True
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # Flatten and remove NaNs
        f, not_nans = do_metric_reduction(data, reduction or self.metric_reduction)
        return (f, not_nans) if self.get_not_nans else f

    def _compute_aurc(self, scores: torch.Tensor, errors: torch.Tensor) -> torch.Tensor:
        """
        Compute AURC from scores and errors.

        Args:
            scores: Confidence scores
            errors: Binary error indicators

        Returns:
            AURC value
        """
        # Sort by scores (descending order - highest confidence first)
        # Use stable=True to ensure consistent ordering for tied values
        sorted_indices = scores.argsort(descending=True, stable=True)
        sorted_errors = errors[sorted_indices]

        # Compute cumulative error rates (risk at each coverage level)
        num_samples = len(sorted_errors)
        coverage_levels = torch.arange(
            1, num_samples + 1, dtype=scores.dtype, device=scores.device
        )
        cumulative_errors = sorted_errors.cumsum(dim=0)
        error_rates = cumulative_errors / coverage_levels

        # Compute coverage as fraction of total samples
        coverage_fractions = coverage_levels / num_samples

        # Compute AUC using trapezoidal rule
        if num_samples < 2:
            return torch.tensor(float("nan"), device=scores.device)

        # AUC computation
        aurc = torch.trapz(error_rates, coverage_fractions)

        # Normalize AURC
        # Standard AURC is not normalized by (1 - 1/N).
        # However, some implementations do this to scale it.
        # We use the raw area under the risk-coverage curve.
        # We must ensure it is 0 for perfect predictions.
        # If errors is all 0s, error_rates is all 0s, AURC is 0.
        # This works IF we only consider the "Predicted Positive" samples (see _compute_tensor fix).

        return aurc
