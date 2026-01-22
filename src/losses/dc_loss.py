import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


__all__ = [
    "DCLoss",
]


class DCLoss(_Loss):
    """
    Distance-based Calibration (DC) Loss.

    This loss measures calibration by comparing the empirical distribution of
    predictions to a theoretical logistic distribution. It uses stratified sampling
    to create bins and computes the L1 distance between sorted logits.

    The loss is computed by:
    1. For each prediction-label pair, creating a stratified sample of points
    2. Converting predictions to logits based on the true label
    3. Sorting both empirical and theoretical logits
    4. Computing the mean absolute difference

    Reference:
        Based on distance-based calibration error metrics for probabilistic predictions.
    """

    def __init__(
        self,
        n_points: int = 50,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            n_points: Number of stratified points to use for calibration measurement. Defaults to 50.
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )

        self.n_points = n_points
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act

    @staticmethod
    def _stratified_logistic(shape, device, dtype):
        """
        Generate stratified logistic samples.

        Creates a grid of evenly-spaced quantiles in logistic space,
        which serves as the theoretical reference distribution.

        Args:
            shape: Shape of the output tensor
            device: Device to create the tensor on
            dtype: Data type for the tensor

        Returns:
            Tensor of stratified logistic samples
        """
        n = int(torch.tensor(shape).prod().item())
        u = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
        return (torch.log(u) - torch.log1p(-u)).view(*shape)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Returns:
            DC loss value
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        # Compute DC loss per batch and channel
        batch_size, num_channels = input.shape[:2]
        losses = torch.zeros(
            batch_size, num_channels, device=input.device, dtype=input.dtype
        )

        for b in range(batch_size):
            for c in range(num_channels):
                # Get predictions and targets for this batch and channel
                q = input[b, c].flatten()
                y = target[b, c].flatten()

                # Clamp to avoid numerical issues
                eps = float(torch.finfo(q.dtype).eps)
                q = q.clamp(eps, 1 - eps)

                # Create stratified points
                r = (
                    torch.arange(self.n_points, device=q.device, dtype=q.dtype) + 0.5
                ) / self.n_points
                r = r.view(1, -1)

                q_exp = q.view(-1, 1)
                y_exp = y.view(-1, 1)

                # Generate empirical distribution based on true labels
                # For y=0: sample in [0, 1-q], for y=1: sample in [1-q, 1]
                u = torch.where(y_exp <= 0.5, r * (1 - q_exp), (1 - q_exp) + r * q_exp)
                u = u.clamp(eps, 1 - eps)
                logit_s = torch.log(u) - torch.log1p(-u)

                # Generate theoretical logistic distribution
                logistic = self._stratified_logistic(
                    logit_s.shape, device=q.device, dtype=q.dtype
                )
                logistic, _ = torch.sort(logistic.flatten())
                logit_s, _ = torch.sort(logit_s.flatten())

                # Compute L1 distance
                losses[b, c] = (logit_s - logistic).abs().mean()

        # Apply reduction
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(losses)
        elif self.reduction == LossReduction.SUM.value:
            return torch.sum(losses)
        elif self.reduction == LossReduction.NONE.value:
            # Maintain broadcastable shape
            broadcast_shape = list(losses.shape[0:2]) + [1] * (len(input.shape) - 2)
            return losses.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )
