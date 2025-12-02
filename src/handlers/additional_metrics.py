from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Callable, Sequence

from ..metrics.additional_metrics import (
    BrierScore,
    CategoricalNLL,
    AURC,
)
from monai.utils.enums import MetricReduction
from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.handlers.utils import write_metrics_reports
from monai.metrics.utils import ignore_background
from monai.handlers import decollate_batch

from monai.config.deviceconfig import IgniteInfo
from monai.utils.misc import is_scalar, ImageMetaKey
from monai.utils.module import min_version, optional_import

from monai.utils.enums import CommonKeys as Keys
from monai.utils.enums import StrEnum

Events, _ = optional_import(
    "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events"
)

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine",
        IgniteInfo.OPT_IMPORT_VERSION,
        min_version,
        "Engine",
        as_type="decorator",
    )

__all__ = [
    "BrierScoreHandler",
    "CategoricalNLLHandler",
    "AURCHandler",
]


class BrierScoreHandler(IgniteMetricHandler):
    """
    Handler to compute Brier Score during training/validation.

    This handler extends IgniteMetricHandler to compute the Brier score,
    which measures the mean squared difference between predicted probabilities
    and actual target values.
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        top_class: bool = False,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Args:
            is_multilabel: Whether this is a multi-label classification task.
            top_class: If True, computes Brier score for top predicted class only.
            metric_reduction: Mode of reduction to apply to the metrics.
            output_transform: Callable to extract y_pred and y from engine.state.output.
            save_details: Whether to save metric computation details per image.
        """
        metric_fn = BrierScore(
            is_multilabel=is_multilabel,
            top_class=top_class,
            metric_reduction=metric_reduction,
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class CategoricalNLLHandler(IgniteMetricHandler):
    """
    Handler to compute Negative Log-Likelihood (NLL) during training/validation.

    This handler extends IgniteMetricHandler to compute the NLL metric,
    which evaluates the performance of probabilistic classification models.
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Args:
            is_multilabel: Whether this is a multi-label classification task.
            metric_reduction: Mode of reduction to apply to the metrics.
            output_transform: Callable to extract y_pred and y from engine.state.output.
            save_details: Whether to save metric computation details per image.
        """
        metric_fn = CategoricalNLL(
            is_multilabel=is_multilabel,
            metric_reduction=metric_reduction,
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class AURCHandler(IgniteMetricHandler):
    """
    Handler to compute Area Under the Risk-Coverage curve (AURC) during training/validation.

    This handler extends IgniteMetricHandler to compute the AURC metric,
    which evaluates the quality of uncertainty estimates for selective classification.
    """

    def __init__(
        self,
        is_multilabel: bool = False,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """
        Args:
            is_multilabel: Whether this is a multi-label classification task.
            metric_reduction: Mode of reduction to apply to the metrics.
            output_transform: Callable to extract y_pred and y from engine.state.output.
            save_details: Whether to save metric computation details per image.
        """
        metric_fn = AURC(
            is_multilabel=is_multilabel,
            metric_reduction=metric_reduction,
        )

        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )
