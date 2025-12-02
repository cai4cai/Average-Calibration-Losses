"""
Tests for additional metrics (AURC, NLL, Brier Score) to ensure they return
correct shapes and produce sensible values.
"""

import pytest
import torch
from monai.utils.enums import MetricReduction

from src.metrics.additional_metrics import BrierScore, CategoricalNLL, AURC

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(params=DEVICES, ids=[str(d) for d in DEVICES])
def device(request):
    return request.param


def extract_tensor_from_result(result):
    """Helper function to extract tensor from result (handles both tensor and tuple)."""
    if isinstance(result, tuple):
        return result[0]  # Extract tensor from (tensor, not_nans) tuple
    return result


class TestBrierScore:
    """Test cases for BrierScore metric."""

    def test_brier_score_shape(self, device):
        """Test that BrierScore returns correct shape [B, C]."""
        metric = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Test data: batch_size=2, num_classes=4, spatial_dims=8x8
        y_pred = torch.softmax(torch.randn(2, 4, 8, 8, device=device), dim=1)
        y_true = torch.randint(0, 4, (2, 1, 8, 8), device=device)

        # Convert to one-hot for Brier score
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should return [B, C] = [2, 4] shape
        assert result_tensor.shape == (
            2,
            4,
        ), f"Expected shape (2, 4), got {result_tensor.shape}"
        assert result_tensor.device.type == device.type

    def test_brier_score_perfect_predictions(self, device):
        """Test BrierScore with perfect predictions (should be 0)."""
        metric = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Create perfect predictions
        batch_size, num_classes = 1, 3
        spatial_size = (4, 4)

        # Create ground truth
        y_true = torch.randint(
            0, num_classes, (batch_size, 1, *spatial_size), device=device
        )

        # Create perfect predictions (one-hot matching ground truth)
        y_pred = torch.zeros(batch_size, num_classes, *spatial_size, device=device)
        y_pred.scatter_(1, y_true, 1.0)  # Perfect one-hot predictions

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Perfect predictions should give Brier score of 0
        assert torch.allclose(
            result_tensor, torch.zeros_like(result_tensor), atol=1e-6
        ), f"Perfect predictions should give Brier score ~0, got {result_tensor}"

    def test_brier_score_value_range(self, device):
        """Test BrierScore produces values in expected range."""
        metric = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Random predictions
        y_pred = torch.softmax(torch.randn(2, 3, 6, 6, device=device), dim=1)
        y_true = torch.randint(0, 3, (2, 1, 6, 6), device=device)

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Brier score should be non-negative and <= 2 (max for binary case)
        assert torch.all(
            result_tensor >= 0
        ), f"Brier score should be non-negative, got {result_tensor}"
        assert torch.all(
            result_tensor <= 2.0
        ), f"Brier score should be <= 2, got {result_tensor}"

    def test_brier_score_multilabel(self, device):
        """Test BrierScore with multi-label data."""
        metric = BrierScore(is_multilabel=True, metric_reduction=MetricReduction.NONE)

        # Multi-label: each channel is independent binary classification
        y_pred = torch.sigmoid(torch.randn(1, 3, 4, 4, device=device))
        y_true = torch.randint(0, 2, (1, 3, 4, 4), device=device).float()

        metric(y_pred, y_true)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should return [B, C] = [1, 3]
        assert result_tensor.shape == (
            1,
            3,
        ), f"Expected shape (1, 3), got {result_tensor.shape}"
        assert torch.all(result_tensor >= 0), "Brier score should be non-negative"


class TestCategoricalNLL:
    """Test cases for CategoricalNLL metric."""

    def test_nll_shape(self, device):
        """Test that CategoricalNLL returns correct shape [B, C]."""
        metric = CategoricalNLL(
            is_multilabel=False, metric_reduction=MetricReduction.NONE
        )

        # Test data: batch_size=3, num_classes=5, spatial_dims=6x6
        y_pred = torch.softmax(torch.randn(3, 5, 6, 6, device=device), dim=1)
        y_true = torch.randint(0, 5, (3, 1, 6, 6), device=device)

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should return [B, C] = [3, 5] shape
        assert result_tensor.shape == (
            3,
            5,
        ), f"Expected shape (3, 5), got {result_tensor.shape}"
        assert result_tensor.device.type == device.type

    def test_nll_perfect_predictions(self, device):
        """Test NLL with perfect predictions (should be close to 0)."""
        metric = CategoricalNLL(
            is_multilabel=False, metric_reduction=MetricReduction.NONE
        )

        # Create perfect predictions
        batch_size, num_classes = 1, 2
        spatial_size = (4, 4)

        # Create ground truth
        y_true = torch.randint(
            0, num_classes, (batch_size, 1, *spatial_size), device=device
        )

        # Create perfect predictions
        y_pred = torch.zeros(batch_size, num_classes, *spatial_size, device=device)
        y_pred.scatter_(1, y_true, 1.0)  # Perfect one-hot predictions

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Perfect predictions should give low NLL
        assert torch.all(
            result_tensor >= 0
        ), f"NLL should be non-negative, got {result_tensor}"

    def test_nll_uniform_predictions(self, device):
        """Test NLL with uniform predictions."""
        metric = CategoricalNLL(
            is_multilabel=False, metric_reduction=MetricReduction.NONE
        )

        # Uniform predictions for 4 classes
        num_classes = 4
        y_true = torch.randint(0, num_classes, (1, 1, 4, 4), device=device)
        y_pred = torch.full(
            (1, num_classes, 4, 4), 1.0 / num_classes, device=device
        )  # uniform probability

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should be valid NLL values
        assert torch.all(
            result_tensor >= 0
        ), f"NLL should be non-negative, got {result_tensor}"
        assert torch.all(
            torch.isfinite(result_tensor)
        ), f"NLL should be finite, got {result_tensor}"

    def test_nll_multilabel(self, device):
        """Test NLL with multi-label data."""
        metric = CategoricalNLL(
            is_multilabel=True, metric_reduction=MetricReduction.NONE
        )

        # Multi-label: independent binary per channel
        y_pred = torch.sigmoid(torch.randn(2, 4, 3, 3, device=device))
        y_true = torch.randint(0, 2, (2, 4, 3, 3), device=device).float()

        metric(y_pred, y_true)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should return [B, C] = [2, 4]
        assert result_tensor.shape == (
            2,
            4,
        ), f"Expected shape (2, 4), got {result_tensor.shape}"
        assert torch.all(result_tensor >= 0), "NLL should be non-negative"


class TestAURC:
    """Test cases for AURC metric."""

    def test_aurc_shape(self, device):
        """Test that AURC returns correct shape [B, C]."""
        metric = AURC(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Test data: batch_size=2, num_classes=3, spatial_dims=5x5
        y_pred = torch.softmax(torch.randn(2, 3, 5, 5, device=device), dim=1)
        y_true = torch.randint(0, 3, (2, 1, 5, 5), device=device)

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should return [B, C] = [2, 3] shape
        assert result_tensor.shape == (
            2,
            3,
        ), f"Expected shape (2, 3), got {result_tensor.shape}"
        assert result_tensor.device.type == device.type

    def test_aurc_perfect_predictions(self, device):
        """Test AURC with perfect predictions (should be close to 0)."""
        metric = AURC(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Create perfect predictions
        batch_size, num_classes = 1, 2
        spatial_size = (6, 6)

        # Create ground truth
        y_true = torch.randint(
            0, num_classes, (batch_size, 1, *spatial_size), device=device
        )

        # Create perfect predictions
        y_pred = torch.zeros(batch_size, num_classes, *spatial_size, device=device)
        y_pred.scatter_(1, y_true, 1.0)  # Perfect one-hot predictions

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Perfect predictions should give good AURC (low values)
        assert torch.all(
            result_tensor >= 0
        ), f"AURC should be non-negative, got {result_tensor}"
        assert torch.all(
            result_tensor <= 1
        ), f"AURC should be <= 1, got {result_tensor}"

    def test_aurc_random_predictions(self, device):
        """Test AURC with random predictions."""
        metric = AURC(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Random predictions
        y_pred = torch.softmax(torch.randn(1, 3, 8, 8, device=device), dim=1)
        y_true = torch.randint(0, 3, (1, 1, 8, 8), device=device)

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        metric(y_pred, y_true_onehot)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # AURC should be in [0, 1] range
        assert torch.all(
            result_tensor >= 0
        ), f"AURC should be non-negative, got {result_tensor}"
        assert torch.all(
            result_tensor <= 1
        ), f"AURC should be <= 1, got {result_tensor}"

    def test_aurc_multilabel(self, device):
        """Test AURC with multi-label data."""
        metric = AURC(is_multilabel=True, metric_reduction=MetricReduction.NONE)

        # Multi-label: independent binary per channel
        y_pred = torch.sigmoid(torch.randn(1, 4, 6, 6, device=device))
        y_true = torch.randint(0, 2, (1, 4, 6, 6), device=device).float()

        metric(y_pred, y_true)
        result = metric.aggregate()
        result_tensor = extract_tensor_from_result(result)

        # Should return [B, C] = [1, 4]
        assert result_tensor.shape == (
            1,
            4,
        ), f"Expected shape (1, 4), got {result_tensor.shape}"


class TestAdditionalMetricsIntegration:
    """Integration tests for all additional metrics together."""

    def test_all_metrics_same_input(self, device):
        """Test all metrics with the same input data."""
        # Create test data
        batch_size, num_classes = 2, 4
        spatial_size = (8, 8)

        y_pred = torch.softmax(
            torch.randn(batch_size, num_classes, *spatial_size, device=device), dim=1
        )
        y_true = torch.randint(
            0, num_classes, (batch_size, 1, *spatial_size), device=device
        )

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        # Initialize all metrics
        brier = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)
        nll = CategoricalNLL(is_multilabel=False, metric_reduction=MetricReduction.NONE)
        aurc = AURC(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # Compute all metrics
        brier(y_pred, y_true_onehot)
        nll(y_pred, y_true_onehot)
        aurc(y_pred, y_true_onehot)

        # Get results
        brier_result = extract_tensor_from_result(brier.aggregate())
        nll_result = extract_tensor_from_result(nll.aggregate())
        aurc_result = extract_tensor_from_result(aurc.aggregate())

        # Check shapes
        expected_shape = (batch_size, num_classes)
        assert (
            brier_result.shape == expected_shape
        ), f"Brier shape: expected {expected_shape}, got {brier_result.shape}"
        assert (
            nll_result.shape == expected_shape
        ), f"NLL shape: expected {expected_shape}, got {nll_result.shape}"
        assert (
            aurc_result.shape == expected_shape
        ), f"AURC shape: expected {expected_shape}, got {aurc_result.shape}"

        # Check devices
        assert brier_result.device.type == device.type
        assert nll_result.device.type == device.type
        assert aurc_result.device.type == device.type

        # Check value ranges
        assert torch.all(brier_result >= 0), "Brier score should be non-negative"
        assert torch.all(nll_result >= 0), "NLL should be non-negative"
        assert torch.all(aurc_result >= 0), "AURC should be non-negative"
        assert torch.all(aurc_result <= 1), "AURC should be <= 1"

    def test_metrics_reset_functionality(self, device):
        """Test that metrics reset properly between computations."""
        metric = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        # First computation
        y_pred1 = torch.softmax(torch.randn(1, 3, 4, 4, device=device), dim=1)
        y_true1 = torch.randint(0, 3, (1, 1, 4, 4), device=device)
        y_true1_onehot = torch.zeros_like(y_pred1)
        y_true1_onehot.scatter_(1, y_true1, 1)

        metric(y_pred1, y_true1_onehot)
        result1 = extract_tensor_from_result(metric.aggregate())

        # Reset and second computation
        metric.reset()
        y_pred2 = torch.softmax(torch.randn(1, 3, 4, 4, device=device), dim=1)
        y_true2 = torch.randint(0, 3, (1, 1, 4, 4), device=device)
        y_true2_onehot = torch.zeros_like(y_pred2)
        y_true2_onehot.scatter_(1, y_true2, 1)

        metric(y_pred2, y_true2_onehot)
        result2 = extract_tensor_from_result(metric.aggregate())

        # Results should be different (very unlikely to be exactly the same)
        assert not torch.allclose(
            result1, result2, atol=1e-6
        ), "Results should be different after reset"

    def test_metrics_with_3d_data(self, device):
        """Test metrics with 3D spatial data."""
        batch_size, num_classes = 1, 3
        spatial_size = (4, 4, 4)  # 3D

        y_pred = torch.softmax(
            torch.randn(batch_size, num_classes, *spatial_size, device=device), dim=1
        )
        y_true = torch.randint(
            0, num_classes, (batch_size, 1, *spatial_size), device=device
        )

        # Convert to one-hot
        y_true_onehot = torch.zeros_like(y_pred)
        y_true_onehot.scatter_(1, y_true, 1)

        # Test all metrics with 3D data
        brier = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)
        nll = CategoricalNLL(is_multilabel=False, metric_reduction=MetricReduction.NONE)
        aurc = AURC(is_multilabel=False, metric_reduction=MetricReduction.NONE)

        brier(y_pred, y_true_onehot)
        nll(y_pred, y_true_onehot)
        aurc(y_pred, y_true_onehot)

        brier_result = extract_tensor_from_result(brier.aggregate())
        nll_result = extract_tensor_from_result(nll.aggregate())
        aurc_result = extract_tensor_from_result(aurc.aggregate())

        # All should return [B, C] shape regardless of spatial dimensions
        expected_shape = (batch_size, num_classes)
        assert brier_result.shape == expected_shape
        assert nll_result.shape == expected_shape
        assert aurc_result.shape == expected_shape


def test_metric_sanity_values():
    """Test that metrics produce sensible values for known cases."""
    device = torch.device("cpu")

    # Test case: 2 classes, simple predictions
    y_true = torch.tensor([[[[0, 1], [1, 0]]]], device=device)  # Ground truth indices

    # Case 1: Perfect predictions (should give low scores)
    y_pred_perfect = torch.zeros(1, 2, 2, 2, device=device)
    y_pred_perfect[0, 0, 0, 0] = 1.0  # Perfect for class 0
    y_pred_perfect[0, 1, 0, 1] = 1.0  # Perfect for class 1
    y_pred_perfect[0, 1, 1, 0] = 1.0  # Perfect for class 1
    y_pred_perfect[0, 0, 1, 1] = 1.0  # Perfect for class 0

    # Convert to one-hot
    y_true_onehot = torch.zeros_like(y_pred_perfect)
    y_true_onehot.scatter_(1, y_true, 1)

    # Test Brier score with perfect predictions
    brier = BrierScore(is_multilabel=False, metric_reduction=MetricReduction.NONE)
    brier(y_pred_perfect, y_true_onehot)
    brier_perfect = extract_tensor_from_result(brier.aggregate())

    # Test NLL with perfect predictions
    nll = CategoricalNLL(is_multilabel=False, metric_reduction=MetricReduction.NONE)
    nll(y_pred_perfect, y_true_onehot)
    nll_perfect = extract_tensor_from_result(nll.aggregate())

    # Perfect predictions should give very low scores
    assert torch.all(
        brier_perfect < 0.1
    ), f"Perfect Brier should be low, got {brier_perfect}"
    assert torch.all(nll_perfect < 0.1), f"Perfect NLL should be low, got {nll_perfect}"


if __name__ == "__main__":
    pytest.main([__file__])
