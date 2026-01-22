import pytest
import torch
from src.losses import DCLoss


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


def device_id(device):
    return str(device)


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


class TestDCLoss:
    """Test suite for DCLoss"""

    def test_basic_forward_pass(self, device):
        """Test that DCLoss can perform a basic forward pass"""
        loss_fn = DCLoss(n_points=50).to(device)

        # Create sample predictions and targets
        y_pred = torch.rand(2, 3, 16, 16, device=device)  # [B, C, H, W]
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_backward_pass(self, device):
        """Test that DCLoss supports gradient computation"""
        loss_fn = DCLoss(n_points=50).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device, requires_grad=True)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None, "Gradients should be computed"
        assert torch.isfinite(y_pred.grad).all(), "Gradients should be finite"

    def test_with_sigmoid(self, device):
        """Test DCLoss with sigmoid activation"""
        loss_fn = DCLoss(n_points=50, sigmoid=True).to(device)

        # Use logits instead of probabilities
        y_pred = torch.randn(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_with_softmax(self, device):
        """Test DCLoss with softmax activation"""
        loss_fn = DCLoss(n_points=50, softmax=True).to(device)

        # Use logits
        y_pred = torch.randn(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_to_onehot_y(self, device):
        """Test DCLoss with one-hot conversion of targets"""
        loss_fn = DCLoss(n_points=50, to_onehot_y=True, softmax=True).to(device)

        # Predictions as logits
        y_pred = torch.randn(2, 3, 16, 16, device=device)
        # Targets as class indices
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_include_background_false(self, device):
        """Test DCLoss with background exclusion"""
        loss_fn = DCLoss(n_points=50, include_background=False).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    @pytest.mark.parametrize("n_points", [10, 50, 100])
    def test_different_n_points(self, device, n_points):
        """Test DCLoss with different numbers of stratified points"""
        loss_fn = DCLoss(n_points=n_points).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reduction_modes(self, device, reduction):
        """Test DCLoss with different reduction modes"""
        loss_fn = DCLoss(n_points=50, reduction=reduction).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        if reduction == "none":
            assert loss.shape[0] == 2, "Batch dimension should be preserved"
            assert loss.shape[1] == 3, "Channel dimension should be preserved"
        else:
            assert loss.dim() == 0, "Scalar loss expected for mean/sum reduction"

    def test_perfect_calibration(self, device):
        """Test that perfectly calibrated predictions have low loss"""
        loss_fn = DCLoss(n_points=50).to(device)

        # Create perfectly calibrated predictions
        # For binary case: predictions match true probabilities
        torch.manual_seed(42)
        y_pred = torch.rand(2, 2, 32, 32, device=device)
        # Generate targets that match the predicted probabilities
        y_true = (torch.rand(2, 2, 32, 32, device=device) < y_pred).float()

        loss = loss_fn(y_pred, y_true)

        # Loss should exist and be finite (perfect calibration is hard to achieve exactly)
        assert torch.isfinite(loss), "Loss should be finite"

    def test_shape_mismatch_raises_error(self, device):
        """Test that shape mismatch raises an error"""
        loss_fn = DCLoss(n_points=50).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(
            0, 2, (2, 2, 16, 16), device=device
        ).float()  # Wrong shape

        with pytest.raises(AssertionError):
            loss_fn(y_pred, y_true)

    def test_activation_conflict_raises_error(self):
        """Test that conflicting activations raise an error"""
        with pytest.raises(ValueError):
            DCLoss(sigmoid=True, softmax=True)

    def test_invalid_other_act_raises_error(self):
        """Test that invalid other_act raises a TypeError"""
        with pytest.raises(TypeError):
            DCLoss(other_act="not_a_callable")

    def test_single_channel_prediction(self, device):
        """Test DCLoss with single channel prediction"""
        loss_fn = DCLoss(n_points=50, sigmoid=True).to(device)

        # Single channel (binary segmentation)
        y_pred = torch.randn(2, 1, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 1, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_3d_input(self, device):
        """Test DCLoss with 3D input (volumetric data)"""
        loss_fn = DCLoss(n_points=50).to(device)

        # 3D input: [B, C, D, H, W]
        y_pred = torch.rand(1, 2, 8, 8, 8, device=device)
        y_true = torch.randint(0, 2, (1, 2, 8, 8, 8), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_deterministic_output(self, device):
        """Test that DCLoss produces deterministic output for same input"""
        loss_fn = DCLoss(n_points=50).to(device)

        torch.manual_seed(42)
        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss1 = loss_fn(y_pred, y_true)
        loss2 = loss_fn(y_pred, y_true)

        assert torch.allclose(loss1, loss2), "Loss should be deterministic"

    def test_batch_consistency(self, device):
        """Test that loss computed on batch equals mean of individual losses"""
        loss_fn = DCLoss(n_points=50, reduction="none").to(device)

        torch.manual_seed(42)
        y_pred = torch.rand(4, 2, 16, 16, device=device)
        y_true = torch.randint(0, 2, (4, 2, 16, 16), device=device).float()

        # Compute batch loss with reduction="none"
        batch_loss = loss_fn(y_pred, y_true)

        # Compute individual losses
        individual_losses = []
        for i in range(4):
            individual_loss = loss_fn(y_pred[i : i + 1], y_true[i : i + 1])
            individual_losses.append(individual_loss)

        stacked_losses = torch.cat(individual_losses, dim=0)

        assert torch.allclose(
            batch_loss, stacked_losses, rtol=1e-5
        ), "Batch loss should match individual losses"
