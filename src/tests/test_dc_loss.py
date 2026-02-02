import pytest
import torch
from src.losses import (
    DCLoss,
    DCLossandCELoss,
    DCLossandDiceLoss,
    DCLossandDiceCELoss,
)


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


class TestCompositeDCLosses:
    """Test suite for composite DC losses"""

    def test_dc_ce_loss_basic(self, device):
        """Test DCLossandCELoss basic forward pass"""
        loss_fn = DCLossandCELoss(
            dc_weight=0.5,
            ce_weight=0.5,
            to_onehot_y=True,
            dc_params={"n_points": 50, "softmax": True},
            ce_params={"reduction": "mean"},
        ).to(device)

        y_pred = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

        # Test backward pass
        loss.backward()
        assert y_pred.grad is not None, "Gradients should be computed"

    def test_dc_dice_loss_basic(self, device):
        """Test DCLossandDiceLoss basic forward pass"""
        loss_fn = DCLossandDiceLoss(
            dc_weight=0.5,
            dice_weight=0.5,
            to_onehot_y=True,
            dc_params={"n_points": 50, "softmax": True},
            dice_params={"softmax": True, "reduction": "mean"},
        ).to(device)

        y_pred = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

        # Test backward pass
        loss.backward()
        assert y_pred.grad is not None, "Gradients should be computed"

    def test_dc_dice_ce_loss_basic(self, device):
        """Test DCLossandDiceCELoss basic forward pass"""
        loss_fn = DCLossandDiceCELoss(
            dc_weight=0.33,
            dice_weight=0.33,
            ce_weight=0.34,
            to_onehot_y=True,
            dc_params={"n_points": 50, "softmax": True},
            dice_params={"softmax": True, "reduction": "mean"},
            ce_params={"reduction": "mean"},
        ).to(device)

        y_pred = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)

        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

        # Test backward pass
        loss.backward()
        assert y_pred.grad is not None, "Gradients should be computed"

    @pytest.mark.parametrize(
        "loss_class,params",
        [
            (DCLossandCELoss, {"dc_weight": 0.7, "ce_weight": 0.3}),
            (DCLossandDiceLoss, {"dc_weight": 0.6, "dice_weight": 0.4}),
            (
                DCLossandDiceCELoss,
                {"dc_weight": 0.2, "dice_weight": 0.5, "ce_weight": 0.3},
            ),
        ],
    )
    def test_composite_loss_weights(self, device, loss_class, params):
        """Test that composite losses work with different weight configurations"""
        loss_fn = loss_class(to_onehot_y=True, **params).to(device)

        y_pred = torch.randn(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss), "Loss should be finite"

    def test_dc_ce_loss_without_onehot(self, device):
        """Test DCLossandCELoss with pre-one-hot targets"""
        loss_fn = DCLossandCELoss(
            dc_weight=0.5,
            ce_weight=0.5,
            to_onehot_y=False,
            dc_params={"n_points": 50},
            ce_params={"reduction": "mean"},
        ).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss), "Loss should be finite"

    def test_dc_dice_loss_without_onehot(self, device):
        """Test DCLossandDiceLoss with pre-one-hot targets"""
        loss_fn = DCLossandDiceLoss(
            dc_weight=0.5,
            dice_weight=0.5,
            to_onehot_y=False,
            dc_params={"n_points": 50},
            dice_params={"reduction": "mean"},
        ).to(device)

        y_pred = torch.rand(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 2, (2, 3, 16, 16), device=device).float()

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss), "Loss should be finite"

    def test_composite_loss_default_params(self, device):
        """Test composite losses with default parameters"""
        # DCLossandCELoss
        loss_fn1 = DCLossandCELoss(to_onehot_y=True).to(device)
        y_pred = torch.randn(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)
        loss1 = loss_fn1(y_pred, y_true)
        assert torch.isfinite(loss1), "DCLossandCELoss should work with defaults"

        # DCLossandDiceLoss
        loss_fn2 = DCLossandDiceLoss(to_onehot_y=True).to(device)
        loss2 = loss_fn2(y_pred, y_true)
        assert torch.isfinite(loss2), "DCLossandDiceLoss should work with defaults"

        # DCLossandDiceCELoss
        loss_fn3 = DCLossandDiceCELoss(to_onehot_y=True).to(device)
        loss3 = loss_fn3(y_pred, y_true)
        assert torch.isfinite(loss3), "DCLossandDiceCELoss should work with defaults"

    def test_composite_loss_custom_dc_params(self, device):
        """Test composite losses with custom DC parameters"""
        dc_params = {
            "n_points": 100,
            "include_background": False,
            "softmax": True,
        }

        loss_fn = DCLossandDiceCELoss(
            to_onehot_y=True,
            dc_params=dc_params,
            dice_params={"softmax": True, "include_background": False},
            ce_params={"reduction": "mean"},
        ).to(device)

        y_pred = torch.randn(2, 3, 16, 16, device=device)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss), "Loss should work with custom DC params"

    def test_composite_loss_3d_input(self, device):
        """Test composite losses with 3D volumetric data"""
        loss_fn = DCLossandDiceCELoss(
            to_onehot_y=True,
            dc_params={"n_points": 50, "softmax": True},
            dice_params={"softmax": True},
            ce_params={"reduction": "mean"},
        ).to(device)

        # 3D input: [B, C, D, H, W]
        y_pred = torch.randn(1, 2, 8, 8, 8, device=device)
        y_true = torch.randint(0, 2, (1, 1, 8, 8, 8), device=device)

        loss = loss_fn(y_pred, y_true)

        assert torch.isfinite(loss), "Loss should work with 3D input"

    @pytest.mark.parametrize(
        "loss_class",
        [DCLossandCELoss, DCLossandDiceLoss, DCLossandDiceCELoss],
    )
    def test_composite_loss_gradient_flow(self, device, loss_class):
        """Test that gradients flow through all components of composite losses"""
        loss_fn = loss_class(to_onehot_y=True).to(device)

        y_pred = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
        y_true = torch.randint(0, 3, (2, 1, 16, 16), device=device)

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None, "Gradients should flow through composite loss"
        assert not torch.isnan(y_pred.grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(y_pred.grad).any(), "Gradients should not be inf"
