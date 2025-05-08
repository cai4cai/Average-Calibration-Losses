import pytest
import unittest.mock as mock
import torch
from sacros.losses import (
    CrossEntropyLoss,
    hard_binned_calibration,
    HardL1ACELoss,
    HardL1ACEandCELoss,
    HardL1ACEandDiceLoss,
    HardL1ACEandDiceCELoss,
    soft_binned_calibration,
    SoftL1ACELoss,
    SoftL1ACEandCELoss,
    SoftL1ACEandDiceLoss,
    SoftL1ACEandDiceCELoss,
)

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


# Define Test Names
def loss_id(loss_tuple):
    loss_class, _ = loss_tuple
    return loss_class.__name__


def device_id(device):
    return str(device)


# Define Fixtures
@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture
def ce_params():
    return {"reduction": "mean"}


@pytest.fixture
def ace_params():
    return {
        "num_bins": 15,
        "include_background": True,
        "softmax": True,
        "reduction": "mean",
    }


@pytest.fixture
def dice_params():
    return {"include_background": True, "softmax": True, "reduction": "mean"}


@pytest.fixture(
    params=[
        (CrossEntropyLoss, {}),
        (HardL1ACELoss, {}),
        (HardL1ACEandCELoss, {"ace_weight": 0.5, "ce_weight": 0.5}),
        (HardL1ACEandDiceLoss, {"ace_weight": 0.5, "dice_weight": 0.5}),
        (
            HardL1ACEandDiceCELoss,
            {"ace_weight": 0.33, "dice_weight": 0.33, "ce_weight": 0.33},
        ),
        (SoftL1ACELoss, {}),
        (SoftL1ACEandCELoss, {"ace_weight": 0.5, "ce_weight": 0.5}),
        (SoftL1ACEandDiceLoss, {"ace_weight": 0.5, "dice_weight": 0.5}),
        (
            SoftL1ACEandDiceCELoss,
            {"ace_weight": 0.33, "dice_weight": 0.33, "ce_weight": 0.33},
        ),
    ],
    ids=loss_id,
)
def loss_fn(request, ce_params, ace_params, dice_params):

    loss_class, extra_params = request.param
    params = {"to_onehot_y": True}
    params.update(extra_params)

    if issubclass(loss_class, (CrossEntropyLoss,)):
        params["ce_params"] = ce_params
    elif issubclass(loss_class, (HardL1ACELoss, SoftL1ACELoss)):
        params.update(ace_params)
    elif issubclass(loss_class, (HardL1ACEandCELoss, SoftL1ACEandCELoss)):
        params["ace_params"] = ace_params
        params["ce_params"] = ce_params
    elif issubclass(loss_class, (HardL1ACEandDiceLoss, SoftL1ACEandDiceLoss)):
        params["ace_params"] = ace_params
        params["dice_params"] = dice_params
    elif issubclass(loss_class, (HardL1ACEandDiceCELoss, SoftL1ACEandDiceCELoss)):
        params["ace_params"] = ace_params
        params["dice_params"] = dice_params
        params["ce_params"] = ce_params

    return loss_class(**params)


# Define simple forward and backward pass tests


def test_forward_pass(loss_fn, device):
    y_pred = torch.randn(
        10, 3, 32, 32, device=device
    )  # Example: batch size 10, 3 channels, 32x32 image
    y_true = torch.randint(
        0, 3, (10, 1, 32, 32), device=device
    ).float()  # Random class labels

    loss = loss_fn(y_pred, y_true)
    assert loss > 0


def test_backward_pass(loss_fn, device):
    y_pred = torch.randn(10, 3, 32, 32, device=device, requires_grad=True)
    y_true = torch.randint(
        0, 3, (10, 1, 32, 32), device=device
    ).float()  # Random class labels

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None
    
    
@pytest.mark.parametrize("loss_class", [HardL1ACELoss, SoftL1ACELoss])
@pytest.mark.parametrize("ignore_empty_classes", [True, False])
@pytest.mark.parametrize("device", DEVICES, ids=device_id)
def test_loss_ignore_empty_classes(loss_class, ignore_empty_classes, device):
    """
    Test HardL1ACELoss and SoftL1ACELoss for handling empty classes based on the ignore_empty_classes flag.
    """
    loss_fn = loss_class(num_bins=5, reduction="mean", ignore_empty_classes=ignore_empty_classes).to(device)

    # Predictions
    y_pred = torch.tensor(
        [
            [[[0.8, 0.2], [0.4, 0.6]], [[0.7, 0.1], [0.3, 0.5]]],
        ],
        device=device,
    )
    
    # Ground truth: second class is empty
    y_true = torch.tensor(
        [
            [[[1, 0], [0, 1]], [[0, 0], [0, 0]]],
        ],
        device=device,
    )

    loss = loss_fn(y_pred, y_true)

    if ignore_empty_classes:
        # Loss for the second channel should be 0 when ignoring empty classes
        assert loss.item() > 0, "Loss should ignore empty classes but be non-zero for populated classes."
    else:
        # Loss should account for all classes, including empty
        assert loss.item() > 0, "Loss should include contributions from empty classes when flag is False."

    

@pytest.mark.parametrize("loss_class", [HardL1ACELoss, SoftL1ACELoss])
@pytest.mark.parametrize("device", DEVICES, ids=device_id)
def test_loss_class_weighting(loss_class, device):
    """
    Test that HardL1ACELoss and SoftL1ACELoss correctly apply class weights.
    """
    # Define class weights (second class has higher weight)
    class_weights = [1.0, 2.0]

    # Instantiate the loss function with weights
    loss_fn = loss_class(num_bins=5, reduction="mean", weight=class_weights).to(device)

    # Input tensor: [Batch, Channels, H, W]
    y_pred = torch.tensor(
        [
            [[[0.8, 0.2], [0.4, 0.6]], [[0.7, 0.1], [0.3, 0.5]]],  # Predictions
        ],
        device=device,
    )
    
    # Ground truth tensor
    y_true = torch.tensor(
        [
            [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],  # Ground truth
        ],
        device=device,
    )

    # Compute loss with weights
    weighted_loss = loss_fn(y_pred, y_true).item()

    # Recompute loss with manual scaling
    loss_fn_no_weight = loss_class(num_bins=5, reduction="none").to(device)
    unweighted_loss = loss_fn_no_weight(y_pred, y_true)  # Shape: [Batch, Channels]

    # Manually compute the mean reduction (as reduction="mean" reduces over batch + spatial dims)
    per_class_loss = torch.mean(unweighted_loss, dim=(0, 2, 3))  # Average over spatial dims
    weighted_per_class_loss = per_class_loss * torch.tensor(class_weights, device=device)
    manual_weighted_loss = torch.mean(weighted_per_class_loss)
    
    assert torch.isclose(
        torch.tensor(weighted_loss),
        manual_weighted_loss,
        atol=1e-6
    ), f"Weighted loss {weighted_loss} did not match manually computed loss {manual_weighted_loss.item()}"


# test binning operations:
test_cases_hard_binned = [
    {
        "case_name": "small_mid",  # small case with probabilities in middle of bin
        "input": [[[[0.7, 0.3], [0.1, 0.9]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.1, 0.3, float("nan"), 0.7, 0.9]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, float("nan"), 1.0, 1.0]]],
        "expected_bin_counts": [[[1.0, 1.0, 0.0, 1.0, 1.0]]],
    },
    {
        "case_name": "large_mid",  # larger case with probabilities in middle of bin
        "input": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "target": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [
            [[0.1, 0.3, torch.nan, 0.7, 0.9], [torch.nan, 0.3, 0.5, 0.7, torch.nan]],
            [
                [torch.nan, 0.3, torch.nan, torch.nan, 0.9],
                [0.1, torch.nan, torch.nan, 0.7, 0.9],
            ],
        ],
        "expected_mean_gt_per_bin": [
            [[0.0, 0.0, torch.nan, 1.0, 1.0], [torch.nan, 1.0, 0.5, 0.0, torch.nan]],
            [
                [torch.nan, 0.0, torch.nan, torch.nan, 1.0],
                [0.0, torch.nan, torch.nan, 1.0, 1.0],
            ],
        ],
        "expected_bin_counts": [
            [[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 1.0, 1.0]],
        ],
    },
    {
        "case_name": "small_off",  # small case with probabilities off center of bin
        "input": [[[[0.65, 0.25], [0.15, 0.95]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.15, 0.25, torch.nan, 0.65, 0.95]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, torch.nan, 1.0, 1.0]]],
        "expected_bin_counts": [[[1.0, 1.0, 0.0, 1.0, 1.0]]],
    },
    {
        "case_name": "small_left_edge",  # small case with probabilities on left bin boundaries
        "input": [[[[0.8, 0.2], [0.4, 0.6]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.2, 0.4, 0.6, 0.8, torch.nan]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, 1.0, 1.0, torch.nan]]],
        "expected_bin_counts": [[[1.0, 1.0, 1.0, 1.0, 0.0]]],
    },
    {
        "case_name": "small_right_edge",  # small case with probabilities on right bin boundaries
        "input": [[[[0.8, 0.2], [0.4, 0.6]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": True,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[torch.nan, 0.2, 0.4, 0.6, 0.8]]],
        "expected_mean_gt_per_bin": [[[torch.nan, 0.0, 0.0, 1.0, 1.0]]],
        "expected_bin_counts": [[[0.0, 1.0, 1.0, 1.0, 1.0]]],
    },
]


@pytest.mark.parametrize(
    "case", test_cases_hard_binned, ids=[c["case_name"] for c in test_cases_hard_binned]
)
def test_hard_binned_calibration(device, case):

    input_tensor = torch.tensor(case["input"], device=device)
    target_tensor = torch.tensor(case["target"], device=device)

    # Use mock.patch to replace torch.linspace
    # This is to avoid floating point precision issues when looking at edge conditions
    mock_boundaries = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device)
    with mock.patch("torch.linspace", return_value=mock_boundaries):
        mean_p_per_bin, mean_gt_per_bin, bin_counts = hard_binned_calibration(
            input_tensor, target_tensor, num_bins=case["num_bins"], right=case["right"]
        )

    expected_mean_p_per_bin = torch.tensor(
        case["expected_mean_p_per_bin"], device=device
    )
    expected_mean_gt_per_bin = torch.tensor(
        case["expected_mean_gt_per_bin"], device=device
    )
    expected_bin_counts = torch.tensor(case["expected_bin_counts"], device=device)

    assert torch.allclose(mean_p_per_bin, expected_mean_p_per_bin, equal_nan=True)
    assert torch.allclose(mean_gt_per_bin, expected_mean_gt_per_bin, equal_nan=True)
    assert torch.allclose(bin_counts, expected_bin_counts)


test_cases_soft_binned = [
    {
        "case_name": "small_mid",  # small case with probabilities in middle of bin
        "input": [[[[0.7, 0.3], [0.1, 0.9]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.1, 0.3, torch.nan, 0.7, 0.9]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, torch.nan, 1.0, 1.0]]],
        "expected_bin_counts": [[[1.0, 1.0, 0.0, 1.0, 1.0]]],
    },
    {
        "case_name": "small_off",  # small case with probabilities off center of bin
        "input": [[[[0.65, 0.25], [0.15, 0.95]]]],
        "target": [[[[1, 0], [0, 1]]]],
        "right": False,
        "num_bins": 5,
        "expected_mean_p_per_bin": [[[0.175, 0.225, 0.650, 0.650, 0.950]]],
        "expected_mean_gt_per_bin": [[[0.0, 0.0, 1.0, 1.0, 1.0]]],
        "expected_bin_counts": [[[1.0, 1.0, 0.25, 0.75, 1.0]]],
    },
]


@pytest.mark.parametrize(
    "case", test_cases_soft_binned, ids=[c["case_name"] for c in test_cases_soft_binned]
)
def test_soft_binned_calibration(device, case):

    input_tensor = torch.tensor(case["input"], device=device)
    target_tensor = torch.tensor(case["target"], device=device)

    # Use mock.patch to replace torch.linspace
    # This is to avoid floating point precision issues when looking at edge conditions
    mock_half_boundaries = torch.tensor(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], device=device
    )
    with mock.patch("torch.linspace", return_value=mock_half_boundaries):
        mean_p_per_bin, mean_gt_per_bin, bin_counts = soft_binned_calibration(
            input_tensor, target_tensor, num_bins=case["num_bins"], right=case["right"]
        )

    expected_mean_p_per_bin = torch.tensor(
        case["expected_mean_p_per_bin"], device=device
    )
    expected_mean_gt_per_bin = torch.tensor(
        case["expected_mean_gt_per_bin"], device=device
    )
    expected_bin_counts = torch.tensor(case["expected_bin_counts"], device=device)

    assert torch.allclose(mean_p_per_bin, expected_mean_p_per_bin, equal_nan=True)
    assert torch.allclose(mean_gt_per_bin, expected_mean_gt_per_bin, equal_nan=True)
    assert torch.allclose(bin_counts, expected_bin_counts)
