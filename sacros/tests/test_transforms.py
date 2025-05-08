import pytest
import torch
from monai.transforms import Compose
from sacros.transforms import (
    AverageAnnotationsd,
    RandomWeightedAverageAnnotationsd,
    ConvertToKits23ClassesSoftmaxd,
)

####################################################################################################
# Test cases for AverageAnnotationsd
####################################################################################################


@pytest.mark.parametrize(
    "case",
    [
        {
            "case_name": "average_annotations_simple",
            "input": {
                "label_1": torch.tensor(
                    [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32
                ),
                "label_2": torch.tensor(
                    [[[[2, 3], [4, 5]], [[6, 7], [8, 9]]]], dtype=torch.float32
                ),
                "label_3": torch.tensor(
                    [[[[3, 4], [5, 6]], [[7, 8], [9, 10]]]], dtype=torch.float32
                ),
            },
            "expected_output": torch.tensor(
                [[[[2, 3], [4, 5]], [[6, 7], [8, 9]]]], dtype=torch.float32
            ),
        }
    ],
    ids=["average_annotations_simple"],
)
def test_average_annotationsd(case):
    transform = AverageAnnotationsd(
        keys=["label_1", "label_2", "label_3"], output_key="label"
    )
    result = transform(case["input"])
    torch.allclose(result["label"], case["expected_output"], rtol=1e-4)
    for key in ["label_1", "label_2", "label_3"]:
        assert key not in result


####################################################################################################
# Test cases for RandomWeightedAverageAnnotationsd
####################################################################################################


@pytest.mark.parametrize(
    "case",
    [
        {
            "case_name": "random_weighted_average_annotations_simple",
            "input": {
                "label_1": torch.tensor(
                    [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32
                ),
                "label_2": torch.tensor(
                    [[[[2, 3], [4, 5]], [[6, 7], [8, 9]]]], dtype=torch.float32
                ),
                "label_3": torch.tensor(
                    [[[[3, 4], [5, 6]], [[7, 8], [9, 10]]]], dtype=torch.float32
                ),
            },
        }
    ],
    ids=["random_weighted_average_annotations_simple"],
)
def test_random_weighted_average_annotationsd(case):
    input_data = case["input"]
    expected_shape = input_data[
        "label_1"
    ].shape  # Store the shape before transformation
    transform = RandomWeightedAverageAnnotationsd(
        keys=["label_1", "label_2", "label_3"], output_key="label"
    )
    result = transform(input_data)
    assert "label" in result
    assert result["label"].shape == expected_shape  # Compare with the stored shape
    for key in ["label_1", "label_2", "label_3"]:
        assert key not in result


# Run a test with Compose to verify integration
def test_compose_transform():
    input_data = {
        "label_1": torch.tensor(
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32
        ),
        "label_2": torch.tensor(
            [[[[2, 3], [4, 5]], [[6, 7], [8, 9]]]], dtype=torch.float32
        ),
        "label_3": torch.tensor(
            [[[[3, 4], [5, 6]], [[7, 8], [9, 10]]]], dtype=torch.float32
        ),
        "image": torch.rand((1, 2, 2, 2), dtype=torch.float32),
    }

    transform = Compose(
        [
            AverageAnnotationsd(
                keys=["label_1", "label_2", "label_3"], output_key="label"
            ),
            # Add more transforms as needed
        ]
    )

    result = transform(input_data)
    expected_label = torch.tensor(
        [[[[2, 3], [4, 5]], [[6, 7], [8, 9]]]], dtype=torch.float32
    )
    torch.allclose(result["label"], expected_label, rtol=1e-4)
    for key in ["label_1", "label_2", "label_3"]:
        assert key not in result
    assert "image" in result  # Ensure other data remains untouched


####################################################################################################
# Test cases for ConvertToKits23ClassesSoftmaxd
####################################################################################################


@pytest.fixture
def softmax_data():
    # Creating a dummy tensor of shape [C, H, W, D] with random softmax probabilities
    C, H, W, D = 4, 16, 16, 16  # Example dimensions
    data = torch.randn(C, H, W, D)
    data = torch.softmax(data, dim=0)
    return {"key": data}


@pytest.fixture
def deep_supervision_data():
    # Creating a list of tensors of varying shapes for deep supervision
    shapes = [(4, 32, 32, 32), (4, 16, 16, 16), (4, 8, 8, 8), (4, 4, 4, 4)]
    data = [torch.softmax(torch.randn(shape), dim=0) for shape in shapes]
    return {"key": data}


def test_convert_to_kits23_classes_softmaxd(softmax_data):
    transform = ConvertToKits23ClassesSoftmaxd(keys=["key"])
    transformed = transform(softmax_data)["key"]

    # Check shapes
    assert transformed.shape == (4, 16, 16, 16), "Output shape is not as expected"
    assert transformed.dtype == torch.float32, "Output type should be float32"

    # Verify probabilities are in [0, 1] range
    assert torch.all(
        (transformed >= 0) & (transformed <= 1)
    ), "Probabilities should be between 0 and 1"

    # Verify the transformation logic
    assert torch.all(
        transformed[1, ...] == softmax_data["key"][2, ...]
    ), "Tumor class not matching"
    assert torch.all(
        transformed[2, ...]
        == (softmax_data["key"][2, ...] + softmax_data["key"][3, ...])
    ), "Kidney Mass class not matching"
    assert torch.all(
        transformed[3, ...]
        == (
            softmax_data["key"][1, ...]
            + softmax_data["key"][2, ...]
            + softmax_data["key"][3, ...]
        )
    ), "Kidney and Masses class not matching"


def test_convert_to_kits23_classes_softmaxd_deep_supervision(deep_supervision_data):
    transform = ConvertToKits23ClassesSoftmaxd(keys=["key"])
    transformed_list = transform(deep_supervision_data)["key"]

    assert isinstance(transformed_list, list), "Output should be a list"
    assert len(transformed_list) == 4, "Output list length is not as expected"

    for idx, transformed in enumerate(transformed_list):
        shape = transformed.shape
        assert shape == (
            4,
            shape[1],
            shape[2],
            shape[3],
        ), f"Output shape for level {idx} is not as expected"
        assert (
            transformed.dtype == torch.float32
        ), f"Output type for level {idx} should be float32"

        # Verify probabilities are in [0, 1] range
        assert torch.all(
            (transformed >= 0) & (transformed <= 1)
        ), f"Probabilities for level {idx} should be between 0 and 1"

        # Verify the transformation logic
        original = deep_supervision_data["key"][idx]
        assert torch.all(
            transformed[1, ...] == original[2, ...]
        ), f"Tumor class not matching for level {idx}"
        assert torch.all(
            transformed[2, ...] == (original[2, ...] + original[3, ...])
        ), f"Kidney Mass class not matching for level {idx}"
        assert torch.all(
            transformed[3, ...]
            == (original[1, ...] + original[2, ...] + original[3, ...])
        ), f"Kidney and Masses class not matching for level {idx}"
