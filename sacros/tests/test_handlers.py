import tempfile
import os
from pathlib import Path
import pytest
import torch
from ignite.engine import Engine, Events

from monai.handlers import from_engine

from sacros.handlers import CalibrationError, ReliabilityDiagramHandler
from sacros.handlers.calibration import BinningDataHandler
from sacros.handlers.calibration import CalibrationErrorHandler
from monai.utils import ImageMetaKey as Key


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(params=DEVICES, ids=[str(d) for d in DEVICES])
def device(request):
    return request.param


test_cases = [
    {
        "case_name": "simple",
        "y_pred": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "y": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "num_bins": 5,
        "include_background": True,
        "calibration_reduction": "expected",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
        "num_iterations": 2,
        "expected_value": [[0.2250]],
    },
    {
        "case_name": "simple_ignore_background",
        "y_pred": [
            [[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]],
            [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]],
        ],
        "y": [
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
            [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
        ],
        "num_bins": 5,
        "include_background": False,
        "calibration_reduction": "expected",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
        "num_iterations": 2,
        "expected_value": [[0.2500]],
    },
]


@pytest.mark.parametrize("case", test_cases, ids=[c["case_name"] for c in test_cases])
def test_calibration_error_handler(device, case):
    y_pred = torch.tensor(case["y_pred"], device=device)
    y = torch.tensor(case["y"], device=device)

    b, c, *_ = y_pred.shape

    c = c if case["include_background"] else c - 1

    handler = CalibrationError(
        num_bins=case["num_bins"],
        include_background=case["include_background"],
        calibration_reduction=case["calibration_reduction"],
        metric_reduction=case["metric_reduction"],
        output_transform=case["output_transform"],
    )

    engine = Engine(lambda e, b: None)
    handler.attach(engine, name="calibration_error")

    for _ in range(case["num_iterations"]):
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)

    engine.fire_event(Events.EPOCH_COMPLETED)

    assert torch.allclose(
        torch.tensor(engine.state.metrics["calibration_error"]),
        torch.tensor(case["expected_value"]),
        atol=1e-4,
    )

    assert engine.state.metric_details["calibration_error"].shape == torch.Size(
        [b * case["num_iterations"], c]
    )


def test_reliability_diagrams(device):

    num_iterations = 4
    num_bins = 20
    shape = (2, 3, 16, 16, 16)
    b, c = shape[:2]

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = ReliabilityDiagramHandler(
            num_classes=c,
            num_bins=num_bins,
            output_dir=temp_dir,
            draw_case_diagrams=True,
            draw_dataset_diagrams=True,
            output_transform=from_engine(["pred", "label"]),
        )

        engine = Engine(lambda e, b: None)
        handler.attach(engine, name="reliability_diagrams")

        for _ in range(num_iterations):
            y_pred = torch.rand(*shape, device=device)
            y = torch.randint(0, 2, shape, device=device)
            engine.state.output = {"pred": y_pred, "label": y}
            engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)

        assert engine.state.metric_details["reliability_diagrams"].shape == torch.Size(
            [b * num_iterations, c, 3, num_bins]
        )

        # Check that temp directory is not empty:
        assert os.listdir(temp_dir) != []


def test_binning_data_handler(device):
    num_iterations = 4
    num_bins = 20
    shape = (2, 3, 16, 16, 16)
    b, c = shape[:2]

    handler = BinningDataHandler(num_bins=num_bins, include_background=True)
    engine = Engine(lambda e, b: None)
    handler.attach(engine)
    
    def _iteration_test(engine):
        binning_data = engine.state.output["iteration_binning"]
        assert binning_data.shape == (b, c, 3, num_bins)
            
    def _epoch_test(engine):
        running_data = engine.state.output["running_binning_data"]
        assert running_data.shape == (num_iterations * b, c, 3, num_bins)
        
        aggregated_data = engine.state.output["aggregated_binning_data"]
        assert aggregated_data.shape == (c, 3, num_bins)

    for _ in range(num_iterations):
        y_pred = torch.rand(shape, device=device)
        y = torch.randint(0, 2, shape, device=device)
        engine.state.output = {"pred": y_pred, "label": y}
        engine.fire_event(Events.ITERATION_COMPLETED)
        _iteration_test(engine)

    engine.fire_event(Events.EPOCH_COMPLETED)
    _epoch_test(engine)


def test_calibration_error_handler(device):
    num_iterations = 4
    num_bins = 5
    shape = (2, 3, 16, 16, 16)
    b, c = shape[:2]

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
        handler = CalibrationErrorHandler(
            num_bins=num_bins,
            include_background=True,
            output_dir=temp_dir,
            # output_transform=from_engine(["pred", "label"]),
            summary_ops="*",
            meta_batch_transform= lambda x: [{Key.FILENAME_OR_OBJ: "image 1"}, {Key.FILENAME_OR_OBJ: "image 2"}],
        )

        engine = Engine(lambda e, b: None)
        handler.attach(engine)

        for _ in range(num_iterations):
            y_pred = torch.rand(*shape, device=device)
            y = torch.randint(0, 2, shape, device=device)
            engine.state.output = [{"pred": y_pred, "label": y}]
            engine.fire_event(Events.ITERATION_COMPLETED)

        engine.fire_event(Events.EPOCH_COMPLETED)

        # Check that the CSV files are created
        assert os.path.exists(os.path.join(temp_dir, "metrics.csv"))
        assert os.path.exists(os.path.join(temp_dir, "micro_ece_raw.csv"))
        assert os.path.exists(os.path.join(temp_dir, "macro_ece_raw.csv"))

        # Check that the metric details are saved correctly
        # TODO: currently not saving the metrics to the engine
        # assert engine.state.metric_details["micro_expected"].shape == torch.Size([num_iterations * b, c])
        # assert engine.state.metric_details["macro_expected"].shape == torch.Size([1, c])
