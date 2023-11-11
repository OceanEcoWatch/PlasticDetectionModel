import io
import sys

# append code directory to sys.path
sys.path.append("code")

from code.inference import input_fn, model_fn, output_fn, predict_fn
from code.marinedebrisdetector.checkpoints import CHECKPOINTS
from code.marinedebrisdetector.model.segmentation_model import SegmentationModel

import pytest
import rasterio


@pytest.fixture
def input_data():
    with open(
        "tests/data/test_image.tiff",
        "rb",
    ) as f:
        return f.read()


@pytest.fixture
def np_data(input_data):
    with rasterio.open(io.BytesIO(input_data)) as src:
        image = src.read()
        meta = src.meta.copy()
        return src, image, meta


@pytest.fixture
def model():
    detector = SegmentationModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS["unet++1"],
        strict=False,
        map_location="cpu",
    )
    return detector


@pytest.fixture
def expected_prediction():
    with open(
        "tests/data/prediction.tiff",
        "rb",
    ) as f:
        return f.read()


def test_model_fn():
    model = model_fn(". ")
    assert hasattr(model, "predict")


def test_input_fn(input_data):
    image, meta = input_fn(input_data, "application/octet-stream")
    assert image.shape == (13, 500, 250)
    assert meta["height"] == 500
    assert meta["width"] == 250
    assert meta["count"] == 13
    assert meta["dtype"] == "uint16"
    assert meta["driver"] == "GTiff"


def test_input_fn_content_type_error(input_data):
    with pytest.raises(ValueError):
        input_fn(b"test", "application/json")


def test_predict_fn(np_data, model, expected_prediction):
    org_src, org_image, org_meta = np_data
    pred_result = predict_fn(np_data[1:], model=model)
    assert isinstance(pred_result, bytes)
    assert pred_result == expected_prediction
    with io.BytesIO(pred_result) as buffer:
        with rasterio.open(buffer) as src:
            image_data = src.read()
            assert image_data.shape == (1, org_image.shape[1], org_image.shape[2])
            assert src.meta["height"] == org_meta["height"]
            assert src.meta["width"] == org_meta["width"]
            assert src.meta["count"] == 1
            assert src.meta["dtype"] == "uint8"
            assert src.meta["driver"] == "GTiff"

            # check that prediction is in same geographic location as input
            assert src.bounds == org_src.bounds


def test_output_fn(expected_prediction):
    output = output_fn(expected_prediction, "application/octet-stream")
    assert isinstance(output, bytes)


def test_output_fn_content_type_error(expected_prediction):
    with pytest.raises(ValueError):
        output_fn(expected_prediction, "application/json")
