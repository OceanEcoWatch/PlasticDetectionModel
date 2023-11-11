import io
from code.inference import input_fn, model_fn, output_fn, predict_fn

import pytest
import rasterio

from tests.conftest import MSE_THRESHOLD
from tests.utils import mse


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

    with io.BytesIO(pred_result) as buffer:
        with rasterio.open(buffer) as src:
            image_data = src.read()
            meta = src.meta.copy()
    assert image_data.shape == (1, org_image.shape[1], org_image.shape[2])
    assert meta["height"] == org_meta["height"]
    assert meta["width"] == org_meta["width"]
    assert meta["count"] == 1
    assert meta["dtype"] == "uint8"
    assert meta["driver"] == "GTiff"

    # check that prediction is in same geographic location as input
    assert src.bounds == org_src.bounds
    assert src.transform == org_src.transform
    assert src.crs == org_src.crs

    # check that prediction is similar to expected prediction
    with rasterio.open(io.BytesIO(expected_prediction)) as src:
        expected_image = src.read()
    assert image_data.shape == expected_image.shape
    assert image_data.dtype == expected_image.dtype

    assert mse(image_data, expected_image) < MSE_THRESHOLD


def test_output_fn(expected_prediction):
    output = output_fn(expected_prediction, "application/octet-stream")
    assert isinstance(output, bytes)


def test_output_fn_content_type_error(expected_prediction):
    with pytest.raises(ValueError):
        output_fn(expected_prediction, "application/json")
