import io

import pytest
import rasterio
import requests

from sagemaker_model.code.inference import input_fn, model_fn, output_fn, predict_fn
from tests.conftest import MSE_THRESHOLD
from tests.utils import mse


def _download_durban():
    url = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/durban_20190424.tif"
    r = requests.get(url)
    return r.content

@pytest.mark.unit
def test_model_fn():
    model = model_fn(". ")
    assert hasattr(model, "predict")


@pytest.mark.unit
def test_input_fn(input_data):
    image, meta = input_fn(input_data, "application/octet-stream")
    assert image.shape == (13, 500, 250)
    assert meta["height"] == 500
    assert meta["width"] == 250
    assert meta["count"] == 13
    assert meta["dtype"] == "uint16"
    assert meta["driver"] == "GTiff"


@pytest.mark.unit
def test_input_fn_content_type_error(input_data):
    with pytest.raises(ValueError):
        input_fn(b"test", "application/json")


@pytest.mark.unit
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

@pytest.mark.unit
@pytest.mark.slow
def test_predict_fn_whole_durban_scene(model):
    expected_pred_path = "tests/data/exp_durban_20190424_prediction.tiff"
    durban_scene = _download_durban()

    with rasterio.open(io.BytesIO(durban_scene)) as org_src:
        org_image = org_src.read()
        org_meta = org_src.meta.copy()

    input_data = input_fn(durban_scene, "application/octet-stream")
    pred_result = predict_fn(input_data, model=model)

    #save prediction to file for later inspection
    with open("tests/data/last_durban_20190424_prediction.tiff", "wb") as f:
        f.write(pred_result)

    with rasterio.open(io.BytesIO(pred_result)) as pred_src:
        image_data = pred_src.read()
        meta = pred_src.meta.copy()

    # check that prediction is in same dimensions as input
    assert image_data.shape == (1, org_image.shape[1], org_image.shape[2])
    assert meta["height"] == org_meta["height"]
    assert meta["width"] == org_meta["width"]
    assert meta["count"] == 1
    assert meta["dtype"] == "uint8"
    assert meta["driver"] == "GTiff"

    # check that prediction is in same geographic location as input
    assert pred_src.bounds == org_src.bounds
    assert pred_src.transform == org_src.transform
    assert pred_src.crs == org_src.crs

    # compare with marinedebrisdetector original expected prediction
    with rasterio.open(expected_pred_path) as exp_src:
        expected_image = exp_src.read()
    assert image_data.shape == expected_image.shape
    assert image_data.dtype == expected_image.dtype

    # check that prediction is similar to expected prediction
    assert mse(image_data, expected_image) < MSE_THRESHOLD




@pytest.mark.unit
def test_output_fn(expected_prediction):
    output = output_fn(expected_prediction, "application/octet-stream")
    assert isinstance(output, bytes)


@pytest.mark.unit
def test_output_fn_content_type_error(expected_prediction):
    with pytest.raises(ValueError):
        output_fn(expected_prediction, "application/json")
