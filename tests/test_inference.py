import numpy as np
import pytest
import rasterio

from sagemaker_model.code.inference import input_fn, model_fn, output_fn, predict_fn
from tests.conftest import TEST_S3_IMAGE_PATH


@pytest.mark.unit
def test_model_fn():
    model = model_fn("sagemaker_model/code")
    assert type(model).__name__ == "SegmentationModel"


@pytest.mark.unit
def test_input_fn(input_data):
    image = input_fn(input_data, "application/octet-stream")
    assert image.shape == (12, 480, 480)


@pytest.mark.integration
def test_input_fn_json_get_from_s3():
    image = input_fn(TEST_S3_IMAGE_PATH, "application/json")
    assert image.shape == (12, 480, 480)


@pytest.mark.unit
def test_input_fn_content_type_error():
    with pytest.raises(ValueError):
        input_fn(b"test", "application/text")


@pytest.mark.unit
def test_predict_fn(np_data, model, expected_y_score):
    src, org_image, meta = np_data
    y_score = predict_fn(org_image, model=model)
    prediction = y_score.reshape(1, meta["height"], meta["width"])

    meta.update(
        {
            "count": prediction.shape[0],
            "height": prediction.shape[1],
            "width": prediction.shape[2],
            "dtype": prediction.dtype,
        }
    )
    with rasterio.open("tests/data/predict_fn_out.tiff", "w+", **meta) as dst:
        dst.write(prediction)

    np.testing.assert_array_equal(y_score, expected_y_score)


@pytest.mark.unit
def test_output_fn_json(expected_y_score, np_data):
    src, image, meta = np_data

    response = output_fn(expected_y_score, "application/octet-stream")

    prediction = np.frombuffer(response, dtype=np.float32).reshape(
        1, meta["height"], meta["width"]
    )
    meta.update(
        {
            "count": prediction.shape[0],
            "height": prediction.shape[1],
            "width": prediction.shape[2],
            "dtype": prediction.dtype,
        }
    )
    with rasterio.open("tests/data/output_fn_out.tiff", "w+", **meta) as dst:
        dst.write(prediction)

    np.testing.assert_almost_equal(prediction, expected_y_score, decimal=6)


@pytest.mark.unit
def test_output_fn_content_type_error(expected_y_score):
    with pytest.raises(ValueError):
        output_fn(expected_y_score, "application/xml")
