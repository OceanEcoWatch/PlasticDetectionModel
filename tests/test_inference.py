import io

import tempfile
from itertools import product

import numpy as np
import pytest
import rasterio
import requests
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

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
    image = input_fn(input_data, "application/octet-stream")
    assert image.shape == (12, 480, 480)


@pytest.mark.unit
def test_input_fn_content_type_error(input_data):
    with pytest.raises(ValueError):
        input_fn(b"test", "application/json")


@pytest.mark.unit
def test_predict_fn(np_data, model, expected_y_score):
    _, org_image, _ = np_data
    y_score = predict_fn(org_image, model=model)
    assert isinstance(y_score, np.ndarray)

    np.testing.assert_allclose(y_score, expected_y_score, rtol=1e-6)


@pytest.mark.integration
@pytest.mark.slow
def test_predict_fn_with_windowing(model, expected_prediction, input_data):
    pred_path = "tests/data/last_2400_1440_prediction.tiff"
    image_size = (480, 480)
    offset = 64

    with rasterio.open(io.BytesIO(input_data)) as org_src:
        meta = org_src.meta.copy()

    model = model.to("cpu")
    model.eval()
    meta["count"] = 1
    meta["dtype"] = "uint8"

    rows = np.arange(0, meta["height"], image_size[0])
    cols = np.arange(0, meta["width"], image_size[1])
    image_window = Window(0, 0, meta["width"], meta["height"])

    with tempfile.NamedTemporaryFile() as tmpfile:
        with rasterio.open(tmpfile.name, "w+", **meta) as dst:
            for r, c in tqdm(
                product(rows, cols), total=len(rows) * len(cols), leave=False
            ):
                H, W = image_size

                window = image_window.intersection(
                    Window(c - offset, r - offset, W + offset, H + offset)
                )

                with rasterio.open(io.BytesIO(input_data)) as src:
                    image = src.read(window=window)
                b, h, w = image.shape
                if b > 12:
                    image = image[:12]

                H, W = image_size
                H, W = H + offset * 2, W + offset * 2

                dh = (H - h) / 2
                dw = (W - w) / 2
                image = np.pad(
                    image,
                    [
                        (0, 0),
                        (int(np.ceil(dh)), int(np.floor(dh))),
                        (int(np.ceil(dw)), int(np.floor(dw))),
                    ],
                )
                y_score = predict_fn(image, model=model)

                # unpad
                y_score = y_score[
                    int(np.ceil(dh)) : y_score.shape[0] - int(np.floor(dh)),
                    int(np.ceil(dw)) : y_score.shape[1] - int(np.floor(dw)),
                ]
                assert y_score.shape[0] == window.height, "unpadding size mismatch"
                assert y_score.shape[1] == window.width, "unpadding size mismatch"

                data = dst.read(window=window)[0] / 255
                overlap = data > 0

                if overlap.any():
                    # smooth transition in overlapping regions
                    dx, dy = np.gradient(overlap.astype(float))  # get border
                    g = np.abs(dx) + np.abs(dy)
                    transition = gaussian_filter(g, sigma=offset / 2)
                    transition /= transition.max()
                    transition[~overlap] = 1.0  # normalize to 1

                    y_score = transition * y_score + (1 - transition) * data

                writedata = (
                    np.expand_dims(y_score, 0).astype(np.float32) * 255
                ).astype(np.uint8)
                dst.write(writedata, window=window)

        tmpfile.seek(0)
        pred_bytes = tmpfile.read()

    # save prediction to file
    with open(pred_path, "wb") as f:
        f.write(pred_bytes)

    with rasterio.open(io.BytesIO(pred_bytes)) as pred_src:
        pred_image = pred_src.read()
        pred_meta = pred_src.meta.copy()

    # check that prediction is in same dimensions as input
    assert pred_image.shape == (1, meta["height"], meta["width"])
    assert pred_meta["height"] == meta["height"]
    assert pred_meta["width"] == meta["width"]
    assert pred_meta["count"] == 1
    assert pred_meta["dtype"] == "uint8"
    assert pred_meta["driver"] == "GTiff"

    # check that prediction is in same geographic location as input
    assert pred_src.bounds == org_src.bounds
    assert pred_src.transform == org_src.transform
    assert pred_src.crs == org_src.crs

    # compare with marinedebrisdetector original expected prediction
    with rasterio.open(io.BytesIO(expected_prediction)) as exp_src:
        expected_image = exp_src.read()
    assert pred_image.shape == expected_image.shape
    assert pred_image.dtype == expected_image.dtype

    # check that prediction is similar to expected prediction
    assert mse(pred_image, expected_image) < MSE_THRESHOLD


@pytest.mark.unit
def test_output_fn(expected_prediction):
    output = output_fn(expected_prediction, "application/octet-stream")
    assert isinstance(output, bytes)


@pytest.mark.unit
def test_output_fn_content_type_error(expected_prediction):
    with pytest.raises(ValueError):
        output_fn(expected_prediction, "application/json")
