import base64
import io
import json

import numpy as np
import pytest
import rasterio

from src.config import ENDPOINT


@pytest.mark.e2e
def test_invoke(input_data: bytes, expected_prediction: bytes):
    with rasterio.open(io.BytesIO(input_data)) as src:
        meta = src.meta.copy()

    with rasterio.open(io.BytesIO(expected_prediction)) as src:
        expected_numpy = src.read()

    encoded_input = base64.b64encode(input_data).decode("utf-8")

    run_request = ENDPOINT.run_sync(
        request_input={"input": {"image": encoded_input}},
        timeout=60,
    )

    np_buffer = base64.b64decode(json.loads(run_request)["prediction"])
    prediction = np.frombuffer(np_buffer, dtype=np.float32).reshape(
        1, meta["height"], meta["width"]
    )
    meta.update(
        dtype="float32",
        count=1,
    )
    with rasterio.open("tests/data/test_invoke.tiff", "w+", **meta) as dst:
        dst.write(prediction)

    np.testing.assert_array_almost_equal(prediction, expected_numpy, decimal=3)
