import io

import boto3
import pytest
import rasterio

from config import CONTENT_TYPE, ENDPOINT_NAME
from tests.conftest import MSE_THRESHOLD
from tests.utils import mse


@pytest.mark.integration
def test_invoke_endpoint(input_data, expected_prediction):
    runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=CONTENT_TYPE,
        Body=input_data,
        Accept=CONTENT_TYPE,
    )
    predictions = response["Body"].read()

    with rasterio.open(io.BytesIO(predictions)) as src:
        pred_image = src.read()
    with rasterio.open(io.BytesIO(expected_prediction)) as src:
        expected_image = src.read()
    assert pred_image.shape == expected_image.shape
    assert pred_image.dtype == expected_image.dtype
    assert mse(pred_image, expected_image) < MSE_THRESHOLD
