import io
import json

import boto3
import numpy as np
import pytest
import rasterio
from moto import mock_sagemaker

from config import (
    CONTENT_TYPE_JSON,
    ENDPOINT_NAME,
    FRAMEWORK,
    FRAMEWORK_VERSION,
    IMAGE_SCOPE,
    MAX_CONCURRENCY,
    MEMORY_SIZE_MB,
    MODEL_SOURCE_DIR,
    PY_VERSION,
    REGION_NAME,
    SAGEMAKER_ROLE,
)
from endpoint.create_endpoint import (
    create_endpoint,
    create_endpoint_config,
    create_model,
    delete_endpoint,
    delete_endpoint_config,
    delete_model,
    wait_endpoint_creation,
)
from endpoint.invoke_endpoint import invoke
from endpoint.upload_model_s3 import upload_model
from tests.conftest import (
    MSE_THRESHOLD,
    TEST_ENDPOINT_CONFIG_NAME,
    TEST_ENDPOINT_NAME,
    TEST_MODEL_NAME,
    TEST_S3_BUCKET_NAME,
    TEST_S3_FILENAME,
    TEST_S3_IMAGE_PATH,
    TEST_S3_MODEL_PATH,
)
from tests.utils import mse


@mock_sagemaker
@pytest.mark.unit
def test_create_endpoint():
    mock_model_name = "test_model_name"
    mock_endpoint_config_name = "test_endpoint_config_name"
    mock_endpoint = "test_endpoint"
    mock_region_name = "us-east-1"

    # mock aws credentials
    boto3.setup_default_session(region_name=mock_region_name)

    model_response = create_model(
        TEST_S3_MODEL_PATH,
        mock_model_name,
        SAGEMAKER_ROLE,
        CONTENT_TYPE_JSON,
        mock_region_name,
        MEMORY_SIZE_MB,
        MAX_CONCURRENCY,
        FRAMEWORK,
        FRAMEWORK_VERSION,
        PY_VERSION,
        IMAGE_SCOPE,
    )
    assert mock_model_name.lower() in model_response["ModelArn"].lower()

    config_response = create_endpoint_config(
        mock_model_name, mock_endpoint_config_name, MEMORY_SIZE_MB, MAX_CONCURRENCY
    )
    assert (
        mock_endpoint_config_name.lower()
        in config_response["EndpointConfigArn"].lower()
    )
    endpoint_response = create_endpoint(
        mock_endpoint_config_name, mock_endpoint_config_name
    )
    assert mock_endpoint.lower() in endpoint_response["EndpointArn"].lower()


@pytest.mark.integration
def test_upload_model_create_invoke_and_delete_endpoint(input_data, expected_y_score):
    delete_endpoint(TEST_ENDPOINT_NAME, REGION_NAME)
    delete_endpoint_config(TEST_ENDPOINT_CONFIG_NAME, REGION_NAME)
    delete_model(TEST_MODEL_NAME, REGION_NAME)
    try:
        upload_model(
            source_dir=MODEL_SOURCE_DIR,
            bucket_name=TEST_S3_BUCKET_NAME,
            object_name=TEST_S3_FILENAME,
        )
        model_response = create_model(
            TEST_S3_MODEL_PATH,
            TEST_MODEL_NAME,
            SAGEMAKER_ROLE,
            CONTENT_TYPE_JSON,
            REGION_NAME,
            MEMORY_SIZE_MB,
            MAX_CONCURRENCY,
            FRAMEWORK,
            FRAMEWORK_VERSION,
            PY_VERSION,
            IMAGE_SCOPE,
        )
        assert TEST_MODEL_NAME.lower() in model_response["ModelArn"].lower()
        endpoint_config_response = create_endpoint_config(
            TEST_MODEL_NAME, TEST_ENDPOINT_CONFIG_NAME, MEMORY_SIZE_MB, MAX_CONCURRENCY
        )
        assert (
            TEST_ENDPOINT_CONFIG_NAME.lower()
            in endpoint_config_response["EndpointConfigArn"].lower()
        )
        endpoint_response = create_endpoint(
            TEST_ENDPOINT_CONFIG_NAME, TEST_ENDPOINT_NAME
        )

        assert TEST_ENDPOINT_NAME.lower() in endpoint_response["EndpointArn"].lower()
        endpoint_response = wait_endpoint_creation(TEST_ENDPOINT_NAME, REGION_NAME)

        assert endpoint_response["EndpointStatus"] == "InService"
        assert endpoint_response["EndpointName"] == TEST_ENDPOINT_NAME
        assert endpoint_response["EndpointConfigName"] == TEST_ENDPOINT_CONFIG_NAME

        response = json.loads(
            invoke(TEST_ENDPOINT_NAME, TEST_S3_IMAGE_PATH, CONTENT_TYPE_JSON)
        )

        y_score = np.frombuffer(response, dtype=np.float32).reshape(
            1, input_data.shape[1], input_data.shape[2]
        )

        assert y_score.shape == expected_y_score.shape
        assert y_score.dtype == expected_y_score.dtype
        np.testing.assert_array_almost_equal(y_score, expected_y_score, decimal=3)

    finally:
        delete_endpoint(TEST_ENDPOINT_NAME, REGION_NAME)
        delete_endpoint_config(TEST_ENDPOINT_CONFIG_NAME, REGION_NAME)
        delete_model(TEST_MODEL_NAME, REGION_NAME)
        client = boto3.client("s3", region_name=REGION_NAME)
        client.delete_object(Bucket=TEST_S3_BUCKET_NAME, Key=TEST_S3_FILENAME)


@pytest.mark.e2e
def test_endpoint_invoke_in_production(input_data, expected_prediction):
    predictions = invoke(
        ENDPOINT_NAME,
        input_data,
        CONTENT_TYPE_JSON,
    )

    with rasterio.open(io.BytesIO(predictions)) as src:
        pred_image = src.read()
    with rasterio.open(io.BytesIO(expected_prediction)) as src:
        expected_image = src.read()
    assert pred_image.shape == expected_image.shape
    assert pred_image.dtype == expected_image.dtype
    assert mse(pred_image, expected_image) < MSE_THRESHOLD
