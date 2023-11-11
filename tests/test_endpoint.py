import io

import boto3
import pytest
import rasterio
from botocore.exceptions import ClientError
from moto import mock_sagemaker

from config import (
    CONTENT_TYPE,
    FRAMEWORK,
    FRAMEWORK_VERSION,
    IMAGE_SCOPE,
    MAX_CONCURRENCY,
    MEMORY_SIZE_MB,
    PY_VERSION,
    REGION_NAME,
    SAGEMAKER_ROLE,
)
from create_endpoint import (
    create_endpoint,
    create_endpoint_config,
    create_model,
    delete_endpoint,
    delete_endpoint_config,
    delete_model,
    wait_endpoint_creation,
)
from invoke import invoke_endpoint
from tests.conftest import (
    MSE_THRESHOLD,
    TEST_ENDPOINT_CONFIG_NAME,
    TEST_ENDPOINT_NAME,
    TEST_MODEL_NAME,
    TEST_S3_MODEL_PATH,
)
from tests.utils import mse


@mock_sagemaker
def test_create_endpoint(aws_credentials):
    model_response = create_model(
        TEST_S3_MODEL_PATH,
        TEST_MODEL_NAME,
        SAGEMAKER_ROLE,
        CONTENT_TYPE,
        REGION_NAME,
        MEMORY_SIZE_MB,
        MAX_CONCURRENCY,
        FRAMEWORK,
        FRAMEWORK_VERSION,
        PY_VERSION,
        IMAGE_SCOPE,
    )
    assert TEST_MODEL_NAME.lower() in model_response["ModelArn"].lower()

    config_response = create_endpoint_config(
        TEST_MODEL_NAME, TEST_ENDPOINT_CONFIG_NAME, MEMORY_SIZE_MB, MAX_CONCURRENCY
    )
    assert (
        TEST_ENDPOINT_CONFIG_NAME.lower()
        in config_response["EndpointConfigArn"].lower()
    )
    endpoint_response = create_endpoint(TEST_ENDPOINT_CONFIG_NAME, TEST_ENDPOINT_NAME)
    assert TEST_ENDPOINT_NAME.lower() in endpoint_response["EndpointArn"].lower()
    endpoint_response = wait_endpoint_creation(TEST_ENDPOINT_NAME)

    assert endpoint_response["EndpointStatus"] == "InService"
    assert endpoint_response["EndpointName"] == TEST_ENDPOINT_NAME
    assert endpoint_response["EndpointConfigName"] == TEST_ENDPOINT_CONFIG_NAME


@pytest.mark.e_2_e
def test_create_and_invoke_and_delete_endpoint(input_data, expected_prediction, caplog):
    try:
        create_model(
            TEST_S3_MODEL_PATH,
            TEST_MODEL_NAME,
            SAGEMAKER_ROLE,
            CONTENT_TYPE,
            REGION_NAME,
            MEMORY_SIZE_MB,
            MAX_CONCURRENCY,
            FRAMEWORK,
            FRAMEWORK_VERSION,
            PY_VERSION,
            IMAGE_SCOPE,
        )
        create_endpoint_config(
            TEST_MODEL_NAME, TEST_ENDPOINT_CONFIG_NAME, MEMORY_SIZE_MB, MAX_CONCURRENCY
        )
        create_endpoint(TEST_ENDPOINT_CONFIG_NAME, TEST_ENDPOINT_NAME)
        wait_endpoint_creation(TEST_ENDPOINT_NAME)

        predictions = invoke_endpoint(TEST_ENDPOINT_NAME, input_data, CONTENT_TYPE)

        with rasterio.open(io.BytesIO(predictions)) as src:
            pred_image = src.read()
        with rasterio.open(io.BytesIO(expected_prediction)) as src:
            expected_image = src.read()
        assert pred_image.shape == expected_image.shape
        assert pred_image.dtype == expected_image.dtype
        assert mse(pred_image, expected_image) < MSE_THRESHOLD

    finally:
        delete_endpoint(TEST_ENDPOINT_NAME)
        delete_endpoint_config(TEST_ENDPOINT_CONFIG_NAME)
        delete_model(TEST_MODEL_NAME)
        client = boto3.client(service_name="sagemaker")
        with pytest.raises(ClientError):
            client.describe_endpoint(EndpointName=TEST_ENDPOINT_NAME)
        with pytest.raises(ClientError):
            client.describe_endpoint_config(
                EndpointConfigName=TEST_ENDPOINT_CONFIG_NAME
            )
        with pytest.raises(ClientError):
            client.describe_model(ModelName=TEST_MODEL_NAME)
