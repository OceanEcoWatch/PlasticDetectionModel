import io

import boto3
import pytest
import rasterio
from moto import mock_sagemaker

from config import (
    CONTENT_TYPE,
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
from endpoint.invoke_endpoint import invoke_endpoint
from endpoint.upload_model_s3 import upload_model
from tests.conftest import (
    MSE_THRESHOLD,
    TEST_ENDPOINT_CONFIG_NAME,
    TEST_ENDPOINT_NAME,
    TEST_MODEL_NAME,
    TEST_S3_BUCKET_NAME,
    TEST_S3_FILENAME,
    TEST_S3_MODEL_PATH,
)
from tests.utils import mse


@mock_sagemaker
def test_create_endpoint():
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


@pytest.mark.integration
def test_upload_model_create_invoke_and_delete_endpoint(
    input_data, expected_prediction
):
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
        endpoint_response = wait_endpoint_creation(TEST_ENDPOINT_NAME)

        assert endpoint_response["EndpointStatus"] == "InService"
        assert endpoint_response["EndpointName"] == TEST_ENDPOINT_NAME
        assert endpoint_response["EndpointConfigName"] == TEST_ENDPOINT_CONFIG_NAME

        predictions = invoke_endpoint(
            TEST_ENDPOINT_NAME, input_data, CONTENT_TYPE, REGION_NAME
        )

        with rasterio.open(io.BytesIO(predictions)) as src:
            pred_image = src.read()
        with rasterio.open(io.BytesIO(expected_prediction)) as src:
            expected_image = src.read()
        assert pred_image.shape == expected_image.shape
        assert pred_image.dtype == expected_image.dtype
        assert mse(pred_image, expected_image) < MSE_THRESHOLD

    finally:
        delete_endpoint(TEST_ENDPOINT_NAME, REGION_NAME)
        delete_endpoint_config(TEST_ENDPOINT_CONFIG_NAME, REGION_NAME)
        delete_model(TEST_MODEL_NAME, REGION_NAME)
        client = boto3.client("s3", region_name=REGION_NAME)
        client.delete_object(Bucket=TEST_S3_BUCKET_NAME, Key=TEST_S3_FILENAME)
