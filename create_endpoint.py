import logging
import time

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker import image_uris
from sagemaker.serverless import ServerlessInferenceConfig

from config import (
    CONTENT_TYPE,
    ENDPOINT_CONFIG_NAME,
    ENDPOINT_NAME,
    FRAMEWORK,
    FRAMEWORK_VERSION,
    IMAGE_SCOPE,
    MAX_CONCURRENCY,
    MEMORY_SIZE_MB,
    MODEL_NAME,
    PY_VERSION,
    REGION_NAME,
    S3_MODEL_PATH,
    SAGEMAKER_ROLE,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def create_model(
    model_path,
    model_name,
    sagemaker_role,
    content_type,
    region_name,
    memory_size_in_mb,
    max_concurrency,
    framework,
    framework_version,
    py_version,
    image_scope,
):
    LOGGER.info("Creating serverless inference model with following parameters:")
    LOGGER.info("model_path: %s", model_path)
    LOGGER.info("model_name: %s", model_name)
    LOGGER.info("content_type: %s", content_type)
    LOGGER.info("region_name: %s", region_name)
    LOGGER.info("memory_size_in_mb: %s", memory_size_in_mb)
    LOGGER.info("max_concurrency: %s", max_concurrency)

    serverless_inference_config = ServerlessInferenceConfig(
        memory_size_in_mb=memory_size_in_mb,
        max_concurrency=max_concurrency,
    )

    image_uri = image_uris.retrieve(
        framework=framework,
        region=region_name,
        version=framework_version,
        py_version=py_version,
        image_scope=image_scope,
        serverless_inference_config=serverless_inference_config,
    )

    sess = sagemaker.Session()

    region = sess.boto_region_name
    sm_client = boto3.client("sagemaker", region_name=region)

    # Define Model specifications
    modelpackage_inference_specification = {
        "InferenceSpecification": {
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_path}],
            "SupportedContentTypes": [content_type],
            "SupportedResponseMIMETypes": [content_type],
        }
    }

    create_model_package_input_dict = {
        "ModelApprovalStatus": "PendingManualApproval",
    }
    create_model_package_input_dict.update(modelpackage_inference_specification)

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        Containers=[
            {
                "Image": image_uri,
                "Mode": "SingleModel",
                "ModelDataUrl": model_path,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_REGION": region_name,
                },
            }
        ],
        ExecutionRoleArn=sagemaker_role,
    )

    LOGGER.info("Model created %s", create_model_response)

    return create_model_response


def create_endpoint_config(
    model_name, endpoint_config_name, memory_size_in_mb, max_concurrency
):
    LOGGER.info("Creating endpoint config %s", endpoint_config_name)
    client = boto3.client(service_name="sagemaker")
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": endpoint_config_name,
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_size_in_mb,
                    "MaxConcurrency": max_concurrency,
                },
            },
        ],
    )
    LOGGER.info("Endpoint config created %s", endpoint_config_response)
    return endpoint_config_response


def create_endpoint(endpoint_config_name, endpoint_name):
    LOGGER.info("Creating endpoint %s", endpoint_name)
    client = boto3.client(service_name="sagemaker")
    endpoint_response = client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    LOGGER.info("Endpoint created %s", endpoint_response)
    return endpoint_response


def delete_model(model_name, region_name):
    client = boto3.client(service_name="sagemaker", region_name=region_name)
    try:
        response = client.delete_model(ModelName=model_name)
        LOGGER.info("Model deleted %s", response)
        return response
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            LOGGER.info("Unable to delete model name %s due to %s", model_name, e)
        else:
            raise e
        return


def delete_endpoint_config(endpoint_config_name, region_name):
    client = boto3.client(service_name="sagemaker", region_name=region_name)
    try:
        response = client.delete_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        LOGGER.info("Endpoint config deleted %s", response)
        return response
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            LOGGER.info(
                "Unable to delete endpoint config name %s due to %s",
                endpoint_config_name,
                e,
            )
        else:
            raise e
        return


def delete_endpoint(endpoint_name, region_name):
    client = boto3.client(service_name="sagemaker", region_name=region_name)
    try:
        response = client.delete_endpoint(EndpointName=endpoint_name)
        LOGGER.info("Endpoint deleted %s", response)
        return response
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            LOGGER.info(
                "Unable to delete endpoint name %s, due to %s", endpoint_name, e
            )
        else:
            raise e
        return


def wait_endpoint_creation(endpoint_name):
    client = boto3.client(service_name="sagemaker")
    describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)

    while describe_endpoint_response["EndpointStatus"] == "Creating":
        describe_endpoint_response = client.describe_endpoint(
            EndpointName=endpoint_name
        )
        LOGGER.info("Endpoint status: %s", describe_endpoint_response["EndpointStatus"])
        if describe_endpoint_response["EndpointStatus"] == "Creating":
            time.sleep(15)
    LOGGER.info("Endpoint ready %s", describe_endpoint_response)
    return describe_endpoint_response


if __name__ == "__main__":
    delete_endpoint(ENDPOINT_NAME)
    delete_endpoint_config(ENDPOINT_CONFIG_NAME)
    delete_model(MODEL_NAME)
    create_model(
        S3_MODEL_PATH,
        MODEL_NAME,
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
        MODEL_NAME, ENDPOINT_CONFIG_NAME, MEMORY_SIZE_MB, MAX_CONCURRENCY
    )
    create_endpoint(ENDPOINT_CONFIG_NAME, ENDPOINT_NAME)
    wait_endpoint_creation(ENDPOINT_NAME)
