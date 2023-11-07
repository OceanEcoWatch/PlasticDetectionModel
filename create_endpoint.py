import logging

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.serverless import ServerlessInferenceConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

model_name = "MarineDebrisDetectorModel"
endpoint_config_name = "MarineDebrisDetectorEndpointConfig"
endpoint_name = "MarineDebrisDetectorEndpoint"
region_name = "eu-central-1"
model_path = "s3://sagemaker-studio-768912473174-0ryazmj34j9/model.tar.gz"
sagemaker_role = "arn:aws:iam::768912473174:role/service-role/AmazonSageMaker-ExecutionRole-20231017T155251"
content_type = "application/octet-stream"
memory_size_in_mb = 2048
max_concurrency = 1


def create_model(
    model_path,
    model_name,
    sagemaker_role,
    content_type,
    region_name,
    framework="pytorch",
    framework_version="2.0.1",
    py_version="py310",
    image_scope="inference",
    memory_size_in_mb=2048,
    max_concurrency=1,
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

    # Initialize sagemaker session
    sess = sagemaker.Session()

    # Initialize boto3 sagemaker client
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
                    "SAGEMAKER_SUBMIT_DIRECTORY": model_path,
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_REGION": region_name,
                    "LOG_LOCATION": "/tmp",
                    "METRICS_LOCATION": "/tmp",
                },
            }
        ],
        ExecutionRoleArn=sagemaker_role,
    )

    LOGGER.info("Model created %s", create_model_response)

    return create_model_response


def create_endpoint_config(
    model_name, endpoint_config_name, memory_size_in_mb=2048, max_concurrency=1
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


def delete_model(model_name):
    LOGGER.info("Deleting model %s", model_name)
    client = boto3.client(service_name="sagemaker")
    response = client.delete_model(ModelName=model_name)
    LOGGER.info("Model deleted %s", response)
    return response


def delete_endpoint_config(endpoint_config_name):
    LOGGER.info("Deleting endpoint config %s", endpoint_config_name)
    client = boto3.client(service_name="sagemaker")
    response = client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    LOGGER.info("Endpoint config deleted %s", response)
    return response


def delete_endpoint(endpoint_name):
    LOGGER.info("Deleting endpoint %s", endpoint_name)
    client = boto3.client(service_name="sagemaker")
    response = client.delete_endpoint(EndpointName=endpoint_name)
    LOGGER.info("Endpoint deleted %s", response)
    return response


if __name__ == "__main__":
    create_model(model_path, model_name, sagemaker_role, content_type, region_name)
    create_endpoint_config(model_name, endpoint_config_name)
    create_endpoint(endpoint_config_name, endpoint_name)
