import logging
from typing import Union

import boto3

LOGGER = logging.getLogger(__name__)


def invoke(
    endpoint_name: str,
    payload: Union[bytes, str],
    content_type: str,
    retry_count: int = 0,
    max_retries: int = 10,
) -> bytes:
    runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=payload,
        Accept=content_type,
    )
    prediction = response["Body"].read()
    return prediction
