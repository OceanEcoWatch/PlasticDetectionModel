import io
import logging
import os
import urllib.parse
from typing import Union

import boto3
import numpy as np
import rasterio
from botocore.exceptions import ClientError, NoCredentialsError

LOGGER = logging.getLogger(__name__)


def create_session(aws_access_key_id: str, aws_secret_access_key: str):
    return boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def get_image(request_body: Union[bytes, str]) -> np.ndarray:
    session = create_session(
        os.environ["AWS_ACCESS_KEY"], os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    s3_uri_parts = urllib.parse.urlparse(request_body)
    bucket_name = s3_uri_parts.netloc
    s3_key = s3_uri_parts.path.lstrip("/")

    content = stream_from_s3(session, bucket_name, s3_key)

    with rasterio.open(io.BytesIO(content)) as src:
        image = src.read()
        return image


def stream_from_s3(aws_session, bucket_name: str, s3_key: str) -> bytes:
    s3 = aws_session.resource("s3")
    s3_object = s3.Object(bucket_name, s3_key)

    s3_response = s3_object.get()
    return s3_response["Body"].read()


def stream_to_s3(
    aws_session, bucket_name: str, object_name: str, data_stream: io.BytesIO
) -> bool:
    s3 = aws_session.resource("s3")
    try:
        s3.upload_fileobj(data_stream, bucket_name, object_name)
        return True
    except NoCredentialsError:
        LOGGER.error("Credentials not available")
        return False
    except ClientError as e:
        LOGGER.error(e)
        return False
