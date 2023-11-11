import io
import logging
import os
import tarfile
from typing import Callable, Optional

import boto3
from botocore.exceptions import NoCredentialsError

LOGGER = logging.getLogger(__name__)


def create_tart_gz_in_memory(
    source_dir: str, exclude: Optional[Callable]
) -> io.BytesIO:
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir), filter=exclude)
    tar_stream.seek(0)
    return tar_stream


def stream_to_s3(bucket_name: str, object_name: str, data_stream: io.BytesIO) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.upload_fileobj(data_stream, bucket_name, object_name)
        return True
    except NoCredentialsError:
        LOGGER.error("Credentials not available")
        return False
    except Exception as e:
        LOGGER.error(e)
        return False


def exclude_pycache(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    if "__pycache__" in tarinfo.name:
        return None

    return tarinfo


def upload_model(
    source_dir, bucket_name, object_name, exlude: Callable = exclude_pycache
) -> bool:
    tar_stream = create_tart_gz_in_memory(source_dir, exlude)
    if stream_to_s3(bucket_name, object_name, tar_stream):
        LOGGER.info("Model uploaded to s3://%s/%s", bucket_name, object_name)
        return True
    else:
        LOGGER.error("Model upload failed")
        return False
