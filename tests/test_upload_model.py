import tarfile

import boto3
from moto import mock_s3

from tests.conftest import TEST_MODEL_SOURCE_DIR, TEST_S3_BUCKET_NAME, TEST_S3_FILENAME
from upload_model_s3 import create_tart_gz_in_memory, exclude_pycache, upload_model


@mock_s3
def test_upload_model():
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=TEST_S3_BUCKET_NAME)

    result = upload_model(
        source_dir=TEST_MODEL_SOURCE_DIR,
        bucket_name=TEST_S3_BUCKET_NAME,
        object_name=TEST_S3_FILENAME,
        exlude=exclude_pycache,
    )
    assert result


def test_create_tart_gz_in_memory():
    result = create_tart_gz_in_memory(
        source_dir=TEST_MODEL_SOURCE_DIR,
        exclude=exclude_pycache,
    )
    with tarfile.open(fileobj=result, mode="r:gz") as tar:
        assert "code/inference.py" in tar.getnames()
        assert "code/marinedebrisdetector" in tar.getnames()
        assert not any("__pycache__" in name for name in tar.getnames())

    assert result.readable()
    assert result.seekable()
