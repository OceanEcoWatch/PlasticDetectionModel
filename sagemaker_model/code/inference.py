import io
import logging
import ssl
import urllib.parse

import boto3
import numpy as np
import rasterio
import torch
from botocore.exceptions import ClientError

from .marinedebrisdetector_mod.checkpoints import CHECKPOINTS
from .marinedebrisdetector_mod.model.segmentation_model import SegmentationModel
from .marinedebrisdetector_mod.predictor import predict

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def create_unverified_https_context():
    """
    Create an unverified HTTPS context for requests.
    This is used to avoid SSL certificate verification errors for macOS users.
    """
    ssl._create_default_https_context = ssl._create_unverified_context


def define_device():
    if torch.cuda.is_available():
        processing_unit = "cuda"
    else:
        processing_unit = "cpu"
    return processing_unit


def model_fn(model_dir):
    device = define_device()
    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS["unet++1"],
        strict=False,
        map_location=device,
    )
    return model.to(device).eval()


def input_fn(request_body: bytes, request_content_type: str) -> np.ndarray:
    if request_content_type == "application/json":
        s3_uri_parts = urllib.parse.urlparse(request_body)
        bucket_name = s3_uri_parts.netloc
        s3_key = s3_uri_parts.path.lstrip("/")

        s3_resource = boto3.resource("s3")
        s3_object = s3_resource.Object(bucket_name, s3_key)
        try:
            s3_response = s3_object.get()
            request_body = s3_response["Body"].read()
            with rasterio.open(io.BytesIO(request_body)) as src:
                image = src.read()
                return image
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                LOGGER.error(
                    f"The specified key does not exist: {s3_key} in bucket {bucket_name}"
                )
            else:
                raise
        with rasterio.open(io.BytesIO(request_body)) as src:
            image = src.read()
            return image

    if request_content_type == "application/octet-stream":
        with rasterio.open(io.BytesIO(request_body)) as src:
            image = src.read()
            return image
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    return predict(
        model,
        image=input_data,
    )


def output_fn(prediction, content_type) -> bytes:
    if content_type == "application/octet-stream" or content_type == "application/json":
        if not isinstance(prediction, np.ndarray):
            raise ValueError(
                f"Prediction is not a numpy array, but {type(prediction)} instead"
            )
        LOGGER.info("Converting prediction to bytes")
        return prediction.tobytes()

    else:
        raise ValueError(f"Unsupported content type: {content_type}")
