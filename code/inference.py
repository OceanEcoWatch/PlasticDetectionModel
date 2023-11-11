import io
import logging
import ssl

import rasterio
import torch
from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import predict

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


def model_fn(model_dir) -> SegmentationModel:
    create_unverified_https_context()
    device = define_device()

    detector = SegmentationModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS["unet++1"],
        strict=False,
        map_location=device,
    )

    LOGGER.info(f"Loaded model from {CHECKPOINTS['unet++1']}")
    return detector


def input_fn(request_body, request_content_type):
    if request_content_type == "application/octet-stream":
        with rasterio.open(io.BytesIO(request_body)) as src:
            image = src.read()
            meta = src.meta.copy()
            return image, meta
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    processing_unit = define_device()
    LOGGER.info(f"Predicting on {processing_unit}")
    prediction = predict(
        model,
        image=input_data[0],
        metadata=input_data[1],
        device=processing_unit,
    )
    LOGGER.info(f"Prediction: {prediction}")
    return prediction


def output_fn(prediction, content_type):
    if content_type == "application/octet-stream":
        return prediction
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
