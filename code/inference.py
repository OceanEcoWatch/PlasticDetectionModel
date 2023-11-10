import io
import json
import logging
import pathlib
import ssl

import numpy as np
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


def get_model_path(model_dir):
    model_path = pathlib.Path(model_dir)
    model_file = list(model_path.glob("*.ckpt"))
    if len(model_file) == 0:
        raise FileNotFoundError(f"Could not find model file in {model_dir}")
    elif len(model_file) > 1:
        raise ValueError(f"Found multiple model files in {model_dir}")
    else:
        return model_file[0]


def model_fn(model_dir):
    """
    Args:
      model_dir: the directory where model is saved.
    Returns:
      SegmentationModel from the model_dir.
    """
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
    if request_content_type == "application/x-npy":
        # Convert binary data back to a NumPy array
        buffer = io.BytesIO(request_body)
        return np.load(buffer)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Args:
      input_data: Returned input data from input_fn
      model: Returned model from model_fn
    Returns:
      The predictions
    """
    processing_unit = define_device()
    LOGGER.info(f"Predicting on {processing_unit}")
    prediction = predict(
        model,
        image=input_data,
        device=processing_unit,
    )
    LOGGER.info(f"Prediction: {prediction}")
    return prediction


def output_fn(prediction, content_type):
    """
    Serialize the prediction output.

    Args:
      prediction: The prediction result from the model.
      content_type: The desired content type of the response.

    Returns:
      Serialized response ready to be returned by the endpoint.
    """
    if content_type == "application/x-npy":
        serialized_prediction = json.dumps(prediction.tolist())
        return serialized_prediction
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# if __name__ == "__main__":
#     from marinedebrisdetector import process
#     from matplotlib import pyplot as plt

#     with open(
#         "images/first_half.tiff",
#         "rb",
#     ) as f:
#         input_data = f.read()

#     model = model_fn(".")
#     window_size = (480, 480)
#     windows = process.preprocess_image(input_data, image_size=window_size)
#     predictions = []
#     windows_output = []
#     images = []
#     for image, window, meta in windows:
#         buffer = io.BytesIO()
#         np.save(buffer, image)
#         image = input_fn(buffer.getvalue(), "application/x-npy")
#         images.append(image)
#         prediction = predict_fn(image, model)
#         predictions.append(prediction)
#         windows_output.append(window)

#         print(prediction)
#         output = output_fn(prediction, "application/x-npy")
#         print(output)
#         plt.imshow(prediction)
#         plt.show()

#     raw_pred_bytes = process.post_process_image(
#         predictions, images, windows_output, meta
#     )
#     print(raw_pred_bytes)
