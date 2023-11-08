import io
import logging
import pathlib

import torch
from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)


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
    device = define_device()

    detector = SegmentationModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS["unet++1"],
        map_location=device,
        strict=False,
    )
    LOGGER.info(f"Loaded model from {CHECKPOINTS['unet++1']}")
    return detector.to(device)


def input_fn(request_body, request_content_type):
    if request_content_type == "application/octet-stream":
        LOGGER.info(f"Request body: {request_body}")
        return io.BytesIO(request_body)
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
    predictor = ScenePredictor(device=processing_unit)
    LOGGER.info(f"Predicting on {processing_unit}")
    prediction = predictor.predict(model, input_data)
    LOGGER.info(f"Prediction: {prediction}")
    return prediction


def output_fn(prediction, content_type):
    if content_type == "application/octet-stream":
        LOGGER.info(f"Returning prediction: {prediction}")
        return prediction
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


if __name__ == "__main__":
    model = model_fn("model")
    print(model)
    with open(
        "/Users/marc.leerink/dev/PlasticDetectionService/images/5cb12a6cbd6df0865947f21170bc432a/response.tiff",
        "rb",
    ) as f:
        input_data = f.read()
    input = input_fn(input_data, "application/octet-stream")

    prediction = predict_fn(input, model)
    print(prediction)
    output = output_fn(prediction, "application/octet-stream")
    print(output)
    print(output)
    print(output)
    print(output)
    print(output)
