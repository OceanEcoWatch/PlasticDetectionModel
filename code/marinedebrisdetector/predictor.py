import logging

import numpy as np
import torch
from marinedebrisdetector.model.segmentation_model import SegmentationModel

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def predict(
    model: SegmentationModel,
    image: np.ndarray,
    activation="sigmoid",
    device="cpu",
) -> np.ndarray:
    if image.shape[1] % 32 != 0 or image.shape[2] % 32 != 0:
        LOGGER.info("Padding image to be divisible by 32")
        image = pad_to_divisible_by_32(image)

    if image.shape[0] > 12:
        LOGGER.info("Only keeping first 12 bands")
        image = image[:12]

    LOGGER.info("Transforming image ")
    torch_image = torch.from_numpy(image.astype(np.float32))
    torch_image = torch_image.to(device) * 1e-4

    with torch.no_grad():
        LOGGER.info("Predicting")
        x = torch_image.unsqueeze(0)
        out = model(x)
        LOGGER.info("Predicted")
        if isinstance(out, tuple):
            LOGGER.info("Tuple output detected")
            y_score, y_pred = out
            LOGGER.info("Converting to numpy")
            y_score = y_score.squeeze().cpu().numpy()
            y_pred = y_pred.squeeze().cpu().numpy()
        else:
            LOGGER.info("Tuple output not detected")
            y_logits = out.squeeze(0)

            if activation == "sigmoid":
                LOGGER.info("Applying sigmoid")
                y_logits = torch.sigmoid(y_logits)
            LOGGER.info("Converting to numpy")
            y_score = y_logits.cpu().detach().numpy()[0]

        return y_score


def pad_to_divisible_by_32(image, pad_value=0):
    """
    Pad an image so that its height and width are divisible by 32.

    Args:
        image (numpy.ndarray): The image to pad.
        pad_value (int, optional): The value to use for padding. Default is 0.

    Returns:
        numpy.ndarray: The padded image.
    """
    bands, h, w = image.shape

    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32

    padding = (
        (0, 0),
        (h_pad // 2, h_pad - h_pad // 2),
        (w_pad // 2, w_pad - w_pad // 2),
    )
    padded_image = np.pad(image, padding)

    return padded_image
