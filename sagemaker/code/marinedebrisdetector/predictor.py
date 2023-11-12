import io
import logging

import numpy as np
import rasterio
import torch
from marinedebrisdetector.model.segmentation_model import SegmentationModel

logging.basicConfig(level=logging.INFO)


LOGGER = logging.getLogger(__name__)


def predict(
    model: SegmentationModel,
    image: np.ndarray,
    metadata: dict,
    activation="sigmoid",
    device="cpu",
) -> bytes:
    metadata["count"] = 1
    metadata["dtype"] = "uint8"
    metadata["driver"] = "GTiff"
    _, h, w = image.shape

    if h % 32 != 0 or w % 32 != 0:
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

    LOGGER.info("Unpad to original size")
    y_score = unpad_to_original_size(y_score, h, w)

    LOGGER.info("Converting to tiff byte stream")
    pred_image_byte_stream = io.BytesIO()
    with rasterio.open(pred_image_byte_stream, "w+", **metadata) as dst:
        writedata = (np.expand_dims(y_score, 0).astype(np.float32) * 255).astype(
            np.uint8
        )
        dst.write(writedata)

    return pred_image_byte_stream.getvalue()


def unpad_to_original_size(
    padded_image: np.ndarray, original_height: int, original_width: int
) -> np.ndarray:
    h_start = (padded_image.shape[0] - original_height) // 2
    w_start = (padded_image.shape[1] - original_width) // 2

    unpadded_image = padded_image[
        h_start : h_start + original_height, w_start : w_start + original_width
    ]
    if unpadded_image.shape[0] != original_height:
        raise ValueError(
            f"unpadding size mismatch: {unpadded_image.shape[0]} != {original_height}"
        )
    if unpadded_image.shape[1] != original_width:
        raise ValueError(
            f"unpadding size mismatch: {unpadded_image.shape[1]} != {original_width}"
        )
    return unpadded_image


def pad_to_divisible_by_32(image: np.ndarray) -> np.ndarray:
    _, h, w = image.shape

    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32

    padding = (
        (0, 0),
        (h_pad // 2, h_pad - h_pad // 2),
        (w_pad // 2, w_pad - w_pad // 2),
    )
    padded_image = np.pad(image, padding)

    return padded_image
