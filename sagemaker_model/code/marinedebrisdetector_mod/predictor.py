import logging

import numpy as np
import torch
from marinedebrisdetector_mod.model.segmentation_model import SegmentationModel

logging.basicConfig(level=logging.INFO)


LOGGER = logging.getLogger(__name__)


def predict(
    model: SegmentationModel,
    image: np.ndarray,
    device: str = "cpu",
    activation: str = "sigmoid",
    image_size: tuple[int, int] = (480, 480),
    offset: int = 64,
) -> np.ndarray:
    b, h, w = image.shape
    if b > 12:
        image = image[:12]

    H, W = image_size
    H, W = H + offset * 2, W + offset * 2

    dh = (H - h) / 2
    dw = (W - w) / 2
    LOGGER.info("Padding image")
    image = np.pad(
        image,
        [
            (0, 0),
            (int(np.ceil(dh)), int(np.floor(dh))),
            (int(np.ceil(dw)), int(np.floor(dw))),
        ],
    )
    LOGGER.info("Transforming image to tensor")
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

    LOGGER.info("Unpadding image")
    y_score = y_score[
        int(np.ceil(dh)) : y_score.shape[0] - int(np.floor(dh)),
        int(np.ceil(dw)) : y_score.shape[1] - int(np.floor(dw)),
    ]
    return y_score
