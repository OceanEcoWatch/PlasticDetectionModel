import io
import logging
from itertools import product

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage.filters import gaussian_filter

LOGGER = logging.getLogger(__name__)


def preprocess_image(
    data: bytes, image_size=(480, 480), offset=64
) -> list[tuple[np.ndarray, Window, dict]]:
    with rasterio.open(io.BytesIO(data)) as src:
        meta = src.meta.copy()
        H, W = image_size
        rows = np.arange(0, meta["height"], H)
        cols = np.arange(0, meta["width"], W)
        image_window = Window(0, 0, meta["width"], meta["height"])

        windows = []
        for r, c in product(rows, cols):
            H, W = image_size
            window = image_window.intersection(
                Window(
                    c - offset,
                    r - offset,
                    W + offset,
                    H + offset,
                )
            )
            image = src.read(window=window)

            if image.shape[0] > 12:
                image = image[:12]

            H, W = H + offset * 2, W + offset * 2

            bands, h, w = image.shape
            dh = (H - h) / 2
            dw = (W - w) / 2
            image = np.pad(
                image,
                [
                    (0, 0),
                    (int(np.ceil(dh)), int(np.floor(dh))),
                    (int(np.ceil(dw)), int(np.floor(dw))),
                ],
            )
            windows.append((image, window, meta))

        return windows


def unpad(y_score: np.ndarray, window: Window, dh: float, dw: float):
    y_score = y_score[
        int(np.ceil(dh)) : y_score.shape[0] - int(np.floor(dh)),
        int(np.ceil(dw)) : y_score.shape[1] - int(np.floor(dw)),
    ]
    if y_score.shape[0] != window.height:
        raise ValueError(
            f"unpadding size mismatch: {y_score.shape[0]} != {window.height}"
        )
    if y_score.shape[1] != window.width:
        raise ValueError(
            f"unpadding size mismatch: {y_score.shape[1]} != {window.width}"
        )
    return y_score


def post_process_image(
    predictions: list[np.ndarray],
    images: list[np.ndarray],
    windows: list[Window],
    meta: dict,
    offset=64,
    window_size=(480, 480),
) -> bytes:
    H, W = window_size
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**meta) as dst:
            for pred, image, window in zip(predictions, images, windows):
                H, W = window_size
                H, W = H + offset * 2, W + offset * 2
                band, h, w = image.shape
                dh = (H - h) / 2
                dw = (W - w) / 2
                pred = unpad(pred, window, dh, dw)

                data = dst.read(window=window)[0] / 255

                overlap = data > 0

                if overlap.any():
                    LOGGER.info("Overlap detected")
                    dx, dy = np.gradient(overlap.astype(float))
                    g = np.abs(dx) + np.abs(dy)
                    transition = gaussian_filter(g, sigma=offset / 2)
                    transition /= transition.max()
                    transition[~overlap] = 1.0  # normalize to 1

                    y_score = transition * pred + (1 - transition) * data

                    writedata = (
                        np.expand_dims(y_score, 0).astype(np.float32) * 255
                    ).astype(np.uint8)
                    dst.write(writedata, window=window)
            return memfile.read()
