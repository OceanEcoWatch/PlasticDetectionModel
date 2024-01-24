from itertools import product

import numpy as np
import rasterio
from rasterio.windows import Window, transform


def mse(imageA, imageB):
    """Compute the mean squared error between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def split_tiff(image_path, window_size=(480, 480)):
    """Split a tiff into smaller tiffs."""
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()

    rows = np.arange(0, meta["height"], window_size[0])
    cols = np.arange(0, meta["width"], window_size[1])

    for row, col in product(rows, cols):
        image_window = Window(col, row, window_size[1], window_size[0])
        with rasterio.open(image_path) as src:
            image_data = src.read(window=image_window)
        meta["height"] = image_window.height
        meta["width"] = image_window.width
        meta["transform"] = transform(image_window, src.transform)
        with rasterio.open(f"tests/data/{row}_{col}.tiff", "w", **meta) as dst:
            dst.write(image_data)


if __name__ == "__main__":
    split_tiff("tests/data/durban_20190424.tiff")
