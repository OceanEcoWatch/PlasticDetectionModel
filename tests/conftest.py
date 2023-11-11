import sys

sys.path.append("code")

import io  # noqa
from code.marinedebrisdetector.checkpoints import CHECKPOINTS  # noqa
from code.marinedebrisdetector.model.segmentation_model import SegmentationModel  # noqa

import pytest  # noqa
import rasterio  # noqa

MSE_THRESHOLD = 0.01


@pytest.fixture
def input_data():
    with open(
        "tests/data/test_image.tiff",
        "rb",
    ) as f:
        return f.read()


@pytest.fixture
def np_data(input_data):
    with rasterio.open(io.BytesIO(input_data)) as src:
        image = src.read()
        meta = src.meta.copy()
        return src, image, meta


@pytest.fixture
def model():
    detector = SegmentationModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINTS["unet++1"],
        strict=False,
        map_location="cpu",
    )
    return detector


@pytest.fixture
def expected_prediction():
    with open(
        "tests/data/prediction.tiff",
        "rb",
    ) as f:
        return f.read()
