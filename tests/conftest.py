import sys

sys.path.append("sagemaker_model/code")

import io  # noqa

import pytest  # noqa
import rasterio  # noqa
from marinedebrisdetector.checkpoints import CHECKPOINTS  # noqa
from marinedebrisdetector.model.segmentation_model import SegmentationModel  # noqa

MSE_THRESHOLD = 0.01
TEST_MODEL_NAME = "TestMarineDebrisDetectorModel"
TEST_ENDPOINT_CONFIG_NAME = "TestMarineDebrisDetectorEndpointConfig"
TEST_ENDPOINT_NAME = "TestMarineDebrisDetectorEndpoint"

TEST_S3_BUCKET_NAME = "test-sagemaker-studio-768912473174-0ryazmj34j9"
TEST_S3_FILENAME = "test-model.tar.gz"
TEST_S3_MODEL_PATH = f"s3://{TEST_S3_BUCKET_NAME}/{TEST_S3_FILENAME}"


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
