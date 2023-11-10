import io
from code.inference import input_fn, model_fn, predict_fn
from code.marinedebrisdetector.model.segmentation_model import SegmentationModel

import pytest


@pytest.fixture
def input_data():
    with open(
        "tests/data/first_half.tiff",
        "rb",
    ) as f:
        input_data = f.read()
    return input_data


def test_model_fn():
    model = model_fn(".")
    assert type(model) == SegmentationModel


def test_input_fn(input_data, content_type="application/octet-stream"):
    input = input_fn(input_data, "application/octet-stream")
    assert isinstance(input, io.BytesIO)


def test_predict_fn(input_data):
    model = model_fn(".")
    input = input_fn(input_data, "application/octet-stream")
    prediction = predict_fn(input, model)
    assert prediction.shape == (1, 1, 480, 480)


# model = model_fn(".")
# print(model)
# with open(
#     "images/first_half.tiff",
#     "rb",
# ) as f:
#     input_data = f.read()
# input = input_fn(input_data, "application/octet-stream")

# prediction = predict_fn(input, model)
# print(prediction)
# output = output_fn(prediction, "application/octet-stream")output = output_fn(prediction, "application/octet-stream")
# output = output_fn(prediction, "application/octet-stream")output = output_fn(prediction, "application/octet-stream")
