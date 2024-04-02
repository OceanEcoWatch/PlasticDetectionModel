import numpy as np
import pytest
import rasterio

from src.marinedebrisdetector_mod.predictor import predict


@pytest.mark.unit
def test_predict_fn(np_data, model, expected_y_score):
    src, org_image, meta = np_data
    y_score = predict(model=model, image=org_image, device="cpu")
    prediction = y_score.reshape(1, meta["height"], meta["width"])

    meta.update(
        {
            "count": 1,
            "height": prediction.shape[1],
            "width": prediction.shape[2],
            "dtype": prediction.dtype,
        }
    )
    with rasterio.open("tests/data/predict_fn_out.tiff", "w+", **meta) as dst:
        dst.write(prediction)

    np.testing.assert_almost_equal(y_score, expected_y_score, decimal=4)
