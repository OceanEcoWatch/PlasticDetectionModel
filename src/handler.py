import base64
import io
import json
import logging
import os

import rasterio
import runpod.serverless
import torch
from marinedebrisdetector_mod.model.segmentation_model import (
    SegmentationModel,
)
from marinedebrisdetector_mod.predictor import predict

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def handler(job):
    job_input = job["input"]
    enc_bytes = job_input["image"]
    image_bytes = base64.b64decode(enc_bytes)

    with rasterio.open(io.BytesIO(image_bytes)) as src:
        image = src.read()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(
        current_dir, "epoch=54-val_loss=0.50-auroc=0.987.ckpt"
    )
    model = (
        SegmentationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=False,
            map_location=device,
        )
        .to(device)
        .eval()
    )

    prediction = predict(model, image=image, device=device)

    base_64_prediction = base64.b64encode(prediction.tobytes()).decode("utf-8")
    return json.dumps({"prediction": base_64_prediction})


runpod.serverless.start({"handler": handler})
