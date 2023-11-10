import io
import json

import boto3
import numpy as np

from config import CONTENT_TYPE, ENDPOINT_NAME


def invoke(endpoint_name: str, content_type: str, payload: bytes) -> str:
    runtime = boto3.client("sagemaker-runtime", region_name="eu-central-1")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=payload,
        Accept=content_type,
    )
    predictions = response["Body"].read()
    return predictions


if __name__ == "__main__":
    from code.marinedebrisdetector import process

    from matplotlib import pyplot as plt

    with open(
        "images/first_half.tiff",
        "rb",
    ) as f:
        input_data = f.read()

    window_size = (128, 128)
    windows = process.preprocess_image(input_data, image_size=window_size)

    for image, window, meta in windows:
        buffer = io.BytesIO()
        np.save(buffer, image)
        payload = buffer.getvalue()
        prediction = invoke(ENDPOINT_NAME, CONTENT_TYPE, payload)
        np.save(buffer, image)
        payload = buffer.getvalue()
        prediction = invoke(ENDPOINT_NAME, CONTENT_TYPE, payload)
        numpy_pred = json.loads(prediction)
        plt.imshow(numpy_pred)
        plt.show()
