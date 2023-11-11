import boto3

from config import CONTENT_TYPE, ENDPOINT_NAME


def invoke(endpoint_name: str, content_type: str, payload: bytes) -> bytes:
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
    import io

    import matplotlib.pyplot as plt
    import rasterio

    with open(
        "tests/data/first_half.tiff",
        "rb",
    ) as f:
        input_data = f.read()

    pred = invoke(ENDPOINT_NAME, CONTENT_TYPE, input_data)
    with io.BytesIO(pred) as buffer:
        with rasterio.open(buffer) as src:
            image_data = src.read()
            plt.imshow(image_data[0])
            plt.show()
