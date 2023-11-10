import boto3

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
    with open(
        "tests/data/first_half.tiff",
        "rb",
    ) as f:
        input_data = f.read()

    pred = invoke(ENDPOINT_NAME, CONTENT_TYPE, input_data)
