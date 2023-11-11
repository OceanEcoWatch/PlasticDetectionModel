import boto3

from config import CONTENT_TYPE, ENDPOINT_NAME, REGION_NAME


def invoke_endpoint(
    endpoint_name: str, input_data: bytes, content_type: str, region_name: str
) -> bytes:
    runtime = boto3.client("sagemaker-runtime", region_name=region_name)

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=input_data,
        Accept=content_type,
    )
    return response["Body"].read()


if __name__ == "__main__":
    with open("images/first_half.tiff", "rb") as f:
        input_data = f.read()
    invoke_endpoint(ENDPOINT_NAME, input_data, CONTENT_TYPE, REGION_NAME)
