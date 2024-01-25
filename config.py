import os

import dotenv

dotenv.load_dotenv()

MODEL_NAME = "MarineDebrisDetectorModel"
ENDPOINT_CONFIG_NAME = "MarineDebrisDetectorEndpointConfig"
ENDPOINT_NAME = "MarineDebrisDetectorEndpoint"
REGION_NAME = "eu-central-1"
SAGEMAKER_ROLE = os.environ["SAGEMAKER_ROLE"]
CONTENT_TYPE_STREAM = "application/octet-stream"
CONTENT_TYPE_JSON = "application/json"
MEMORY_SIZE_MB = 4096
MAX_CONCURRENCY = 1
FRAMEWORK = "pytorch"
FRAMEWORK_VERSION = "2.0.1"
PY_VERSION = "py310"
IMAGE_SCOPE = "inference"
MODEL_SOURCE_DIR = "sagemaker_model/code"

S3_BUCKET_NAME = "sagemaker-studio-768912473174-0ryazmj34j9"
S3_MODEL_NAME = "model.tar.gz"
S3_MODEL_PATH = f"s3://{S3_BUCKET_NAME}/{S3_MODEL_NAME}"
