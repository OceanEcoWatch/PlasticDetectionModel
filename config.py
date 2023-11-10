import os

import dotenv

dotenv.load_dotenv()

MODEL_NAME = "MarineDebrisDetectorModel"
ENDPOINT_CONFIG_NAME = "MarineDebrisDetectorEndpointConfig"
ENDPOINT_NAME = "MarineDebrisDetectorEndpoint"
REGION_NAME = "eu-central-1"
MODEL_PATH = "s3://sagemaker-studio-768912473174-0ryazmj34j9/model.tar.gz"
SAGEMAKER_ROLE = os.environ["SAGEMAKER_ROLE"]
CONTENT_TYPE = "application/octet-stream"
MEMORY_SIZE_MB = 3072
MAX_CONCURRENCY = 1
FRAMEWORK = "pytorch"
FRAMEWORK_VERSION = "2.0.1"
PY_VERSION = "py310"
IMAGE_SCOPE = "inference"
