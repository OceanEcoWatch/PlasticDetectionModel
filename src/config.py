import os

import dotenv
import runpod
from runpod import Endpoint

dotenv.load_dotenv(override=True)

runpod.api_key = os.environ["RUNPOD_API_KEY"]
ENDPOINT = Endpoint("i1dp5odzq2kbgc")
