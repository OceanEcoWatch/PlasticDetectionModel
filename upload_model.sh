#!/bin/bash

# delete the old model
rm -rf model.tar.gz

# zip the folders for sagemaker deployment
tar -cvpzf model.tar.gz --exclude='*__pycache__*' code

python3 upload_model_s3.py
