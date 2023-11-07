#!/bin/bash

# delete the old model
rm -rf model.tar.gz

# zip the folders for sagemaker deployment
tar -cvpzf model.tar.gz model code


python upload_model_s3.py
