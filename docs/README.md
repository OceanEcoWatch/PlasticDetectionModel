# PlasticDetectionModel

[![Endpoint Available](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/e2e_test.yml/badge.svg?branch=main)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/e2e_test.yml)
[![Unit Tests](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/unit_tests.yml)
[![Format and Lint](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/format_lint.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/format_lint.yml)

PlasticDetectionModel is a machine learning model used for detecting marine debris, specifically plastic, from Sentinel-2 L2A satellite images. The model is customized from the [marinedebrisdetector](https://github.com/MarcCoru/marinedebrisdetector/tree/main) to work in memory on a serverless cloud infrastructure. This repository contains the code and resources necessary for deploying the model on a [Runpod](https://www.runpod.io/serverless-gpu) serverless GPU. This repository is part of a larger pipeline aimed at identifying marine debris on a schedule: [PlasticDetectionService](https://github.com/OceanEcoWatch/PlasticDetectionService).
Ultimately, the predictions are displayed on our [mapping application](https://github.com/OceanEcoWatch/website), deployed here: https://oceanecowatch.org/en

## CI/CD Diagram

![CI/CD Diagram](https://github.com/OceanEcoWatch/PlasticDetectionModel/blob/main/docs/PlasticDetectionModel.png?raw=true)

## Installation

### Add .env

Add a .env file with the following keys

```
RUNPOD_API_KEY=<your_runpod_api_key>
```
### Pull the image from Docker Hub
https://hub.docker.com/repository/docker/oceanecowatch/plasticdetectionmodel/general
```
docker pull oceanecowatch/plasticdetectionmodel:latest
```

### Run the Docker container
```
docker run -d -p 8080:8080 --name plastic_detection_model oceanecowatch/plasticdetectionmodel:latest
```
## Deploy on Runpod
To deploy this Dockerized version on a Runpod serverless instance, follow these steps:

- Log in to your Runpod account and create a new serverless instance.
- In the deployment settings, specify the Docker image to use: oceanecowatch/marinext:latest.
- Deploy the instance and monitor the logs to ensure everything is running smoothly.
For detailed instructions, refer to the Runpod documentation.
