# PlasticDetectionModel

[![Endpoint Available](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/e2e_test.yml/badge.svg?branch=main)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/e2e_test.yml)
[![Deployed](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/deploy_prod.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/deploy_prod.yml)
[![Integration Tests](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/integration_tests.yml)
[![Unit Tests](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/unit_tests.yml)
[![Format and Lint](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/format_lint.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionModel/actions/workflows/format_lint.yml)


PlasticDetectionModel is a machine learning model used for detecting marine debris, specifically plastic, from Sentinel-2 L2A satellite images. The model is customized from the [marinedebrisdetector](https://github.com/MarcCoru/marinedebrisdetector/tree/main) to work in memory on a serverless cloud infrastructure. This repository contains the code and resources necessary for deploying the model on [Sagemaker serverless inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html). This repository is part of a larger pipeline aimed at identifying marine debris on a schedule. [PlasticDetectionService](https://github.com/OceanEcoWatch/PlasticDetectionService). 
Ultimately, the predictions will be displayed on our [mapping application](https://github.com/OceanEcoWatch/website)


## CI/CD Diagram

![CI/CD Diagram](https://github.com/OceanEcoWatch/PlasticDetectionModel/blob/main/docs/PlasticDetectionModel.png?raw=true)
