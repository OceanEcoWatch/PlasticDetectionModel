import boto3

PATH_TO_MODEL = "/Users/marc.leerink/dev/PlasticDetectionService/sagemaker/model.tar.gz"
BUCKET_NAME = "sagemaker-studio-768912473174-0ryazmj34j9"
FILENAME = "model.tar.gz"


def upload_model_s3(path_to_model, bucket_name, filename):
    s3 = boto3.client("s3")
    s3.upload_file(path_to_model, bucket_name, filename)
    print(f"Uploaded {path_to_model} to s3://{bucket_name}/{filename}")


if __name__ == "__main__":
    upload_model_s3(PATH_TO_MODEL, BUCKET_NAME, FILENAME)
