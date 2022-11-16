import os
from pathlib import Path

from Config import ConfigClassificationTrain
import uuid
import mlflow
import json
from minio import Minio
from minio.error import S3Error
from core.config import get_app_settings
from predict import predict

# print(os.environ)
settings = get_app_settings()
user_id = settings.user_id
project_id = settings.project_id
dataset_version_id = settings.dataset_version_id
pipeline_id = settings.pipeline_id
# print(settings)

minioClient = Minio(settings.minio_endpoint,
                    access_key=settings.minio_access_key,
                    secret_key=settings.minio_secret_key,
                    secure=False)
model_bucket = "model"


def compare(config):
    output = {
        "status": "Running",
        "best_model_minio_result": None,
        "latest_model_result": None,
        "is_lastest_better": False
    }

    with open('compare.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    minioClient.fput_object("pipeline",
                            f"{user_id}/{project_id}/{dataset_version_id}/{pipeline_id}/compare.json",
                            "compare.json")
    mlflow.set_tracking_uri("http://10.255.187.41:5120")
    name_experiment = f"image-classification-{user_id}-{project_id}-{pipeline_id}"
    mlflow.set_experiment(name_experiment)

    result_1 = predict(config, weight_path=f"weights/{config.model_name}/best.pth")
    print(f"result 1: {result_1}")
    try:
        Path("minio").mkdir(parents=True, exist_ok=True)
        minioClient.fget_object(model_bucket,
                                f"{settings.user_id}/{settings.project_id}/{config.model_name}/best.pth",
                                f"minio/{config.model_name}/best.pth")
        result_2 = predict(config, weight_path=f"minio/{config.model_name}/best.pth")
        print(f"result 2: {result_2}")

    except S3Error:
        print("object not found")
        result_2 = result_1

    if result_1 >= result_2:
        output["is_lastest_better"] = True
        minioClient.fput_object(model_bucket,
                                f"{settings.user_id}/{settings.project_id}/{config.model_name}/best.pth",
                                f"weights/{config.model_name}/best.pth")

    output["latest_model_result"] = float(result_1)
    output["best_model_minio_result"] = float(result_2)
    output["status"] = "Done"

    with open('compare.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    mlflow.log_params(output)
    minioClient.fput_object("pipeline",
                            f"{settings.user_id}/{settings.project_id}/{settings.dataset_version_id}/{settings.pipeline_id}/compare.json",
                            "compare.json")


if __name__ == "__main__":
    classes = json.load(open('data/classes.json'))
    num_classes = len(classes)

    config_dict = {
        "epochs": 100,
        "num-classes": num_classes
    }
    config = ConfigClassificationTrain(config_dict)
    compare(config)
