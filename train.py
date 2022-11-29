import copy
import os
import time
from pathlib import Path
import mlflow
import torch
import torchvision.transforms as transforms
from minio import Minio
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import MyDataset
from Config import ConfigClassificationTrain
from get_model import get_model
from core.config import get_app_settings
import uuid
import json
import requests
from core.config import get_app_settings

# print(os.environ)

settings = get_app_settings()
# print(settings)


user_id = settings.user_id
project_id = settings.project_id
dataset_version_id = settings.dataset_version_id
pipeline_id = settings.pipeline_id

minioClient = Minio(settings.minio_endpoint,
                    access_key=settings.minio_access_key,
                    secret_key=settings.minio_secret_key,
                    secure=False)


model_bucket = "model"
pipeline_bucket = "pipeline"

found = minioClient.bucket_exists(model_bucket)
if not found:
    minioClient.make_bucket(model_bucket)

found = minioClient.bucket_exists(pipeline_bucket)
if not found:
    minioClient.make_bucket(pipeline_bucket)

mlflow.set_tracking_uri("http://10.255.187.41:5120")
name_experiment = f"image-classification-{user_id}-{project_id}-{pipeline_id}"
mlflow.set_experiment(name_experiment)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu=True, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        params = {}
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # if phase == 'train':
            #     loss_train = epoch_loss
            #     acc_train = epoch_acc

            if phase == 'val' and epoch_acc >= best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc
                # best_val_train_loss = loss_train
                # best_val_train_acc = acc_train
                params = {
                    "best_val_loss": float(epoch_loss),
                    "best_val_acc": float(epoch_acc),
                    "best epoch": int(epoch)
                }
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # mlflow.log_params(best_acc)
    model.load_state_dict(best_model_wts)
    return model, params


def main(config):
    img_size = 224
    transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load_dataset
    train_dataset = MyDataset.CustomImageDataset(annotation_folder=f"data/training/labels",
                                                 img_folder="data/training/images", transform=transform)
    val_dataset = MyDataset.CustomImageDataset(annotation_folder="data/validation/labels",
                                               img_folder="data/validation/images", transform=transform)
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset)
    }
    batch_size = config.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    dataloaders = {
        "train": DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=nw,
                            ),

        "val": DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=nw,
                          )
    }
    # load model

    model = get_model(config.model_name, config.num_classes)

    print(f"model name: {config.model_name}")
    print(f"num_classes: {config.num_classes}")

    loss_fn = torch.nn.CrossEntropyLoss()
    # classes = json.load(open("data/classes_test.json"))
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_ft, params = train_model(model=model, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                                   criterion=loss_fn, optimizer=optimizer, scheduler=exp_lr_scheduler,
                                   num_epochs=config.epochs)
    params["model"] = config.model_name
    params["learning_rate"] = config.lr
    params["epochs"] = config.epochs
    mlflow.log_params(params=params)
    Path(f"weights/{config.model_name}").mkdir(parents=True, exist_ok=True)
    torch.save(model_ft.state_dict(), f"weights/{config.model_name}/best.pth")
    minioClient.fput_object(model_bucket, f"{user_id}/{project_id}/{dataset_version_id}/{pipeline_id}/best.pth", f"weights/{config.model_name}/best.pth")
    res = requests.post("http://10.255.187.50:8089/api/v1/model/create",
    json={
        "model_name": config.model_name,
        "model_url": f"{user_id}/{project_id}/{dataset_version_id}/{pipeline_id}/best.pth",
        "model_result": params,
        "pipeline_id": int(settings.pipeline_id),
        "project_id": int(settings.project_id),
        "description": ""
    }, headers={"token": user_id}, proxies = {
        "http_proxy": "http://10.255.188.84:3128",
        "https_proxy": "http://10.255.188.84:3128"
    })
    print(res.json())
    # minioClient.fput_object(bucket, "weights/best.pt", "weights/best.pt")
    # minio_client.fput_object(bucket, "classes.json", "classes.json")


if __name__ == "__main__":
    classes = json.load(open('data/classes.json'))
    num_classes = len(classes)
    # print(f"classes: {classes}")
    # print(f"len: {num_classes}")

    config_dict = {
        "epochs": 5,
        "num-classes": num_classes
    }
    config = ConfigClassificationTrain(config_dict)
    main(config)
