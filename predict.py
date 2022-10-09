import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import MyDataset
from Config import ConfigClassificationPredict
from get_model import get_model


def predict(config, weight_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    batch_size = config.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    test_dataset = MyDataset.CustomImageDataset(annotation_folder="data/testing/labels",
                                                img_folder="data/testing/images", transform=data_transform)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=nw,
                                 )

    # read class_indict
    # json_path = 'classes.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # create model
    model = get_model(config.model_name, config.num_classes)
    # load model weights
    # model_weight_path = config.pretrained_weights
    # print(model_weight_path)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Calculate Accuracy
    correct = 0
    total = 0
    for imgs, labels in test_dataloader:
        print(imgs.shape)
        imgs = Variable(imgs.cuda())
        labels = Variable(labels.cuda())
        outputs = model(imgs)

        _, preds = torch.max(outputs.data, 1)

        total += len(labels)

        correct += torch.sum(preds == labels.data)

    accuracy = 100 * correct / float(total)
    print(f"accuracy: {accuracy}")
    return accuracy


if __name__ == '__main__':
    classes = json.load(open('data/classes.json'))
    num_classes = len(classes)
    config_dict = {
        "epochs": 100,
        "num-classes": num_classes
    }
    config = ConfigClassificationPredict(config_dict)
    predict(config, weight_path='weights/best.pth')
