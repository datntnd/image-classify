import torch
from torch import nn
from torchvision import models

state_dict = {
    "resnet18": "pretrained_weights/resnet18-f37072fd.pth",
    "mobilenet_v2": "pretrained_weights/mobilenet_v2-b0353104.pth",
    "efficientnet_b0": "pretrained_weights/efficientnet_b0_rwightman-3dd342df.pth",
    "convnext_tiny": "pretrained_weights/convnext_tiny-983f1562.pth"
}


def get_model(model_name, num_classes):
    global model
    assert model_name in ['resnet18', 'efficientnet_b0', 'mobilenet_v2', 'convnext_tiny'], \
        f"No have {model_name} in models list"
    if model_name == 'resnet18':
        model = models.resnet18()
        model.load_state_dict(torch.load(state_dict[model_name]))
        model.fc = nn.Linear(512, num_classes)
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
        model.load_state_dict(torch.load(state_dict[model_name]))
        model.classifier[1] = nn.Linear(1280, num_classes)
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0()
        model.load_state_dict(torch.load(state_dict[model_name]))
        model.classifier[1] = nn.Linear(1280, num_classes)
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny()
        model.load_state_dict(torch.load(state_dict[model_name]))
        model.classifier[3] = nn.Linear(1280, num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


if __name__ == "__main__":
    model = get_model(model_name='efficientnet_b0', num_classes=100)
    model.eval()
