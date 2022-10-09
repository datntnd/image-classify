import bentoml
from PIL import Image
import json
from Config import ConfigClassificationPredict
from get_model import get_model
import torch
from torchvision import transforms

classes = json.load(open('data/classes.json'))
num_classes = len(classes)
config_dict = {
    "epochs": 100,
    "num-classes": num_classes
}
config = ConfigClassificationPredict(config_dict)
weight_path = f"weights/{config.model_name}/best.pth"
# weight_path = 'weights/best.pth'
model = get_model(config.model_name, config.num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(weight_path, map_location=device))

saved_model = bentoml.pytorch.save_model(
    config.model_name,
    model
)
print(f"Saved model: {saved_model}")

# Test compare server and model predict.
# img = Image.open("data/testing/images/450128527_fd35742d44_jpg.jpg")
# img_size = 224
# data_transform = transforms.Compose(
#     [transforms.Resize(int(img_size)),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# # Transform
# model.to("cpu")
# input = data_transform(img)
# print(input.shape)
# input = input.unsqueeze(0)
# model.eval()
# output = model(input)
# print(output)
