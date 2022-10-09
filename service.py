from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
import torch
from PIL.Image import Image as PILImage
from torchvision import transforms

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray

from Config import ConfigClassificationTrain

if TYPE_CHECKING:
    from numpy.typing import NDArray

config = ConfigClassificationTrain({})
# print(f"config: {config.service_name}")
# print(f"config: {config.model_name}")


runner = bentoml.pytorch.get(f"{config.model_name}:latest").to_runner()

svc = bentoml.Service(name=config.service_name, runners=[runner])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> NDArray[t.Any]:
    assert isinstance(f, PILImage)

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Transform
    input = data_transform(f)

    # unsqueeze batch dimension, in case you are dealing with a single image
    input = input.unsqueeze(0)
    print(f"img shape: {input.shape}")

    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension. Then we will also add one batch
    # dimension
    output_tensor = await runner.async_run(input)
    print(f"output: {output_tensor}")
    _, preds = torch.max(output_tensor, 1)
    print(f"preds: {preds}")
    return to_numpy(preds)
