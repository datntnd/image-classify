import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def read_txt(file):
    f = open(file, "r")
    return f.read().strip('\n').strip(" ")


class CustomImageDataset(Dataset):
    def __init__(self, annotation_folder, img_folder, transform=None, target_transform=None):
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.annotation_dir = os.listdir(annotation_folder)
        self.img_dir = os.listdir(img_folder)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = json.load(open("data/classes.json"))

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        img_path = f"{self.img_folder}/{self.img_dir[idx]}"
        image = Image.open(img_path)
        label = self.classes[read_txt(f"{self.annotation_folder}/{self.annotation_dir[idx]}")]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, int(label)

    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


if __name__ == "__main__":
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = CustomImageDataset(annotation_folder="data/label", img_folder="data/img")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )
