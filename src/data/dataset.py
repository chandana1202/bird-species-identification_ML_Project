import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGES_DIR = os.path.join("data", "CUB_200_2011", "images")


def resolve_path(path, images_dir=IMAGES_DIR):
    """Rebuild image path so the CSV works across Windows and Colab."""
    rel = path.replace("\\", "/").split("images/", 1)[1]
    return os.path.join(images_dir, *rel.split("/"))


def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.6, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class BirdDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None,
        images_dir=IMAGES_DIR
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = resolve_path(
            row["path"],
            self.images_dir
        )

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, int(row["label"])