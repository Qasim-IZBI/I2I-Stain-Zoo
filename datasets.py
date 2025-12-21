# datasets.py

import os
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(dir_path):
    images = []
    assert os.path.isdir(dir_path), f"{dir_path} is not a valid directory"
    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class UnpairedImageDataset(Dataset):
    """
    Expects:
      root/trainA/*.jpg
      root/trainB/*.jpg
    """

    def __init__(self, root, phase="train", load_size=286, crop_size=256):
        super().__init__()
        self.dir_A = os.path.join(root, phase + "A", 'images')
        self.dir_B = os.path.join(root, phase + "B", 'images')

        self.paths_A = sorted(make_dataset(self.dir_A))
        self.paths_B = sorted(make_dataset(self.dir_B))

        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)

        # Data augmentation similar to CycleGAN
        self.transform = transforms.Compose(
            [
                transforms.Resize((load_size, load_size), Image.BICUBIC),
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return max(self.size_A, self.size_B)

    def __getitem__(self, index):
        path_A = self.paths_A[index % self.size_A]
        path_B = random.choice(self.paths_B)  # random unpaired B

        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {
            "A": img_A,
            "B": img_B,
            "path_A": path_A,
            "path_B": path_B,
        }
