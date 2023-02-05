from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted

class StylePhotoDataset(Dataset):
    def __init__(self, root_photo, root_style, transform=None):
        self.root_photo = root_photo
        self.root_style = root_style
        self.transform = transform
        self.photo_images = os.listdir(root_photo)
        self.photo_images = natsorted(self.photo_images)
        self.style_images = os.listdir(root_style)
        self.length_dataset = max(len(self.photo_images), len(self.style_images)) # 1000, 1500
        self.photo_len = len(self.photo_images)
        self.style_len = len(self.style_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        photo_img = self.photo_images[index % self.photo_len]
        style_img = self.style_images[index % self.style_len]

        photo_path = os.path.join(self.root_photo, photo_img)
        style_path = os.path.join(self.root_style, style_img)

        photo_img = np.array(Image.open(photo_path).convert("RGB"))
        style_img = np.array(Image.open(style_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=photo_img, image0=style_img)
            photo_img = augmentations["image0"]
            style_img = augmentations["image"]

        return photo_img, style_img
