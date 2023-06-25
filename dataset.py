from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, domainB_dir, domainA_dir, transform):
        self.root_domainB = domainB_dir
        self.root_domainA = domainA_dir
        self.transform = transform

        self.domainB_images = os.listdir(domainB_dir)
        self.domainA_images = os.listdir(domainA_dir)
        self.domainB_length = len(self.domainB_images)
        self.domainA_length = len(self.domainA_images)

    def __len__(self):
        return max(len(self.domainB_images), len(self.domainA_images))

    def __getitem__(self, index):
        domainB_img = self.domainB_images[index % self.domainB_length]
        domainA_img = self.domainA_images[index % self.domainA_length]

        domainB_path = os.path.join(self.root_domainB, domainB_img)
        domainA_path = os.path.join(self.root_domainA, domainA_img)

        domainB_img = np.array(Image.open(domainB_path).convert("RGB"))
        domainA_img = np.array(Image.open(domainA_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=domainB_img, image0=domainA_img)
            domainA_img = augmentations["image"]
            domainB_img = augmentations["image0"]

        return domainB_img, domainA_img
