#!/usr/bin/env python
# coding: utf-8

# In[1]:

from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, summer_dir, winter_dir, transform):
        self.root_summer = summer_dir
        self.root_winter = winter_dir
        self.transform = transform

        self.summer_images = os.listdir(summer_dir)
        self.winter_images = os.listdir(winter_dir)
        self.summer_length = len(self.summer_images)
        self.winter_length = len(self.winter_images)

    def __len__(self):
        return max(len(self.summer_images), len(self.winter_images))

    def __getitem__(self, index):
        summer_img = self.summer_images[index % self.summer_length]
        winter_img = self.winter_images[index % self.winter_length]

        summer_path = os.path.join(self.root_summer, summer_img)
        winter_path = os.path.join(self.root_winter, winter_img)

        summer_img = np.array(Image.open(summer_path).convert("RGB"))
        winter_img = np.array(Image.open(winter_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=summer_img, image0=winter_img)
            summer_img = augmentations["image"]
            winter_img = augmentations["image0"]

        return summer_img, winter_img
