

#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "tree/main/Datasets/trainWinterr"
TEST_DIR = "tree/main/Datasets/testWinter"
TRAIN_DIR_ALT="tree/main/Datasets/trainSummer"
TEST_DIR_ALT ="tree/main/Datasets/testSummer"
LEARNING_RATE = 1e-5
GAMMA_CYCLE = 0.1
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5 # 0.5*lambda_cycle
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = False
TRAIN_MODEL = True
DATASET_ORIGINAL= True                            
BCE= False
BETTER= False

CHECKPOINT_GEN_A = "C:\\Users\\Asus\\Desktop\\MaskCycleGAN\Save_Weights\\gen_a.pth.tar"     #!!!!!
CHECKPOINT_GEN_B= "C:\\Users\\Asus\\Desktop\\CycleGAN\\Save_Weights\\gen_b.pth.tar"
CHECKPOINT_DISC_A = "C:\\Users\\Asus\\Desktop\\CycleGAN\\Save_Weights\\disc_a.pth.tar"
CHECKPOINT_DISC_B = "C:\\Users\\Asus\\Desktop\\CycleGAN\\Save_Weights\\disc_b.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
