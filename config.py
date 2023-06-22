import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Datasets/Train"
TEST_DIR = "Datasets/Test"

LEARNING_RATE = 1e-5
GAMMA_CYCLE = 0.1
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 5 # 0.5*lambda_cycle
NUM_EPOCHS = 1
LOAD_MODEL = True
SAVE_MODEL = False
TRAIN_MODEL = False
TRANSFORMATION = "HorseToZebra"                          
BETTER= False

CHECKPOINT_GEN_A = "MyDrive/Checkpoints/HorseToZebra/gen_a.pth.tar"     
CHECKPOINT_GEN_B= "MyDrive/Checkpoints/HorseToZebra/gen_b.pth.tar"
CHECKPOINT_DISC_A = "MyDrive/Checkpoints/HorseToZebra/disc_a.pth.tar"
CHECKPOINT_DISC_B = "MyDrive/Checkpoints/HorseToZebra/disc_b.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
