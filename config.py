import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Datasets/Train"
TEST_DIR = "Datasets/Test"

LEARNING_RATE = 2e-5
GAMMA_CYCLE = 0.1
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0 # 0.5*lambda_cycle
NUM_EPOCHS = 10
LOAD_MODEL = false
SAVE_MODEL = True
TRAIN_MODEL = True
TRANSFORMATION = "HorseToZebra"                          
BETTER= False

CHECKPOINT_GEN_A = "../drive/MyDrive/Checkpoints/HorseToZebra5/gen_a.pth.tar"     
CHECKPOINT_GEN_B= "../drive/MyDrive/Checkpoints/HorseToZebra5/gen_b.pth.tar"
CHECKPOINT_DISC_A = "../drive/MyDrive/Checkpoints/HorseToZebra5/disc_a.pth.tar"
CHECKPOINT_DISC_B = "../drive/MyDrive/Checkpoints/HorseToZebra5/disc_b.pth.tar"

NEW_CHECKPOINT_GEN_A = "../drive/MyDrive/Checkpoints/HorseToZebraIdentity10/gen_a.pth.tar"     
NEW_CHECKPOINT_GEN_B= "../drive/MyDrive/Checkpoints/HorseToZebraIdentity10/gen_b.pth.tar"
NEW_CHECKPOINT_DISC_A = "../drive/MyDrive/Checkpoints/HorseToZebraIdentity10/disc_a.pth.tar"
NEW_CHECKPOINT_DISC_B = "../drive/MyDrive/Checkpoints/HorseToZebraIdentity10/disc_b.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
