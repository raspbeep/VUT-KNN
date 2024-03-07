import os

import torch
import albumentations as a
from albumentations.pytorch import ToTensorV2


def get_device():
    filename = 'device'

    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            device = file.read().strip()
            if device in ['cuda', 'cpu', 'mps']:
                return device

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Save the device type to the file
    with open(filename, 'w') as file:
        print(f'[USING {device}]')
        file.write(device)

    return device

DEVICE = get_device()

# directories
DATA_DIR = 'data'
TRAIN_DIR = f'{DATA_DIR}/train'
VAL_DIR = f'{DATA_DIR}/val'
C1_TRAIN_DIR = f'{TRAIN_DIR}/class1'
C2_TRAIN_DIR = f'{TRAIN_DIR}/class2'
C1_VAL_DIR = f'{VAL_DIR}/class1'
C2_VAL_DIR = f'{VAL_DIR}/class2'

# values from the paper
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_CYCLES = 10
LAMBDA_IDENTITY = 0.5 * LAMBDA_CYCLES
NUM_WORKERS = 2
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True

# maybe useful for transparent images in dataset
# slaps it on white background, transparent one is not representable in RGB
ADD_WHITE_BACKGROUND=False

CHECKPOINT_ALL='checkpoint.pth.tar'

transforms = a.Compose(
    [
        a.Resize(width=256, height=256),
        a.HorizontalFlip(p=0.5),
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
],
    additional_targets={'image0': 'image'}
)

