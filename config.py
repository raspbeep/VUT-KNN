import torch
import albumentations as a
from albumentations.pytorch import ToTensorV2

printed = False

def get_device():
    global printed
    if torch.cuda.is_available():
        if not printed:
            print('[USING CUDA]')
        printed = True
        return 'cuda'
    if torch.backends.mps.is_available():
        if not printed:
            print('[USING MPS]')
        printed = True
        return 'mps'
    if not printed:
        print('[USING CPU]')
    printed = True
    return 'cpu'


DEVICE = get_device(printed)

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
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLES = 10
NUM_WORKERS = 6
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_C1 = 'gen1.pth.tar'
CHECKPOINT_GEN_C2 = 'gen2.pth.tar'
CHECKPOINT_DISC_C2 = 'critic1.pth.tar'
CHECKPOINT_DISC_C1 = 'critic2.pth.tar'

transforms = a.Compose(
    [
        a.Resize(width=256, height=256),
        a.HorizontalFlip(p=0.5),
        a.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
],
    additional_targets={'image0': 'image'}
)

