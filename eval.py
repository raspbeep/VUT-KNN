import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import Class1Class2Dataset
from discriminator import Discriminator
from generator_model import Generator
from image_pool import ImagePool
from utils import load_from_checkpoint

def val_fn(gen_c1, gen_c2, loader, epoch, save_path):
    loop = tqdm(loader, leave=True)

    print('saving to: ', save_path)

    for idx, (c1, c2) in enumerate(loop):
        c1 = c1.to(config.DEVICE)
        c2 = c2.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_c1 = gen_c1(c2)
            fake_c2 = gen_c2(c1)

        save_image(c1 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_c1_real.png')
        save_image(c2 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_c2_real.png')
        save_image(fake_c1 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_h(c2).png')
        save_image(fake_c2 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_g(c1).png')

def main():
    disc_c1 = Discriminator(in_channels=3).to(config.DEVICE)
    disc_c2 = Discriminator(in_channels=3).to(config.DEVICE)
    gen_c2 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_c1 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    pool_c1 = ImagePool(config.IMAGE_BUFFER_CAP)
    pool_c2 = ImagePool(config.IMAGE_BUFFER_CAP)

    opt_disc = optim.Adam(
        list(disc_c1.parameters()) + list(disc_c2.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_c2.parameters()) + list(gen_c1.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if config.LOAD_MODEL:
        epoch = load_from_checkpoint(gen_c1, gen_c2, opt_gen, disc_c1, disc_c2, opt_disc, config.LEARNING_RATE, pool_c1, pool_c2)
        if epoch is None:
            epoch = 0
        else:
            print(f'Forward pass at checkpoint {epoch}')
    else:
        epoch = 0

    val_dataset = Class1Class2Dataset(
        root_c1='./data/' + config.C1_VAL_DIR,
        root_c2='./data/' + config.C1_VAL_DIR,
        transform=config.val_transforms,
    )

    loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f'[EVAL AT EPOCH {epoch}]')
    val_fn(
        gen_c1,
        gen_c2,
        loader,
        epoch,
        './saved_images'
    )

if __name__ == "__main__":
    main()
