import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import Class1Class2Dataset
from discriminator import Discriminator
from generator_model import Generator
from utils import load_checkpoint, save_checkpoint


def train_fn(disc_c1, disc_c2, gen_c1, gen_c2, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, save_path):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    print('saving to: ', save_path)

    for idx, (c1, c2) in enumerate(loop):
        c1 = c1.to(config.DEVICE)
        c2 = c2.to(config.DEVICE)        

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_c1 = gen_c1(c2)
            disc_c1_real = disc_c1(c1)
            disc_c1_fake = disc_c1(fake_c1.detach())
            H_reals += disc_c1_real.mean().item()
            H_fakes += disc_c1_fake.mean().item()
            disc_c1_real_loss = mse(disc_c1_real, torch.ones_like(disc_c1_real))
            disc_c1_fake_loss = mse(disc_c1_fake, torch.zeros_like(disc_c1_fake))
            disc_c1_loss = disc_c1_real_loss + disc_c1_fake_loss

            fake_c2 = gen_c2(c1)
            disc_c2_real = disc_c2(c2)
            disc_c2_fake = disc_c2(fake_c2.detach())
            disc_c2_real_loss = mse(disc_c2_real, torch.ones_like(disc_c2_real))
            disc_c2_fake_loss = mse(disc_c2_fake, torch.zeros_like(disc_c2_fake))
            disc_c2_loss = disc_c2_real_loss + disc_c2_fake_loss

            disc_loss = (disc_c1_loss + disc_c2_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            disc_c1_fake = disc_c1(fake_c1)
            disc_c2_fake = disc_c2(fake_c2)
            gen_c1_loss = mse(disc_c1_fake, torch.ones_like(disc_c1_fake))
            gen_c2_loss = mse(disc_c2_fake, torch.ones_like(disc_c2_fake))

            # cycle loss
            cycle_c2 = gen_c2(fake_c1)
            cycle_c1 = gen_c1(fake_c2)
            cycle_c2_loss = l1(c2, cycle_c2)
            cycle_c1_loss = l1(c1, cycle_c1)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # identity_zebra = gen_c2(c2)
            # identity_horse = gen_c1(c1)
            # identity_zebra_loss = l1(c2, identity_zebra)
            # identity_horse_loss = l1(c1, identity_horse)

            # add all togethor
            G_loss = (
                gen_c2_loss
                + gen_c1_loss
                + cycle_c1_loss * config.LAMBDA_CYCLES
                + cycle_c2_loss * config.LAMBDA_CYCLES
                # + identity_horse_loss * config.LAMBDA_IDENTITY
                # + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            # reals horse
            save_image(c1 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_c1_real.png')
            # reals zebra
            save_image(c2 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_c2_real.png')
            # fakes zebra
            save_image(fake_c1 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_h(c2).png')
            # fakes horse
            save_image(fake_c2 * 0.5 + 0.5, save_path + f'/epoch_{epoch}_idx_{idx}_g(c1).png')

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main(save_path=None):
    disc_c1 = Discriminator(in_channels=3).to(config.DEVICE)
    disc_c2 = Discriminator(in_channels=3).to(config.DEVICE)
    gen_c2 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_c1 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
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

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_C1,
            gen_c1,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_C1,
            gen_c2,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_C1,
            disc_c1,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_C2,
            disc_c2,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = Class1Class2Dataset(
        root_c1=config.C1_TRAIN_DIR,
        root_c2=config.C2_TRAIN_DIR,
        transform=config.transforms,
    )
    val_dataset = Class1Class2Dataset(
        root_c1=config.C1_VAL_DIR,
        root_c2=config.C2_VAL_DIR,
        transform=config.transforms,
    )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_c1,
            disc_c2,
            gen_c1,
            gen_c2,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            epoch,
            save_path
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_c1, opt_gen, filename=config.CHECKPOINT_GEN_C1)
            save_checkpoint(gen_c2, opt_gen, filename=config.CHECKPOINT_GEN_C1)
            save_checkpoint(disc_c1, opt_disc, filename=config.CHECKPOINT_DISC_C1)
            save_checkpoint(disc_c2, opt_disc, filename=config.CHECKPOINT_DISC_C2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGAN Training Script")
    parser.add_argument('--save', type=str, default='./saved_images',)
    
    args = parser.parse_args()
    
    main(save_path=args.save)

