import torch
from dataset import StylePhotoDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator_resnet import Generator
from generator_unet import Generator


def train_fn(disc_S, disc_P, gen_P, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    print(loader)

    S_reals = 0
    S_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (photo, style) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        style = style.to(config.DEVICE)

        # Train Two Discriminators S(style) and P(photo)
        with torch.cuda.amp.autocast():
            fake_style = gen_S(photo)
            D_S_real = disc_S(style)
            D_S_fake = disc_S(fake_style.detach())
            S_reals += D_S_real.mean().item()
            S_fakes += D_S_fake.mean().item()
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            fake_photo = gen_P(style)
            D_P_real = disc_P(photo)
            D_P_fake = disc_P(fake_photo.detach())
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss

            D_loss = (D_S_loss + D_P_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_S_fake = disc_S(fake_style)
            D_P_fake = disc_P(fake_photo)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

            # cycle loss
            cycle_photo = gen_P(fake_style)
            cycle_style = gen_S(fake_photo)
            cycle_photo_loss = l1(photo, cycle_photo)
            cycle_style_loss = l1(style, cycle_style)

            # identity loss
            # identity_photo = gen_P(photo)
            # identity_style = gen_H(style)
            # identity_photo_loss = l1(photo, identity_photo)
            # identity_style_loss = l1(style, identity_style)

            G_loss = (
                loss_G_P
                + loss_G_S
                + cycle_photo_loss * config.LAMBDA_CYCLE
                + cycle_style_loss * config.LAMBDA_CYCLE
                # + identity_style_loss * config.LAMBDA_IDENTITY
                # + identity_photo_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # if idx < 50:

        save_image(fake_photo*0.5+0.5, f"ukiyoe_unet_sobel2_result/{idx}.png")

        loop.set_postfix(S_real=S_reals/(idx+1), S_fake=S_fakes/(idx+1))


def main():
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    disc_P = Discriminator(in_channels=3).to(config.DEVICE)
    if config.GENERATOR == "resnet":
        gen_P = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
        gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    if config.GENERATOR == "unet":
        gen_P = Generator(in_channels=3, features=64).to(config.DEVICE)
        gen_S = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_P.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_P.parameters()) + list(gen_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_S, gen_S, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_S, disc_S, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_P, disc_P, opt_disc, config.LEARNING_RATE,
        )

    dataset = StylePhotoDataset(
        root_style=config.TRAIN_DIR+"/trainA", root_photo=config.TRAIN_DIR+"/trainB", transform=config.transforms
    )
    val_dataset = StylePhotoDataset(
        root_style=config.VAL_DIR+"/testA", root_photo=config.VAL_DIR+"/testB", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_S, disc_P, gen_P, gen_S, val_loader,
                 opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        if config.SAVE_MODEL:
            save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_P)
            save_checkpoint(disc_S, opt_disc,
                            filename=config.CHECKPOINT_CRITIC_S)
            save_checkpoint(disc_P, opt_disc,
                            filename=config.CHECKPOINT_CRITIC_P)


if __name__ == "__main__":
    main()
