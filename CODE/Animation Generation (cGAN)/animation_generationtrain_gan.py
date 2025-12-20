"""
train_gan.py
----------------------------------
Train a conditional GAN for shadow puppet animation generation.

Input: pose heatmaps (.npy) and paired target images
Output: trained generator and discriminator weights
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models.unet_generator import UNetGenerator
from models.patchgan_discriminator import PatchGANDiscriminator


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train cGAN for animation generation")
    parser.add_argument("--heatmap_dir", type=str, required=True,
                        help="Directory of input pose heatmaps (.npy)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of paired target images")
    parser.add_argument("--output_dir", type=str, default="gan_checkpoints")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--label_smoothing", action="store_true")
    parser.add_argument("--instance_noise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# -----------------------------
# Dataset
# -----------------------------
class PairedDataset(Dataset):
    def __init__(self, heatmap_dir, image_dir, image_size):
        self.heatmap_dir = heatmap_dir
        self.image_dir = image_dir
        self.names = sorted([
            f.replace(".npy", "")
            for f in os.listdir(heatmap_dir) if f.endswith(".npy")
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        heatmap = np.load(os.path.join(self.heatmap_dir, name + ".npy"))
        heatmap = torch.from_numpy(heatmap).float()

        img = Image.open(os.path.join(self.image_dir, name + ".png")).convert("RGB")
        img = self.transform(img)

        return heatmap, img, name


# -----------------------------
# Utility
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Training loop
# -----------------------------
def train():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PairedDataset(args.heatmap_dir, args.image_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    opt_g = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_g_loss = float("inf")
    patience, wait = 15, 0

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        g_loss_epoch, d_loss_epoch = 0.0, 0.0

        for heatmap, real_img, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
            heatmap = heatmap.to(device)
            real_img = real_img.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            fake_img = generator(heatmap).detach()

            if args.instance_noise:
                noise = torch.randn_like(real_img) * 0.05
                real_img = real_img + noise
                fake_img = fake_img + noise

            real_pred = discriminator(heatmap, real_img)
            fake_pred = discriminator(heatmap, fake_img)

            real_label = torch.ones_like(real_pred)
            fake_label = torch.zeros_like(fake_pred)

            if args.label_smoothing:
                real_label *= 0.9

            loss_d = (
                criterion_gan(real_pred, real_label) +
                criterion_gan(fake_pred, fake_label)
            ) * 0.5

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            fake_img = generator(heatmap)
            pred = discriminator(heatmap, fake_img)

            loss_g_gan = criterion_gan(pred, torch.ones_like(pred))
            loss_g_l1 = criterion_l1(fake_img, real_img) * args.lambda_l1
            loss_g = loss_g_gan + loss_g_l1

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_loss_epoch += loss_g.item()
            d_loss_epoch += loss_d.item()

        g_loss_epoch /= len(loader)
        d_loss_epoch /= len(loader)

        print(f"[Epoch {epoch+1}] G_loss: {g_loss_epoch:.4f}, D_loss: {d_loss_epoch:.4f}")

        # Early stopping on generator loss
        if g_loss_epoch < best_g_loss:
            best_g_loss = g_loss_epoch
            wait = 0
            torch.save(generator.state_dict(),
                       os.path.join(args.output_dir, "generator_best.pth"))
            torch.save(discriminator.state_dict(),
                       os.path.join(args.output_dir, "discriminator_best.pth"))
        else:
            wait += 1
            if wait >= patience:
                print("[INFO] Early stopping triggered.")
                break

        # Save example outputs
        if (epoch + 1) % 10 == 0:
            save_image(fake_img[:4],
                       os.path.join(args.output_dir, f"epoch_{epoch+1}.png"),
                       normalize=True)

    print("[INFO] GAN training completed.")


if __name__ == "__main__":
    train()
