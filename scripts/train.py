"""
Training Script — U-Net Segmentation + ESRGAN Super-Resolution.
Trains on medical image datasets (DICOM/NIfTI).

Usage:
  python scripts/train.py --model unet --data_dir data/train --epochs 100
  python scripts/train.py --model esrgan --data_dir data/train --epochs 50
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.segmentation import UNet
from models.super_resolution import ESRGANGenerator, Discriminator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class MedicalSliceDataset(Dataset):
    """
    Loads preprocessed medical image slices from .npy files.
    Expected directory structure:
      data/train/images/*.npy  — float32 arrays (H, W)
      data/train/masks/*.npy   — uint8 arrays (H, W) for segmentation
    """

    def __init__(self, data_dir: str, mode: str = "segmentation", target_size: int = 256):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        self.image_files = sorted((self.data_dir / "images").glob("*.npy"))
        logger.info(f"Dataset: {len(self.image_files)} slices found in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.load(str(self.image_files[idx])).astype(np.float32)
        img = self._resize(img, self.target_size)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        if self.mode == "segmentation":
            mask_file = self.data_dir / "masks" / self.image_files[idx].name
            if mask_file.exists():
                mask = np.load(str(mask_file)).astype(np.long)
                mask = self._resize(mask.astype(np.float32), self.target_size).astype(np.int64)
                return img_tensor, torch.from_numpy(mask)
            return img_tensor, torch.zeros(self.target_size, self.target_size, dtype=torch.long)

        elif self.mode == "superresolution":
            # Low-res: downsample 4× then return both
            import cv2
            lr_size = self.target_size // 4
            lr = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
            return torch.from_numpy(lr).unsqueeze(0), img_tensor

        return img_tensor, img_tensor

    def _resize(self, arr, size):
        import cv2
        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────
# U-Net Training
# ──────────────────────────────────────────────

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training U-Net on {device}")

    dataset = MedicalSliceDataset(args.data_dir, mode="segmentation", target_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = UNet(in_channels=1, num_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="logs/unet")
    weights_path = Path("models/weights")
    weights_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        writer.add_scalar("Loss/train", avg_loss, epoch)
        logger.info(f"[U-Net] Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(weights_path / "unet_medical.pth"))
            logger.info(f"  ✓ Best model saved (loss={best_loss:.4f})")

    writer.close()
    logger.info("U-Net training complete.")


# ──────────────────────────────────────────────
# ESRGAN Training
# ──────────────────────────────────────────────

def train_esrgan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ESRGAN on {device}")

    dataset = MedicalSliceDataset(args.data_dir, mode="superresolution", target_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    generator = ESRGANGenerator(in_channels=1, num_rrdb=8).to(device)
    discriminator = Discriminator(in_channels=1).to(device)

    g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    d_opt = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.9, 0.99))

    pixel_loss = nn.L1Loss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    weights_path = Path("models/weights")
    weights_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir="logs/esrgan")

    for epoch in range(args.epochs):
        g_total = 0.0
        d_total = 0.0

        for lr_imgs, hr_imgs in loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            B = lr_imgs.size(0)

            # ── Discriminator step ──
            d_opt.zero_grad()
            fake_hr = generator(lr_imgs).detach()
            real_labels = torch.ones(B, 1, device=device)
            fake_labels = torch.zeros(B, 1, device=device)
            d_real = adversarial_loss(discriminator(hr_imgs), real_labels)
            d_fake = adversarial_loss(discriminator(fake_hr), fake_labels)
            d_loss = (d_real + d_fake) * 0.5
            d_loss.backward()
            d_opt.step()

            # ── Generator step ──
            g_opt.zero_grad()
            fake_hr = generator(lr_imgs)
            g_pixel = pixel_loss(fake_hr, hr_imgs)
            g_adv = adversarial_loss(discriminator(fake_hr), real_labels)
            g_loss = g_pixel * 1.0 + g_adv * 0.001
            g_loss.backward()
            g_opt.step()

            g_total += g_loss.item()
            d_total += d_loss.item()

        avg_g = g_total / len(loader)
        avg_d = d_total / len(loader)
        writer.add_scalar("Loss/generator", avg_g, epoch)
        writer.add_scalar("Loss/discriminator", avg_d, epoch)
        logger.info(f"[ESRGAN] Epoch {epoch+1}/{args.epochs} | G: {avg_g:.4f} | D: {avg_d:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), str(weights_path / "esrgan_medical.pth"))
            logger.info(f"  ✓ Generator checkpoint saved at epoch {epoch+1}")

    writer.close()
    logger.info("ESRGAN training complete.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train medical AI models")
    parser.add_argument("--model", choices=["unet", "esrgan", "3dcnn"], required=True)
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model == "unet":
        train_unet(args)
    elif args.model == "esrgan":
        train_esrgan(args)
    else:
        logger.error(f"Training for '{args.model}' not implemented in this script.")


if __name__ == "__main__":
    main()

