"""
train_recognition.py
----------------------------------
Train action recognition model based on HRNet-W32 + AAM
using pose heatmap sequences.

Input: preprocessed keypoint heatmaps (.npy)
Output: trained model weights
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

from models.hrnet_aam import HRNetAAM


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train action recognition model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing heatmap sequences")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Path to action labels file (txt)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of action classes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# -----------------------------
# Dataset definition
# -----------------------------
class ActionDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.samples = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                name, label = line.strip().split()
                self.samples.append((name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        heatmap = np.load(os.path.join(self.data_dir, name))
        heatmap = torch.from_numpy(heatmap).float()
        return heatmap, label


# -----------------------------
# Training function
# -----------------------------
def train():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ActionDataset(args.data_dir, args.label_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = HRNetAAM(num_classes=args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    patience = 15
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for heatmaps, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            heatmaps = heatmaps.to(device)
            labels = labels.to(device)

            outputs = model(heatmaps)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pth")
            )
        else:
            wait += 1
            if wait >= patience:
                print("[INFO] Early stopping triggered.")
                break

    p
