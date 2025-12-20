"""
inference_recognition.py
----------------------------------
Inference script for action recognition using HRNet-W32 + AAM.

Input: preprocessed keypoint heatmaps (.npy)
Output: classification accuracy and per-sample predictions
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.hrnet_aam import HRNetAAM


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Action recognition inference")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test heatmap files (.npy)")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Path to test labels file (txt)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of action classes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="predictions.txt",
                        help="File to save per-sample predictions")
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
        return heatmap, label, name


# -----------------------------
# Inference function
# -----------------------------
def inference():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ActionDataset(args.data_dir, args.label_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    model = HRNetAAM(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for heatmaps, labels, names in tqdm(dataloader, desc="Inference"):
            heatmaps = heatmaps.to(device)
            labels = labels.to(device)

            outputs = model(heatmaps)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for name, pred, gt in zip(names, preds.cpu().numpy(), labels.cpu().numpy()):
                predictions.append(f"{name}\t{pred}\t{gt}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"[INFO] Action classification accuracy: {accuracy * 100:.2f}%")

    # Save per-sample predictions
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("sample\tpredicted_label\tground_truth\n")
        for line in predictions:
            f.write(line + "\n")

    print(f"[INFO] Predictions saved to {args.output_file}")


if __name__ == "__main__":
    inference()
