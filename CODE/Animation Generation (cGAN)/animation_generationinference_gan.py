"""
inference_gan.py
----------------------------------
Inference script for conditional GAN animation generation.

Input: pose heatmaps (.npy)
Output: generated shadow puppet images
"""

import os
import argparse
import numpy as np
import torch
from torchvision.utils import save_image

from models.unet_generator import UNetGenerator


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GAN inference for animation generation")
    parser.add_argument("--heatmap_dir", type=str, required=True,
                        help="Directory containing pose heatmaps (.npy)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained generator checkpoint")
    parser.add_argument("--output_dir", type=str, default="gan_results")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# -----------------------------
# Main inference
# -----------------------------
def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load generator
    generator = UNetGenerator()
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.to(device)
    generator.eval()

    heatmap_files = sorted([
        f for f in os.listdir(args.heatmap_dir) if f.endswith(".npy")
    ])

    with torch.no_grad():
        for fname in heatmap_files:
            heatmap_path = os.path.join(args.heatmap_dir, fname)
            heatmap = np.load(heatmap_path)

            # (K, H, W) -> (1, K, H, W)
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).float().to(device)

            fake_img = generator(heatmap)

            out_name = fname.replace(".npy", ".png")
            out_path = os.path.join(args.output_dir, out_name)

            # Save image in [0,1] range
            save_image(fake_img, out_path, normalize=True)

            print(f"[INFO] Generated: {out_path}")

    print("[INFO] Inference completed.")


if __name__ == "__main__":
    main()
