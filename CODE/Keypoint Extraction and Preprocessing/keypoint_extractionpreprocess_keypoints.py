"""
preprocess_keypoints.py
----------------------------------
Convert extracted keypoints (JSON format) into
normalized multi-channel heatmaps.

Each joint is rendered as a 2D Gaussian heatmap.
"""

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess keypoints to heatmaps")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing keypoint JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated heatmaps (.npy)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Output heatmap resolution (default: 256)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma for heatmap generation"
    )
    return parser.parse_args()


def generate_gaussian_heatmap(center_x, center_y, width, height, sigma):
    """Generate a single 2D Gaussian heatmap."""
    x = np.arange(0, width, 1, np.float32)
    y = np.arange(0, height, 1, np.float32)
    y = y[:, np.newaxis]

    heatmap = np.exp(
        -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
    )
    return heatmap


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith(".json")
    ])

    print(f"[INFO] Found {len(json_files)} keypoint files.")

    for json_name in tqdm(json_files):
        json_path = os.path.join(args.input_dir, json_name)

        with open(json_path, "r", encoding="utf-8") as f:
            persons = json.load(f)

        # Skip empty detections
        if len(persons) == 0:
            continue

        # Use the first detected person
        keypoints = persons[0]["keypoints"]

        # Sort joints by index to ensure consistent ordering
        joint_items = sorted(keypoints.items(), key=lambda x: int(x[0]))

        heatmaps = []

        for _, joint in joint_items:
            x, y = joint["x"], joint["y"]

            # Normalize coordinates to target resolution
            x_norm = np.clip(x / 1.0, 0, 1) * args.image_size
            y_norm = np.clip(y / 1.0, 0, 1) * args.image_size

            heatmap = generate_gaussian_heatmap(
                x_norm,
                y_norm,
                args.image_size,
                args.image_size,
                args.sigma
            )
            heatmaps.append(heatmap)

        heatmaps = np.stack(heatmaps, axis=0)  # [K, H, W]

        output_path = os.path.join(
            args.output_dir,
            json_name.replace(".json", ".npy")
        )
        np.save(output_path, heatmaps.astype(np.float32))

    print("[INFO] Heatmap generation completed.")


if __name__ == "__main__":
    main()
