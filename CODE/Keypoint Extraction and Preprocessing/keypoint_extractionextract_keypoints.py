"""
extract_keypoints.py
----------------------------------
Extract human pose keypoints from input images or video frames
using PaddleHub pose estimation models.

This script is used for shadow puppet action analysis.
"""

import os
import cv2
import json
import argparse
import paddlehub as hub
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract pose keypoints using PaddleHub")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images or video frames"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted keypoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openpose_body25",
        help="PaddleHub pose model name (default: openpose_body25)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load PaddleHub pose estimation model
    pose_model = hub.Module(name=args.model)

    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    print(f"[INFO] Found {len(image_files)} images for keypoint extraction.")

    for img_name in tqdm(image_files):
        img_path = os.path.join(args.input_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] Failed to load image: {img_name}")
            continue

        # PaddleHub expects BGR images
        result = pose_model.keypoint_detection(
            images=[image],
            use_gpu=True
        )

        keypoints_data = []

        if result and "data" in result[0]:
            persons = result[0]["data"]
            for person in persons:
                keypoints_data.append({
                    "keypoints": person["keypoints"],
                    "score": person.get("score", 1.0)
                })

        output_path = os.path.join(
            args.output_dir,
            img_name.replace(".jpg", ".json").replace(".png", ".json")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(keypoints_data, f, ensure_ascii=False, indent=2)

    print("[INFO] Keypoint extraction completed.")


if __name__ == "__main__":
    main()
