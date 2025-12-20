"""
evaluate_keypoint_error.py
----------------------------------
Evaluate keypoint error between
generated animation and reference motion.

Metric: Mean Per-Joint Position Error (MPJPE)
"""

import argparse
import numpy as np
import os


# -----------------------------
# Load keypoint sequences
# -----------------------------
def load_keypoints(path):
    """
    Load keypoint sequence from .npy file.

    Expected shape:
        (T, N, 2)
    """
    kp = np.load(path)
    assert kp.ndim == 3 and kp.shape[2] == 2, \
        "Keypoints must have shape (T, N, 2)"
    return kp


# -----------------------------
# MPJPE computation
# -----------------------------
def compute_mpjpe(pred, gt):
    """
    Compute Mean Per-Joint Position Error (MPJPE).

    Args:
        pred: np.ndarray (T, N, 2)
        gt:   np.ndarray (T, N, 2)

    Returns:
        mpjpe: float
    """
    assert pred.shape == gt.shape, "Predicted and GT shapes must match"

    # Euclidean distance per joint per frame
    error = np.linalg.norm(pred - gt, axis=2)  # (T, N)
    mpjpe = error.mean()
    return mpjpe


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate keypoint position error (MPJPE)"
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Ground truth keypoint sequence (.npy)"
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Predicted/generated keypoint sequence (.npy)"
    )
    args = parser.parse_args()

    gt_kp = load_keypoints(args.gt)
    pred_kp = load_keypoints(args.pred)

    mpjpe = compute_mpjpe(pred_kp, gt_kp)

    print("===================================")
    print(f"GT sequence   : {args.gt}")
    print(f"Pred sequence : {args.pred}")
    print(f"Metric        : MPJPE (pixels)")
    print(f"Keypoint Error: {mpjpe:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
