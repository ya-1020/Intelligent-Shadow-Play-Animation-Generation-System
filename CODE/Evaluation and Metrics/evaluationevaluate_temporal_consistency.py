"""
evaluate_temporal_consistency.py
----------------------------------
Evaluate temporal consistency of generated
animation using keypoint motion smoothness.

Metrics:
- Mean Velocity Difference
- Mean Acceleration Difference
"""

import argparse
import numpy as np


# -----------------------------
# Load keypoint sequence
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
# Temporal consistency metrics
# -----------------------------
def compute_velocity(kp):
    """
    Compute per-frame velocity.

    Returns:
        velocity: (T-1, N, 2)
    """
    return kp[1:] - kp[:-1]


def compute_acceleration(velocity):
    """
    Compute per-frame acceleration.

    Returns:
        acceleration: (T-2, N, 2)
    """
    return velocity[1:] - velocity[:-1]


def mean_motion_difference(motion):
    """
    Compute mean L2 norm of motion differences.

    Args:
        motion: np.ndarray (T, N, 2)

    Returns:
        mean_diff: float
    """
    diff = np.linalg.norm(motion, axis=2)  # (T, N)
    return diff.mean()


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal consistency of keypoint sequences"
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        help="Keypoint sequence (.npy)"
    )
    args = parser.parse_args()

    kp = load_keypoints(args.keypoints)

    velocity = compute_velocity(kp)
    acceleration = compute_acceleration(velocity)

    vel_diff = mean_motion_difference(velocity)
    acc_diff = mean_motion_difference(acceleration)

    print("===================================")
    print(f"Keypoint file          : {args.keypoints}")
    print("Metrics:")
    print(f"  Mean Velocity Diff   : {vel_diff:.4f}")
    print(f"  Mean Acceleration Diff: {acc_diff:.4f}")
    print("Lower values indicate better temporal consistency")
    print("===================================")


if __name__ == "__main__":
    main()
