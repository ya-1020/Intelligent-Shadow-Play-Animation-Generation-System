"""
dtw_alignment.py
----------------------------------
Dynamic Time Warping (DTW) alignment
for evaluating temporal consistency of
shadow puppet action sequences.

Input: keypoint sequences (T, N, 2)
Output: DTW distance
"""

import numpy as np
import argparse
import os


# -----------------------------
# DTW implementation
# -----------------------------
def dtw_distance(seq1, seq2, window=None):
    """
    Compute DTW distance between two sequences.

    Args:
        seq1: np.ndarray, shape (T1, D)
        seq2: np.ndarray, shape (T2, D)
        window: int or None, Sakoe-Chiba window size

    Returns:
        dtw_dist: float
    """
    T1, T2 = len(seq1), len(seq2)

    if window is None:
        window = max(T1, T2)
    else:
        window = max(window, abs(T1 - T2))

    dtw = np.full((T1 + 1, T2 + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(max(1, i - window), min(T2 + 1, i + window)):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1]
            )

    return dtw[T1, T2]


# -----------------------------
# Sequence preprocessing
# -----------------------------
def load_keypoint_sequence(path):
    """
    Load keypoint sequence from .npy file.

    Expected shape: (T, N, 2)
    Returns flattened sequence: (T, N*2)
    """
    seq = np.load(path)
    T, N, C = seq.shape
    assert C == 2, "Keypoints must be 2D coordinates"
    return seq.reshape(T, N * 2)


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="DTW alignment evaluation")
    parser.add_argument("--ref", type=str, required=True,
                        help="Reference keypoint sequence (.npy)")
    parser.add_argument("--gen", type=str, required=True,
                        help="Generated keypoint sequence (.npy)")
    parser.add_argument("--window", type=int, default=10,
                        help="DTW window size (frames)")
    args = parser.parse_args()

    ref_seq = load_keypoint_sequence(args.ref)
    gen_seq = load_keypoint_sequence(args.gen)

    dist = dtw_distance(ref_seq, gen_seq, window=args.window)

    print("===================================")
    print(f"Reference sequence: {args.ref}")
    print(f"Generated sequence: {args.gen}")
    print(f"DTW window size   : w = {args.window}")
    print(f"DTW distance      : {dist:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
