import numpy as np
import os

os.makedirs("example_data", exist_ok=True)

# Parameters
T = 20   # number of frames
N = 17   # number of keypoints

# ---- Ground truth sequence (smooth motion) ----
gt = np.zeros((T, N, 2), dtype=np.float32)
for t in range(T):
    gt[t, :, 0] = np.linspace(50, 150, N) + t * 1.5   # x motion
    gt[t, :, 1] = np.linspace(80, 200, N)             # y static

# ---- Generated sequence (slightly noisy) ----
pred = gt + np.random.normal(scale=2.0, size=gt.shape)

np.save("example_data/gt_keypoints.npy", gt)
np.save("example_data/pred_keypoints.npy", pred)

print("Example keypoint sequences saved.")
