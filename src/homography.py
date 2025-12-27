import cv2
import numpy as np

def get_homography(frame):
    h, w, _ = frame.shape

    # Approximate volleyball court corners (adjust once per video)
    src = np.array([
        [0.15 * w, 0.2 * h],
        [0.85 * w, 0.2 * h],
        [0.85 * w, 0.85 * h],
        [0.15 * w, 0.85 * h],
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [600, 0],
        [600, 400],
        [0, 400],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    return H


def project_points(points, H):
    if not points or H is None:
        return []

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2)
