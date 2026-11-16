"""
threshold.py - binarization (turn a grayscale image into pure black & white).
Written from scratch with NumPy only.
"""

import numpy as np


def apply_threshold(gray, t):
    """Pixels brighter than t become white (255), the rest black (0)."""
    return np.where(gray > t, 255, 0).astype(np.uint8)


def global_threshold(gray, t=127):
    """Global thresholding: we choose one fixed cutoff ourselves."""
    return apply_threshold(gray, t)


def otsu_threshold_value(gray):
    """Find the best cutoff automatically (Otsu's method).

    Idea: try every possible cutoff t (0-255). For each t the image splits
    into a dark group and a bright group. Pick the t where the two groups are
    the most different from each other (largest 'between-class variance').
    """
    # 1. histogram: how many pixels have each brightness 0..255
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    prob = hist / hist.sum()                 # fraction of pixels per brightness

    levels = np.arange(256)
    omega = np.cumsum(prob)                  # fraction of pixels in dark group (<= t)
    mu = np.cumsum(prob * levels)            # running brightness sum for dark group
    mu_total = mu[-1]                        # average brightness of whole image

    # 2. between-class variance for every t at once
    denom = omega * (1.0 - omega)
    valid = denom > 0                        # avoid dividing by zero
    sigma_b = np.zeros(256)
    sigma_b[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denom[valid]

    # 3. the best t is where variance is largest
    return int(np.argmax(sigma_b))


def otsu_threshold(gray):
    """Otsu thresholding: find the best cutoff, then apply it."""
    t = otsu_threshold_value(gray)
    return apply_threshold(gray, t), t


def local_mean(gray, window):
    """Average brightness around every pixel (window x window neighbourhood).

    Speed trick: an 'integral image'. We pre-compute running sums once, then
    the sum of ANY rectangle needs only 4 lookups. Much faster than
    re-adding a 31x31 box for every single pixel.
    """
    r = window // 2
    h, w = gray.shape

    padded = np.pad(gray.astype(np.float64), r, mode="edge")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)))      # zero row/col at top-left

    box_sum = (integral[window:window + h, window:window + w]
               - integral[0:h, window:window + w]
               - integral[window:window + h, 0:w]
               + integral[0:h, 0:w])
    return box_sum / (window * window)


def adaptive_threshold(gray, window=31, C=10):
    """Adaptive (local mean) thresholding.

    A pixel becomes BLACK (ink) only if it is darker than the average of its
    neighbours by more than C. Otherwise it is WHITE (paper).

    window : size of the neighbourhood (must be odd). Make it bigger than the
             thickness of the letters.
    C      : how much darker than its surroundings a pixel must be.
    """
    if window % 2 == 0:
        window += 1
    mean = local_mean(gray, window)
    return np.where(gray > mean - C, 255, 0).astype(np.uint8)