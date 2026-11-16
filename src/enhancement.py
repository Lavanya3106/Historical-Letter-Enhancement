"""
enhancement.py - contrast enhancement written from scratch (NumPy only).
Input and output are grayscale uint8 images.
"""

import numpy as np
from src.utils import to_uint8


def histogram(gray):
    """Count how many pixels have each brightness 0..255."""
    return np.bincount(gray.ravel(), minlength=256)


def contrast_stretch(gray, low_pct=1, high_pct=99):
    """Stretch the used brightness range to fill 0..255.

    We ignore the darkest 1% and brightest 1% of pixels (low_pct / high_pct)
    so a few odd pixels (a black speck, a bright glare) can't ruin the range.

    new = (old - low) / (high - low) * 255
    """
    cdf = np.cumsum(histogram(gray)) / gray.size * 100     # % of pixels up to each level
    low = int(np.searchsorted(cdf, low_pct))               # first level reaching low_pct %
    high = int(np.searchsorted(cdf, high_pct))
    if high <= low:                                        # flat image, nothing to stretch
        return gray.copy()

    stretched = (gray.astype(float) - low) / (high - low) * 255
    return to_uint8(stretched)


def gamma_correction(gray, gamma=0.7):
    """Brighten or darken with a curve.

    new = 255 * (old / 255) ** gamma
    gamma < 1 -> brighter,  gamma > 1 -> darker.
    """
    lut = 255 * (np.arange(256) / 255.0) ** gamma          # pre-compute for all 256 levels
    return to_uint8(lut)[gray]


def histogram_equalization(gray):
    """Spread brightness values so every level is used about equally.

    Uses the cumulative histogram (CDF) as the brightness curve:
    new = (cdf(old) - cdf_min) / (total - cdf_min) * 255
    """
    hist = histogram(gray)
    cdf = np.cumsum(hist)
    cdf_min = cdf[hist > 0][0]                             # CDF at the first used level
    total = gray.size
    if total == cdf_min:                                   # single-colour image
        return gray.copy()

    lut = (cdf - cdf_min) / (total - cdf_min) * 255
    return to_uint8(np.round(lut))[gray]


def clahe(gray, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE = Contrast Limited Adaptive Histogram Equalization.

    1. Cut the image into tiles (tile_grid = rows x columns).
    2. For each tile build a brightness curve (like histogram equalization),
       but CLIP tall histogram bars first so the boost stays gentle.
       clip_limit: 1 = almost no change, bigger = stronger contrast.
    3. Each pixel blends the curves of its 4 nearest tiles (bilinear
       interpolation) so there are no visible tile borders.
    """
    gy, gx = tile_grid
    h, w = gray.shape
    th = -(-h // gy)                  # tile height (rounded up)
    tw = -(-w // gx)                  # tile width  (rounded up)

    # pad so the image divides evenly into tiles
    padded = np.pad(gray, ((0, th * gy - h), (0, tw * gx - w)), mode="reflect")

    # --- step 1 + 2: one lookup table (brightness curve) per tile ---
    luts = np.zeros((gy, gx, 256))
    clip = max(1.0, clip_limit * th * tw / 256)      # max allowed height of a bar
    for i in range(gy):
        for j in range(gx):
            tile = padded[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            hist = np.bincount(tile.ravel(), minlength=256).astype(float)
            excess = np.maximum(hist - clip, 0).sum()        # what we cut off...
            hist = np.minimum(hist, clip) + excess / 256     # ...is shared evenly
            cdf = np.cumsum(hist)
            luts[i, j] = cdf / cdf[-1] * 255

    # --- step 3: blend the 4 nearest tile curves for every pixel ---
    ys = (np.arange(h) + 0.5) / th - 0.5             # pixel row in "tile-centre" units
    xs = (np.arange(w) + 0.5) / tw - 0.5
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    wy = (ys - y0)[:, None]                          # how far between the 2 tile rows
    wx = (xs - x0)[None, :]
    y1 = np.clip(y0 + 1, 0, gy - 1)[:, None]
    x1 = np.clip(x0 + 1, 0, gx - 1)[None, :]
    y0 = np.clip(y0, 0, gy - 1)[:, None]
    x0 = np.clip(x0, 0, gx - 1)[None, :]

    top_left = luts[y0, x0, gray]
    top_right = luts[y0, x1, gray]
    bottom_left = luts[y1, x0, gray]
    bottom_right = luts[y1, x1, gray]

    top = top_left * (1 - wx) + top_right * wx
    bottom = bottom_left * (1 - wx) + bottom_right * wx
    return to_uint8(np.round(top * (1 - wy) + bottom * wy))