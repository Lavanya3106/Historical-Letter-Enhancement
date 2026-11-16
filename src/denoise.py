"""
denoise.py - noise removal filters written from scratch (NumPy only).
Input and output are grayscale uint8 images.
"""

import numpy as np
from src.utils import convolve2d, to_uint8


def mean_filter(gray, size=3):
    """Replace each pixel with the average of its size x size neighbours."""
    kernel = np.ones((size, size)) / (size * size)
    return to_uint8(convolve2d(gray, kernel))


def gaussian_kernel(size=5, sigma=1.0):
    """Build a bell-shaped weight grid: centre pixel counts most."""
    r = size // 2
    ax = np.arange(-r, r + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))     # 1D bell curve
    kernel = np.outer(g, g)                        # 2D bell = 1D x 1D
    return kernel / kernel.sum()                   # weights add up to 1


def gaussian_filter(gray, size=5, sigma=1.0):
    """Weighted average where closer neighbours matter more."""
    kernel = gaussian_kernel(size, sigma)
    return to_uint8(convolve2d(gray, kernel))


def median_filter(gray, size=3):
    """Replace each pixel with the MIDDLE value of its neighbours.

    Trick for speed: stack size*size shifted copies of the image, so for
    every pixel its neighbours line up in one column. Then take the median
    down that column.
    """
    r = size // 2
    h, w = gray.shape
    padded = np.pad(gray, r, mode="edge")

    shifted = []
    for i in range(size):
        for j in range(size):
            shifted.append(padded[i:i + h, j:j + w])

    stack = np.stack(shifted, axis=0)              # (size*size, h, w)
    return np.median(stack, axis=0).astype(np.uint8)