"""
utils.py - basic toolbox for the Historical Letter Enhancement project.
No cv2 / skimage / scipy used. Only NumPy for math, Pillow for opening/saving
images, and matplotlib for displaying them.
"""

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def load_image(path, max_width=800):
    """Open an image file and return it as a NumPy array (height x width x 3).

    Phone photos are huge, so we shrink them to max_width to keep our
    from-scratch algorithms fast.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # fixes phone photos that load sideways
    img = img.convert("RGB")

    w, h = img.size
    if w > max_width:
        new_h = int(h * max_width / w)
        img = img.resize((max_width, new_h))

    return np.array(img, dtype=np.uint8)


def to_grayscale(rgb):
    """Convert a colour image to grayscale using the standard weights.

    gray = 0.299*R + 0.587*G + 0.114*B
    (green gets the most weight because our eyes are most sensitive to it)
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def convolve2d(img, kernel):
    """Slide a small kernel (window of weights) over the image.

    At every pixel: multiply the neighbours by the kernel weights and add
    them up. Used later for the Mean and Gaussian filters.

    Trick for speed: instead of looping over every pixel, we loop over the
    kernel positions only (e.g. 9 times for a 3x3 kernel) and shift the
    whole image at once.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape

    padded = np.pad(img.astype(float), ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((h, w), dtype=float)

    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]

    return out


def to_uint8(img):
    """Clip values to 0-255 and convert to a proper image array."""
    return np.clip(img, 0, 255).astype(np.uint8)


def save_image(arr, path):
    """Save a NumPy array as an image file."""
    Image.fromarray(arr).save(path)


def show_images(images, titles, save_path=None):
    """Show several images side by side (handy for before/after)."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 6))
    if len(images) == 1:
        axes = [axes]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()