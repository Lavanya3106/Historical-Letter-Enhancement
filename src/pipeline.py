"""
pipeline.py - chains all the algorithms together.
The GUI (and any script) just calls run_pipeline() with the options it wants.
"""

import os
import numpy as np
from src.utils import to_grayscale, save_image
from src.preprocess import crop_to_document
from src.enhancement import contrast_stretch, gamma_correction, histogram_equalization, clahe
from src.denoise import mean_filter, gaussian_filter, median_filter
from src.threshold import global_threshold, otsu_threshold, adaptive_threshold
from src.morphology import (binary_to_ink, ink_to_binary, opening, closing,
                            remove_small_components, clear_border)

ENHANCE_OPTIONS = ["None", "Contrast Stretch", "Gamma Correction",
                   "Histogram Equalization", "CLAHE"]
DENOISE_OPTIONS = ["None", "Mean Filter", "Gaussian Filter", "Median Filter"]
THRESHOLD_OPTIONS = ["Global", "Otsu", "Adaptive"]
CLEANUP_OPTIONS = ["None", "Remove Specks + Clear Border", "Opening", "Closing"]


def run_pipeline(rgb, rotate=0, crop=True, crop_margin=0.03,
                 enhance="CLAHE", denoise="Median Filter", threshold="Adaptive",
                 cleanup="Remove Specks + Clear Border",
                 C=25, window=51, global_t=127, min_size=15):
    """Run the full restoration pipeline on a colour image (NumPy array).

    rotate : 0, 90, 180 or 270 degrees (counter-clockwise)
    Returns (final_binary_image, steps) where steps is a dict of the
    intermediate images (handy for showing / debugging).
    """
    steps = {}
    steps["Original"] = np.ascontiguousarray(np.rot90(rgb, (rotate // 90) % 4))

    # 1. grayscale + rotate + crop away the table
    gray = to_grayscale(rgb)
    gray = np.ascontiguousarray(np.rot90(gray, (rotate // 90) % 4))
    img = crop_to_document(gray, crop_margin) if crop else gray
    steps["Grayscale / Cropped"] = img

    # 2. enhancement
    if enhance == "Contrast Stretch":
        img = contrast_stretch(img)
    elif enhance == "Gamma Correction":
        img = gamma_correction(img, 0.7)
    elif enhance == "Histogram Equalization":
        img = histogram_equalization(img)
    elif enhance == "CLAHE":
        img = clahe(img, clip_limit=2.0, tile_grid=(8, 8))
    steps["Enhanced"] = img

    # 3. noise removal
    if denoise == "Mean Filter":
        img = mean_filter(img, 3)
    elif denoise == "Gaussian Filter":
        img = gaussian_filter(img, 5, 1.0)
    elif denoise == "Median Filter":
        img = median_filter(img, 3)
    steps["Denoised"] = img

    # 4. binarization
    if threshold == "Global":
        binary = global_threshold(img, global_t)
    elif threshold == "Otsu":
        binary, _ = otsu_threshold(img)
    else:
        binary = adaptive_threshold(img, window=int(window), C=C)
    steps["Binarized"] = binary

    # 5. cleanup
    ink = binary_to_ink(binary)
    if cleanup == "Remove Specks + Clear Border":
        ink = clear_border(remove_small_components(ink, int(min_size)), 0.015)
    elif cleanup == "Opening":
        ink = opening(ink, 3)
    elif cleanup == "Closing":
        ink = closing(ink, 3)
    final = ink_to_binary(ink)
    steps["Final"] = final

    return final, steps


def save_steps(steps, folder):
    """Save every intermediate stage as a numbered PNG inside `folder`.

    Example output:  00_original.png, 01_grayscale_cropped.png, 02_enhanced.png ...
    """
    os.makedirs(folder, exist_ok=True)
    for i, (name, arr) in enumerate(steps.items()):
        clean = name.lower().replace(" / ", "_").replace(" ", "_")
        save_image(arr, os.path.join(folder, f"{i:02d}_{clean}.png"))