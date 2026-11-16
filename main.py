"""
main.py - runs the pipeline.
Run from the project folder with:  python main.py
"""

import os
from src.utils import load_image, to_grayscale, show_images
from src.threshold import adaptive_threshold
from src.preprocess import crop_to_document
from src.denoise import median_filter
from src.enhancement import clahe
from src.morphology import (binary_to_ink, ink_to_binary, opening, closing,
                            remove_small_components, clear_border)

DATASET_DIR = "dataset"
VALID_EXT = (".jpg", ".jpeg", ".png")

files = sorted(f for f in os.listdir(DATASET_DIR) if f.lower().endswith(VALID_EXT))
print(f"Found {len(files)} images")

# change this number to test other images
IMAGE_NUMBER = 13

path = os.path.join(DATASET_DIR, files[IMAGE_NUMBER])
gray = to_grayscale(load_image(path, max_width=1600))

# --- our best pipeline so far: Crop -> CLAHE -> Median -> Adaptive (C=25) ---
cropped = crop_to_document(gray)
enhanced = clahe(cropped, clip_limit=2.0, tile_grid=(8, 8))
denoised = median_filter(enhanced, 3)
binary = adaptive_threshold(denoised, window=51, C=25)

# --- cleanup with morphology ---
ink = binary_to_ink(binary)
opened = opening(ink, 3)
closed = closing(ink, 3)
cleaned = clear_border(remove_small_components(ink, min_size=15), frac=0.015)

show_images(
    [binary, ink_to_binary(opened), ink_to_binary(closed), ink_to_binary(cleaned)],
    ["Before cleanup", "Opening 3x3", "Closing 3x3", "Remove specks + clear border"],
)