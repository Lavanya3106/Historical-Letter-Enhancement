"""
process_dataset.py - runs the pipeline on EVERY image in the dataset folder and
saves each intermediate stage (deliverable: "intermediate outputs for every stage").

Run from the project folder with:  python process_dataset.py

Result:  output/<image name>/00_original.png, 01_grayscale_cropped.png,
         02_enhanced.png, 03_denoised.png, 04_binarized.png, 05_final.png
"""

import os
import time
from src.utils import load_image
from src.pipeline import run_pipeline, save_steps

DATASET_DIR = "dataset"
OUTPUT_DIR = "output"
VALID_EXT = (".jpg", ".jpeg", ".png")

files = sorted(f for f in os.listdir(DATASET_DIR) if f.lower().endswith(VALID_EXT))
print(f"Found {len(files)} images. Processing...\n")

start = time.time()
for n, name in enumerate(files, 1):
    try:
        rgb = load_image(os.path.join(DATASET_DIR, name), max_width=1600)
        _, steps = run_pipeline(rgb)            # default settings (best pipeline)
        save_steps(steps, os.path.join(OUTPUT_DIR, os.path.splitext(name)[0]))
        print(f"[{n}/{len(files)}] {name}  done")
    except Exception as e:                      # one bad image should not stop the rest
        print(f"[{n}/{len(files)}] {name}  FAILED: {e}")

print(f"\nFinished in {time.time() - start:.0f}s. Results are in the '{OUTPUT_DIR}' folder.")