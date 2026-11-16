"""
compare_algorithms.py - builds the ALGORITHM COMPARISON TABLE.

For each stage (enhancement, noise removal, binarization, cleanup) it runs every
algorithm on the dataset images, measures a few numbers, averages them, and writes:

    results/comparison_summary.md          <- paste these tables into the report
    results/comparison_<stage>.csv         <- one summary table per stage (opens in Excel)
    results/comparison_per_image.csv       <- every single measurement

Run from the project folder with:  python compare_algorithms.py

Fair test: when comparing one stage, the stages BEFORE it are fixed to our best
settings (Crop -> CLAHE -> Median -> Adaptive C=25), so only one thing changes.
We have no "perfect clean copy" of real old letters, so every measure below
works on a single image (no reference needed).
"""

import os
import csv
import time
import numpy as np

from src.utils import load_image, to_grayscale, convolve2d
from src.preprocess import crop_to_document
from src.enhancement import (contrast_stretch, gamma_correction,
                             histogram_equalization, clahe)
from src.denoise import mean_filter, gaussian_filter, median_filter
from src.threshold import (global_threshold, otsu_threshold,
                           otsu_threshold_value, adaptive_threshold)
from src.morphology import (binary_to_ink, erosion, opening, closing,
                            remove_small_components, clear_border)

# ----------------------------------------------------------------- settings
DATASET_DIR = "dataset"
RESULTS_DIR = "results"
VALID_EXT = (".jpg", ".jpeg", ".png")
MAX_IMAGES = 30        # compare on 30 evenly-spaced images (fast). Set to None for ALL images.
MIN_SPECK = 15         # ink blobs smaller than this many pixels count as specks


# ------------------------------------------------------- the algorithms
def identity(x):
    return x

ENHANCERS = {
    "None": identity,
    "Contrast Stretch": contrast_stretch,
    "Gamma Correction (0.7)": lambda g: gamma_correction(g, 0.7),
    "Histogram Equalization": histogram_equalization,
    "CLAHE": lambda g: clahe(g, 2.0, (8, 8)),
}
DENOISERS = {
    "None": identity,
    "Mean Filter 3x3": lambda g: mean_filter(g, 3),
    "Gaussian Filter 5x5": lambda g: gaussian_filter(g, 5, 1.0),
    "Median Filter 3x3": lambda g: median_filter(g, 3),
}
BINARIZERS = {
    "Global (t=127)": lambda g: global_threshold(g, 127),
    "Otsu": lambda g: otsu_threshold(g)[0],
    "Adaptive (C=25)": lambda g: adaptive_threshold(g, 51, 25),
}
CLEANERS = {
    "None": identity,
    "Opening 3x3": lambda ink: opening(ink, 3),
    "Closing 3x3": lambda ink: closing(ink, 3),
    "Remove Specks + Clear Border": lambda ink: clear_border(remove_small_components(ink, MIN_SPECK), 0.015),
}


# ---------------------------------------------------------- the measures
def separation(gray):
    """How well ink stands out from paper: (paper brightness - ink brightness) / paper unevenness.
    Higher = text is easier to pick out of the background."""
    t = otsu_threshold_value(gray)
    paper, ink = gray[gray > t].astype(float), gray[gray <= t].astype(float)
    if paper.size == 0 or ink.size == 0:
        return 0.0
    return float((paper.mean() - ink.mean()) / max(paper.std(), 1e-9))


def paper_mask(gray):
    """Pixels brighter than Otsu's cutoff = paper (not ink)."""
    return gray > otsu_threshold_value(gray)


def background_variation(gray):
    """How uneven the paper looks: std/mean of paper pixels in %. Lower = flatter, cleaner."""
    paper = gray[paper_mask(gray)].astype(float)
    return float(paper.std() / max(paper.mean(), 1) * 100) if paper.size else 0.0


def noise_level(gray, mask):
    """Tiny pixel-to-pixel jitter on the paper (std of pixel minus its 3x3 average)."""
    residual = gray.astype(float) - convolve2d(gray, np.ones((3, 3)) / 9)
    return float(residual[mask].std())


def gradient(gray):
    """How sharply brightness changes at each pixel (edge strength)."""
    g = gray.astype(float)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) / 2
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) / 2
    return np.hypot(gx, gy)


def speck_percent(ink):
    """% of ink pixels that belong to tiny blobs (< MIN_SPECK pixels)."""
    total = ink.sum()
    if total == 0:
        return 0.0
    kept = remove_small_components(ink, MIN_SPECK).sum()
    return float((total - kept) / total * 100)


def timed(fn, x):
    start = time.perf_counter()
    out = fn(x)
    return out, time.perf_counter() - start


# arrows show which direction is better
STAGES = {
    "Enhancement": ["Contrast (std) ↑", "Background variation % ↓", "Ink-paper separation ↑", "Time (s) ↓"],
    "Noise Removal": ["Noise level ↓", "Edge retention % ↑", "Time (s) ↓"],
    "Binarization": ["Ink coverage %", "Speck % of ink ↓", "Time (s) ↓"],
    "Cleanup": ["Ink retained % ↑", "Speck % of ink ↓", "Time (s) ↓"],
}


# ------------------------------------------------------------- main work
def measure_image(rgb, name, records):
    def add(stage, algo, values):
        for metric, v in zip(STAGES[stage], values):
            records.append((name, stage, algo, metric, v))

    gray = to_grayscale(rgb)
    cropped = crop_to_document(gray)

    # ---- Stage 1: enhancement (input = cropped grayscale) ----
    for algo, fn in ENHANCERS.items():
        out, t = timed(fn, cropped)
        add("Enhancement", algo, [float(out.std()), background_variation(out), separation(out), t])
    enhanced = ENHANCERS["CLAHE"](cropped)             # fixed for the next stages

    # ---- Stage 2: noise removal (input = CLAHE output) ----
    paper = paper_mask(enhanced)
    grad_in = gradient(enhanced)
    letters = remove_small_components(
        binary_to_ink(adaptive_threshold(enhanced, 51, 25)), MIN_SPECK)
    edges = letters & ~erosion(letters, 3)             # outline of the letters (specks excluded)
    if not edges.any():
        edges = grad_in >= np.percentile(grad_in, 95)
    base_edge = grad_in[edges].mean()
    for algo, fn in DENOISERS.items():
        out, t = timed(fn, enhanced)
        retention = gradient(out)[edges].mean() / max(base_edge, 1e-9) * 100
        add("Noise Removal", algo, [noise_level(out, paper), float(retention), t])
    denoised = DENOISERS["Median Filter 3x3"](enhanced)

    # ---- Stage 3: binarization (input = median-filtered CLAHE) ----
    for algo, fn in BINARIZERS.items():
        out, t = timed(fn, denoised)
        ink = binary_to_ink(out)
        add("Binarization", algo, [float(ink.mean() * 100), speck_percent(ink), t])
    ink_base = binary_to_ink(BINARIZERS["Adaptive (C=25)"](denoised))

    # ---- Stage 4: cleanup (input = adaptive binary result) ----
    base_pixels = max(ink_base.sum(), 1)
    for algo, fn in CLEANERS.items():
        out, t = timed(fn, ink_base)
        add("Cleanup", algo, [float(out.sum() / base_pixels * 100), speck_percent(out), t])


def summarise(records):
    """Average every measure over all images -> {stage: {algorithm: {metric: mean}}}."""
    bucket = {}
    for _, stage, algo, metric, v in records:
        bucket.setdefault(stage, {}).setdefault(algo, {}).setdefault(metric, []).append(v)
    return {s: {a: {m: float(np.mean(v)) for m, v in ms.items()} for a, ms in algos.items()}
            for s, algos in bucket.items()}


def write_outputs(records, summary, n_images):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "comparison_per_image.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "stage", "algorithm", "measure", "value"])
        w.writerows(records)

    md = [f"# Algorithm Comparison Table\n", f"Averages over {n_images} images. "
          "Arrows show the better direction (↑ higher is better, ↓ lower is better).\n"]
    for stage, metrics in STAGES.items():
        rows = summary[stage]
        header = ["Algorithm"] + metrics
        with open(os.path.join(RESULTS_DIR, f"comparison_{stage.lower().replace(' ', '_')}.csv"),
                  "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for algo, vals in rows.items():
                w.writerow([algo] + [round(vals[m], 3) for m in metrics])

        md.append(f"\n## {stage}\n")
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        for algo, vals in rows.items():
            md.append("| " + " | ".join([algo] + [f"{vals[m]:.2f}" for m in metrics]) + " |")

    with open(os.path.join(RESULTS_DIR, "comparison_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(DATASET_DIR) if f.lower().endswith(VALID_EXT))
    if MAX_IMAGES and len(files) > MAX_IMAGES:
        files = [files[i] for i in np.linspace(0, len(files) - 1, MAX_IMAGES).astype(int)]
    print(f"Comparing algorithms on {len(files)} images...\n")

    records, used = [], 0
    start = time.time()
    for n, name in enumerate(files, 1):
        try:
            rgb = load_image(os.path.join(DATASET_DIR, name), max_width=1600)
            measure_image(rgb, name, records)
            used += 1
            print(f"[{n}/{len(files)}] {name} done")
        except Exception as e:
            print(f"[{n}/{len(files)}] {name} FAILED: {e}")

    if used == 0:
        raise SystemExit("No image could be processed.")

    table = write_outputs(records, summarise(records), used)
    print(f"\nFinished in {time.time() - start:.0f}s.\n")
    print(table)
    print(f"\nSaved to the '{RESULTS_DIR}' folder (comparison_summary.md + CSV files).")