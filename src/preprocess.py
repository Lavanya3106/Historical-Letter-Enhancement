"""
preprocess.py - get rid of the table/cloth around the paper before enhancing.
"""

import numpy as np
from src.threshold import otsu_threshold


def crop_to_document(gray, margin=0.03):
    """Crop the image down to just the paper.

    Idea: Otsu splits the photo into 'bright paper' and 'dark background'.
    We look at which rows and columns are mostly white, and keep only those.
    (Using 'mostly white' instead of 'any white pixel' means a few stray
    specks in the background can't fool us.)

    margin: extra shrink on every side (fraction of the size) to avoid
            dark corners of a tilted page.
    """
    mask, _ = otsu_threshold(gray)
    mask = mask > 0

    col_frac = mask.mean(axis=0)       # fraction of white pixels in each column
    row_frac = mask.mean(axis=1)       # fraction of white pixels in each row

    cols = np.where(col_frac > 0.5 * col_frac.max())[0]
    rows = np.where(row_frac > 0.5 * row_frac.max())[0]

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    # shrink inward a little
    dh = int((bottom - top) * margin)
    dw = int((right - left) * margin)
    top, bottom = top + dh, bottom - dh
    left, right = left + dw, right - dw

    return gray[top:bottom + 1, left:right + 1]