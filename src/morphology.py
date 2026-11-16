"""
morphology.py - clean up binary (black & white) images. From scratch, NumPy only.

CONVENTION: these functions work on an INK MASK:  True/1 = ink, False/0 = paper.
Use binary_to_ink() to convert our usual black/white image into an ink mask,
and ink_to_binary() to convert back for display.
"""

import numpy as np


def binary_to_ink(binary):
    """Our binarized image has ink = 0 (black). Make ink = True."""
    return binary == 0


def ink_to_binary(ink):
    """Back to the usual picture: ink = black (0), paper = white (255)."""
    return np.where(ink, 0, 255).astype(np.uint8)


def _shifted_stack(mask, size, pad_value):
    """All size x size shifted copies of the mask (the 'neighbours' of each pixel)."""
    r = size // 2
    h, w = mask.shape
    padded = np.pad(mask.astype(np.uint8), r, mode="constant", constant_values=pad_value)
    return [padded[i:i + h, j:j + w] for i in range(size) for j in range(size)]


def erosion(ink, size=3):
    """A pixel stays ink only if ALL its neighbours are ink  -> ink shrinks."""
    result = None
    for s in _shifted_stack(ink, size, pad_value=1):
        result = s if result is None else np.minimum(result, s)
    return result.astype(bool)


def dilation(ink, size=3):
    """A pixel becomes ink if ANY neighbour is ink  -> ink grows."""
    result = None
    for s in _shifted_stack(ink, size, pad_value=0):
        result = s if result is None else np.maximum(result, s)
    return result.astype(bool)


def opening(ink, size=3):
    """Erosion then dilation: removes small specks, keeps big shapes."""
    return dilation(erosion(ink, size), size)


def closing(ink, size=3):
    """Dilation then erosion: fills small holes and gaps in letters."""
    return erosion(dilation(ink, size), size)


def remove_small_components(ink, min_size=15):
    """Delete ink blobs smaller than min_size pixels.

    Find each group of touching ink pixels (8-neighbour connectivity) and
    count its pixels. Tiny groups = specks -> removed. A thin pen stroke is
    one long connected group, so it survives (unlike with opening).
    """
    h, w = ink.shape
    grid = ink.tolist()                                  # plain lists are faster here
    visited = [bytearray(w) for _ in range(h)]
    out = ink.copy()

    ys, xs = np.nonzero(ink)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0][x0]:
            continue
        visited[y0][x0] = 1
        stack = [(y0, x0)]
        blob = []
        while stack:
            y, x = stack.pop()
            blob.append((y, x))
            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (x - 1, x, x + 1):
                    if 0 <= nx < w and grid[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = 1
                        stack.append((ny, nx))
        if len(blob) < min_size:
            for y, x in blob:
                out[y, x] = False
    return out


def clear_border(ink, frac=0.015):
    """Wipe a thin strip along the edges (page-border streaks live there)."""
    h, w = ink.shape
    by, bx = int(h * frac), int(w * frac)
    out = ink.copy()
    out[:by, :] = False
    out[h - by:, :] = False
    out[:, :bx] = False
    out[:, w - bx:] = False
    return out