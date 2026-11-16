# Algorithm Comparison Table

Averages over 30 images. Arrows show the better direction (↑ higher is better, ↓ lower is better).


## Enhancement

| Algorithm | Contrast (std) ↑ | Background variation % ↓ | Ink-paper separation ↑ | Time (s) ↓ |
|---|---|---|---|---|
| None | 27.29 | 8.33 | 5.63 | 0.00 |
| Contrast Stretch | 51.09 | 11.72 | 5.55 | 0.03 |
| Gamma Correction (0.7) | 24.93 | 6.36 | 6.24 | 0.01 |
| Histogram Equalization | 74.28 | 18.97 | 3.51 | 0.01 |
| CLAHE | 33.74 | 9.96 | 4.92 | 0.11 |

## Noise Removal

| Algorithm | Noise level ↓ | Edge retention % ↑ | Time (s) ↓ |
|---|---|---|---|
| None | 7.33 | 100.00 | 0.00 |
| Mean Filter 3x3 | 2.67 | 58.86 | 0.08 |
| Gaussian Filter 5x5 | 2.20 | 54.00 | 0.17 |
| Median Filter 3x3 | 4.40 | 73.40 | 0.15 |

## Binarization

| Algorithm | Ink coverage % | Speck % of ink ↓ | Time (s) ↓ |
|---|---|---|---|
| Global (t=127) | 30.84 | 3.92 | 0.01 |
| Otsu | 21.65 | 1.74 | 0.01 |
| Adaptive (C=25) | 10.92 | 3.95 | 0.09 |

## Cleanup

| Algorithm | Ink retained % ↑ | Speck % of ink ↓ | Time (s) ↓ |
|---|---|---|---|
| None | 100.00 | 3.95 | 0.00 |
| Opening 3x3 | 74.29 | 5.70 | 0.01 |
| Closing 3x3 | 108.21 | 2.16 | 0.01 |
| Remove Specks + Clear Border | 92.71 | 0.06 | 0.37 |
