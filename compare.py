# compare.py
# David Owen

import cv2
import numpy as np
import sys

from crop import *
from cells import *

M = 8  # Margin to be included in cell image.
P = 4  # Padding not to be include in template image.
C = 0.4  # Match result threshold.

def compare(cropped, cell_a, cell_b):
    x0, y0 = cell_a[0]
    x1, y1 = cell_a[1]
    a = cropped[(y0 - M):(y1 + M), (x0 - M):(x1 + M)]
    ta = cropped[(y0 + P):(y1 - P), (x0 + P):(x1 - P)]
    
    x0, y0 = cell_b[0]
    x1, y1 = cell_b[1]
    b = cropped[(y0 - M):(y1 + M), (x0 - M):(x1 + M)]
    tb = cropped[(y0 + P):(y1 - P), (x0 + P):(x1 - P)]

    m = cv2.matchTemplate(a, tb, cv2.TM_CCOEFF_NORMED)
    n = cv2.matchTemplate(b, ta, cv2.TM_CCOEFF_NORMED)
    
    return (cv2.minMaxLoc(m)[1] + cv2.minMaxLoc(n)[1]) / 2

if __name__ == '__main__':
    a = sys.argv[1]
    n = a[:a.find('.')]
    original = cv2.imread(a)
    cropped = crop(original, m=30, f=6, fast=True)
    cells = cells(cropped)
    erased = np.copy(cropped)
    
    for column in cells:
        for cell in column[1:]:
            adjust_borders(erased, cell)
            erase_borders(erased, cell, borders_adjusted=True)
    
    highlighted = np.copy(cropped)
    k = 0

    for column in cells[1:-1]:
        color = COLORS[k]
        k = (k + 5) % len(COLORS)
        # for i in range(2, len(column) - 2):
        i = 2
        # if i < len(COLORS):
        x0, y0 = column[i][0]
        x1, y1 = column[i][1]
        cv2.rectangle(highlighted, (x0 + 5, y0 + 5),
                (x1 - 5, y1 - 5), color, -1)
    
        for j in range(i + 1, len(column) - 1):
            c = compare(erased, column[i], column[j])

            if c > C:
                x0, y0 = column[j][0]
                x1, y1 = column[j][1]
                cv2.rectangle(highlighted, (x0 + 5, y0 + 5),
                        (x1 - 5, y1 - 5), color, -1)

    cv2.addWeighted(cropped, 0.85, highlighted, 0.15, 0, dst=cropped)
    cv2.imwrite(n + '_compare.jpg', original)
