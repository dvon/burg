# compare.py
# David Owen

import cv2
import numpy as np
import sys

from crop import *
from cells import *

M = 8  # Margin to be included in cell image.
P = 4  # Padding not to be include in template image.
S = 0.4  # Match result score threshold.

# TODO Memo this for efficiency...
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
   
    for column in cells[1:-1]:
    # column = cells[5]
        scores = [False] * len(column)
        
        for i in range(2, len(column) - 2):
            for j in range(i + 1, len(column) - 1):
                score = compare(erased, column[i], column[j])
                
                if score > S:
                    if not scores[i] or (score > scores[i][0]):
                        scores[i] = (score, j)
                    
                    if not scores[j] or (score > scores[j][0]):
                        scores[j] = (score, i)

        # TODO Lookup connected components algorithm and
        #      do this right...
        groups = []
        
        for i in range(len(scores)):
            if scores[i]:
                j = scores[i][1]
            
                for g in groups:
                    if j in g:
                        g.append(i)
                        break
                else:
                    groups.append([i])

                scores[i] = False
                
                while scores[j]:
                    groups[-1].append(j)
                    k = scores[j][1]
                    scores[j] = False
                    j = k

        k = 0
        
        for i in range(min(len(groups), len(COLORS))):
            for j in groups[i]:
                x0, y0 = column[j][0]
                x1, y1 = column[j][1]
                cv2.rectangle(highlighted, (x0 + 5, y0 + 5),
                        (x1 - 5, y1 - 5), COLORS[k], -1)
            
            k = (k + 5) % len(COLORS)

    cv2.addWeighted(cropped, 0.85, highlighted, 0.15, 0, dst=cropped)
    cv2.imwrite(n + '_compare.jpg', original)
