# compare.py
# David Owen

import cv2
import numpy as np
import sys

from crop import *
from cells import *

M = 8  # Margin to be included in cell image.
P = 4  # Padding not to be include in template image.
S = 0.5  # Match result score threshold.
T = 3  # Number of (top) cell matches to consider when making groups.
B = 245 # Cell with average above this considered blank (max 255).

def compare(cropped, cell_a, cell_b):
    x0, y0 = cell_a[0]
    x1, y1 = cell_a[1]
    a = cropped[(y0 - M):(y1 + M), (x0 - M):(x1 + M)]
    ta = cropped[(y0 + P):(y1 - P), (x0 + P):(x1 - P)]
    
    if np.mean(ta) > B: return 0
    
    x0, y0 = cell_b[0]
    x1, y1 = cell_b[1]
    b = cropped[(y0 - M):(y1 + M), (x0 - M):(x1 + M)]
    tb = cropped[(y0 + P):(y1 - P), (x0 + P):(x1 - P)]

    if np.mean(tb) > B: return 0

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
        scores = [False] * len(column)
        
        for i in range(2, len(column) - 2):
            for j in range(i + 1, len(column) - 1):
                score = compare(erased, column[i], column[j])
                
                if score > S:
                    if not scores[i]:
                        scores[i] = []
                    
                    if not scores[j]:
                        scores[j] = []
                    
                    scores[i].append((score, j))
                    scores[j].append((score, i))
        
        for i in range(len(scores)):
            if scores[i]:
                scores[i].sort()
                scores[i].reverse()
                scores[i] = scores[i][:T]

        groups = []
        
        def find_group(i):  # DFS for connected components.
            m = scores[i]
            scores[i] = False

            for k in range(len(m)):
                j = m[k][1]
                q = False
                
                for g in groups[:-1]:
                    if j in g:
                        g.append(i)
                        q = True
                
                if q: break
                
                if scores[j]:
                    groups[-1].append(j)
                    find_group(j)
                        
        for i in range(len(scores)):
            if scores[i]:
                groups.append([i])
                find_group(i)
                
                if len(groups[-1]) == 1:
                    groups = groups[:-1]
       
        print(groups)
        k = 0
        
        for i in range(min(len(groups), len(COLORS))):
            for j in groups[i]:
                x0, y0 = column[j][0]
                x1, y1 = column[j][1]
                cv2.rectangle(highlighted, (x0 + 5, y0 + 5),
                        (x1 - 5, y1 - 5), COLORS[k], -1)
            
            k = (k + 5) % len(COLORS)

    cv2.addWeighted(cropped, 0.85, highlighted, 0.15, 0, dst=cropped)
    cv2.imwrite(n + '_compare_closest.jpg', original)
