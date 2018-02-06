# cells.py
# David Owen

import cv2
import numpy as np
import sys

from crop import *

T = 200  # Threshold value for pre-HoughLines processing.
V = 1475  # HoughLines score needed for a vertical line.
H = 1875  # HoughLines score needed for a horizontal line.
E = 0.01  # How close to vertical / horizontal must lines be?
F = 5  # How far apart must lines be?

B = 4  # How far away from approximate border line should we
       # look for a real border to erase?
M = 0.35  # Threshold factor for border match.

# Colors for highlighting columns (accent colors from
# ethanschoonover.com/solarized).
COLORS = [(0, 137, 181), (22, 75, 203), (47, 50, 220),
          (130, 54, 211), (196, 113, 108), (210, 139, 38),
          (152, 161, 42), (0, 153, 133)]

def lines(cropped):
    h, w = cropped.shape[:2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Find vertical lines (each represented by x coordinate).
    scharr_x = cv2.Scharr(gray, cv2.CV_8U, 1, 0)
    k = np.ones((7, 3), dtype=np.uint8)
    cv2.morphologyEx(scharr_x, cv2.MORPH_CLOSE, k, dst=scharr_x)
    thresh = cv2.threshold(scharr_x, T, 255, cv2.THRESH_BINARY)[1]

    lines = cv2.HoughLines(thresh, 1, np.pi / 180, V)
    xs = []

    for line in lines:
        rho, theta = line[0]

        if abs(theta) < E:
            x = int(round(np.cos(theta) * rho)) - 3
            xs.append(x)

    xs.append(w)

    # Find horizontal lines (each represented by y coordinate).
    scharr_y = cv2.Scharr(gray, cv2.CV_8U, 0, 1)
    k = np.ones((3, 7), dtype=np.uint8)
    cv2.morphologyEx(scharr_y, cv2.MORPH_CLOSE, k, dst=scharr_y)
    thresh = cv2.threshold(scharr_y, T, 255, cv2.THRESH_BINARY)[1]

    lines = cv2.HoughLines(thresh, 1, np.pi / 180, H)
    ys = []

    for line in lines:
        rho, theta = line[0]

        if abs(theta - np.pi / 2) < E:
            y = int(round(np.sin(theta) * rho)) - 3
            ys.append(y)

    return sorted(xs), sorted(ys)

def filter_lines(xs, ys):

    for x in xs:
        for i in range(1, F):
            if (x + i) in xs:
                xs.remove(x + i)

    for y in ys:
        for i in range(1, F):
            if (y + i) in ys:
                ys.remove(y + i)

def draw_lines(cropped, xs, ys):
    h, w = cropped.shape[:2]

    for x in xs:
        cv2.line(cropped, (x, 0), (x, h), (0, 20, 180), 2)

    for y in ys:
        cv2.line(cropped, (0, y), (w, y), (0, 20, 180), 2)

def cells(cropped):
    xs, ys = lines(cropped)
    filter_lines(xs, ys)
    columns = []

    for i in range(len(xs) - 1):
        columns.append([])

        for j in range(len(ys) - 1):
            columns[-1].append(
                    [(xs[i], ys[j]), (xs[i + 1], ys[j + 1])])

    return columns

def erase_borders(cropped, cell, borders_adjusted=False,
        color=(255, 255, 255)):

    x0, y0 = cell[0]
    x1, y1 = cell[1]
   
    original = cropped[(y0 - B):(y1 + B), (x0 - B):(x1 + B)]
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    if borders_adjusted:
        b = 3 * B // 2
    else:
        b = B * 2

    gray[b:-b, b:-b] = 255

    h, w = gray.shape[:2]

    # Make vertical template, erase vertical borders.
    t = np.zeros((h, 1), dtype=np.uint8)
    m_x = cv2.matchTemplate(gray, t, cv2.TM_SQDIFF)
    v_x = cv2.minMaxLoc(m_x)[1]

    for x in np.where(m_x < M * v_x)[1]:
        cv2.line(original, (x, 0), (x, h), color, 1)

    # Make horizontal template, erase horizontal borders.
    t = np.zeros((1, w), dtype=np.uint8)
    m_y = cv2.matchTemplate(gray, t, cv2.TM_SQDIFF)
    v_y = cv2.minMaxLoc(m_y)[1]

    for y in np.where(m_y < M * v_y)[0]:
        cv2.line(original, (0, y), (w, y), color, 1)

    # Clean up leftover border bits.
    for x in np.where(m_x < M * v_x)[0]:
        k = np.ones((1, 5), dtype=np.uint8)
        cv2.morphologyEx(cropped[y0:y1, (x0 + x - 2):(x0 + x + 3)],
                cv2.MORPH_CLOSE, k,
                dst=cropped[y0:y1, (x0 + x - 2):(x0 + x + 3)])

    for y in np.where(m_y < M * v_y)[0]:
        k = np.ones((5, 1), dtype=np.uint8)
        cv2.morphologyEx(cropped[(y0 + y - 2):(y0 + y + 3), x0:x1],
                cv2.MORPH_CLOSE, k,
                dst=cropped[(y0 + y - 2):(y0 + y + 3), x0:x1])

def adjust_borders(cropped, cell):
    x0, y0 = cell[0]
    x1, y1 = cell[1]
    gray = cv2.cvtColor(
            cropped[(y0 - B):(y1 + B), (x0 - B):(x1 + B)],
            cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    gray[(B * 2):-(B * 2), (B * 2):-(B * 2)] = 255

    # Find top of top border.
    t = np.zeros((h, 1), dtype=np.uint8)
    m_x = cv2.matchTemplate(gray, t, cv2.TM_SQDIFF)
    v_x = cv2.minMaxLoc(m_x)[1]
    ll = rr = False

    for x in np.where(m_x < M * v_x)[1]:
        if (x < B * 2) and ((not ll) or (x < ll)):
            ll = x
        if (x > B * 2) and ((not rr) or (x > rr)):
            rr = x
    
    if not ll:
        ll = x0
    else:
        ll += x0 - B

    if not rr:
        rr = x1
    else:
        rr += x0 - B

    # Find left side of left border, right side of right border.
    t = np.zeros((1, w), dtype=np.uint8)
    m_y = cv2.matchTemplate(gray, t, cv2.TM_SQDIFF)
    v_y = cv2.minMaxLoc(m_y)[1]
    tt = bb = False

    for y in np.where(m_y < M * v_y)[0]:
        if (y < B * 2) and ((not tt) or (y < tt)):
            tt = y
        if (y > B * 2) and ((not bb) or (y > bb)):
            bb = y
    
    if not tt:
        tt = y0
    else:
        tt += y0 - B
    
    if not bb:
        bb = y1
    else:
        bb += y0 - B
    
    cell[0] = (ll, tt)
    cell[1] = (rr, bb)

def draw_cells(cropped, cells):

    for column in cells:
        for cell in column[1:]:
            adjust_borders(cropped, cell)
            erase_borders(cropped, cell, borders_adjusted=True)

    copy = np.copy(cropped)
    c = 0
    b = 5

    for column in cells[1:-1]:
        for cell in column[2:-1]:
            x0, y0 = cell[0]
            x1, y1 = cell[1]
            cv2.rectangle(copy, (x0 + b, y0 + b),
                    (x1 - b, y1 - b), COLORS[c], -1)

        c = (c + 5) % len(COLORS)

    cv2.addWeighted(cropped, 0.85, copy, 0.15, 0, dst=cropped)

if __name__ == '__main__':
    a = sys.argv[1]
    n = a[:a.find('.')]
    original = cv2.imread(a)
    cropped = crop(original, m=30, f=6, fast=True)
    # draw_lines(cropped, *lines(cropped))
    # cv2.imwrite(n + '_lines.jpg', original)
    draw_cells(cropped, cells(cropped))
    cv2.imwrite(n + '_cells.jpg', original)
