# crop.py
# David Owen

K = 9  # corner match kernel size

import cv2
import numpy as np
import sys

def corners(original, f=4):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    h, w = original.shape[:2]
    k = np.ones((K, K), dtype=np.uint8)

    # top left corner template
    t = 255 * np.ones((K * 3, K * 3), dtype=np.uint8)
    t[K:, K:(K * 2)] = 0
    t[K:(K * 2), (K * 2):] = 0

    def match(roi):
        cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, dst=roi)
        m = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        
        return cv2.minMaxLoc(m)[3]
    
    tl = match(gray[:(h // f), :(w // f)])
    
    # modify top left template for top right
    t[K:(K * 2), :K] = 0
    t[K:(K * 2), (K * 2):] = 255

    tr = match(gray[:(h // f), ((f - 1) * w // f):])
    tr = (tr[0] + ((f - 1) * w // f), tr[1])

    # modify top right template for bottom right
    t[:K, K:(K * 2)] = 0
    t[(K * 2):, K:(K * 2)] = 255

    br = match(gray[((f - 1) * h // f):, ((f - 1) * w // f):])
    br = (br[0] + ((f - 1) * w // f), br[1] + ((f - 1) * h // f))

    # modify bottom right template for bottom left
    t[K:(K * 2), :K] = 255
    t[K:(K * 2), (K * 2):] = 0

    bl = match(gray[((f - 1) * h // f):, :(w // f)])
    bl = (bl[0], bl[1] + ((f - 1) * h // f))

    return tl, tr, bl, br

def draw_corners(original, tl, tr, bl, br):
    cv2.rectangle(original, tl,
            (tl[0] + K * 3, tl[1] + K * 3), (0, 20, 180), -1)
    cv2.rectangle(original, tr,
            (tr[0] + K * 3, tr[1] + K * 3), (0, 20, 180), -1)
    cv2.rectangle(original, br,
            (br[0] + K * 3, br[1] + K * 3), (0, 20, 180), -1)
    cv2.rectangle(original, bl,
            (bl[0] + K * 3, bl[1] + K * 3), (0, 20, 180), -1)

def deskew(original, tl, tr, bl, br):
    lx = (tl[0] + bl[0]) // 2
    ty = (tl[1] + tr[1]) // 2
    rx = (tr[0] + br[0]) // 2
    by = (bl[1] + br[1]) // 2
    tl_new = (lx, ty)
    tr_new = (rx, ty)
    bl_new = (lx, by)
    br_new = (rx, by)

    old_corners = np.array([tl, tr, bl, br])
    new_corners = np.array([tl_new, tr_new, bl_new, br_new])
    homography = cv2.findHomography(old_corners, new_corners)[0]
    cv2.warpPerspective(original, homography, original.shape[:2],
            dst=original)
            
    return lx, rx, ty, by

def crop(original, m=20, f=4):
    tl, tr, bl, br = corners(original, f)
    lx, rx, ty, by = deskew(original, tl, tr, bl, br)
    
    return original[
            (ty - m + 3 * K // 2):(by + m + 3 * K // 2),
            (lx - m + 3 * K // 2):(rx + m + 3 * K // 2)]

if __name__ == '__main__':
    a = sys.argv[1]
    original = cv2.imread(a)
    n = a[:a.find('.')]
    cropped = crop(original, m=30, f=6)
    cv2.imwrite(n + '_cropped.jpg', cropped)
    draw_corners(original, *corners(original, f=6))
    cv2.imwrite(n + '_corners.jpg', original)
