"""Connected-component baseline for clearly separated grains, not touching piles."""
import time
import cv2
import numpy as np


def detect_grains(image):
    started = time.perf_counter()
    height, width = image.shape[:2]
    scale = min(1., 1800 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    inverted = np.mean(border > 0) > .5
    if inverted:
        mask = cv2.bitwise_not(mask)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4, mask.size * .00001)
    points = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        # Background fragments at the border are not complete isolated grains.
        if x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]:
            continue
        if area > mask.size * .25:
            continue
        cx, cy = centers[i]
        points.append(dict(x=round(float(cx / scale), 2), y=round(float(cy / scale), 2),
                           r=round(float(min(w, h) / (2 * scale)), 2),
                           area=round(float(area / scale**2), 2)))
    points.sort(key=lambda p: (round(p['y'] / 20), p['x']))
    return points, dict(framework='SeparatedGrains', mode='connected_components',
                        diagnosis='detections_returned' if points else 'no_final_detections',
                        elapsed_seconds=round(time.perf_counter() - started, 3),
                        threshold=threshold, polarity='dark' if inverted else 'bright',
                        limitation='Each connected foreground region counts as one; touching grains may merge; border fragments excluded.')
