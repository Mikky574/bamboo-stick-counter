"""Offline CPU experiment for touching grains; deliberately not a count route.

Image pixels only. The sweep exposes parameter sensitivity; its region counts
are segmentation candidates, never validated object counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed, find_boundaries


def experiment(image, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.zeros((height, width), bool)
    margin = max(2, round(min(height, width) * .04))
    border[:margin] = border[-margin:] = True
    border[:, :margin] = border[:, -margin:] = True
    results, previews = [], []
    for luminance_weight in (1., .5, .25):
        transformed = lab.copy()
        transformed[..., 0] *= luminance_weight
        background = np.median(transformed[border], axis=0)
        difference = transformed - background
        distance = np.linalg.norm(difference, axis=2)
        _, seed = cv2.threshold(np.clip(distance, 0, 255).astype(np.uint8), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        direction = np.median(difference[seed > 0], axis=0)
        direction /= max(1., float(np.linalg.norm(direction)))
        projection = np.clip(difference @ direction, 0, 255).astype(np.uint8)
        _, raw = cv2.threshold(projection, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for closing in (3, 5, 9):
            mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((closing, closing), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask[:] = 0
            cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
            mask = mask > 0
            minimum_area = max(8, mask.size * .00005)
            labels, _ = ndi.label(mask)
            sizes = np.bincount(labels.ravel())
            mask &= sizes[labels] >= minimum_area
            distances = ndi.distance_transform_edt(mask)
            if distances.max() <= 0:
                continue
            for peak_fraction in (.05, .1, .2, .3):
                peaks = h_maxima(distances, distances.max() * peak_fraction)
                markers, marker_count = ndi.label(peaks)
                segmented = watershed(-distances, markers, mask=mask)
                areas = np.bincount(segmented.ravel())[1:]
                candidate_count = int(np.count_nonzero(areas >= minimum_area))
                record = dict(luminance_weight=luminance_weight, closing=closing,
                              h_fraction=peak_fraction, marker_count=int(marker_count),
                              candidate_count=candidate_count,
                              region_areas=[int(x) for x in areas],
                              distance_max=float(distances.max()))
                results.append(record)
                if luminance_weight == .5 and closing == 3:
                    preview = image.copy()
                    preview[find_boundaries(segmented, mode='inner')] = (0, 0, 255)
                    for label in range(1, int(segmented.max()) + 1):
                        ys, xs = np.where(segmented == label)
                        if len(xs) < minimum_area:
                            continue
                        x, y = int(xs.mean()), int(ys.mean())
                        cv2.circle(preview, (x, y), 9, (20, 20, 20), -1)
                        cv2.putText(preview, str(label), (x-5, y+4), cv2.FONT_HERSHEY_SIMPLEX,
                                    .35, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(preview, f'h={peak_fraction}; {candidate_count} regions (NOT verified count)',
                                (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
                    previews.append(preview)
    counts = [row['candidate_count'] for row in results]
    report = dict(method='distance-transform + h-maxima seeded watershed parameter sweep',
                  inference_device='cpu', uses_category_or_truth=False,
                  valid_object_count=False,
                  reason='Watershed regions may merge or split real objects; no acceptance rule is calibrated.',
                  minimum_candidate_regions=min(counts) if counts else None,
                  maximum_candidate_regions=max(counts) if counts else None,
                  candidates=results)
    (output/'watershed_sweep.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if previews:
        montage = np.vstack([np.hstack(previews[:2]), np.hstack(previews[2:4])])
        ok, encoded = cv2.imencode('.jpg', montage)
        if ok:
            encoded.tofile(output/'watershed_sensitivity.jpg')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    decoded = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit('Image decode failed.')
    report = experiment(decoded, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'candidates'}, indent=2))
