"""Whole-rod silhouette counting for separated objects on a pale neutral ground.

Only image pixels enter this function. Paint, printed characters and internal
dark lines do not create another rod: contours are filled before shape checks.
The caller must establish that the scene depicts laid-out rods. This restricted
fallback rejects wide/branched/touching groups instead of splitting by texture.
"""
import time

import cv2
import numpy as np


PARAMETERS = dict(saturation_thresholds=[35, 45, 55], dark_value=110,
                  minimum_area_fraction=.0008, minimum_aspect=18,
                  minimum_weak_foreground_coverage=.78,
                  maximum_centerline_residual_width=.22,
                  maximum_threshold_center_drift_width=.55,
                  maximum_threshold_area_ratio=1.55,
                  maximum_processing_side=1800)


def _one_mask(mask, minimum_area):
    """Get external silhouettes; each retained component must be one long rod."""
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    filled = np.zeros_like(mask)
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x+w >= mask.shape[1]-1 or y+h >= mask.shape[0]-1:
            return None
        sides = sorted(cv2.minAreaRect(contour)[1])
        if sides[1] / max(sides[0], 1) < PARAMETERS['minimum_aspect']:
            return None
        isolated = np.zeros((h, w), np.uint8)
        cv2.drawContours(isolated, [contour - [x, y]], -1, 255, cv2.FILLED)
        yy, xx = np.where(isolated)
        xy = np.column_stack((xx+x, yy+y)).astype(np.float32)
        mean = xy.mean(axis=0)
        direction = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
        cross = np.array([-direction[1], direction[0]])
        uv = (xy-mean) @ np.column_stack((cross, direction))
        lo, hi = np.quantile(uv[:, 1], [.01, .99])
        centers, widths = [], []
        for v in np.linspace(lo+(hi-lo)*.12, hi-(hi-lo)*.12, 15):
            section = uv[np.abs(uv[:, 1]-v) <= max(1.5, (hi-lo)/150), 0]
            if len(section) < 3:
                return None
            left, right = np.quantile(section, [.01, .99])
            centers.append(((left+right)/2, v))
            widths.append(right-left)
        median_width = max(float(np.median(widths)), 1.)
        centers = np.array(centers)
        fit = np.polyfit(centers[:, 1], centers[:, 0], 1)
        residual = float(np.max(np.abs(centers[:, 0]-np.polyval(fit, centers[:, 1]))))
        if residual > median_width*PARAMETERS['maximum_centerline_residual_width']:
            return None
        # A localized painted circle can widen a silhouette; broad branches,
        # crossings and repeated large width changes are unresolved.
        if np.max(widths) > median_width*2.2 or np.sum(np.array(widths)>median_width*1.5) > 3:
            return None
        endpoints = []
        for v in [lo+(hi-lo)*.08, hi-(hi-lo)*.08]:
            point = mean+cross*np.polyval(fit, v)+direction*v
            endpoints.append([float(point[0]), float(point[1])])
        center = np.mean(endpoints, axis=0)
        shapes.append(dict(center=center, width=median_width, area=area,
                           aspect=sides[1]/max(sides[0], 1), segment=endpoints))
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    return (shapes, filled) if shapes else None


def detect_rod_silhouettes(image):
    """Return (points, meta) or None for unresolved geometry (BGR uint8 input)."""
    started = time.perf_counter()
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 HxWx3 BGR pixels')
    height, width = image.shape[:2]
    if min(height, width) < 16:
        return None
    scale = min(1., PARAMETERS['maximum_processing_side']/max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    background = np.median(border, axis=0)
    if background[0] < 180 or np.linalg.norm(background[1:]-128) > 18:
        return None
    if np.mean(np.linalg.norm(border-background, axis=1) < 22) < .90:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    chroma = np.linalg.norm(lab[:, :, 1:]-background[1:], axis=-1)
    weak = (chroma > 8) | (lab[:, :, 0] < 120)
    results = []
    coverages = []
    for threshold in PARAMETERS['saturation_thresholds']:
        mask = (((hsv[:, :, 1] > threshold) & (hsv[:, :, 2] > 45)) |
                (hsv[:, :, 2] < PARAMETERS['dark_value'])).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        result = _one_mask(mask, height*width*PARAMETERS['minimum_area_fraction'])
        if result is None:
            return None
        shapes, filled = result
        coverage = np.logical_and(filled > 0, weak).sum()/max(int(weak.sum()), 1)
        if coverage < PARAMETERS['minimum_weak_foreground_coverage']:
            return None
        results.append(shapes)
        coverages.append(float(coverage))
    if len({len(shapes) for shapes in results}) != 1:
        return None
    reference = results[1]
    # Require the same locations and occupied extent, not just the same number.
    for alternative in (results[0], results[2]):
        remaining = set(range(len(alternative)))
        for shape in reference:
            match = min(remaining, key=lambda j: np.linalg.norm(alternative[j]['center']-shape['center']))
            other = alternative[match]
            remaining.remove(match)
            if np.linalg.norm(other['center']-shape['center']) > max(2, shape['width']*PARAMETERS['maximum_threshold_center_drift_width']):
                return None
            if max(shape['area'], other['area'])/max(1., min(shape['area'], other['area'])) > PARAMETERS['maximum_threshold_area_ratio']:
                return None
    points = [dict(x=round(float(s['center'][0])/scale, 2),
                   y=round(float(s['center'][1])/scale, 2), r=3.,
                   segment=[[round(x/scale, 2), round(y/scale, 2)] for x, y in s['segment']])
              for s in reference]
    points.sort(key=lambda point: (point['y'], point['x']))
    return points, dict(framework='RodSilhouetteConsensus', device='cpu',
        category='laid_rods', detected_category='laid_rods',
        mode='filled_external_silhouette_consensus', prediction_valid=True,
        diagnosis='detections_returned', count_unit='individual_visible_whole_rod',
        manual_exemplars=False, texture_seams_used=False,
        elapsed_seconds=round(time.perf_counter()-started, 4),
        foreground_coverage=[round(c, 4) for c in coverages],
        aspect_ratios=[round(s['aspect'], 3) for s in reference],
        processing_scale=round(scale, 5), parameters=PARAMETERS.copy(),
        limitation=('Restricted to separate slender silhouettes on a pale neutral background. '
                    'Paint and printing inside each filled contour do not split a rod. '
                    'Wide touching groups and crossings are rejected. Completely hidden or '
                    'flush-overlapping rods cannot be inferred from a single silhouette.'),
        segment_definition='Approximate inner axis from 8% to 92% of visible silhouette length.')
