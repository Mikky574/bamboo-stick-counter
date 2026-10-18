"""Conservative, category-free CPU fast path for separated repeated particles.

The gate consumes image pixels only. Rejection returns None, never a count.
Agreement of color segmentations is a heuristic, not an accuracy guarantee.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


_WEIGHTS = (1.0, 0.5, 0.25)
_LIMITATION = (
    "Only clearly separated, similarly sized particles on a contrasting, simple "
    "background. The category is unspecified. Touching/overlapping objects, "
    "textured backgrounds, mixed object sizes, and objects cut by the image edge "
    "must use another method or be photographed separately. Segmentation "
    "agreement is a heuristic and does not prove the count is correct."
)


def _fill_external(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _candidate(base_lab, border, scale, luminance_weight):
    lab = base_lab.copy()
    lab[..., 0] *= luminance_weight
    background = np.median(lab[border], axis=0)
    distance = np.linalg.norm(lab-background, axis=2)
    _, seed = cv2.threshold(np.clip(distance, 0, 255).astype(np.uint8),
                            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not np.any(seed):
        return None
    direction = np.median(lab[seed > 0], axis=0)-background
    direction /= max(1., float(np.linalg.norm(direction)))
    # Signed contrast suppresses cast shadows in the opposite color direction.
    projection = np.clip((lab-background) @ direction, 0, 255).astype(np.uint8)
    threshold, raw = cv2.threshold(projection, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    filled = _fill_external(raw)
    number, labels, stats, centers = cv2.connectedComponentsWithStats(filled, connectivity=8)
    minimum_area = max(8., filled.size*.000025)
    initial_areas = [int(stat[4]) for stat in stats[1:] if stat[4] >= minimum_area]
    if not initial_areas:
        return None
    diameter = float(np.sqrt(np.median(initial_areas)*4/np.pi))
    closing_size = max(3, min(9, int(round(diameter*.10)) | 1))
    if closing_size > 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
        filled = _fill_external(cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel))
        number, labels, stats, centers = cv2.connectedComponentsWithStats(filled, connectivity=8)
    height, width = filled.shape
    foreground_area = int(np.count_nonzero(filled))
    border_area = tiny_area = 0
    areas, solidities, elongations, points = [], [], [], []
    for index in range(1, number):
        x, y, w, h, area = [int(value) for value in stats[index]]
        if area < minimum_area:
            tiny_area += area
            continue
        if x == 0 or y == 0 or x+w == width or y+h == height:
            border_area += area
        local = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        solidity = cv2.contourArea(contour)/max(1., cv2.contourArea(cv2.convexHull(contour)))
        side_a,side_b = cv2.minAreaRect(contour)[1]
        elongations.append(max(side_a,side_b)/max(1.,min(side_a,side_b)))
        areas.append(area)
        solidities.append(solidity)
        points.append({"x": round(float(centers[index][0]/scale), 2),
                       "y": round(float(centers[index][1]/scale), 2),
                       "r": round(float(np.sqrt(area/np.pi)/scale), 2)})
    if not points:
        return None
    contrast = float(np.median(distance[filled > 0]))
    metrics = {
        "foreground_fraction": foreground_area/filled.size,
        "background_p95_delta": float(np.percentile(distance[border], 95)),
        "foreground_median_delta": contrast,
        "border_mask_fraction": float(np.mean(filled[border] > 0)),
        "border_component_fraction": border_area/max(1, foreground_area),
        "tiny_fragment_fraction": tiny_area/max(1, foreground_area),
        "largest_foreground_share": float(np.max(areas))/max(1, foreground_area),
        "area_p90_p10": float(np.percentile(areas, 90)/max(1., np.percentile(areas, 10))),
        "area_min_median": float(np.min(areas)/max(1., np.median(areas))),
        "solidity_p10": float(np.percentile(solidities, 10)),
        "solidity_min": float(np.min(solidities)),
        "elongation_median": float(np.median(elongations)),
        "threshold": float(threshold), "closing_size": closing_size,
    }
    checks = (
        5 <= len(points) <= 400,
        .005 <= metrics["foreground_fraction"] <= .25,
        metrics["background_p95_delta"] <= max(12., contrast*.28),
        metrics["border_mask_fraction"] <= .005,
        metrics["border_component_fraction"] == 0.,
        metrics["tiny_fragment_fraction"] <= .01,
        metrics["largest_foreground_share"] <= .25,
        metrics["area_p90_p10"] <= 3.,
        # Quantiles alone miss small duplicate seed-hilum fragments.
        metrics["area_min_median"] >= .25,
        metrics["solidity_p10"] >= .85,
        metrics["solidity_min"] >= .80,
        # A touching pair of chopsticks can be one convex component. Long rods
        # need instance separation; never count pairs as independent particles.
        metrics["elongation_median"] <= 7.,
    )
    if not all(checks):
        return None
    return {"points": points, "statistics": metrics, "luminance_weight": luminance_weight}


def detect_separated(image):
    """Return (original-image points, metadata) only when the fast gate passes.

    Input is a BGR uint8 OpenCV image. No category or external metadata is used.
    None means the fast path abstained, not that there are zero objects.
    """
    started = time.perf_counter()
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 32):
        return None
    scale = min(1., 1024/max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.zeros(small.shape[:2], bool)
    margin = max(2, int(min(small.shape[:2])*.04))
    border[:margin] = border[-margin:] = True
    border[:, :margin] = border[:, -margin:] = True
    candidates = []
    for weight in _WEIGHTS:
        candidate = _candidate(lab, border, scale, weight)
        if candidate is not None:
            candidates.append(candidate)
    if len(candidates) < 2 or len({len(row["points"]) for row in candidates}) != 1:
        return None
    reference = candidates[0]
    reference_xy = np.array([[p["x"], p["y"]] for p in reference["points"]])
    diameter = float(np.median([2*p["r"] for p in reference["points"]]))
    max_normalized_shift = 0.
    for candidate in candidates[1:]:
        other_xy = np.array([[p["x"], p["y"]] for p in candidate["points"]])
        distances = np.linalg.norm(reference_xy[:, None, :]-other_xy[None, :, :], axis=2)
        matches = np.argmin(distances, axis=1)
        if len(np.unique(matches)) != len(matches):
            return None
        normalized_shift = float(np.max(distances[np.arange(len(matches)), matches])/max(1., diameter))
        if normalized_shift > .25:
            return None
        max_normalized_shift = max(max_normalized_shift, normalized_shift)
    points = sorted(reference["points"], key=lambda p: (round(p["y"]/20), p["x"]))
    meta = {
        "framework": "AutomaticParticles", "detected_category": "particles",
        "prediction_valid": True, "diagnosis": "detections_returned",
        "mode": "category_free_background_color_consensus", "device": "cpu",
        "selected_weights": [row["luminance_weight"] for row in candidates],
        "tested_weights": list(_WEIGHTS),
        "reason": "At least two color projections passed all shape/background checks with equal counts and matching positions.",
        "shape_statistics": reference["statistics"],
        "consensus_max_normalized_shift": round(max_normalized_shift, 5),
        "elapsed_seconds": round(time.perf_counter()-started, 4),
        "limitation": _LIMITATION,
    }
    return points, meta
