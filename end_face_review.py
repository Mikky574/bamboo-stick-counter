"""Restricted consistency checks for compact end faces mixed with long shafts.

These checks operate on model boxes, not saved annotations or sample identity.
Uniform end-face center suppression is not appropriate for overlapping insects,
mixed-size particles or whole rods; the worker only calls it on this end route.
"""
import math
from statistics import median


def mixed_shaft_evidence(points, exemplars):
    if len(exemplars) < 2:
        return None
    sizes = [(b[2]-b[0], b[3]-b[1]) for b in exemplars]
    if any(min(s) <= 0 or max(s)/min(s) > 2.2 for s in sizes):
        return None
    typical_long = median(max(s) for s in sizes)
    shafts = []
    for point in points:
        b = point['box']; w, h = b[2]-b[0], b[3]-b[1]
        if min(w, h) > 0 and max(w, h)/min(w, h) > 2.8 and max(w, h) > 3*typical_long:
            shafts.append(point)
    if not shafts:
        return None
    return dict(shaft_box_candidates=len(shafts), original_prediction_count=len(points),
                seed_long_side=round(typical_long, 2),
                trigger='compact_end_examples_with_long_shaft_predictions')


def suppress_uniform_end_centers(points):
    """Merge close center proposals only for a dense, nearly uniform face set."""
    report = dict(applied=False, removed=0,
                  limitation='Assumes compact, similar-size, non-overlapping end faces')
    if len(points) < 10:
        return points, report
    widths = []
    aspects = []
    for p in points:
        b = p['box']; w, h = b[2]-b[0], b[3]-b[1]
        if min(w, h) <= 0:
            return points, report
        widths.append(min(w, h)); aspects.append(max(w, h)/min(w, h))
    ordered = sorted(widths)
    low, high = ordered[int((len(ordered)-1)*.1)], ordered[int((len(ordered)-1)*.9)]
    if median(aspects) > 2.2 or high/max(low, 1e-6) > 1.5:
        return points, report
    separation = .7*median(widths)
    retained = []
    def same_scale(a, b):
        ab, bb = a['box'], b['box']
        am, bm = min(ab[2]-ab[0], ab[3]-ab[1]), min(bb[2]-bb[0], bb[3]-bb[1])
        return max(am, bm)/min(am, bm) <= 1.5
    for p in sorted(points, key=lambda p: p['score'], reverse=True):
        if not any(same_scale(p, q) and math.hypot(p['x']-q['x'], p['y']-q['y']) < separation
                   for q in retained):
            retained.append(p)
    report.update(applied=True, removed=len(points)-len(retained),
                  minimum_center_separation=round(separation, 2),
                  median_minor_side=round(median(widths), 2))
    return retained, report
