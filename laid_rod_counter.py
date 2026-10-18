"""Conservative CPU counting of visible rods in supported layouts.

The caller must first establish that the image shows laid-out rods/chopsticks.
Input is BGR pixels only: no filename, sample metadata, stored labels or counts.
Returns None when the foreground or the per-component separation is unresolved.
"""
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


PARAMETERS = dict(
    min_area_fraction=.0008, strong_saturation=45, dark_value=110,
    weak_chroma=8, min_coverage=.78, min_component_aspect=4,
    min_single_aspect=18, seam_prominence=35, cross_sections=9,
    seam_support_minimum=6, seam_normalized_tolerance=.07,
)


def detect_laid_rods(image):
    """Prefer separate whole silhouettes before inspecting internal seam texture.

    A painted line is not evidence of another instance. For fully separate,
    slender contours, surface texture must not alter their count. Wide touching
    groups use the restricted seam routine. Framed packaging and contrasting
    close-up terminal faces have their own visible-candidate checks.
    """
    from rod_silhouette import detect_rod_silhouettes
    started = time.perf_counter()
    result = detect_rod_silhouettes(image)
    silhouette_seconds = time.perf_counter() - started
    if result is None:
        result = _detect_laid_rods_seams(image)
    if result is None:
        from boxed_rod_counter import detect_boxed_rods
        result = detect_boxed_rods(image)
    if result is None:
        from closeup_rod_counter import detect_closeup_rods
        result = detect_closeup_rods(image)
    if result is not None:
        result[1].update(silhouette_check_seconds=round(silhouette_seconds, 4),
                         elapsed_seconds=round(time.perf_counter()-started, 4))
    return result


def _detect_laid_rods_seams(image):
    """Return (points, metadata), or None; image must be uint8 HxWx3 BGR.

    A colored foreground component can contain several touching rods. Each is
    aligned by PCA, and narrow dark internal boundaries must recur at similar
    positions in at least six of nine transverse sections. Texture and hidden
    seams can still cause mistakes, so this is a restricted scene method.
    The returned segment marks an approximate inner axis, not full endpoints.
    """
    started = time.perf_counter()
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an HxWx3 BGR image')
    if image.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR pixels')
    height, width = image.shape[:2]
    if min(height, width) < 8:
        return None
    if max(height,width)>1800:
        scale=1800/max(height,width)
        result=_detect_laid_rods_seams(cv2.resize(image,None,fx=scale,fy=scale))
        if result is None:return None
        points,metadata=result
        for point in points:
            for key in ('x','y','r'):point[key]=round(point[key]/scale,2)
            point['segment']=[[round(x/scale,2),round(y/scale,2)] for x,y in point['segment']]
        metadata.update(processing_scale=scale,elapsed_seconds=round(time.perf_counter()-started,4))
        return points,metadata

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    border = np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]
    background = np.median(border, axis=0)
    if background[0] < 180 or np.linalg.norm(background[1:] - 128) > 15:
        return None

    chroma = np.linalg.norm(lab[:, :, 1:].astype(float) - background[1:], axis=-1)
    weak = (chroma > 8) | (lab[:, :, 0] < 120)
    strong = (((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45)) |
              (hsv[:, :, 2] < 110)).astype('uint8') * 255
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong)
    ids = [i for i in range(1, n) if stats[i, 4] >= height * width * .0008]
    if not ids:
        return None
    kept = np.isin(labels, ids)
    coverage = float(np.logical_and(kept, weak).sum() / max(1, weak.sum()))
    if coverage < .78:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centerlines, component_details = [], []
    for ident in ids:
        yy, xx = np.where(labels == ident)
        xy = np.column_stack([xx, yy]).astype(float)
        short, long = sorted(cv2.minAreaRect(xy.astype('float32'))[1])
        aspect = long / max(1, short)
        if aspect < 4:
            return None

        mean = xy.mean(axis=0)
        v = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
        u = np.array([-v[1], v[0]])
        uv = (xy - mean) @ np.column_stack([u, v])
        umin, vmin = uv.min(axis=0) - 5
        umax, vmax = uv.max(axis=0) + 5
        rw, rh = int(umax - umin) + 1, int(vmax - vmin) + 1
        uu, vv = np.meshgrid(np.arange(rw) + umin, np.arange(rh) + vmin)
        px = mean[0] + uu * u[0] + vv * v[0]
        py = mean[1] + uu * u[1] + vv * v[1]
        rect = cv2.remap(gray, px.astype('float32'), py.astype('float32'),
                         cv2.INTER_LINEAR, borderValue=255)
        mask = cv2.remap((labels == ident).astype('uint8') * 255,
                         px.astype('float32'), py.astype('float32'), cv2.INTER_NEAREST)
        sections, hits = [], []
        for fraction in np.linspace(.2, .8, 9):
            y = int(fraction * rh)
            indices = np.where(mask[max(0, y - 2):y + 3].mean(axis=0) > 100)[0]
            if len(indices) < 3:
                continue
            left, right = int(indices[0]), int(indices[-1])
            section_width = right - left
            profile = gaussian_filter1d(
                rect[max(0, y - 4):y + 5].mean(axis=0).astype(float), .6)
            highpass = profile - gaussian_filter1d(profile, max(2, section_width * .13))
            peaks, props = find_peaks(
                -highpass, prominence=35, distance=max(2, section_width * .18))
            boundaries = [dict(
                x=int(p), position=float((p - left) / max(section_width, 1)),
                prominence=round(float(prominence), 2))
                for p, prominence in zip(peaks, props['prominences'])
                if left + section_width * .2 < p < right - section_width * .2]
            for boundary in boundaries:
                hits.append((len(sections), boundary['position'], boundary['prominence']))
            sections.append(dict(y=y, left=left, right=right, boundaries=boundaries))
        if len(sections) < 8:
            return None

        clusters = []
        for position in np.arange(.2, .801, .04):
            selected = [hit for hit in hits if abs(hit[1] - position) < .07]
            support = len({hit[0] for hit in selected})
            if support >= 6:
                center = float(np.median([hit[1] for hit in selected]))
                if all(abs(center - cluster['position']) > .15 for cluster in clusters):
                    clusters.append(dict(position=center, support=support))
        clusters.sort(key=lambda cluster: cluster['position'])
        if len(clusters) > 2:
            return None
        if not clusters and aspect < 18:
            return None
        if not clusters and len({hit[0] for hit in hits}) >= 3:
            return None

        count = len(clusters) + 1
        axis_samples = [[] for _ in range(count)]
        for section in sections:
            left, right = section['left'], section['right']
            limits = [float(left)]
            for cluster in clusters:
                near = [boundary for boundary in section['boundaries']
                        if abs(boundary['position'] - cluster['position']) < .09]
                limits.append(float(max(near, key=lambda p: p['prominence'])['x'])
                              if near else left + (right - left) * cluster['position'])
            limits.append(float(right))
            for j in range(count):
                axis_samples[j].append((section['y'], (limits[j] + limits[j + 1]) / 2))

        for samples in axis_samples:
            ya = np.array([p[0] for p in samples])
            xa = np.array([p[1] for p in samples])
            coef = np.polyfit(ya, xa, 1)
            endpoints = []
            for fraction in [.12, .88]:
                cy = fraction * rh
                cx = float(np.polyval(coef, cy))
                point = mean + (cx + umin) * u + (cy + vmin) * v
                endpoints.append([round(float(point[0]), 2), round(float(point[1]), 2)])
            centerlines.append(endpoints)
        component_details.append(dict(
            component_id=int(ident), count=count, aspect=round(aspect, 3),
            section_counts=[len(section['boundaries']) + 1 for section in sections],
            stable_boundaries=[dict(position=round(cluster['position'], 4),
                                    support=cluster['support']) for cluster in clusters]))

    points = [dict(x=round(sum(p[0] for p in line) / 2, 2),
                   y=round(sum(p[1] for p in line) / 2, 2), r=3.0, segment=line)
              for line in centerlines]
    points.sort(key=lambda point: (point['y'], point['x']))
    metadata = dict(
        framework='LaidRodSections', device='cpu', detected_category='laid_rods',
        category='laid_rods', mode='longitudinal_seam_consensus', prediction_valid=True,
        diagnosis='detections_returned',
        count_unit='individual_whole_rod_not_pair', manual_exemplars=False,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        foreground_coverage=round(coverage, 4), section_counts=component_details,
        consensus=dict(transverse_sections=9, minimum_matching_sections=6,
                       normalized_boundary_tolerance=.07,
                       description='Count stable internal seams plus one; reject unresolved components.'),
        parameters=PARAMETERS.copy(),
        limitation=('Only light neutral backgrounds and elongated, separated groups with visible '
                    'stable seams. Dense fans, broad touching groups, low color contrast and '
                    'overlap may be rejected. Longitudinal wood grain can mimic a seam; '
                    'these development parameters are not an accuracy guarantee.'),
        segment_definition='Approximate inner axis from 12% to 88% of component length.')
    return points, metadata
