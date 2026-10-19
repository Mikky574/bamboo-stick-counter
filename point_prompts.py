"""Validate user-selected visual examples in uploaded-image pixel coordinates.

Only current request points are accepted. Saved truth/annotations and filenames
are never used to choose examples. A point and diameter define one visual box.
"""
import math


def validate_exemplar_points(points, width, height):
    if not isinstance(points, list) or not 1 <= len(points) <= 3:
        raise ValueError('请点选 1–3 个目标作为示例。')
    normalized = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError('示例点格式错误。')
        values = []
        for key in ('x', 'y', 'diameter'):
            value = point.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('示例点坐标和大小必须是有限数字。')
            try:
                value = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError('示例点坐标和大小必须是有限数字。') from error
            if not math.isfinite(value):
                raise ValueError('示例点坐标和大小必须是有限数字。')
            values.append(value)
        x, y, diameter = values
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError('示例点必须位于图片内。')
        if not 8 <= diameter <= 400:
            raise ValueError('示例目标直径必须在 8–400 原图像素之间。')
        if any(math.hypot(x-old['x'], y-old['y']) < 2 for old in normalized):
            raise ValueError('示例点重复，请选择不同的目标。')
        normalized.append(dict(x=x, y=y, diameter=diameter))
    return normalized


def exemplar_boxes(points, width, height):
    points = validate_exemplar_points(points, width, height)
    return [[max(0., p['x']-p['diameter']/2), max(0., p['y']-p['diameter']/2),
             min(float(width), p['x']+p['diameter']/2),
             min(float(height), p['y']+p['diameter']/2)] for p in points]
