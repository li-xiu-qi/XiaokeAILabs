"""Validate the recorded point-in-box result before aggregation."""
import math


def validate_result(row):
    box, point = row['bbox'], row['point_normalized']
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
            or not (0 <= box[0] <= box[2] <= 1 and 0 <= box[1] <= box[3] <= 1)):
        raise ValueError('invalid normalized bounding box')
    if point is None:
        expected = False
    else:
        if (not isinstance(point, list) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point)):
            raise ValueError('invalid normalized point')
        expected = box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
    if not isinstance(row['hit'], bool) or row['hit'] != expected:
        raise ValueError('recorded hit disagrees with point and bounding box')
