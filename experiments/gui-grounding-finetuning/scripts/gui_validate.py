"""Validate the recorded point-in-box result before aggregation."""
import math
import re


def parse_point(answer):
    """Parse the generated 0–1000 coordinate pair into normalized coordinates."""
    if not isinstance(answer, str):
        raise ValueError("prediction must be a string")
    text = answer.strip()
    if text.startswith("(") or text.endswith(")"):
        if not (text.startswith("(") and text.endswith(")")):
            return None
        text = text[1:-1]
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*", text)
    return [float(match.group(i)) / 1000 for i in (1, 2)] if match else None


def validate_result(row):
    box, point = row['bbox'], row['point_normalized']
    if point != parse_point(row['prediction']):
        raise ValueError('recorded point disagrees with prediction')
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
            or not (0 <= box[0] <= box[2] <= 1 and 0 <= box[1] <= box[3] <= 1)):
        raise ValueError('invalid normalized bounding box')
    if point is None:
        expected = False
    else:
        if (not isinstance(point, list) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)):
            raise ValueError('invalid normalized point')
        expected = (0 <= point[0] <= 1 and 0 <= point[1] <= 1
                    and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])
    if not isinstance(row['hit'], bool) or row['hit'] != expected:
        raise ValueError('recorded hit disagrees with point and bounding box')
