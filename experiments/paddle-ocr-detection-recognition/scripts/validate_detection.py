"""Validate paired detection outputs before the recognition stage."""
import math


def validate_detections(boxes, scores):
    if len(boxes) != len(scores):
        raise ValueError('detection boxes and scores have different lengths')
    for box, score in zip(boxes, scores):
        if (not isinstance(score, (int, float)) or not math.isfinite(score)
                or len(box) != 4 or any(len(p) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in box)):
            raise ValueError('invalid detection polygon or score')
        if min(p[0] for p in box) >= max(p[0] for p in box) or min(p[1] for p in box) >= max(p[1] for p in box):
            raise ValueError('degenerate detection polygon cannot be cropped')
