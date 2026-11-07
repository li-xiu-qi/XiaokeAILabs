"""Validate the sample identity and vector norms before ranking."""
import math


def validate_sample(rows):
    names = [row.get('filename') for row in rows]
    if (len(names) != 100 or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('The sampled image pool must contain 100 distinct, nonempty filenames')


def validate_norms(norms):
    if any(not math.isfinite(float(n)) or float(n) <= 0 for n in norms):
        raise ValueError('Cannot rank using zero or non-finite embedding norms')
