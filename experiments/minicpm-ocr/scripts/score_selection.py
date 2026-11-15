"""Choose the highest scored candidate without treating rendering errors as scores."""
import math


def best_scored_index(scores, verdicts):
    if len(scores) != len(verdicts):
        raise ValueError("scores and verdicts must have equal lengths")
    valid = [i for i, (score, verdict) in enumerate(zip(scores, verdicts))
             if verdict == "scored" and math.isfinite(score)]
    return max(valid, key=lambda i: scores[i]) if valid else None
