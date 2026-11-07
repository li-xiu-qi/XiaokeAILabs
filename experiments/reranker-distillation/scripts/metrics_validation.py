"""Reject invalid fixed-pool inputs before calculating ranking metrics."""
import math

def validate_metrics_inputs(part, scores):
    if not part:
        raise ValueError('Evaluation query set is empty')
    for qid, row in part.items():
        ids = [str(did) for did, _ in row['pool']]
        rel = [str(did) for did in row['relevant']]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'{qid}: empty pool or duplicate candidate IDs')
        if not rel or len(rel) != len(set(rel)):
            raise ValueError(f'{qid}: empty or duplicate relevance IDs')
        for did in ids:
            key = f'{qid}:{did}'
            if key not in scores:
                raise ValueError(f'{qid}: missing score for candidate {did}')
            value = scores[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{qid}: invalid score for candidate {did}')
