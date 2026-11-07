"""Validate pointwise targets and count single-process optimizer updates."""
import math


def optimizer_steps_per_run(batches_per_epoch, accumulation, epochs):
    if batches_per_epoch < 1 or accumulation < 1 or epochs < 1:
        raise ValueError('batch count, accumulation and epochs must be positive')
    return ((batches_per_epoch + accumulation - 1) // accumulation) * epochs


def normalized_label(item, min_label, max_label):
    if 'label' not in item or isinstance(item['label'], bool) or not isinstance(item['label'], (int, float)):
        raise ValueError('label must be an explicitly supplied number')
    if not math.isfinite(item['label']) or not math.isfinite(min_label) or not math.isfinite(max_label):
        raise ValueError('label and label bounds must be finite')
    if min_label >= max_label or not min_label <= item['label'] <= max_label:
        raise ValueError('label must be within valid increasing bounds')
    return (item['label'] - min_label) / (max_label - min_label)
