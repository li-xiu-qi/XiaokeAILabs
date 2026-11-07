"""Validate input IDs and compare generation metadata without video dependencies."""
import re


def validate_entries(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest split must contain at least one entry')
    seen = set()
    for entry in entries:
        key = entry.get('id')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', key) or key in seen:
            raise ValueError(f'unsafe or duplicate video ID: {key!r}')
        if not isinstance(entry.get('prompt'), str) or not entry['prompt'].strip():
            raise ValueError(f'missing prompt: {key}')
        if isinstance(entry.get('seed'), bool) or not isinstance(entry.get('seed'), int):
            raise ValueError(f'invalid integer seed: {key}')
        seen.add(key)


def validate_pair(left, right, video_names):
    for field in ('model', 'size', 'frames', 'steps', 'fps', 'guidance', 'split'):
        if field not in left or field not in right or left[field] != right[field]:
            raise ValueError(f'generation setting mismatch: {field}')
    if left.get('adapter') is not False or right.get('adapter') is not True:
        raise ValueError('expected baseline without adapter and LoRA with adapter')
    for data in (left, right):
        validate_entries(data['entries'])
    a = {e['id']: e for e in left['entries']}
    b = {e['id']: e for e in right['entries']}
    if a.keys() != b.keys() or {f'{key}.mp4' for key in a} != set(video_names):
        raise ValueError('manifest IDs differ from paired videos')
    for key in a:
        if (a[key]['prompt'], a[key]['seed']) != (b[key]['prompt'], b[key]['seed']):
            raise ValueError(f'prompt or seed mismatch for {key}')
