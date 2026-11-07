"""Validate query isolation before using development data as a training gate."""


def validate_splits(pools):
    names = ('train', 'dev', 'test')
    for name in names:
        if name not in pools or not pools[name]:
            raise ValueError(f'Missing or empty {name} split')
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = set(pools[left]) & set(pools[right])
            if shared:
                raise ValueError(f'{left} and {right} share {len(shared)} query IDs')
