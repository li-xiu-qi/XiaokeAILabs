"""Validate both 256-image reviews and calculate descriptive paired statistics."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ARMS = ('base', 'self', 'teacher_student', 'teacher_model')
FIELDS = ('single_mug', 'color_correct', 'background_correct', 'severe_artifact')


def load_review(review, key_path, frozen):
    key = json.loads(key_path.read_text(encoding='utf-8'))
    with review.open(encoding='utf-8-sig', newline='') as f:
        scores = list(csv.DictReader(f))
    by_blind = {r['blind_id']: r for r in key}
    ratings = {r['blind_id']: r for r in scores}
    if len(key) != 256 or len(by_blind) != 256 or len(scores) != 256 or len(ratings) != 256 or set(by_blind) != set(ratings):
        raise ValueError('review or mapping missing/duplicate blind IDs')
    expected = {(arm, id_) for arm in ARMS for id_ in frozen}
    observed = {(r['arm'], r['id']) for r in key}
    if len(observed) != 256 or observed != expected:
        raise ValueError('mapping does not contain the frozen four-arm Cartesian set')
    result = {}
    for blind, item in by_blind.items():
        row = ratings[blind]
        prompt = frozen[item['id']]['prompt']
        if not re.fullmatch(r'item-\d{3}', blind):
            raise ValueError('invalid blind ID')
        if row['prompt'] != prompt or item['prompt'] != prompt or item['seed'] != frozen[item['id']]['seed']:
            raise ValueError('prompt or seed mismatch')
        if any(row.get(field) not in ('0', '1', 'unknown') for field in FIELDS):
            raise ValueError('missing or invalid rating')
        result[item['arm'], item['id']] = {f: row[f] for f in FIELDS}
    return result


def success(row):
    if any(row[f] == 'unknown' for f in FIELDS):
        return None
    return int(all(row[f] == '1' for f in FIELDS[:3]) and row[FIELDS[3]] == '0')


def summarize(scores, ids):
    result = {'arms': {}, 'paired_vs_base': {}}
    for arm in ARMS:
        vals = [success(scores[arm, i]) for i in ids]
        scenes = {}
        for id_, value in zip(ids, vals):
            scenes.setdefault(id_.rsplit('-', 1)[0], []).append(value)
        result['arms'][arm] = {
            'image_success': vals.count(1), 'image_failure': vals.count(0), 'image_unknown': vals.count(None),
            'scenes': {scene: {'success': v.count(1), 'failure': v.count(0), 'unknown': v.count(None)} for scene, v in scenes.items()}}
    for arm in ARMS[1:]:
        scene_delta = {}
        for id_ in ids:
            a, b = success(scores['base', id_]), success(scores[arm, id_])
            scene_delta.setdefault(id_.rsplit('-', 1)[0], []).append(None if a is None or b is None else b - a)
        flat = [x for values in scene_delta.values() for x in values]
        result['paired_vs_base'][arm] = {'improved': flat.count(1), 'worse': flat.count(-1), 'unchanged': flat.count(0),
                                        'unknown_pairs': flat.count(None), 'scene_differences': scene_delta}
    return result


def run(manifest, expected_hash, review1, key1, review2, key2, out):
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash.lower():
        raise ValueError('frozen manifest hash mismatch')
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    frozen = {r['id']: r for r in rows}
    if len(rows) != 64 or len(frozen) != 64 or any(not re.fullmatch(r'test-\d{3}-[01]', i) for i in frozen):
        raise ValueError('invalid frozen IDs')
    pairs = {}
    for i in frozen:
        scene, n = i.rsplit('-', 1)
        pairs.setdefault(scene, set()).add(n)
    if len(pairs) != 32 or any(s != {'0', '1'} for s in pairs.values()):
        raise ValueError('invalid 32-scene two-seed grouping')
    if out.exists():
        raise ValueError('output already exists')
    first = load_review(review1, key1, frozen)
    second = load_review(review2, key2, frozen)
    disagreements = {}
    for field in FIELDS:
        same = sum(first[k][field] == second[k][field] for k in first)
        disagreements[field] = {'agree': same, 'disagree': 256 - same,
                                'unknown_in_either': sum('unknown' in (first[k][field], second[k][field]) for k in first)}
    composite = [(success(first[k]), success(second[k])) for k in first]
    disagreements['composite'] = {'agree_including_unknown': sum(a == b for a, b in composite),
                                   'disagree_including_unknown': sum(a != b for a, b in composite),
                                   'unknown_in_either': sum(a is None or b is None for a, b in composite)}
    payload = {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'reviewer_1': summarize(first, list(frozen)),
               'reviewer_2': summarize(second, list(frozen)), 'reviewer_agreement': disagreements,
               'interpretation': 'Descriptive matched results; two seeds belong to the same scene and are not independent scenes.'}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Recomputed ratings; omitted images cannot be checked against their recorded hashes')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    for name in ('review1', 'key1', 'review2', 'key2', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.manifest_sha256, a.review1, a.key1, a.review2, a.key2, a.out)
