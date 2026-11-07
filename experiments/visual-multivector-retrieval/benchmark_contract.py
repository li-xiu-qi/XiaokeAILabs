"""Selection and alignment checks for the fixed-candidate benchmark."""
import hashlib
import json


def select_rows(data, count):
    if count < 1:
        raise ValueError('count must be positive')
    rows, seen = [], set()
    for row in data:
        name = row['image_filename']
        if not isinstance(name, str) or not name or not isinstance(row['query'], str) or not row['query'].strip():
            raise ValueError('every row must have a nonempty filename and query')
        if name not in seen:
            seen.add(name)
            rows.append(row)
            if len(rows) == count:
                return rows
    raise ValueError(f'expected {count} distinct images, found {len(rows)}')


def selection_sha256(rows):
    content = [(row['image_filename'], row['query']) for row in rows]
    data = json.dumps(content, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()
