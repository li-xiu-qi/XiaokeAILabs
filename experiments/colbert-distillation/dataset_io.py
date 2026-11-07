"""Read corpus/query rows without silently overwriting duplicate IDs."""
import json


def read_records(path):
    records = []
    seen = set()
    with path.open(encoding='utf8') as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get('_id')
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError(f'{path}:{line_no}: missing, non-string or duplicate _id')
            seen.add(identifier)
            records.append(row)
    return records
