"""Check query isolation and ranking inputs before model evaluation."""


def validate_candidate_rows(train, test, queries, docs):
    train_ids = [row['id'] for row in train]
    test_ids = [row['id'] for row in test]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError('query IDs must be unique within each split')
    if set(train_ids) & set(test_ids):
        raise ValueError('training and test query IDs overlap')
    for split, rows, count in (('train', train, 4), ('test', test, 30)):
        for row in rows:
            candidates = row['candidates']
            if len(candidates) != count or len(set(candidates)) != count:
                raise ValueError(f'{split} {row["id"]}: expected {count} distinct candidates')
            if row['id'] not in queries or any(did not in docs for did in candidates):
                raise ValueError(f'{split} {row["id"]}: unknown query or candidate document')
            if split == 'test':
                relevant = row['relevant']
                if not relevant or len(set(relevant)) != len(relevant) or any(did not in docs for did in relevant):
                    raise ValueError(f'test {row["id"]}: invalid or unknown relevance documents')
