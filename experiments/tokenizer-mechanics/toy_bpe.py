"""A tiny byte-BPE teaching example; not a production tokenizer."""
from collections import Counter

CORPUS = ["low lower lowest", "new newer newest", "low low lower"]
EXAMPLES = ["low lower", "newest", "你好🙂", "a  b\n"]


def merge(sequence, pair):
    out, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            out.append(sequence[i] + sequence[i + 1])
            i += 2
        else:
            out.append(sequence[i])
            i += 1
    return out


def train(corpus, count=12):
    sequences = [[bytes([b]) for b in text.encode('utf-8')] for text in corpus]
    vocabulary = {bytes([b]): b for b in range(256)}
    merges = []
    for _ in range(count):
        pairs = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
        candidates = sorted(pairs, key=lambda pair: (-pairs[pair], pair))
        pair = next((p for p in candidates if p[0] + p[1] not in vocabulary), None)
        if pair is None:
            break
        merged = pair[0] + pair[1]
        vocabulary[merged] = len(vocabulary)
        merges.append(pair)
        sequences = [merge(seq, pair) for seq in sequences]
    return merges, vocabulary


def encode(text, merges, vocabulary):
    pieces = [bytes([b]) for b in text.encode('utf-8')]
    for pair in merges:
        pieces = merge(pieces, pair)
    return [vocabulary[piece] for piece in pieces]


def decode(ids, vocabulary):
    inverse = {value: key for key, value in vocabulary.items()}
    return b''.join(inverse[i] for i in ids).decode('utf-8')


def main():
    merges, vocabulary = train(CORPUS)
    inverse = {value: key for key, value in vocabulary.items()}
    for i, (left, right) in enumerate(merges, 1):
        print(f'merge {i:02}: {left!r} + {right!r} -> {left + right!r}')
    for text in EXAMPLES:
        ids = encode(text, merges, vocabulary)
        assert decode(ids, vocabulary) == text
        print(f'{text!r}: bytes={len(text.encode("utf-8"))}, tokens={len(ids)}, ids={ids}')
        print('  token bytes:', [inverse[i].hex() for i in ids])


if __name__ == '__main__':
    main()
