"""Compare encodings on identical texts. First use may download vocabulary files."""
import importlib.metadata
import tiktoken

SAMPLES = ['low lower', '你好🙂', 'a  b\n', 'The quick brown fox jumps over the lazy dog.']


def main():
    print('tiktoken', importlib.metadata.version('tiktoken'))
    for name in ('r50k_base', 'cl100k_base'):
        enc = tiktoken.get_encoding(name)
        print('\n', name)
        for text in SAMPLES:
            ids = enc.encode(text, disallowed_special=())
            raw = [enc.decode_single_token_bytes(token) for token in ids]
            assert b''.join(raw).decode('utf-8') == text
            print(repr(text), 'tokens=', len(ids), 'ids=', ids,
                  'token bytes=', [x.hex() for x in raw])


if __name__ == '__main__':
    main()
