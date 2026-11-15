"""Executable adaptation of llm-mechanics/tokenizer/ngram_explanation.ipynb."""
import re


def generate_ngrams(text: str, n: int, word_based: bool = True):
    if n < 1:
        raise ValueError('n must be positive')
    if word_based:
        units = re.sub(r'[^\w\s]', '', text.lower()).split()
        return [tuple(units[i:i+n]) for i in range(len(units)-n+1)]
    text = re.sub(r'\s+', ' ', text).strip()
    return [text[i:i+n] for i in range(len(text)-n+1)]


if __name__ == '__main__':
    print('English word bigrams:', generate_ngrams('The quick brown fox.', 2))
    print('Chinese character bigrams:', generate_ngrams('我爱自然语言处理技术', 2, False))
