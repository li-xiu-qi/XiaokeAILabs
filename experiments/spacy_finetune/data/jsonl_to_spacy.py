import spacy
from spacy.tokens import Doc, DocBin
import json

nlp = spacy.blank("zh")
doc_bin = DocBin()

with open("senter_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        tokens = example["tokens"]
        sent_starts = example["sent_starts"]
        if len(tokens) != len(sent_starts):
            raise ValueError(f"tokens 数与 sent_starts 不一致: {tokens}\nsent_starts: {sent_starts}")
        doc = Doc(nlp.vocab, words=tokens)
        for i, token in enumerate(doc):
            token.is_sent_start = sent_starts[i]
        doc_bin.add(doc)

doc_bin.to_disk("senter_train.spacy")
print("已生成 senter_train.spacy ")
