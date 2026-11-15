"""Exercise the training entry point on synthetic data with a tiny random model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch installation and GPU are required.")
    print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True)
    with tempfile.TemporaryDirectory(prefix="colbert-synthetic-") as temp:
        root = Path(temp)
        model = root / "model"
        data = root / "data"
        prepared = root / "prepared"
        for path in (model, data, prepared):
            path.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        vocab += [f"[unused{i}]" for i in range(100)]
        vocab += ["science", "text", "question", "first", "second", "document", "answer", "other", "wrong"]
        (model / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        BertTokenizerFast(vocab_file=str(model / "vocab.txt")).save_pretrained(str(model))
        config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64, max_position_embeddings=512)
        BertModel(config).save_pretrained(str(model))
        documents = {"d0": "science document answer", "d1": "other document wrong",
                     "d2": "first document text", "d3": "second document text"}
        queries = {f"q{i}": "science question" for i in range(4)}
        queries.update({"qt": "science question", "qdev": "science question"})
        write_jsonl(data / "corpus.jsonl", [{"_id": key, "text": text} for key, text in documents.items()])
        write_jsonl(data / "queries.jsonl", [{"_id": key, "text": text} for key, text in queries.items()])
        item = {"relevant": ["d0"], "unjudged": ["d1", "d2", "d3"],
                "pool": [["d0", 2.0], ["d1", 1.0]]}
        pools = {"train": {f"q{i}": item for i in range(4)}, "dev": {"qdev": item}, "test": {"qt": item}}
        (prepared / "pools.json").write_text(json.dumps(pools), encoding="utf-8")
        pairs = root / "pairs.jsonl"
        write_jsonl(pairs, [{"query_id": f"q{i}", "positive_doc_id": "d0", "unjudged_doc_id": "d1"}
                            for i in range(4)])
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")), "--data", str(data),
                        "--prepared", str(prepared), "--pairs", str(pairs), "--model", str(model),
                        "--out", str(root / "result"), "--smoke"], env=env, check=True)
    print("Synthetic GPU smoke completed.")


if __name__ == "__main__":
    main()
