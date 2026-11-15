"""Exercise teacher scoring and both student training losses on synthetic inputs."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import BertConfig, BertModel, BertTokenizerFast

import train


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch installation and GPU are required.")
    train.check()
    torch.cuda.set_per_process_memory_fraction(0.20)
    torch.manual_seed(train.SEED)
    print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True)

    with tempfile.TemporaryDirectory(prefix="embedding-distillation-synthetic-") as temp:
        model_dir = Path(temp) / "model"
        model_dir.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "science", "question",
                 "document", "answer", "other", "first", "second", "third"]
        (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt")).save_pretrained(str(model_dir))
        config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64)
        BertModel(config).save_pretrained(str(model_dir))

        def model(device):
            encoder = models.Transformer(str(model_dir), max_seq_length=64)
            pooling = models.Pooling(encoder.get_word_embedding_dimension())
            return SentenceTransformer(modules=[encoder, pooling], device=device)

        # Exercise the public two-stage CLI, including its development-set gate.
        saved_model = Path(temp) / "sentence-model"
        model("cpu").save(str(saved_model))
        data_dir = Path(temp) / "data"
        data_dir.mkdir()
        ids = [f"d{i}" for i in range(30)]
        docs = [{"_id": d, "text": "science document answer"} for d in ids]
        query_sets = {"train": [f"train{i}" for i in range(128)],
                      "dev": ["dev0", "dev1"], "test": ["test0", "test1"]}
        for filename, records in (("corpus.jsonl", docs),
                                  ("queries.jsonl", [{"_id": q, "text": "science question"}
                                                     for group in query_sets.values() for q in group])):
            (data_dir / filename).write_text("".join(json.dumps(row) + "\n" for row in records),
                                             encoding="utf-8")
        pool = {"relevant": [ids[0]], "unjudged": ids[1:],
                "pool": [[d, float(30 - i)] for i, d in enumerate(ids)]}
        pools = Path(temp) / "pools.json"
        pools.write_text(json.dumps({split: {q: pool for q in names}
                                     for split, names in query_sets.items()}), encoding="utf-8")
        output = Path(temp) / "output"
        for stage in ("teacher", "student"):
            subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")),
                            "--stage", stage, "--smoke", "--data", str(data_dir),
                            "--pools", str(pools), "--out", str(output),
                            "--teacher", str(saved_model), "--student", str(saved_model),
                            "--min-teacher-gap", "0.05"], check=True)
        assert (output / "smoke" / "stopped.json").is_file()

        documents = {"d0": "science document answer", "d1": "other document",
                     "d2": "first document", "d3": "second document"}
        queries = {"q0": "science question", "q1": "first question"}
        rows = [{"id": q, "candidates": ["d0", "d1", "d2", "d3"]} for q in queries]
        teacher_model = model("cuda")
        scores, _ = train.calc(teacher_model, documents, queries, rows, [], 2, teacher=True)
        del teacher_model
        torch.cuda.empty_cache()
        for mode in ("supervised", "distilled"):
            student = model("cuda")
            first_parameter = next(student.parameters())
            before = first_parameter.detach().clone()
            result = train.train_student(student, mode, rows, documents, queries, scores, smoke=True)
            if result["steps"] != 2 or torch.equal(before, first_parameter.detach()):
                raise RuntimeError(f"{mode} did not update the model in two steps")
            print(mode, result, flush=True)
            del before, student, first_parameter
            torch.cuda.empty_cache()
    print("Synthetic GPU scoring and training smoke completed.", flush=True)


if __name__ == "__main__":
    main()
