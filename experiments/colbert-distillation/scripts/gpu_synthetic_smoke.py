"""Exercise the distillation training function on CUDA with synthetic data."""

import importlib.util
import tempfile
from pathlib import Path

import torch
from pylate import models
from transformers import BertConfig, BertModel, BertTokenizerFast


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires an available CUDA GPU")

    with tempfile.TemporaryDirectory(prefix="colbert-distillation-smoke-") as tmp:
        model_path = Path(tmp) / "tiny-model"
        model_path.mkdir()
        vocab = (
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            + [f"[unused{i}]" for i in range(100)]
            + ["science", "text", "question", "first", "second", "third", "fourth",
               "document", "answer", "other", "wrong"]
        )
        (model_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        BertTokenizerFast(vocab_file=str(model_path / "vocab.txt")).save_pretrained(str(model_path))
        config = BertConfig(
            vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, max_position_embeddings=512,
        )
        torch.manual_seed(42)
        BertModel(config).save_pretrained(str(model_path))

        run_path = Path(__file__).resolve().with_name("run.py")
        spec = importlib.util.spec_from_file_location("distill_run", run_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = models.ColBERT(
            model_name_or_path=str(model_path), device="cuda", query_length=32,
            document_length=180, local_files_only=True,
        )
        docs = {
            "d0": "science document answer", "d1": "other document wrong",
            "d2": "first document text", "d3": "second document text",
        }
        queries = {f"q{i}": "science question" for i in range(4)}
        rows = [{"id": f"q{i}", "candidates": ["d0", "d1", "d2", "d3"]} for i in range(4)]
        teacher = {f"q{i}": [1.0, 0.2, 0.1, -0.1] for i in range(4)}
        print("CUDA device:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, flush=True)
        result = module.train(model, docs, queries, rows, teacher, "distilled", True)
        print("Synthetic GPU distillation result:", result, flush=True)


if __name__ == "__main__":
    main()
