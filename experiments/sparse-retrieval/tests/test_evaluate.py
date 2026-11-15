import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "evaluate.py"
spec = importlib.util.spec_from_file_location("sparse_eval", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class EvaluationTests(unittest.TestCase):
    def test_mrr_ties_and_misses(self):
        result = ev.metrics(
            {"q1": {"b": 2.0, "a": 2.0}, "q2": {"a": 1.0, "b": 0.0}},
            {"q1": {"b"}, "q2": {"b"}},
            top_k=1,
        )
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["per_query"]["q1"]["top_doc_ids"], ["a"])
        self.assertEqual(ev.metrics({"q": {"b": 2.0, "a": 2.0}}, {"q": {"b"}})["mrr"], 0.5)
        self.assertEqual(ev.metrics({"q": {"a": 0.0, "b": 0.0}}, {"q": {"a"}})["mrr"], 0.0)

    def test_cjk_and_bm25(self):
        self.assertEqual(ev.terms("AI技术 OpenAI"), ["ai", "技", "术", "openai"])
        scores = ev.bm25_scores({"q": "稀疏检索"}, {"a": "稀疏检索", "b": "美食推荐"})
        self.assertGreater(scores["q"]["a"], scores["q"]["b"])
        with self.assertRaises(ValueError):
            ev.bm25_scores({"q": "!!!"}, {"a": "apple"})

    def test_qrels_validation_and_end_to_end_bm25(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)

            def save(filename, records):
                path = root / filename
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8",
                )
                return path

            corpus = save(
                "corpus.jsonl",
                [{"doc_id": "d1", "text": "apple fruit"}, {"doc_id": "d2", "text": "cat pet"}],
            )
            queries = save(
                "queries.jsonl",
                [{"query_id": "q1", "text": "apple"}, {"query_id": "q2", "text": "cat"}],
            )
            qrels = save(
                "qrels.jsonl",
                [
                    {"query_id": "q1", "doc_id": "d1", "relevance": 1},
                    {"query_id": "q2", "doc_id": "d2", "relevance": 1},
                ],
            )
            output = root / "out.json"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--method",
                "bm25",
                "--corpus",
                str(corpus),
                "--queries",
                str(queries),
                "--qrels",
                str(qrels),
                "--out",
                str(output),
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["mrr"], 1.0)
            self.assertEqual(payload["candidate_documents"], 2)
            prior = output.read_bytes()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), prior)
            qrels.write_text(
                '{"query_id":"q1","doc_id":"missing","relevance":1}\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                ev.read_qrels(qrels, {"q1": "apple"}, {"d1": "apple"})

    def test_splade_batch_scores_and_active_term_counts(self):
        import torch

        old_transformers, old_model = sys.modules.get("transformers"), sys.modules.get("model")

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def __call__(self, texts, **kwargs):
                ids = [[1, 0] if text == "apple" else [0, 1] for text in texts]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), 2)}

        class Model(torch.nn.Module):
            def __init__(self, path):
                super().__init__()

            def forward(self, input_ids, attention_mask):
                return input_ids.float()

        sys.modules["transformers"] = types.SimpleNamespace(
            AutoTokenizer=Tokenizer, __version__="stub"
        )
        sys.modules["model"] = types.SimpleNamespace(SpladeModel=Model)
        try:
            with tempfile.TemporaryDirectory() as directory:
                args = types.SimpleNamespace(
                    device="cpu",
                    model=directory,
                    checkpoint=None,
                    batch_size=1,
                    max_len_q=64,
                    max_len_d=256,
                )
                scores, meta = ev.splade_scores(
                    args, {"q1": "apple", "q2": "cat"}, {"d1": "apple", "d2": "cat"}
                )
                self.assertEqual(
                    scores, {"q1": {"d1": 1.0, "d2": 0.0}, "q2": {"d1": 0.0, "d2": 1.0}}
                )
                self.assertEqual(meta["mean_doc_active_terms_gt_1e-4"], 1.0)
        finally:
            for name, original in [("transformers", old_transformers), ("model", old_model)]:
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original


if __name__ == "__main__":
    unittest.main()
