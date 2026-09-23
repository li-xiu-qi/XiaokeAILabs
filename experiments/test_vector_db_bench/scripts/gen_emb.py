"""生成 20 Newsgroups 的 sentence embedding，存 data/20news_emb.npz。

要点：
- 去邮件 header（空行前是 header，之后才是正文），否则正文会被 256 token 截断挤掉
- max_seq_length=256：本机 CPU 40 docs/s，全集约 8 分钟；384/512 要 17-28 分钟
- 五库对比只看同一份向量的一致性，模型精度不影响结论
"""
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "20news"
MAX_SEQ = 256
MAX_CHARS = 1200


def load_split(split_dir, per_cat=None):
    texts, labels, names = [], [], []
    for cat_dir in sorted(split_dir.iterdir()):
        files = sorted(cat_dir.iterdir())
        if per_cat:
            files = files[:per_cat]
        for fp in files:
            try:
                raw = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            parts = raw.split("\n\n", 1)
            body = parts[1] if len(parts) > 1 else raw
            body = " ".join(body.split())
            if len(body) < 30:
                continue
            texts.append(body[:MAX_CHARS])
            labels.append(cat_dir.name)
            names.append(split_dir.name + "/" + cat_dir.name + "/" + fp.name)
    return texts, labels, names


def main():
    t00 = time.perf_counter()
    model = SentenceTransformer(str(ROOT / "models" / "all-MiniLM-L6-v2"))
    model.max_seq_length = MAX_SEQ

    out = {}
    for split_dir in sorted(BASE.iterdir()):
        if not split_dir.is_dir():
            continue
        per_cat = 30 if "test" in split_dir.name else None
        texts, labels, names = load_split(split_dir, per_cat=per_cat)
        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=64, show_progress_bar=False)
        print(split_dir.name + ": " + str(len(emb)) + " docs, "
              + str(round(time.perf_counter() - t0, 1)) + "s, dim=" + str(emb.shape[1]),
              flush=True)
        out[split_dir.name] = (emb, labels, names)

    first = sorted(out)[0]
    np.savez_compressed(
        ROOT / "data" / "20news_emb.npz",
        train_emb=out["20news-bydate-train"][0],
        test_emb=out["20news-bydate-test"][0],
        train_labels=out["20news-bydate-train"][1],
        test_labels=out["20news-bydate-test"][1],
        train_names=out["20news-bydate-train"][2],
        test_names=out["20news-bydate-test"][2],
    )
    print("saved, total " + str(round(time.perf_counter() - t00, 1)) + "s")


if __name__ == "__main__":
    main()
