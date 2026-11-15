"""生成 20 Newsgroups 的 sentence embedding，存 data/20news_emb.npz。

要点：
- 去邮件 header（空行前是 header，之后才是正文），否则正文会被 256 token 截断挤掉
- max_seq_length=256：本机 CPU 40 docs/s，全集约 8 分钟；384/512 要 17-28 分钟
- 五库对比只看同一份向量的一致性，模型精度不影响结论
"""
import argparse
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
if __package__:
    from .paired_texts import text_digest
else:
    from paired_texts import text_digest

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=BASE,
                        help="20 Newsgroups 解压目录")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "all-MiniLM-L6-v2",
                        help="本地模型目录")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "20news_emb.npz",
                        help="输出路径，已有文件不会覆盖")
    parser.add_argument("--texts-output", type=Path, default=None,
                        help="同时保存与向量同序的文本 NPZ；已有文件不会覆盖")
    args = parser.parse_args()
    outputs = [args.output] + ([args.texts_output] if args.texts_output is not None else [])
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("向量和文本必须使用不同的输出路径")
    for output in outputs:
        if os.path.lexists(output):
            raise FileExistsError(f"输出已存在，请选择新路径：{output}")
    t00 = time.perf_counter()
    model = SentenceTransformer(str(args.model))
    model.max_seq_length = MAX_SEQ

    out = {}
    for split_dir in sorted(args.data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        per_cat = 30 if "test" in split_dir.name else None
        texts, labels, names = load_split(split_dir, per_cat=per_cat)
        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=64, show_progress_bar=False)
        print(split_dir.name + ": " + str(len(emb)) + " docs, "
              + str(round(time.perf_counter() - t0, 1)) + "s, dim=" + str(emb.shape[1]),
              flush=True)
        out[split_dir.name] = (emb, labels, names, texts)

    train = out["20news-bydate-train"]
    test = out["20news-bydate-test"]
    train_hash, test_hash = text_digest(train[3]), text_digest(test[3])
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    # Open both paths exclusively before writing; interrupted runs may leave
    # partial files and must be retried with new output paths.
    from contextlib import ExitStack
    with ExitStack() as stack:
        output = stack.enter_context(args.output.open("xb"))
        text_output = (stack.enter_context(args.texts_output.open("xb"))
                       if args.texts_output is not None else None)
        np.savez_compressed(
            output,
            train_emb=train[0], test_emb=test[0],
            train_labels=train[1], test_labels=test[1],
            train_names=train[2], test_names=test[2],
            train_text_sha256=train_hash, test_text_sha256=test_hash,
        )
        if text_output is not None:
            np.savez_compressed(text_output, train_texts=train[3], test_texts=test[3],
                                train_text_sha256=train_hash, test_text_sha256=test_hash)
    print("saved, total " + str(round(time.perf_counter() - t00, 1)) + "s")


if __name__ == "__main__":
    main()
