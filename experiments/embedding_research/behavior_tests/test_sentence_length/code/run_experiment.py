# -*- coding: utf-8 -*-
"""入口：跑全量实验，落盘 raw_sims.csv。

用法（DGX Spark）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiment.py

单模型失败不拖死全局：记录到 failed 列表，末尾集中报告。
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import lab  # noqa: E402


def main() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    corpus = lab.build_corpus()
    print(f"corpus rows: {len(corpus)}", flush=True)

    text_a = [r["text_a"] for r in corpus]
    text_b = [r["text_b"] for r in corpus]

    all_results, failed = [], []
    for spec in config.MODELS:
        t0 = time.time()
        try:
            model = lab.load_model(spec["hf_id"])
            emb_a = lab.encode(model, text_a)
            emb_b = lab.encode(model, text_b)
            denom = np.clip(
                np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1), 1e-12, None
            )
            sims = np.sum(emb_a * emb_b, axis=1) / denom
            for row, s in zip(corpus, sims):
                all_results.append({
                    "model": spec["name"],
                    "lang": spec["lang"],
                    "lang_pair": row["lang_pair"],
                    "theme_id": row["theme_id"],
                    "condition": row["condition"],
                    "sim": float(s),
                })
            print(f"{spec['name']:14s} done in {time.time() - t0:.1f}s", flush=True)
            del model, emb_a, emb_b
        except Exception as exc:
            failed.append(spec["name"])
            print(f"{spec['name']:14s} FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        out = os.path.join(config.RESULTS_DIR, "raw_sims.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"saved -> {out}  ({len(df)} rows)")
    if failed:
        print(f"failed models: {failed}")


if __name__ == "__main__":
    main()
