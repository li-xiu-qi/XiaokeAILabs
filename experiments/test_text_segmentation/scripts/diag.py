# -*- coding: utf-8 -*-
"""诊断：逐算法检查实现正确性。"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
from corpus_loader import load_corpus
from model_hub import TextEncoder
from algo_bm25_boundary import BM25Splitter
from algo_semantic_breakpoint import SemanticSplitter
from token_map import sentence_token_spans

enc = TextEncoder.get()
docs = load_corpus("corpus", n_docs=50, cache_path="corpus/wiki_eval_300.jsonl")

print("=" * 72)
print("检查1：BM25 是否有尺寸上限约束")
print("=" * 72)
import inspect
sig = inspect.signature(BM25Splitter.__init__)
print("BM25Splitter.__init__ 参数:", list(sig.parameters.keys()))
print("→ 没有 max_tokens 参数")

sp = BM25Splitter(percentile=25, min_tokens=64)
over = []
for d in docs:
    seg = sp.split(d.text)
    sizes = [b - a for a, b in seg.chunk_token_spans()]
    if sizes:
        m = max(sizes)
        if m > 512:
            over.append((d.doc_id, m, len(sizes)))
print(f"设定意图 max_tokens=512，实际超过的文档: {len(over)}/{len(docs)}")
print(f"其中最大块: {max((o[1] for o in over), default=0)} token")

print()
print("=" * 72)
print("检查2：semantic 相似度信号在真实文本上的分布")
print("=" * 72)
d = docs[0]
spans = sentence_token_spans(d.text, enc.tok)
sents = [s for s, _, _ in spans]
embs = enc.encode(sents)
sims = np.array([float(embs[i] @ embs[i + 1]) for i in range(len(embs) - 1)])
print(f"文档《{d.title}》{len(sents)}句")
print(f"相邻句相似度: min={sims.min():.3f} p25={np.percentile(sims,25):.3f} "
      f"中位={np.median(sims):.3f} p75={np.percentile(sims,75):.3f} max={sims.max():.3f}")
print()
print("关键相邻对（块1粘连处）:")
for i in range(min(3, len(sims))):
    print(f"  句{i+1}->句{i+2}: sim={sims[i]:.3f}")
    print(f"     「{sents[i][:26]}」->「{sents[i+1][:26]}")
print()
print(f"阈值0.60时低于阈值的相邻对: {int((sims<0.60).sum())}/{len(sims)}")
print(f"阈值0.50时低于阈值的相邻对: {int((sims<0.50).sum())}/{len(sims)}")
print(f"阈值0.30时低于阈值的相邻对: {int((sims<0.30).sum())}/{len(sims)}")

print()
print("=" * 72)
print("检查3：semantic 的 min_tokens 约束是否压制了有效切分")
print("=" * 72)
for mt in [0, 64, 128]:
    sp = SemanticSplitter(threshold=0.60, max_tokens=512, min_tokens=mt)
    nb = sum(len(sp.split(x.text).boundaries) for x in docs[:20])
    print(f"  min_tokens={mt:3d}: 20篇共切 {nb} 刀")
