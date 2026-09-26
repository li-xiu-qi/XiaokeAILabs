# -*- coding: utf-8 -*-
"""跨模态编码引擎。

封装 jina-embeddings-v4 的加载与编码，处理两件事。

一是 transformers 5.16.1 的兼容补丁。jina-v4 的 remote code 基于 Qwen2.5-VL，
其 processor 的 __init__ 直接访问 tokenizer.convert_tokens_to_ids，而新版
ProcessorMixin 不把 tokenizer 传进去，必崩。补丁见 mm_patch.py。

二是编码口径统一。文本与图像走同一个 task adapter，统一用 retrieval。余弦对
向量缩放免疫，因此一律用 raw embedding（不归一化）算余弦，避开模型默认归一化
差异带来的不可比。

用法：
    lab = CrossModalLab()
    te = lab.encode_text(["a dog on grass"])
    ie = lab.encode_image(["/path/to/img.jpg"])
    lab.close()
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_patch  # noqa: E402

MODEL_ID = "jinaai/jina-embeddings-v4"

# 只有 jina-v4 需要兼容补丁。它的 remote code 基于 Qwen2.5-VL，processor 的
# __init__ 直接访问 tokenizer.convert_tokens_to_ids，新版 ProcessorMixin 不传
# tokenizer 必崩。补丁的 fallback 是「拿 jina 的 processor 类手动构造」，对标准
# CLIP 系模型（如 BAAI/BGE-VL-base，auto_map 指向 CLIPModel）会直接构造失败。
# 因此按模型判断，非 jina-v4 一律走原生加载路径。
PATCHED_MODELS = ("jina-embeddings-v4",)


class CrossModalLab:
    def __init__(self, model_id: str = MODEL_ID, task: str = "retrieval"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.model_id = model_id
        self.task = task
        if any(k in model_id for k in PATCHED_MODELS):
            mm_patch.apply_all(model_id)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id, trust_remote_code=True)
        # sentence-transformers 6.x 对自定义 remote code 读不到维度，返回 None。
        # 不硬编码，首次编码后用真实维度回填。
        self.dim = None
        # 6.x 新增模态白名单，取自第一个模块的 modalities，jina 的 remote code
        # 没声明 image，会被 preprocess 拦下来。custom_st.tokenize 本身接受
        # PIL.Image，补上声明即可，不改权重。
        first = self.model[0]
        mods = list(getattr(first, "modalities", ["text"]))
        if "image" not in mods:
            mods.append("image")
            first.modalities = mods

    def _fix_dim(self, arr: np.ndarray) -> None:
        if self.dim is None and arr.size:
            self.dim = int(arr.shape[-1])

    def encode_text(self, texts, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float64)
        out = self.model.encode(list(texts), task=self.task,
                                normalize_embeddings=False,
                                batch_size=batch_size, show_progress_bar=False)
        arr = np.asarray(out, dtype=np.float64)
        self._fix_dim(arr)
        return arr

    def encode_image(self, paths, batch_size: int = 8) -> np.ndarray:
        """按路径编码图像。路径不存在或读不动的位置填 NaN 行，由调用方过滤。"""
        from PIL import Image
        good_idx, imgs = [], []
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
                good_idx.append(i)
            except Exception:
                continue
        if self.dim is None:
            probe = self.model.encode(["probe"], task=self.task,
                                      normalize_embeddings=False)
            self._fix_dim(np.asarray(probe, dtype=np.float64))
        emb = np.full((len(paths), self.dim), np.nan, dtype=np.float64)
        if imgs:
            out = self.model.encode(imgs, task=self.task,
                                    normalize_embeddings=False,
                                    batch_size=batch_size, show_progress_bar=False)
            emb[good_idx] = np.asarray(out, dtype=np.float64)
        return emb

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """逐行配对算余弦，自动跳过含 NaN 的行（返回 NaN）。"""
        va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        na = np.linalg.norm(va, axis=1)
        nb = np.linalg.norm(vb, axis=1)
        denom = np.clip(na * nb, 1e-12, None)
        return np.sum(va * vb, axis=1) / denom

    def close(self):
        if getattr(self, "model", None) is not None:
            del self.model


def self_check(root: str, model_id: str = MODEL_ID):
    """最小自检：程序生成两张图，验证图文同空间且相关对高于无关对。

    返回 (通过?, 读数 dict)。读数只供冒烟，不作结论。
    """
    import tempfile
    from PIL import Image, ImageDraw

    tmp = tempfile.mkdtemp(prefix="mmcheck_")

    def make(name, kind, fg, bg):
        im = Image.new("RGB", (224, 224), bg)
        d = ImageDraw.Draw(im)
        if kind == "circle":
            d.ellipse([52, 52, 172, 172], fill=fg)
        else:
            d.rectangle([52, 52, 172, 172], fill=fg)
        p = os.path.join(tmp, name)
        im.save(p)
        return p

    red = make("red_circle.png", "circle", (200, 30, 30), (240, 240, 240))
    blue = make("blue_square.png", "square", (30, 60, 200), (250, 250, 250))

    lab = CrossModalLab(model_id)
    try:
        texts = ["a red circle on a light background",
                 "a blue square on a white background",
                 "unrelated text about tax law"]
        te = lab.encode_text(texts)
        ie = lab.encode_image([red, blue])
        c = CrossModalLab.cosine
        reads = {
            "dim": lab.dim,
            "t_red_vs_i_red": float(c(te[0:1], ie[0:1])[0]),
            "t_red_vs_i_blue": float(c(te[0:1], ie[1:2])[0]),
            "t_blue_vs_i_blue": float(c(te[1:2], ie[1:2])[0]),
            "t_unrel_vs_i_red": float(c(te[2:3], ie[0:1])[0]),
            "i_red_vs_i_blue": float(c(ie[0:1], ie[1:2])[0]),
            "t_red_vs_t_blue": float(c(te[0:1], te[1:2])[0]),
        }
    finally:
        lab.close()
    ok = reads["t_red_vs_i_red"] > reads["t_unrel_vs_i_red"]
    return ok, reads


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    print("模型:", mid)
    ok, reads = self_check(".", mid)
    print("自检", "通过" if ok else "未通过")
    for k, v in reads.items():
        print(f"  {k:20s} {v:+.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    sys.exit(0 if ok else 1)
