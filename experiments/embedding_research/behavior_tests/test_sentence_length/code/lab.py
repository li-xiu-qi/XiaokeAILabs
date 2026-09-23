# -*- coding: utf-8 -*-
"""核心引擎：加载一次模型，批量编码全部条件，算余弦相似度。

与旧 notebook 的关键差异：
  - 模型只加载一次（旧的每次 get_embeddings 都重新 new SentenceTransformer）
  - 全部文本一次性 encode
  - transformers 5.16.1 与 jina-v3 remote code 的兼容问题在这层 patch
  - 余弦相似度对向量缩放免疫，统一用 raw embedding 算，避开模型默认归一化差异
"""

import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

# 条件 -> (第一侧槽位, 第二侧槽位)。单侧条件出 a/b 两变体，分析时取平均，
# 避免「永远只加同一侧」造成的两侧不对称偏差。
COND_SLOTS: Dict[str, Tuple[str, str]] = {
    "base": ("a", "b"),
    "rel_one_a": ("rel", "b"),
    "rel_one_b": ("a", "rel"),
    "irr_one_a": ("irr", "b"),
    "irr_one_b": ("a", "irr"),
    "rel_both": ("rel", "rel"),
    "irr_both": ("irr", "irr"),
}


def patch_transformers() -> None:
    """transformers 5.16.1 在 _move_missing_keys_from_meta_to_device 里引用
    self.all_tied_weights_keys.keys()，该属性已从实例上移除，jina-v3 的 remote
    code（XLMRobertaLoRA）加载时必触发 AttributeError。挂空类属性 + 包一层
    try/except，只兜这一个错误，其他路径不变。"""
    from transformers import modeling_utils

    PTM = modeling_utils.PreTrainedModel
    PTM.all_tied_weights_keys = {}
    orig = PTM._move_missing_keys_from_meta_to_device

    def safe(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except AttributeError:
            return None

    PTM._move_missing_keys_from_meta_to_device = safe


def load_model(hf_id: str):
    patch_transformers()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(hf_id, trust_remote_code=True)


def encode(model, texts: Sequence[str]) -> np.ndarray:
    return np.asarray(
        model.encode(list(texts), normalize_embeddings=False, show_progress_bar=False),
        dtype=np.float64,
    )


def cosine_pairs(emb: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """按行下标配对算余弦相似度。"""
    va, vb = emb[ia], emb[ib]
    denom = np.clip(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12, None)
    return np.sum(va * vb, axis=1) / denom


def _side(text_base: str, text_rel: str, text_irr: str) -> Dict[str, str]:
    """一侧的槽位表：a/b 为基准，rel/irr 为加长版。"""
    return {"a": text_base, "b": text_base, "rel": text_rel, "irr": text_irr}


def _sides(t, nz_a, nz_b, ne_a, ne_b):
    """三种语言对下，每侧（第一 / 第二）的槽位表。两侧噪声句取不同池。"""
    zh_first = _side(t["a"], t["a"] + t["a_rel"], t["a"] + nz_a)
    zh_second = _side(t["b"], t["b"] + t["b_rel"], t["b"] + nz_b)
    en_first = _side(t["en_a"], t["en_a"] + " " + t["en_a_rel"], t["en_a"] + " " + ne_a)
    en_second = _side(t["en_b"], t["en_b"] + " " + t["en_b_rel"], t["en_b"] + " " + ne_b)
    return {
        "zh-zh": (zh_first, zh_second),
        "en-en": (en_first, en_second),
        "zh-en": (zh_first, en_second),
    }


def build_corpus() -> List[dict]:
    """铺开全部 (theme, lang_pair, condition) 的文本对。"""
    from themes import THEMES, noise_a_zh, noise_a_en, noise_b_zh, noise_b_en

    rows = []
    for idx, t in enumerate(THEMES):
        nz_a, nz_b = noise_a_zh(idx), noise_b_zh(idx)
        ne_a, ne_b = noise_a_en(idx), noise_b_en(idx)
        for lp, (first, second) in _sides(t, nz_a, nz_b, ne_a, ne_b).items():
            for cond, (sa, sb) in COND_SLOTS.items():
                rows.append({
                    "theme_id": t["id"],
                    "lang_pair": lp,
                    "condition": cond,
                    "text_a": first[sa],
                    "text_b": second[sb],
                })
    return rows
