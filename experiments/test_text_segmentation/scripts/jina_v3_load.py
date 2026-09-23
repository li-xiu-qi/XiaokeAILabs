# -*- coding: utf-8 -*-
"""加载 jina-embeddings-v3 并验证可用性。

v3 的 SentenceTransformer custom_st.py 内部调用 AutoModel 加载 flash 实现
（XLMRobertaLoRA），该自定义类未定义 all_tied_weights_keys，
transformers 4.5x 的 _move_missing_keys_from_meta_to_device 会访问它而崩。
补丁栈复用 jina_v4_diag2.py 的全部补桩，外加 all_tied_weights_keys 回退。
"""
import sys, os, types
sys.path.insert(0, "scripts")

# --- 补桩 1：transformers.onnx 假模块 ---
try:
    import transformers.onnx
except ModuleNotFoundError:
    _fake = types.ModuleType("transformers.onnx")
    class OnnxConfig: pass
    _fake.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = _fake

# --- 补桩 2：find_pruneable_heads_and_indices ---
import transformers.pytorch_utils as _putils
if not hasattr(_putils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        return (set(heads) - set(already_pruned_heads), list(range(n_heads * head_size)))
    _putils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

# --- 补桩 3：PretrainedConfig.__getattr__ ---
import transformers.configuration_utils as _cu
def _patched_getattr(self, key):
    if key == "add_cross_attention": return False
    if key == "chunk_size_feed_forward": return 0
    if key == "is_decoder": return False
    if key == "cross_attention_hidden_size": return None
    raise AttributeError(key)
_cu.PretrainedConfig.__getattr__ = _patched_getattr

# --- 补桩 4：ROPE_INIT_FUNCTIONS["default"] ---
import transformers.modeling_rope_utils as _rope
if "default" not in _rope.ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        import torch
        base = getattr(config, "rope_theta", 10000.0)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, 1.0
    _rope.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

# --- 补桩 5：tied weights（_tied_weights_keys list→dict）---
import transformers.modeling_utils as _mu
_orig_get_tied = _mu.PreTrainedModel.get_expanded_tied_weights_keys
def _patched_get_tied(self, all_submodels=False):
    for cls in type(self).__mro__:
        if hasattr(cls, "_tied_weights_keys") and isinstance(cls._tied_weights_keys, list):
            cls._tied_weights_keys = {k: k for k in cls._tied_weights_keys}
            break
    return _orig_get_tied(self, all_submodels)
_mu.PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_tied

# --- 补桩 6：all_tied_weights_keys 回退（v3 专属，XLMRobertaLoRA 未定义）---
_orig_missing = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
def _patched_missing(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        tw = getattr(type(self), "_tied_weights_keys", [])
        if isinstance(tw, list):
            tw = {k: k for k in tw}
        self.all_tied_weights_keys = tw if isinstance(tw, dict) else {}
    return _orig_missing(self, *args, **kwargs)
_mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_missing

# --- 加载 v3 ---
from sentence_transformers import SentenceTransformer
import torch
m = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
m = m.to("cuda" if torch.cuda.is_available() else "cpu")
print("V3 LOADED dim=", m.get_sentence_embedding_dimension(), file=sys.stderr)

# --- 验证编码（task label + prompt_name）---
for task, pname in [("retrieval.passage", "passage"), ("retrieval.query", "query")]:
    e = m.encode(["The quick brown fox jumps over the lazy dog.",
                  "一只敏捷的棕色狐狸跳过了懒狗。"],
                 task=task, prompt_name=pname)
    print(f"encode OK task={task} prompt_name={pname} shape={e.shape}", file=sys.stderr)

# passage/query 相似度 sanity check
ep = m.encode(["A study on machine learning models.",
               "The cat sat on the mat."], task="retrieval.passage")
eq = m.encode(["papers about neural networks"], task="retrieval.query")
import numpy as np
ep = ep / (np.linalg.norm(ep, axis=1, keepdims=True) + 1e-9)
eq = eq / (np.linalg.norm(eq, axis=1, keepdims=True) + 1e-9)
sims = eq @ ep.T
print("sim ML-vs-paper-query:", float(sims[0, 0]), " cat-vs-paper-query:", float(sims[0, 1]), file=sys.stderr)
print("V3 READY")
