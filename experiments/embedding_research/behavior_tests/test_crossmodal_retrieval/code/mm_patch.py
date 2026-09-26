# -*- coding: utf-8 -*-
"""多模态 embedding 的 transformers 5.x 兼容补丁集。

jina-embeddings-v4 的 remote code 基于 Qwen2.5-VL，其 processor 的 __init__
直接访问 tokenizer.convert_tokens_to_ids。transformers 5.16.1 的
ProcessorMixin.from_pretrained 不把 tokenizer 传进去（preprocessor_config.json
的 tokenizer_class 是 None，attributes 推断不到），tokenizer 为 None，必崩。

直接 patch remote code 的类无效，transformers 的动态模块机制会另建一个类对象。
有效做法是 patch AutoProcessor.from_pretrained：原生路径崩了就改走手动构造，
显式传入 tokenizer、image_processor、video_processor。
"""
import glob
import importlib.util
import os
import sys
import types

MODELS_DIR = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")


def _load_remote_package(model_id: str):
    """按 model_id 精确定位 remote code 目录。

    目录结构是 <MODELS_DIR>/<org>/<模型名连字符转下划线>/<hash>/。
    原先靠 mtime 取最新的 fallback，在缓存里只有一个 remote code 模型时能跑，
    第二个模型一下载就会被抢走（2026-09-26 实测：下载 BGE-VL 后 jina-v4 的
    patch 指向了 BGE-VL 的目录，报找不到 modeling_jina_embeddings_v4.py）。
    改为精确匹配，匹配不到直接报错，不猜。
    """
    # transformers 的动态模块目录把连字符写成 _hyphen_，不是下划线。
    # 例：jina-embeddings-v4 -> jina_hyphen_embeddings_hyphen_v4
    want = model_id.split("/")[-1].replace("-", "_hyphen_")
    cands = [d for d in glob.glob(os.path.join(MODELS_DIR, "*", "*", "*"))
             if os.path.basename(os.path.dirname(d)) == want
             and glob.glob(os.path.join(d, "modeling_*.py"))]
    if not cands:
        raise FileNotFoundError(
            f"未找到 {model_id} 的 remote code 目录（期望目录名 {want!r}）。"
            f"可能是模型没加载过，或缓存被清。")
    target = max(cands, key=os.path.getmtime)
    pkg_name = "mm_remote_" + os.path.basename(os.path.dirname(target)).replace("-", "_")
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [target]
        sys.modules[pkg_name] = pkg
    return pkg_name, target


def patch_jina_v4(model_id: str = "jinaai/jina-embeddings-v4"):
    pkg, d = _load_remote_package(model_id)
    mod_name = f"{pkg}.modeling_jina_embeddings_v4"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(d, "modeling_jina_embeddings_v4.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = sys.modules[mod_name]

    proc_cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.endswith("Processor") and hasattr(obj, "__init__"):
            proc_cls = obj
            break
    if proc_cls is None:
        raise AttributeError("remote code 里找不到 Processor 类")
    return proc_cls


def patch_auto_processor(model_id: str, proc_cls):
    from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor
    from transformers.models.auto import processing_auto

    orig = processing_auto.AutoProcessor.from_pretrained.__func__

    @classmethod
    def patched(cls, pretrained, **kwargs):
        try:
            proc = orig(cls, pretrained, **kwargs)
            if getattr(proc, "tokenizer", None) is not None:
                return proc
        except Exception:
            pass
        tok = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, use_fast=True)
        img = AutoImageProcessor.from_pretrained(pretrained, trust_remote_code=True)
        extra = {}
        try:
            from transformers import AutoVideoProcessor
            extra["video_processor"] = AutoVideoProcessor.from_pretrained(
                pretrained, trust_remote_code=True)
        except Exception:
            extra["video_processor"] = None
        proc = proc_cls(image_processor=img, tokenizer=tok, **extra)
        # ProcessorMixin 5.x 的位置参数顺序与 remote code 的按位置传递不一致，
        # 传入的 tokenizer 会落到别的属性上，self.tokenizer 仍是 None。这里按名字补挂。
        for attr, val in (("tokenizer", tok), ("image_processor", img)):
            if getattr(proc, attr, None) is None:
                setattr(proc, attr, val)
        if getattr(proc, "video_processor", "missing") is None:
            proc.video_processor = extra.get("video_processor")
        return proc

    processing_auto.AutoProcessor.from_pretrained = patched
    AutoProcessor.from_pretrained = patched


def patch_transformers_tied_keys():
    from transformers import modeling_utils
    PTM = modeling_utils.PreTrainedModel
    if not hasattr(PTM, "all_tied_weights_keys"):
        PTM.all_tied_weights_keys = {}
    if getattr(PTM, "_mm_patched", False):
        return
    orig = PTM._move_missing_keys_from_meta_to_device

    def safe(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except AttributeError:
            return None

    PTM._move_missing_keys_from_meta_to_device = safe
    PTM._mm_patched = True


def apply_all(model_id: str = "jinaai/jina-embeddings-v4"):
    patch_transformers_tied_keys()
    proc_cls = patch_jina_v4(model_id)
    patch_auto_processor(model_id, proc_cls)
    return proc_cls
