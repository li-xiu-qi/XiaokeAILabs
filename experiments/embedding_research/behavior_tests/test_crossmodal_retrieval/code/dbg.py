import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_patch
cls = mm_patch.apply_all()
print("patch 返回:", cls.__name__)
from transformers import AutoProcessor
import transformers.models.auto.processing_auto as pa
print("AutoProcessor.from_pretrained 是否被换:", AutoProcessor.from_pretrained.__func__ is not pa.AutoProcessor.from_pretrained.__func__)
print("patched 函数名:", AutoProcessor.from_pretrained.__func__.__name__)
p = AutoProcessor.from_pretrained("jinaai/jina-embeddings-v4", use_fast=True)
print("返回类型:", type(p).__name__)
print("有 tokenizer 属性:", hasattr(p, "tokenizer"), "值:", type(getattr(p, "tokenizer", None)).__name__)
print("image_token_id:", getattr(p, "image_token_id", "MISSING"))
