#!/usr/bin/env python3
"""Bespoke Nimble 部署到 GPU 机器。下载 pinned 基座 + LoRA 适配器，merge 成完整权重。

环境：torch 2.14.0+cu130，transformers 5.16.1，peft 0.20.0
README 要求的 transformers 5.17.0 / peft 0.21.0 比现有高一个小版本，先用现有的试。
merge 在 CPU 上跑，bf16，需要同时放基座与合并后权重（各约 18 GB）。
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = "bespokelabs/Bespoke-Nimble-9B"
NIMBLE_DIR = Path.home() / "nimble"

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

log("host: %s" % os.uname().nodename)

from huggingface_hub import snapshot_download
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

log("torch %s  transformers %s  peft %s" % (
    torch.__version__,
    __import__("transformers").__version__,
    __import__("peft").__version__))

# ---- 1. 下载适配器仓库 ----
t0 = time.time()
log("下载适配器仓库 %s ..." % REPO)
snapshot = Path(snapshot_download(REPO))
log("适配器就绪 %.1f s -> %s" % (time.time() - t0, snapshot))

# ---- 2. 校验 prompt 契约 ----
contract_file = snapshot / "schema_config.json"
contract = json.loads(contract_file.read_text()) if contract_file.exists() else {}
prompt_hash = hashlib.sha256(
    (NIMBLE_DIR / "nimble" / "scoring" / "parallel_schema.py").read_bytes()).hexdigest()
if contract:
    log("契约 task=%r  prompt_code_sha256=%s" % (contract.get("task"), contract.get("prompt_code_sha256")))
    assert contract["task"] == "schema_candidate_classification_v1", "task 名不符"
    assert contract["prompt_code_sha256"] == prompt_hash, "prompt 契约不符， scorer 代码被改过"
    log("prompt 契约校验通过")
else:
    log("警告：没有 schema_config.json，跳过契约校验")

base_repo = contract.get("model", "Qwen/Qwen3.5-9B")
base_rev = contract.get("revision")
log("pinned 基座: %s  revision=%s" % (base_repo, base_rev))

# ---- 3. 下载基座权重（大头，约 18 GB）----
t0 = time.time()
log("下载基座权重（约 18 GB，走 hf-mirror）...")
snapshot_download(base_repo, revision=base_rev)
log("基座下载完成 %.1f s" % (time.time() - t0))

# ---- 4. merge ----
log("加载基座到 CPU（bf16）...")
t0 = time.time()
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_repo, revision=base_rev, dtype=torch.bfloat16, device_map="cpu")
log("基座加载 %.1f s" % (time.time() - t0))

log("挂适配器并 merge ...")
t0 = time.time()
adapter = PeftModel.from_pretrained(base, snapshot)
merged = adapter.merge_and_unload(safe_merge=True)
log("merge 完成 %.1f s" % (time.time() - t0))

model_path = Path.home() / ".cache" / "models" / ("nimble-9b-" + snapshot.name)
model_path.mkdir(parents=True, exist_ok=True)
log("保存合并后权重到 %s ..." % model_path)
merged.save_pretrained(model_path)
AutoTokenizer.from_pretrained(snapshot).save_pretrained(model_path)
log("保存完成")

# ---- 5. 落一份配置，供推理脚本用 ----
cfg = {
    "model_path": str(model_path.resolve()),
    "model_id": REPO,
    "base_model": base_repo,
    "base_revision": base_rev,
    "adapter_commit": snapshot.name,
    "contract": contract,
    "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
    "peft": __import__("peft").__version__,
}
Path.home().joinpath("nimble-config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
log("全部完成。model_path=%s" % cfg["model_path"])
