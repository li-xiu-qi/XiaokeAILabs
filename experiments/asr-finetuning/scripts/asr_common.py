# -*- coding: utf-8 -*-
"""asr_common.py — ASR 微调实验共享模块。

职责：
  - 运行契约：输出写 XL_OUT_DIR、数据读 XL_DATA_DIR、运行标识取 XL_RUN_ID；
    未显式指定时回退到实验目录下被 .gitignore 忽略的 .work/；设置
    XL_REQUIRE_EXTERNAL_ENV=1 时禁止本地回退并报错退出。
  - 模型与数据集引用：ASR_MODEL / ASR_MODEL_REVISION、ASR_DATASET /
    ASR_DATASET_REVISION。默认要求固定 revision（ASR_REQUIRE_REVISION=1），
    确认可复现性影响后可显式关闭。
  - 文本归一化、WER 汇总、manifest 读写、训练/测试集隔离与输出目录保护。

本模块不导入 torch / datasets / transformers，保证无重依赖的静态检查与
单元测试可以在任何 Python 3.9+ 环境运行。
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_SAMPLE_RATE = 16000  # Whisper 特征提取器要求 16 kHz 单声道

DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_DATASET = "openslr/librispeech_asr"
DEFAULT_DATASET_CONFIG = "clean"

_TRUE = {"1", "true", "yes", "on"}


def _flag(value):
    """把环境变量字符串解析为布尔值。"""
    return value is not None and str(value).strip().lower() in _TRUE


def require_external_env():
    """XL_REQUIRE_EXTERNAL_ENV=1 时禁止 .work/ 本地回退。"""
    return _flag(os.environ.get("XL_REQUIRE_EXTERNAL_ENV"))


def require_revision():
    """是否强制固定 model / dataset revision，默认开。"""
    return _flag(os.environ.get("ASR_REQUIRE_REVISION", "1"))


def _resolve_env(name, default_factory, msg_hint):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if require_external_env():
        sys.exit(f"{name} 未设置（XL_REQUIRE_EXTERNAL_ENV=1 禁止本地回退）：{msg_hint}")
    return default_factory()


def run_id():
    """本次运行的标识（XL_RUN_ID）。"""
    return _resolve_env("XL_RUN_ID", lambda: "local", msg_hint="请显式设置 XL_RUN_ID")


def out_dir():
    """本次运行的输出目录（XL_OUT_DIR），不存在则创建。"""
    value = _resolve_env(
        "XL_OUT_DIR", lambda: str(ROOT / ".work" / "out" / run_id()), msg_hint="请显式设置 XL_OUT_DIR"
    )
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir():
    """数据目录（XL_DATA_DIR），不存在则创建。"""
    value = _resolve_env(
        "XL_DATA_DIR", lambda: str(ROOT / ".work" / "data"), msg_hint="请显式设置 XL_DATA_DIR"
    )
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_ref():
    """返回 (模型 id 或本地路径, revision 或 None)。"""
    model = _resolve_env(
        "ASR_MODEL", lambda: DEFAULT_MODEL, msg_hint="请显式设置 ASR_MODEL（模型 id 或本地路径）"
    )
    revision = os.environ.get("ASR_MODEL_REVISION", "").strip() or None
    if revision is None and require_revision():
        sys.exit(
            "ASR_MODEL_REVISION 未设置：默认要求固定模型 revision；"
            "确认可复现性影响后可设置 ASR_REQUIRE_REVISION=0 关闭"
        )
    return model, revision


def dataset_ref():
    """返回 (数据集 id 或本地路径, revision 或 None)。"""
    dataset = _resolve_env(
        "ASR_DATASET", lambda: DEFAULT_DATASET, msg_hint="请显式设置 ASR_DATASET（数据集 id 或本地路径）"
    )
    revision = os.environ.get("ASR_DATASET_REVISION", "").strip() or None
    if revision is None and require_revision():
        sys.exit(
            "ASR_DATASET_REVISION 未设置：默认要求固定数据集 revision；"
            "确认可复现性影响后可设置 ASR_REQUIRE_REVISION=0 关闭"
        )
    return dataset, revision


_NON_KEEP = re.compile(r"[^a-z0-9'\- ]+")


def normalize_text(text):
    """Whisper 风格的英文归一化（简化版，与评测两侧一致使用）。

    NFKC -> 弯引号转直引号 -> 小写 -> 仅保留英文字母、数字、撇号、连字符和
    空格 -> 折叠空白。完整的 Whisper EnglishTextNormalizer 还会展开数字读法
    与缩略语，本实验不做；参考与假设文本走同一函数，保证口径一致。
    """
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.lower()
    t = _NON_KEEP.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def wer_percent(references, hypotheses):
    """词错误率 WER（百分比）。references 与 hypotheses 需等长，依赖 jiwer。"""
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError("计算 WER 需要 jiwer：pip install jiwer") from exc
    if len(references) != len(hypotheses):
        raise ValueError("references 与 hypotheses 长度不一致")
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    total_words = sum(len(r.split()) for r in refs)
    if total_words == 0:
        raise ValueError("归一化后参考文本为空，无法计算 WER；请检查 manifest 的 text 字段")
    return jiwer.wer(refs, hyps) * 100.0


def read_manifest(path):
    """读取 JSONL manifest，返回 dict 列表；空文件或非法 JSON 报错。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} 不是 JSON 对象")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def manifest_ids(path):
    """manifest 中的 id 集合（用于训练/测试隔离检查）。"""
    ids = set()
    for row in read_manifest(path):
        if "id" not in row:
            raise ValueError(f"manifest 行缺少 id 字段：{path}")
        ids.add(str(row["id"]))
    return ids


def check_split_isolation(train_path, test_path):
    """校验训练集与测试集无重复 id，重叠即报错。"""
    train_ids = manifest_ids(train_path)
    test_ids = manifest_ids(test_path)
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError(
            f"训练集与测试集存在 {len(overlap)} 条重复 id（前 5：{overlap[:5]}），请重新划分"
        )


def filter_by_duration(rows, min_duration, max_duration):
    """按时长过滤 manifest 行，返回 (保留行, 丢弃数)。"""
    kept = []
    dropped = 0
    for row in rows:
        try:
            duration = float(row.get("duration"))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (min_duration <= duration <= max_duration):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def ensure_fresh_dir(path):
    """拒绝已存在的非空目录或非目录路径，防止覆盖已有产物。"""
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"输出路径已存在且不是空目录：{path}；请换用不存在的新路径")
    if path.is_dir() and any(path.iterdir()):
        raise ValueError(f"输出目录非空：{path}；为避免覆盖已有产物，请换用新目录")
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_paths_outside(base, paths):
    """确保 manifest 等输入路径不在输出目录内，避免训练写坏输入。"""
    base = Path(base).resolve()
    for item in paths:
        if item is None:
            continue
        resolved = Path(item).resolve()
        if resolved == base or base in resolved.parents:
            raise ValueError(f"输入路径位于输出目录内：{resolved}（输出 {base}）；请分开存放")
