#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_tokenizer_domains.py — 多分词器多域评估脚本（域偏差与词表规模实验）。

支持同时评估多个 tokenizer.json，计算：
  - 基础压缩率: utf8_bytes, chars, tokens, bytes_per_token, chars_per_token
  - 域膨胀: token_inflation_vs_climbmix_val（相对 climbmix_val 的 token 比例）
  - 一致性: roundtrip_ok（编解码无损）
  - 词表利用率: vocab_utilization（实际使用的 token 占词表比例）
  - token_bytes.pt 公式审计: len(id_to_token.encode('utf-8')) vs ByteLevel 解码真实字节数

用法：
    # 单分词器（域切片来自 fixtures/，输出到 $XL_OUT_DIR/tokenizer/）：
    python scripts/eval_tokenizer_domains.py \
        --tokenizer-path "$XL_OUT_DIR/tokenizer/tokenizer.json" \
        --tokenizer-name current --out "$XL_OUT_DIR/tokenizer/domain_compression_eval.json"

    # 多分词器：
    python scripts/eval_tokenizer_domains.py \
        --tokenizer-path tok_a.json --tokenizer-name a \
        --tokenizer-path tok_b.json --tokenizer-name b

    # 追加真实语料域（parquet 或 txt，路径经参数注入，不进脚本）：
    python scripts/eval_tokenizer_domains.py ... \
        --extra-domain climbmix_val=/path/to/shard_06542.parquet \
        --extra-domain smoltalk=/path/to/train-00000-of-00004.parquet \
        --audit-source /path/to/shard_06542.parquet --baseline-domain climbmix_val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# 域切片固定来自实验目录下的 fixtures/（可用 --fixtures-dir 覆盖）
_SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
DEFAULT_MAX_CHARS = 100_000
EOS = "<|endoftext|>"
PAD = "<|pad|>"

# 默认域：fixtures/ 下的五个固定域切片（跨 tokenizer 口径一致）
DOMAINS = [
    ("python_code", FIXTURES_DIR / "python_code.txt"),
    ("latex_math", FIXTURES_DIR / "latex_math.txt"),
    ("chinese_tech", FIXTURES_DIR / "chinese_tech.txt"),
    ("mixed_zh_en_code", FIXTURES_DIR / "mixed_zh_en_code.txt"),
    ("digits_arithmetic", FIXTURES_DIR / "digits_arithmetic.txt"),
]


def build_domains(fixtures_dir: Path | None = None,
                  extra_domains: list[str] | None = None) -> list[tuple[str, Path]]:
    """组装评估域列表：fixtures 五域 + --extra-domain NAME=PATH 注入的真实语料域。"""
    base = Path(fixtures_dir) if fixtures_dir else FIXTURES_DIR
    domains = [
        ("python_code", base / "python_code.txt"),
        ("latex_math", base / "latex_math.txt"),
        ("chinese_tech", base / "chinese_tech.txt"),
        ("mixed_zh_en_code", base / "mixed_zh_en_code.txt"),
        ("digits_arithmetic", base / "digits_arithmetic.txt"),
    ]
    for item in extra_domains or []:
        if "=" not in item:
            raise ValueError(f"--extra-domain 需要 NAME=PATH 形式，得到: {item}")
        name, _, path = item.partition("=")
        domains.append((name.strip(), Path(path.strip())))
    return domains


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _chars(text: str) -> int:
    return len(text)


def _tokens(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text).ids)


def _roundtrip_ok(tok: Tokenizer, text: str) -> bool:
    return tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text


def _vocab_utilization(tok: Tokenizer, text: str) -> float:
    """返回文本中实际用到的 token id 占 vocab_size 的比例。"""
    ids = tok.encode(text).ids
    return len(set(ids)) / tok.get_vocab_size()


def _load_parquet_text(path: str, max_chars: int) -> str:
    """流式读取 parquet 的 text 列，按 doc_cap 截断并限制 max_chars。

    支持两种常见 schema：
    1. 扁平 text 列（如 climbmix）
    2. 嵌套 messages 列（如 smoltalk，提取 content 拼接）
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("读取 Parquet 语料需要 pyarrow 库，请先 pip install pyarrow。")
    pf = pq.ParquetFile(path)
    chunks: list[str] = []
    total = 0
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        col_names = rg.column_names
        if "text" in col_names:
            texts = rg.column("text").to_pylist()
            for doc in texts:
                if not doc or not doc.strip():
                    continue
                if len(doc) > 10_000:
                    doc = doc[:10_000]
                chunks.append(doc)
                total += len(doc)
                if total >= max_chars:
                    return "".join(chunks)
        elif "messages" in col_names:
            # smoltalk 等对话数据集：提取每条消息的 content 拼接
            rows = rg.column("messages").to_pylist()
            for msgs in rows:
                if not msgs:
                    continue
                parts: list[str] = []
                for m in msgs:
                    if isinstance(m, dict):
                        c = m.get("content", "")
                        if c:
                            parts.append(str(c))
                doc = "\n".join(parts)
                if not doc.strip():
                    continue
                if len(doc) > 10_000:
                    doc = doc[:10_000]
                chunks.append(doc)
                total += len(doc)
                if total >= max_chars:
                    return "".join(chunks)
        else:
            # 回退：取第一列
            first = rg.column(col_names[0]).to_pylist()
            for doc in first:
                if not doc or not str(doc).strip():
                    continue
                doc = str(doc)
                if len(doc) > 10_000:
                    doc = doc[:10_000]
                chunks.append(doc)
                total += len(doc)
                if total >= max_chars:
                    return "".join(chunks)
    return "".join(chunks)


def _load_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def _load_domain_text(name: str, path, max_chars: int) -> str:
    if name in ("climbmix_val", "smoltalk"):
        if not Path(path).exists():
            print(f"[warn] {path} 不存在，跳过域 {name}")
            return ""
        return _load_parquet_text(path, max_chars)
    return _load_text(path, max_chars)


def evaluate_domain(tok: Tokenizer, name: str, source, max_chars: int) -> dict:
    """对单个域切片做全面评估，返回指标字典。"""
    text = _load_domain_text(name, source, max_chars)
    if not text:
        return {"name": name, "skipped": True, "reason": "no data"}

    n_chars = _chars(text)
    n_bytes = _utf8_bytes(text)
    enc = tok.encode(text)
    n_tokens = len(enc.ids)
    used_ids = set(enc.ids)

    return {
        "name": name,
        "skipped": False,
        "utf8_bytes": n_bytes,
        "chars": n_chars,
        "tokens": n_tokens,
        "bytes_per_token": round(n_bytes / max(n_tokens, 1), 6),
        "chars_per_token": round(n_chars / max(n_tokens, 1), 6),
        "roundtrip_ok": _roundtrip_ok(tok, text),
        "vocab_utilization": round(len(used_ids) / tok.get_vocab_size(), 6),
        "unique_token_ids": len(used_ids),
    }


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / HuggingFace ByteLevel 预分词器的 256 字节到可打印 Unicode 字符的双射表。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_LEVEL_CHAR_SET: set[str] = set(_gpt2_bytes_to_unicode().values())


def audit_token_bytes(
    tok: Tokenizer,
    token_bytes_pt: str | Path | None,
    reference_text: str,
) -> dict:
    """审计 token_bytes.pt 的字节长度公式：
    对比 len(id_to_token.encode('utf-8')) 与 ByteLevel 还原后的真实原始 UTF-8 字节数。
    同时按真实 token_id 下标估算公式偏差对参考文本 total_bytes 的影响。
    """
    vocab_size = tok.get_vocab_size()
    special_ids = {tok.token_to_id(EOS), tok.token_to_id(PAD)}
    special_ids.discard(None)

    formula_bytes: list[int] = [0] * vocab_size
    true_bytes: list[int] = [0] * vocab_size
    mismatches = 0
    total_formula = 0
    total_true = 0

    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        token_str = tok.id_to_token(token_id)
        f_bytes = len(token_str.encode("utf-8"))
        # ByteLevel 将 0..255 的每个原始字节一对一映射到单个 Unicode 字符，
        # 因此符号串字符数即原始字节数（含不完整多字节前缀 token）
        if token_str and all(ch in _BYTE_LEVEL_CHAR_SET for ch in token_str):
            t_bytes = len(token_str)
        else:
            t_bytes = len(token_str.encode("utf-8"))

        formula_bytes[token_id] = f_bytes
        true_bytes[token_id] = t_bytes
        total_formula += f_bytes
        total_true += t_bytes
        if f_bytes != t_bytes:
            mismatches += 1

    # 在参考文本上按真实 token_id 查表核对总字节数
    enc = tok.encode(reference_text)
    ref_ids = [i for i in enc.ids if i not in special_ids]
    ref_total_formula = sum(formula_bytes[i] for i in ref_ids if i < vocab_size)
    ref_total_true = sum(true_bytes[i] for i in ref_ids if i < vocab_size)
    ref_raw_utf8_bytes = len(reference_text.encode("utf-8"))

    # 加载 disk 上的 token_bytes.pt 做交叉核对
    disk_loaded = None
    disk_path = None
    if token_bytes_pt and Path(token_bytes_pt).exists():
        try:
            import torch
            disk_loaded = torch.load(token_bytes_pt, weights_only=True).tolist()
            disk_path = str(token_bytes_pt)
        except Exception:
            disk_loaded = None

    return {
        "vocab_size": vocab_size,
        "non_special_tokens": vocab_size - len(special_ids),
        "formula_vs_true_mismatches": mismatches,
        "formula_total_bytes": total_formula,
        "true_total_bytes": total_true,
        "formula_minus_true_bytes": total_formula - total_true,
        "formula_overestimate_pct": round(
            (total_formula - total_true) / max(total_true, 1) * 100, 4
        ),
        "reference_text_chars": len(reference_text),
        "reference_raw_utf8_bytes": ref_raw_utf8_bytes,
        "reference_total_formula_bytes": ref_total_formula,
        "reference_total_true_bytes": ref_total_true,
        "reference_formula_minus_true_bytes": ref_total_formula - ref_total_true,
        "reference_formula_overestimate_pct": round(
            (ref_total_formula - ref_total_true) / max(ref_total_true, 1) * 100, 4
        ),
        "disk_token_bytes_pt_path": disk_path,
        "disk_token_bytes_pt_length": len(disk_loaded) if disk_loaded is not None else None,
        "disk_matches_formula": (
            disk_loaded == formula_bytes if disk_loaded is not None else None
        ),
    }


def compute_token_inflation(domain_results: dict, baseline_name: str = "climbmix_val") -> dict:
    """计算各域相对 baseline 的 token 膨胀率。"""
    baseline = domain_results.get(baseline_name)
    if not baseline or baseline.get("skipped"):
        return {}
    base_bpt = baseline["bytes_per_token"]
    inflation = {}
    for name, res in domain_results.items():
        if res.get("skipped"):
            continue
        if name == baseline_name:
            inflation[name] = 1.0
        else:
            inflation[name] = round(res["bytes_per_token"] / max(base_bpt, 1e-9), 4)
    return inflation


def load_audit_reference(source: str | Path | None, max_chars: int = 100_000) -> str:
    """读取 token_bytes 审计用参考文本（parquet/txt，截取固定长度保证确定性）。"""
    if not source:
        return ""
    path = Path(source)
    if not path.exists():
        return ""
    if path.suffix.lower() == ".parquet":
        return _load_parquet_text(path, max_chars)
    return _load_text(path, max_chars)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def evaluate_tokenizer(
    tok_path: Path,
    name: str,
    token_bytes_pt: Path | None,
    max_chars: int,
    domains: list[tuple[str, Path]],
    baseline_name: str | None = None,
    audit_source: str | Path | None = None,
) -> dict:
    """对单个分词器执行完整评估。"""
    print(f"[eval] loading tokenizer: {tok_path}  (name={name})")
    tok = Tokenizer.from_file(str(tok_path))
    vocab_size = tok.get_vocab_size()
    eos_id = tok.token_to_id(EOS)
    pad_id = tok.token_to_id(PAD)

    # 域评估
    domain_results = {}
    for dname, dsrc in domains:
        t0 = time.time()
        res = evaluate_domain(tok, dname, dsrc, max_chars)
        res["eval_seconds"] = round(time.time() - t0, 4)
        domain_results[dname] = res
        if not res.get("skipped"):
            print(
                f"  {dname}: chars={res['chars']:,} tokens={res['tokens']:,} "
                f"bpt={res['bytes_per_token']} roundtrip={res['roundtrip_ok']}"
            )

    # token_inflation：基准域缺省取第一个未跳过域
    baseline = baseline_name or next(
        (n for n, r in domain_results.items() if not r.get("skipped")), None
    )
    inflation = compute_token_inflation(domain_results, baseline_name=baseline) if baseline else {}

    # token_bytes 审计：参考文本来自 --audit-source，缺省回落内置短文本
    ref_text = load_audit_reference(audit_source, max_chars=max(100_000, max_chars))
    if not ref_text:
        ref_text = (
            "Hello world! This is a verification sentence.\n"
            "Numbers: 123, 4567, 89. Contractions: I'm, you're, it's.\n"
            "Special chars: @#$%^&*(). Unicode: 深度学习与大语言模型。"
        )
    tb_audit = audit_token_bytes(tok, token_bytes_pt, ref_text)

    # 特殊 token 信息
    special_tokens = {}
    if eos_id is not None:
        special_tokens[EOS] = eos_id
    if pad_id is not None:
        special_tokens[PAD] = pad_id

    return {
        "tokenizer_name": name,
        "tokenizer_path": str(tok_path),
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "domains": domain_results,
        "token_inflation": {"baseline": baseline, "values": inflation},
        "token_bytes_audit": tb_audit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="多分词器多域评估（域偏差与词表规模实验）")
    ap.add_argument(
        "--tokenizer-path",
        action="append",
        dest="tokenizer_paths",
        required=True,
        help="tokenizer.json 路径（可多次指定）",
    )
    ap.add_argument(
        "--tokenizer-name",
        action="append",
        dest="tokenizer_names",
        required=True,
        help="分词器名称（与 --tokenizer-path 一一对应）",
    )
    ap.add_argument(
        "--tokenizer-bytes",
        action="append",
        dest="tokenizer_bytes_paths",
        default=None,
        help="token_bytes.pt 路径（与 --tokenizer-path 一一对应，缺省自动推断）",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="每个域的最大评估字符数（默认 100k）",
    )
    ap.add_argument(
        "--fixtures-dir", type=str, default=None, dest="fixtures_dir",
        help="域切片目录（缺省实验目录 fixtures/）",
    )
    ap.add_argument(
        "--extra-domain", action="append", dest="extra_domains", default=None,
        help="追加评估域 NAME=PATH（parquet 或 txt，可多次）",
    )
    ap.add_argument(
        "--baseline-domain", type=str, default=None, dest="baseline_domain",
        help="token 膨胀率的基准域（缺省取第一个未跳过域）",
    )
    ap.add_argument(
        "--audit-source", type=str, default=None, dest="audit_source",
        help="token_bytes 审计的参考文本（parquet/txt，缺省用内置短文本）",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 JSON 路径（缺省 stdout）",
    )
    args = ap.parse_args()

    if args.tokenizer_bytes_paths and len(args.tokenizer_bytes_paths) != len(args.tokenizer_paths):
        print("[error] --tokenizer-bytes 数量必须与 --tokenizer-path 一致", file=sys.stderr)
        return 1

    domains = build_domains(
        fixtures_dir=Path(args.fixtures_dir) if args.fixtures_dir else None,
        extra_domains=args.extra_domains,
    )

    results = []
    for idx, (tok_path, tok_name) in enumerate(zip(args.tokenizer_paths, args.tokenizer_names)):
        p = Path(tok_path)
        if not p.exists():
            print(f"[error] tokenizer 不存在: {p}", file=sys.stderr)
            return 1
        # 定位 token_bytes.pt
        tb_path = None
        if args.tokenizer_bytes_paths and args.tokenizer_bytes_paths[idx]:
            tb_path = Path(args.tokenizer_bytes_paths[idx])
        else:
            candidate = p.parent / "token_bytes.pt"
            if candidate.exists():
                tb_path = candidate
        report = evaluate_tokenizer(
            p, tok_name, tb_path, args.max_chars, domains,
            baseline_name=args.baseline_domain, audit_source=args.audit_source,
        )
        results.append(report)

    out_json = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
        print(f"[eval] 报告已写入: {args.out}")
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
