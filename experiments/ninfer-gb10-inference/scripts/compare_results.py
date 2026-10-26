#!/usr/bin/env python3
"""Render the FP8/NVFP4 result files as one auditable Markdown comparison."""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def bench_index(data):
    return {row["concurrency"]: row for row in data["summary"]} if data else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvfp4-bench")
    parser.add_argument("--fp8-bench")
    parser.add_argument("--nvfp4-quality")
    parser.add_argument("--fp8-quality")
    parser.add_argument("--nvfp4-smoke")
    parser.add_argument("--fp8-smoke")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    nv_bench, fp_bench = load(args.nvfp4_bench), load(args.fp8_bench)
    nv_quality, fp_quality = load(args.nvfp4_quality), load(args.fp8_quality)
    nv_smoke, fp_smoke = load(args.nvfp4_smoke), load(args.fp8_smoke)
    nv_index, fp_index = bench_index(nv_bench), bench_index(fp_bench)
    levels = sorted(set(nv_index) | set(fp_index))

    lines = [
        "# DGX Spark：FP8 与 NVFP4 W4A4 + DFlash2 对照结果",
        "",
        "两档均使用 SGLang、DFlash2 草稿 8、512 token 上限与 xhigh 思考。吞吐是客户端端到端口径，包含 prefill 与排队，不等于纯 decode tok/s。小样本质量探针不能替代完整公开测评。",
        "",
        "## 并发性能",
        "",
        "| 并发 | NVFP4 总输出 tok/s | FP8 总输出 tok/s | FP8/NVFP4 | NVFP4 单路中位 tok/s | FP8 单路中位 tok/s | NVFP4 接受率 | FP8 接受率 | NVFP4 接受长度 | FP8 接受长度 | 完成（NV/FP） |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    comparison = {"performance": [], "quality": {}, "smoke": {}}
    for level in levels:
        nv, fp = nv_index.get(level), fp_index.get(level)
        ratio = (
            fp["aggregate_output_tps"] / nv["aggregate_output_tps"]
            if nv and fp and nv["aggregate_output_tps"]
            else None
        )
        row = {
            "concurrency": level,
            "nvfp4": nv,
            "fp8": fp,
            "fp8_over_nvfp4_aggregate_tps": ratio,
        }
        comparison["performance"].append(row)

        def cell(data, key, digits=1):
            return "n/a" if not data else f"{data[key]:.{digits}f}"

        done = (
            f"{nv['finished_stop']}/{nv['requests']} / {fp['finished_stop']}/{fp['requests']}"
            if nv and fp
            else "n/a"
        )
        lines.append(
            f"| {level} | {cell(nv, 'aggregate_output_tps')} | {cell(fp, 'aggregate_output_tps')} | "
            f"{('n/a' if ratio is None else f'{ratio:.2f}x')} | {cell(nv, 'median_request_e2e_output_tps')} | "
            f"{cell(fp, 'median_request_e2e_output_tps')} | {percent(nv['draft_accept_rate']) if nv else 'n/a'} | "
            f"{percent(fp['draft_accept_rate']) if fp else 'n/a'} | {cell(nv, 'draft_accept_length', 2) if nv and nv['draft_accept_length'] is not None else 'n/a'} | "
            f"{cell(fp, 'draft_accept_length', 2) if fp and fp['draft_accept_length'] is not None else 'n/a'} | {done} |"
        )

    if nv_quality or fp_quality:
        lines += [
            "",
            "## 小样本质量探针",
            "",
            "自动题共 10 道；编程题只检查语法和函数名，不计入自动正确率。截断与空答单独列出。",
            "",
            "| 档位 | 自动题通过 | 截断 | 空答 | 编程语法通过 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, data in [("NVFP4 W4A4", nv_quality), ("FP8", fp_quality)]:
            if not data:
                continue
            summary = data["summary"]
            comparison["quality"][name] = summary
            lines.append(
                f"| {name} | {summary['automated_passed']}/{summary['automated_total']} | {summary['automated_truncated']} | {summary['automated_empty']} | {summary['code_syntax_passed']}/{summary['code_total']} |"
            )

    if nv_smoke or fp_smoke:
        lines += ["", "## 文本与视觉冒烟", "", "| 档位 | 总体通过 | 检查项 |", "| --- | --- | --- |"]
        for name, data in [("NVFP4 W4A4", nv_smoke), ("FP8", fp_smoke)]:
            if not data:
                continue
            comparison["smoke"][name] = {"passed": data["passed"], "checks": data["checks"]}
            checks = ", ".join(f"{key}={value}" for key, value in data["checks"].items())
            lines.append(f"| {name} | {data['passed']} | {checks} |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
