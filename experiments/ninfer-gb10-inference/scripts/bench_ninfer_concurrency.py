#!/usr/bin/env python3
"""Concurrency benchmark using llama.cpp's native prompt/decode/draft timings."""
import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "A distributed cache loses one of six replicas after a network partition. "
    "Explain how quorum reads and writes should be configured, and identify the main "
    "consistency trade-off.",
    "Design a Python function that finds the longest palindromic substring. "
    "Explain the algorithm, its complexity, and three edge cases before giving the "
    "implementation.",
    "A training job uses 42 GB at batch size 8 and 58 GB at batch size 16. "
    "Estimate the fixed and per-sample memory costs, then predict usage at batch size 32. "
    "Show your reasoning.",
    "Compare FP8 and NVFP4 inference for a 27B model on unified-memory hardware. "
    "Discuss bandwidth, kernel support, quality risk, and how to benchmark them fairly.",
    "Write a robust plan for migrating a production vector database without downtime. "
    "Include validation, rollback, and the metrics that would trigger a rollback.",
    "Solve this scheduling problem: five jobs take 3, 7, 2, 9, and 4 minutes. "
    "Two identical workers start at time zero. Minimize makespan and explain why the "
    "assignment is optimal.",
    "Explain why speculative decoding can reduce latency without changing the target "
    "model's output distribution. Then describe when it can make throughput worse.",
    "Review this design: a public benchmark reuses the same prompt eight times, "
    "reports only aggregate tokens/s, and discards failed requests. Identify every "
    "methodological problem and propose a better protocol.",
]


def post(
    base_url,
    model,
    prompt,
    max_tokens,
    seed,
    timeout,
    enable_thinking=False,
    reasoning_effort=None,
    return_spec_tokens_details=False,
):
    template_kwargs = {"enable_thinking": enable_thinking}
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "chat_template_kwargs": template_kwargs,
        "return_timings": True,
    }
    if return_spec_tokens_details:
        payload["return_spec_tokens_details"] = True
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    latency = time.perf_counter() - started
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "latency_s": latency,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": completion_tokens,
        "e2e_output_tps": completion_tokens / latency if latency > 0 else 0,
        "timings": timings,
        "spec": (body.get("sglext") or {}).get("spec_tokens_details") or {},
        "error": None,
    }


def run_round(args, level, round_index):
    barrier = threading.Barrier(level)

    def worker(index):
        prompt = PROMPTS[index] + (
            f"\n\nBenchmark request identifier: r{round_index}-c{level}-i{index}."
        )
        barrier.wait()
        try:
            return post(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.seed + round_index * 100 + level * 10 + index,
                args.timeout,
                args.enable_thinking,
                args.reasoning_effort,
                args.return_spec_tokens_details,
            )
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": None,
                "completion_tokens": 0,
                "timings": {},
                "spec": {},
            }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
        records = list(pool.map(worker, range(level)))
    wall_s = time.perf_counter() - started
    return {"concurrency": level, "round": round_index, "wall_s": wall_s, "records": records}


def aggregate(runs):
    records = [record for run in runs for record in run["records"]]
    good = [record for record in records if not record.get("error")]
    total_tokens = sum(record["completion_tokens"] for record in good)
    total_wall = sum(run["wall_s"] for run in runs)
    draft = sum((record["timings"].get("draft_n") or 0) for record in good)
    accepted = sum((record["timings"].get("draft_n_accepted") or 0) for record in good)
    decode_rates = [(record["timings"].get("predicted_per_second") or 0) for record in good]
    prefill_rates = [(record["timings"].get("prompt_per_second") or 0) for record in good]
    spec_correct = sum((record["spec"].get("spec_num_correct_drafts") or 0) for record in good)
    spec_proposed = sum((record["spec"].get("spec_num_proposed_drafts") or 0) for record in good)
    spec_verify = sum((record["spec"].get("spec_verify_ct") or 0) for record in good)
    return {
        "runs": len(runs),
        "requests": len(records),
        "errors": len(records) - len(good),
        "finished_stop": sum(record.get("finish_reason") == "stop" for record in good),
        "finished_length": sum(record.get("finish_reason") == "length" for record in good),
        "total_output_tokens": total_tokens,
        "total_wall_s": total_wall,
        "aggregate_output_tps": total_tokens / total_wall if total_wall else 0,
        "median_request_e2e_output_tps": (
            statistics.median([record["e2e_output_tps"] for record in good]) if good else 0
        ),
        "median_server_decode_tps": statistics.median(decode_rates) if decode_rates else 0,
        "median_server_prefill_tps": statistics.median(prefill_rates) if prefill_rates else 0,
        "median_request_latency_s": (
            statistics.median([record["latency_s"] for record in good]) if good else 0
        ),
        "draft_accept_rate": (
            accepted / draft if draft else (spec_correct / spec_proposed if spec_proposed else None)
        ),
        "draft_accept_length": total_tokens / spec_verify if spec_verify else None,
        "spec_correct_drafts": spec_correct,
        "spec_proposed_drafts": spec_proposed,
        "spec_verify_ct": spec_verify,
        "draft_n": draft,
        "draft_n_accepted": accepted,
        "cached_prompt_tokens": sum((record["timings"].get("cache_n") or 0) for record in good),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8835/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="llama.cpp build-force-cublas")
    parser.add_argument("--context-total", type=int, default=262144)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--return-spec-tokens-details", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    levels = [int(value) for value in args.levels.split(",")]

    warmup = post(
        args.base_url,
        args.model,
        "Warm up the serving path with a short answer.",
        64,
        args.seed,
        args.timeout,
        args.enable_thinking,
        args.reasoning_effort,
        args.return_spec_tokens_details,
    )
    result = {
        "engine": args.engine,
        "model": args.model,
        "base_url": args.base_url,
        "protocol": {
            "levels": levels,
            "rounds_per_level": args.rounds,
            "requests_per_round": "equal to concurrency",
            "max_tokens": args.max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "enable_thinking": args.enable_thinking,
            "reasoning_effort": args.reasoning_effort,
            "return_spec_tokens_details": args.return_spec_tokens_details,
            "context_total": args.context_total,
            "throughput_definition": (
                "sum(completion_tokens) / sum(client wall time); native "
                "predicted_per_second is kept separately"
            ),
        },
        "warmup": warmup,
        "runs": [],
        "summary": [],
    }
    for level in levels:
        level_runs = []
        for round_index in range(args.rounds):
            run = run_round(args, level, round_index)
            result["runs"].append(run)
            level_runs.append(run)
            print(
                f"finished c={level} round={round_index + 1}/{args.rounds} "
                f"wall={run['wall_s']:.2f}s",
                flush=True,
            )
        summary = aggregate(level_runs)
        summary["concurrency"] = level
        result["summary"].append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
