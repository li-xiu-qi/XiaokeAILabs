"""Train a cross-encoder, then compare it on identical SciFact candidate pools."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import transformers

from prepare_scifact import digest, load_jsonl, write_json


def memory_available_gib():
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return None


def check_memory(config):
    available = memory_available_gib()
    if available is not None and available < config["memory_floor_gib"]:
        raise RuntimeError(f"System memory reached floor: {available:.2f} GiB")
    return available


def tokenize(tokenizer, pairs, config):
    queries, documents = zip(*pairs)
    return tokenizer(
        list(queries), list(documents), padding=True, pad_to_multiple_of=8,
        truncation="only_second", max_length=config["max_length"], return_tensors="pt"
    )


def score_pool(model, tokenizer, queries, docs, partition, config):
    model.eval()
    pairs = [(qid, did) for qid, row in partition.items() for did, _ in row["pool"]]
    scores = {}
    times = []
    batch_size = config["eval_batch_pairs"]
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            if start % (batch_size * 20) == 0:
                check_memory(config)
            batch = pairs[start:start + batch_size]
            t0 = time.perf_counter()
            inputs = tokenize(tokenizer, [(queries[q], docs[d]) for q, d in batch], config).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**inputs).logits[:, 0]
            logits = output.float().cpu().tolist()
            if not all(math.isfinite(x) for x in logits):
                raise RuntimeError("Non-finite evaluation score")
            times.append(time.perf_counter() - t0)
            scores.update(zip(batch, logits))
    return scores, {"pairs": len(pairs), "batches": len(times), "wall_seconds": sum(times)}


def training_batches(train, config, steps):
    rng = random.Random(config["seed"])
    qids = list(train)
    batch_size = config["train_batch_queries"]
    emitted = 0
    while emitted < steps:
        rng.shuffle(qids)
        for start in range(0, len(qids), batch_size):
            group = []
            for qid in qids[start:start + batch_size]:
                row = train[qid]
                group.append((qid, rng.choice(row["relevant"]), rng.choice(row["unjudged"])))
            yield group
            emitted += 1
            if emitted == steps:
                break


def train_model(model, tokenizer, queries, docs, train, config, steps):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], fused=True
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(config["warmup_steps"], steps // 10), num_training_steps=steps
    )
    model.train()
    log = []
    history = []
    t0 = time.perf_counter()
    for step, batch in enumerate(training_batches(train, config, steps), 1):
        if (step - 1) % 10 == 0:
            check_memory(config)
        pairs = [(queries[qid], docs[did]) for qid, pos, neg in batch for did in (pos, neg)]
        inputs = tokenize(tokenizer, pairs, config).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = model(**inputs).logits.float().reshape(-1, 2)
        loss = F.softplus(scores[:, 1] - scores[:, 0]).mean()
        scalar = loss.item()
        if not math.isfinite(scalar):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        lr_used = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        log.append({
            "step": step, "loss": scalar, "lr": lr_used, "pairs": len(pairs),
            "wall_seconds": time.perf_counter() - t0,
        })
        history.extend(batch)
        if step == 1 or step % 20 == 0 or step == steps:
            print(f"train {step}/{steps} loss={scalar:.5f} elapsed={log[-1]['wall_seconds']:.1f}s", flush=True)
    return log, history


def metrics(ranking, relevant):
    dcg = sum(1 / math.log2(rank + 2) for rank, did in enumerate(ranking[:10]) if did in relevant)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
    mrr = next((1 / rank for rank, did in enumerate(ranking[:10], 1) if did in relevant), 0.0)
    return {"ndcg_at_10": dcg / idcg, "mrr_at_10": mrr}


def evaluate(partition, initial, trained, history, config):
    rows = []
    training_positive = {pos for _, pos, _ in history}
    training_docs = training_positive | {neg for _, _, neg in history}
    for qid, item in partition.items():
        relevant = set(item["relevant"])
        bm25 = [did for did, _ in item["pool"]]
        start = sorted(bm25, key=lambda did: -initial[qid, did])
        end = sorted(bm25, key=lambda did: -trained[qid, did])
        present = relevant.intersection(bm25)
        rows.append({
            "query_id": qid,
            "relevant_doc_ids": sorted(relevant),
            "relevant_doc_ids_seen_as_training_positive": sorted(relevant & training_positive),
            "relevant_doc_ids_seen_in_training_pairs": sorted(relevant & training_docs),
            "candidates_in_bm25_order": [
                {"doc_id": did, "bm25": score, "before": initial[qid, did], "after": trained[qid, did]}
                for did, score in item["pool"]
            ],
            "relevant_ranks": {
                name: {did: rank for rank, did in enumerate(ranked, 1) if did in relevant}
                for name, ranked in (("bm25", bm25), ("initial", start), ("finetuned", end))
            },
            "hit_at_pool_size": bool(present),
            "recall_at_pool_size": len(present) / len(relevant),
            "metrics": {
                name: metrics(ranked, relevant)
                for name, ranked in (("bm25", bm25), ("initial", start), ("finetuned", end))
            },
        })

    def aggregate(group):
        if not group:
            return {"queries": 0}
        means = {
            name: {metric: float(np.mean([row["metrics"][name][metric] for row in group]))
                   for metric in ("ndcg_at_10", "mrr_at_10")}
            for name in ("bm25", "initial", "finetuned")
        }
        deltas = np.array([
            row["metrics"]["finetuned"]["ndcg_at_10"] - row["metrics"]["bm25"]["ndcg_at_10"]
            for row in group
        ])
        rng = np.random.default_rng(config["seed"])
        samples = rng.choice(deltas, size=(config["bootstrap_resamples"], len(group)), replace=True)
        ci = np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()
        directions = Counter("up" if d > 1e-10 else "down" if d < -1e-10 else "tied" for d in deltas)
        return {
            "queries": len(group),
            "candidate_hit_rate": float(np.mean([row["hit_at_pool_size"] for row in group])),
            "candidate_recall": float(np.mean([row["recall_at_pool_size"] for row in group])),
            "rankings": means,
            "finetuned_vs_bm25_ndcg_at_10": {
                "delta": float(deltas.mean()), "paired_bootstrap_ci95": ci,
                "query_directions": {key: directions[key] for key in ("up", "down", "tied")},
            },
        }

    return rows, {
        "all": aggregate(rows),
        "no_relevant_doc_seen_as_training_positive": aggregate(
            [row for row in rows if not row["relevant_doc_ids_seen_as_training_positive"]]
        ),
        "no_relevant_doc_seen_in_training_pairs": aggregate(
            [row for row in rows if not row["relevant_doc_ids_seen_in_training_pairs"]]
        ),
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = json.loads((args.prepared / "data-audit.json").read_text(encoding="utf-8"))
    pools_file = args.prepared / "pools.json"
    if audit["config_sha256"] != digest(args.config) or audit["pools_sha256"] != digest(pools_file):
        raise ValueError("Prepared pools were made with different inputs")
    for name, sha in audit["source_sha256"].items():
        if digest(args.data / name) != sha:
            raise ValueError(f"Dataset file differs from audited version: {name}")
    partitions = json.loads(pools_file.read_text(encoding="utf-8"))
    train = partitions["train"]
    if args.smoke:
        partition = dict(list(partitions["dev"].items())[:config["smoke_queries"]])
        steps = config["smoke_steps"]
    else:
        partition = partitions["test"]
        steps = config["steps"]
    queries = {qid: row["text"] for qid, row in load_jsonl(args.data / "queries.jsonl").items()}
    docs = {
        did: row["title"] + "\n" + row["text"]
        for did, row in load_jsonl(args.data / "corpus.jsonl").items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        raise FileExistsError(f"Results already exist under {args.out}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This run requires a CUDA GPU with native bfloat16 support")
    memory_start = check_memory(config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model_path = str(args.model_path) if args.model_path else config["model"]
    source = {"local_files_only": True} if args.model_path else {"revision": config["model_revision"]}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, attn_implementation="sdpa", **source
    ).to("cuda")
    # Initialize reproducibly even if from_pretrained consumes different random draws
    # when resolving cached .bin weights vs a Hub safetensors file.
    torch.manual_seed(config["seed"])
    with torch.no_grad():
        for layer in (model.classifier.dense, model.classifier.out_proj):
            layer.weight.normal_(mean=0.0, std=model.config.initializer_range)
            layer.bias.zero_()

    print(f"evaluating initial scores on {len(partition)} queries", flush=True)
    initial, timing_before = score_pool(model, tokenizer, queries, docs, partition, config)
    print(f"initial scoring complete in {timing_before['wall_seconds']:.1f}s", flush=True)
    log, history = train_model(model, tokenizer, queries, docs, train, config, steps)
    print(f"evaluating trained scores on {len(partition)} queries", flush=True)
    trained, timing_after = score_pool(model, tokenizer, queries, docs, partition, config)
    rows, aggregate = evaluate(partition, initial, trained, history, config)

    saved_model = None
    if args.save_model:
        saved = args.out / "final-model"
        pending = args.out / "final-model.partial"
        if saved.exists() or pending.exists():
            raise FileExistsError(f"Checkpoint already exists under {args.out}")
        started_save = time.perf_counter()
        model.save_pretrained(pending)
        tokenizer.save_pretrained(pending)
        file_sizes = {f.name: f.stat().st_size for f in sorted(pending.iterdir()) if f.is_file()}
        pending.rename(saved)
        saved_model = {
            "directory": saved.name,
            "files_bytes": file_sizes,
            "total_bytes": sum(file_sizes.values()),
            "wall_seconds": time.perf_counter() - started_save,
        }

    write_jsonl(args.out / "per-query.jsonl", rows)
    write_jsonl(args.out / "training-curve.jsonl", log)
    write_jsonl(args.out / "training-pairs.jsonl", [
        {"query_id": qid, "positive_doc_id": pos, "unjudged_doc_id": neg} for qid, pos, neg in history
    ])
    write_json(args.out / "split-query-ids.json", {name: sorted(group) for name, group in partitions.items()})
    weights_sha = {}
    if args.model_path:
        for name in ("pytorch_model.bin", "model.safetensors", "config.json", "tokenizer.json"):
            path = args.model_path / name
            if path.is_file():
                weights_sha[name] = digest(path)
    delta = aggregate["all"]["finetuned_vs_bm25_ndcg_at_10"]
    low, high = delta["paired_bootstrap_ci95"]
    decision = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
    result = {
        "mode": "smoke" if args.smoke else "formal",
        "evaluation_split": "dev_subset" if args.smoke else "beir_scifact_test",
        "start_utc": started,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": audit["config_sha256"],
        "pools_sha256": audit["pools_sha256"],
        "code_sha256": {
            "prepare_scifact.py": digest(Path(__file__).with_name("prepare_scifact.py")),
            Path(__file__).name: digest(Path(__file__)),
        },
        "hardware_software": {
            "gpu": torch.cuda.get_device_name(),
            "cuda_compute_capability": list(torch.cuda.get_device_capability()),
            "platform": platform.system() + "/" + platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "training_and_eval_autocast": "bfloat16",
        },
        "model_snapshot_sha256": weights_sha,
        "saved_checkpoint": saved_model,
        "training": {
            "steps": len(log),
            "triplets": len(history),
            "distinct_queries_used": len({q for q, _, _ in history}),
            "distinct_positive_doc_ids": len({p for _, p, _ in history}),
            "first_20_loss_mean": float(np.mean([row["loss"] for row in log[:20]])),
            "last_20_loss_mean": float(np.mean([row["loss"] for row in log[-20:]])),
        },
        "wall_seconds": {
            "total": time.perf_counter() - t0,
            "training": log[-1]["wall_seconds"],
            "initial_eval": timing_before["wall_seconds"],
            "trained_eval": timing_after["wall_seconds"],
        },
        "memory_gib": {
            "system_available_before": memory_start,
            "system_available_after": memory_available_gib(),
            "peak_torch_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "primary_comparison_outcome": decision if not args.smoke else None,
        "evaluation": aggregate,
    }
    write_json(args.out / "summary.json", result)
    print(json.dumps(result["evaluation"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
