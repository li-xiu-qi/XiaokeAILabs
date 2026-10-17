"""训练入口：在 prepared 数据上全参微调，选优、温度校准、测试一条龙。

用法：
  python scripts/train.py --backbone models/Qwen3-0.6B-Base \
      --run-dir runs/agentjev_v1

流程顺序由代码固定：先训练并按 dev 软交叉熵选 checkpoint，再用独立的
calibration 划分拟合每类问题的温度，最后才第一次读 test。run 目录已
存在且含 selection.json 时直接报错，防止重跑覆盖，换个目录即可。
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from agentjev.model import AgentJevModel  # noqa: E402
from jev_service.contract import encode_paths  # noqa: E402


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load(split, prepared_dir):
    return [
        json.loads(line)
        for line in (prepared_dir / f"{split}_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def tokenize(rows, tok):
    for r in rows:
        r["paths"] = encode_paths(
            [{"state": r["state"], "questions": [r["question"]]}],
            tok,
            max_tokens=2048,
        )[0]
    return rows


def batch(rows, pad):
    paths = [seq for r in rows for seq in r["paths"]]
    length = max(map(len, paths))
    cmax = max(len(r["target"]) for r in rows)
    ids = torch.full((len(paths), length), pad, dtype=torch.long, device="cuda:0")
    att = torch.zeros_like(ids)
    mask = torch.zeros(len(rows), cmax, dtype=torch.bool, device="cuda:0")
    targets = torch.zeros(len(rows), cmax, device="cuda:0")
    qi, ci, ends = [], [], []
    p = 0
    for i, r in enumerate(rows):
        mask[i, : len(r["target"])] = True
        targets[i, : len(r["target"])] = torch.tensor(r["target"], device="cuda:0")
        for j, seq in enumerate(r["paths"]):
            ids[p, : len(seq)] = torch.tensor(seq, device="cuda:0")
            att[p, : len(seq)] = 1
            qi.append(i)
            ci.append(j)
            ends.append(len(seq) - 1)
            p += 1
    return {
        "input_ids": ids,
        "attention_mask": att,
        "cand_mask": mask,
        "target": targets,
        "q_index": torch.tensor(qi, device="cuda:0"),
        "cand_index": torch.tensor(ci, device="cuda:0"),
        "cand_end_pos": torch.tensor(ends, device="cuda:0"),
    }


def probabilities(logits, temp=1.0):
    z = np.asarray(logits, dtype=np.float64) / temp
    z -= z.max()
    p = np.exp(z)
    return (p / p.sum()).tolist()


def metrics(rows):
    correct, ces, briers, soft, confidence, score_errors = [], [], [], [], [], []
    for r in rows:
        p = np.asarray(r["probs"])
        y = np.asarray(r["target"])
        guess, gold = int(p.argmax()), int(y.argmax())
        correct.append(float(guess == gold))
        ces.append(float(-(y * np.log(np.clip(p, 1e-12, 1))).sum()))
        briers.append(float(((p - y) ** 2).sum()))
        soft.append(float(y[guess]))
        confidence.append(float(p.max()))
        if r["type"] == "score":
            score_errors.append(abs(float(np.arange(len(p)) @ (p - y))))
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        ix = [
            i
            for i, c in enumerate(confidence)
            if lo <= c < (lo + 0.1 if lo < 0.9 else 1.0000001)
        ]
        if ix:
            ece += len(ix) / len(rows) * abs(
                np.mean([correct[i] for i in ix]) - np.mean([confidence[i] for i in ix])
            )
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "soft_cross_entropy": float(np.mean(ces)),
        "brier_sum": float(np.mean(briers)),
        "gold_mass_at_selected_option": float(np.mean(soft)),
        "ece_10_bins_vs_gold_argmax": float(ece),
        "score_expectation_mae": float(np.mean(score_errors)) if score_errors else None,
    }


def report(rows):
    return {
        "overall": metrics(rows),
        "by_type": {
            k: metrics([r for r in rows if r["type"] == k])
            for k in sorted({r["type"] for r in rows})
        },
        "by_workflow": {
            k: metrics([r for r in rows if r["workflow"] == k])
            for k in sorted({r["workflow"] for r in rows})
        },
    }


def evaluate(model, rows, pad):
    model.eval()
    predictions = []
    for start in range(0, len(rows), 8):
        chunk = rows[start : start + 8]
        b = batch(chunk, pad)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b)["logits"].float().cpu().tolist()
        for r, z in zip(chunk, logits):
            z = z[: len(r["target"])]
            predictions.append(
                {k: r[k] for k in ("id", "case_id", "workflow", "target")}
                | {"type": r["question"]["type"], "logits": z, "probs": probabilities(z)}
            )
    return predictions


def calibrate(rows):
    temperatures = {}
    for kind in sorted({r["type"] for r in rows}):
        selected = [r for r in rows if r["type"] == kind]
        candidates = []
        for t in np.exp(np.linspace(np.log(0.25), np.log(4), 81)):
            ce = np.mean(
                [
                    -(np.asarray(r["target"]))
                    @ np.log(np.clip(probabilities(r["logits"], t), 1e-12, 1))
                    for r in selected
                ]
            )
            candidates.append((float(ce), float(t)))
        ce, t = min(candidates)
        temperatures[kind] = {
            "temperature": t,
            "calibration_n": len(selected),
            "soft_cross_entropy": ce,
        }
    return temperatures


def apply_calibration(rows, temperatures):
    return [
        dict(r, probs=probabilities(r["logits"], temperatures[r["type"]]["temperature"]))
        for r in rows
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="本地底座目录（Qwen3-0.6B）")
    parser.add_argument("--protocol", default=str(ROOT / "scripts" / "protocol.json"))
    parser.add_argument("--prepared", default=str(ROOT / "prepared"))
    parser.add_argument("--run-dir", default=str(ROOT / "runs" / "agentjev_v1"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "selection.json").exists():
        raise RuntimeError("selection.json 已存在，换一个 --run-dir 再跑")

    torch.manual_seed(protocol["seed"])
    rng = random.Random(protocol["seed"])
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    pad = tok.pad_token_id or tok.eos_token_id

    prepared_dir = Path(args.prepared)
    train_rows = tokenize(load("train", prepared_dir), tok)
    dev_rows = tokenize(load("dev", prepared_dir), tok)
    lengths = [len(seq) for r in train_rows + dev_rows for seq in r["paths"]]
    save(
        run_dir / "input_audit.json",
        {
            "max_path_tokens": max(lengths),
            "mean_path_tokens": float(np.mean(lengths)),
            "truncated": 0,
            "train_questions": len(train_rows),
            "dev_questions": len(dev_rows),
            "input_fields": ["state", "question", "candidate description"],
        },
    )

    model = AgentJevModel(args.backbone).to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    base = evaluate(model, dev_rows, pad)
    best_ce = metrics(base)["soft_cross_entropy"]
    best_step = 0
    save(run_dir / "initial_dev.json", report(base))
    print("initial_dev", json.dumps(metrics(base)), flush=True)

    groups = [
        {
            "params": [p for n, p in model.named_parameters() if ".backbone." in n],
            "lr": protocol["backbone_lr"],
        },
        {
            "params": [p for n, p in model.named_parameters() if ".backbone." not in n],
            "lr": protocol["head_lr"],
        },
    ]
    opt = torch.optim.AdamW(groups, weight_decay=protocol.get("weight_decay", 0.01))
    steps = protocol["max_steps"]
    warmup = protocol.get("warmup_steps", 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup)
        * 0.5
        * (1 + math.cos(math.pi * max(0, s - warmup) / max(1, steps - warmup))),
    )

    order = list(range(len(train_rows)))
    rng.shuffle(order)
    position = 0
    start = time.monotonic()
    with (run_dir / "events.jsonl").open("w", buffering=1) as events:
        for step in range(1, steps + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(protocol["gradient_accumulation"]):
                ix = []
                for _ in range(protocol["microbatch_questions"]):
                    if position == len(order):
                        rng.shuffle(order)
                        position = 0
                    ix.append(order[position])
                    position += 1
                b = batch([train_rows[i] for i in ix], pad)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(b)["logits"]
                logits = logits.float().masked_fill(~b["cand_mask"], float("-inf"))
                logp = torch.log_softmax(logits, -1).masked_fill(~b["cand_mask"], 0)
                p = logp.exp() * b["cand_mask"]
                ce = -(b["target"] * logp).sum(-1).mean()
                loss = ce + 0.1 * ((p - b["target"]) ** 2).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                (loss / protocol["gradient_accumulation"]).backward()
                total_loss += float(loss.detach()) / protocol["gradient_accumulation"]
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.get("grad_clip", 1.0)
            )
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite gradient")
            opt.step()
            schedule.step()
            rec = {"step": step, "loss": total_loss, "seconds": round(time.monotonic() - start, 2)}
            if step % 100 == 0:
                pred = evaluate(model, dev_rows, pad)
                m = metrics(pred)
                rec["dev"] = m
                if m["soft_cross_entropy"] < best_ce:
                    best_ce = m["soft_cross_entropy"]
                    best_step = step
                    tmp = run_dir / "best.tmp"
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "step": step,
                            "role": "typed_decisions",
                            "input_schema": "agentjev.decision.v1",
                            "max_path_tokens": 2048,
                            "protocol": protocol,
                        },
                        tmp,
                    )
                    tmp.replace(run_dir / "best.pt")
                save(run_dir / f"dev_step_{step}.json", report(pred))
            events.write(json.dumps(rec) + "\n")
            if step % 20 == 0:
                print(json.dumps(rec), flush=True)
    del opt

    selection = {
        "best_step": best_step,
        "dev_soft_cross_entropy": best_ce,
        "checkpoint": str(run_dir / "best.pt") if best_step else None,
        "selection_completed_before_test": True,
        "training_seconds": time.monotonic() - start,
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    save(run_dir / "selection.json", selection)

    ck = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    del ck

    calibration_rows = tokenize(load("calibration", prepared_dir), tok)
    cal_pred = evaluate(model, calibration_rows, pad)
    temps = calibrate(cal_pred)
    save(run_dir / "calibration_predictions.json", cal_pred)
    save(run_dir / "temperatures.json", temps)

    test_rows = tokenize(load("test", prepared_dir), tok)
    test_pred = evaluate(model, test_rows, pad)
    cal_test = apply_calibration(test_pred, temps)
    save(run_dir / "test_predictions.json", test_pred)
    save(run_dir / "test_calibrated_predictions.json", cal_test)

    final = {
        "selection": selection,
        "trained": report(test_pred),
        "trained_calibrated": report(cal_test),
        "temperatures": temps,
        "limitation": "Agreement with public teacher labels; not actual agent task success.",
    }
    save(run_dir / "report.json", final)
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
