#!/usr/bin/env python3
"""Train a small LoRA adapter with the original pairwise DPO objective."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DATA_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"


def extract_pair(row):
    """Use only single-turn pairs with exactly the same user prompt."""
    chosen, rejected = row["chosen"], row["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if [x["role"] for x in chosen] != ["user", "assistant"]:
        return None
    if [x["role"] for x in rejected] != ["user", "assistant"]:
        return None
    prompt = chosen[0]["content"]
    if prompt != rejected[0]["content"]:
        return None
    a, b = chosen[1]["content"], rejected[1]["content"]
    if not all(isinstance(x, str) and x.strip() for x in (prompt, a, b)) or a == b:
        return None
    return {"prompt": prompt, "chosen": a, "rejected": b}


def select_pairs(rows, size, seed):
    pairs = [item for row in rows if (item := extract_pair(row)) is not None]
    random.Random(seed).shuffle(pairs)
    selected, seen = [], set()
    for pair in pairs:
        digest = hashlib.sha256(pair["prompt"].encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        selected.append({**pair, "prompt_sha256": digest})
        seen.add(digest)
        if len(selected) == size:
            return selected
    raise ValueError(f"only {len(selected)} distinct valid pairs; requested {size}")


def make_tokens(tokenizer, row, max_prompt, max_completion):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, list):
        prompt = prompt["input_ids"]
    prompt_was_truncated = len(prompt) > max_prompt
    prompt = prompt[-max_prompt:]
    answers = []
    truncated = []
    for name in ("chosen", "rejected"):
        ids = tokenizer.encode(row[name], add_special_tokens=False)
        truncated.append(len(ids) > max_completion - 1)
        ids = ids[: max_completion - 1] + [tokenizer.eos_token_id]
        if len(ids) < 2:
            raise ValueError("empty completion")
        answers.append((prompt + ids, len(prompt)))
    return answers, {
        "prompt_truncated": prompt_was_truncated,
        "chosen_truncated": truncated[0],
        "rejected_truncated": truncated[1],
        "identical_after_truncation": answers[0][0] == answers[1][0],
    }


def tokenize_pairs(tokenizer, pairs, max_prompt, max_completion):
    kept_pairs, tokens = [], []
    stats = {"sampled": len(pairs), "prompt_truncated": 0, "chosen_truncated": 0,
             "rejected_truncated": 0, "identical_after_truncation": 0}
    for pair in pairs:
        pair_tokens, info = make_tokens(tokenizer, pair, max_prompt, max_completion)
        for key, value in info.items():
            stats[key] += int(value)
        if not info["identical_after_truncation"]:
            kept_pairs.append(pair)
            tokens.append(pair_tokens)
    stats["valid_after_truncation"] = len(tokens)
    if not tokens:
        raise ValueError("no non-identical pairs remain after truncation")
    return kept_pairs, tokens, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-prompt", type=int, default=192)
    parser.add_argument("--max-completion", type=int, default=192)
    args = parser.parse_args()
    if min(args.train_size, args.test_size, args.steps, args.accumulation) <= 0:
        parser.error("sizes, steps and accumulation must be positive")
    if args.max_prompt < 1 or args.max_completion < 2 or args.beta <= 0:
        parser.error("invalid token limits or beta")

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this experiment requires a BF16-capable CUDA GPU")
    if os.getenv("XL_REQUIRE_EXTERNAL_ENV") == "1" and (
        not os.getenv("XL_OUT_DIR") or not os.getenv("XL_DATA_DIR")
    ):
        raise RuntimeError("strict mode requires XL_OUT_DIR and XL_DATA_DIR")
    exp_dir = Path(__file__).resolve().parents[1]
    out = Path(os.getenv("XL_OUT_DIR", exp_dir / ".work" / "run"))
    data = Path(os.getenv("XL_DATA_DIR", exp_dir / ".work" / "data"))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    data.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train = select_pairs(
        load_dataset(DATASET, split="train_prefs", revision=DATA_REVISION, cache_dir=str(data)),
        args.train_size,
        args.seed,
    )
    test = select_pairs(
        load_dataset(DATASET, split="test_prefs", revision=DATA_REVISION, cache_dir=str(data)),
        args.test_size,
        args.seed + 1,
    )
    if {x["prompt_sha256"] for x in train} & {x["prompt_sha256"] for x in test}:
        raise ValueError("selected train and test prompts overlap")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, cache_dir=str(data))
    if tokenizer.eos_token_id is None:
        raise ValueError("model tokenizer has no EOS token")
    train, train_tokens, train_stats = tokenize_pairs(
        tokenizer, train, args.max_prompt, args.max_completion
    )
    test, test_tokens, test_stats = tokenize_pairs(
        tokenizer, test, args.max_prompt, args.max_completion
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, cache_dir=str(data), torch_dtype=torch.bfloat16
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    def score(seq):
        ids, prompt_len = seq
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
        logits = model(input_ids=inputs).logits[:, :-1, :].float()
        labels = inputs[:, 1:].clone()
        labels[:, : prompt_len - 1] = -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            reduction="sum", ignore_index=-100,
        )
        return -loss

    def score_pair(pair):
        return score(pair[0]), score(pair[1])

    def reference(pairs):
        model.eval()
        values = []
        with torch.no_grad(), model.disable_adapter():
            for pair in pairs:
                a, b = score_pair(pair)
                values.append((a.item(), b.item()))
        return values

    print("computing frozen reference scores", flush=True)
    train_ref = reference(train_tokens)
    test_ref = reference(test_tokens)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    order = list(range(len(train_tokens)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(args.accumulation):
            idx = order[(step * args.accumulation + micro) % len(order)]
            chosen, rejected = score_pair(train_tokens[idx])
            ref_chosen, ref_rejected = train_ref[idx]
            margin = (chosen - rejected) - (ref_chosen - ref_rejected)
            loss = -F.logsigmoid(args.beta * margin)
            (loss / args.accumulation).backward()
            total += loss.item() / args.accumulation
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        losses.append(total)
        if (step + 1) % 8 == 0 or step == 0:
            print(json.dumps({"step": step + 1, "loss": total}), flush=True)
    model.eval()
    results = []
    with torch.no_grad():
        for pair, (ref_a, ref_b) in zip(test_tokens, test_ref):
            a, b = score_pair(pair)
            base_margin = ref_a - ref_b
            new_margin = (a - b).item()
            results.append({"base_margin": base_margin, "trained_margin": new_margin})
    n = len(results)
    rng = random.Random(args.seed + 2)
    deltas = [
        int(x["trained_margin"] > 0) - int(x["base_margin"] > 0) for x in results
    ]
    bootstrap = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000)
    )
    report = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET,
        "dataset_revision": DATA_REVISION,
        "train_split": "train_prefs",
        "test_split": "test_prefs",
        "train_size": len(train),
        "test_size": n,
        "truncation": {"train": train_stats, "test": test_stats},
        "exact_prompt_overlap": 0,
        "parameters": vars(args),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "base_pair_accuracy": sum(x["base_margin"] > 0 for x in results) / n,
        "trained_pair_accuracy": sum(x["trained_margin"] > 0 for x in results) / n,
        "pair_accuracy_change_bootstrap_95pct": [bootstrap[50], bootstrap[1949]],
        "mean_change_in_reference_relative_margin": sum(
            x["trained_margin"] - x["base_margin"] for x in results
        ) / n,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    model.save_pretrained(out / "adapter")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "test_margins.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out / "sample_prompt_hashes.json").write_text(
        json.dumps({name: [x["prompt_sha256"] for x in items] for name, items in
                    (("train", train), ("test", test))}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
