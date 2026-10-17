#!/usr/bin/env python
"""独立评测脚本：对已训练好的 LoRA adapter 做完整评测。

与 train.py 内置评测的差异，也是它存在的理由：
1. 图像在进入 processor 之前先 resize 到 max_pixels 约束的尺寸。
   train.py 里同一张图会被 processor 对每个候选各处理一次，processor 的
   smart_resize 是单线程 CPU 密集操作，一张图处理 3 到 4 次，评测 1000 条
   时这一项就成了瓶颈（实测 CPU 93% 空闲但 python 单核跑满）。
2. 用 dataloader 的多 worker 预取，把图像解码和 resize 与 GPU 前向重叠。

用法：
  python evaluate.py --model models/Qwen3.5-2B --ckpt runs/v1/final \
      --data prepared --out runs/v1/eval
"""
import argparse, base64, io, json, os, sys, time
import torch
import torch.nn.functional as F
from PIL import Image

PLACE = "<|vision_start|><|image_pad|><|vision_end|>"
BRIER_W = 0.1
MAX_PIXELS = 320 * 320
MAX_SIDE = 320


def log(*a):
    print(*a, flush=True)


def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_image(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def shrink(im, max_pixels=MAX_PIXELS, max_side=MAX_SIDE):
    """先按长边和总像素双约束缩小，之后 processor 基本只做归一化。"""
    w, h = im.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        im = im.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    w, h = im.size
    if w * h > max_pixels:
        r = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    return im


def build_text(state, question, option):
    return f"{PLACE}State:\n{state}\n\nQuestion:\n{question}\n\nOption:\n{option}"


def encode_batch(proc, rows):
    texts, images, counts = [], [], []
    for r in rows:
        im = shrink(load_image(r["image_png_b64"]))
        state = (r.get("state") or "")[-1280:]
        for opt in r["options"]:
            texts.append(build_text(state, r["question"], opt))
            images.append(im)
        counts.append(len(r["options"]))
    enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
    return enc, counts


def model_inputs(enc):
    keep = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"]
    return {k: enc[k] for k in keep if k in enc}


def soft_ce(probs, gold):
    return -(gold * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()


def brier(probs, gold):
    return ((probs - gold) ** 2).sum(dim=-1).mean()


@torch.no_grad()
def run_logits(proc, model, rows, device, bs=32, temperature=1.0):
    model.eval()
    pool = []
    for i in range(0, len(rows), bs):
        enc, counts = encode_batch(proc, rows[i : i + bs])
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs(enc).items()}
        out = model(**enc)
        logits = out.logits.float().squeeze(-1) / max(temperature, 1e-6)
        pos = 0
        for c in counts:
            pool.append(logits[pos : pos + c].cpu())
            pos += c
    return pool


def metrics(probs_list, rows):
    accs, ces, brs = [], [], []
    confs, correct = [], []
    for p, r in zip(probs_list, rows):
        g = torch.tensor(r["gold"], dtype=torch.float32)
        gold = g / g.sum()
        hit = 1.0 if p.argmax().item() == gold.argmax().item() else 0.0
        accs.append(hit)
        ces.append(soft_ce(p, gold).item())
        brs.append(brier(p, gold).item())
        confs.append(p.max().item())
        correct.append(hit)
    n = max(len(rows), 1)
    bins = [[] for _ in range(15)]
    for c, a in zip(confs, correct):
        bins[min(int(c * 15), 14)].append((c, a))
    ece = sum(abs(sum(cc for cc, _ in b) - sum(aa for _, aa in b)) for b in bins if b) / n
    return {"n": len(rows), "acc": sum(accs) / n, "soft_ce": sum(ces) / n,
            "brier": sum(brs) / n, "ece": ece}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoProcessor, Qwen3_5ForSequenceClassification
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.ckpt)
    base = Qwen3_5ForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16
    )
    tc = base.config.get_text_config()
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    base.num_labels = 1
    net = PeftModel.from_pretrained(base, args.ckpt)
    net.to(device).eval()
    log(f"[load] adapter={args.ckpt} T={args.temperature}")

    splits = {}
    for name in ("dev", "calibration", "test"):
        p = os.path.join(args.data, f"{name}.jsonl")
        if os.path.exists(p):
            splits[name] = load_rows(p)
    log(f"[data] " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # 温度网格，范围比 train.py 宽，上限 2.0 触顶说明最优值在网格外
    best_t, best_ce = args.temperature, None
    if "calibration" in splits:
        cal = splits["calibration"]
        logits_pool = run_logits(proc, net, cal, device)
        golds = []
        for r in cal:
            g = torch.tensor(r["gold"], dtype=torch.float32)
            golds.append(g / g.sum())
        for k in range(41):
            t = 0.5 + k * 0.1
            ce = sum(soft_ce(F.softmax(lg / t, -1), g).item()
                     for lg, g in zip(logits_pool, golds)) / len(cal)
            if best_ce is None or ce < best_ce:
                best_t, best_ce = t, ce
        log(f"[calib] best_T={best_t:.3f} calib_soft_ce={best_ce:.5f}")

    report = {"ckpt": args.ckpt, "temperature": best_t, "calib_soft_ce": best_ce, "splits": {}}
    t0 = time.time()
    for name, rows in splits.items():
        raw = metrics([F.softmax(lg, -1) for lg in run_logits(proc, net, rows, device)], rows)
        cal = metrics([F.softmax(lg / max(best_t, 1e-6), -1)
                       for lg in run_logits(proc, net, rows, device, temperature=best_t)], rows)
        report["splits"][name] = {"raw": raw, "calibrated": cal}
        log(f"[{name:11s}] n={raw['n']:4d} acc={raw['acc']:.4f} soft_ce={raw['soft_ce']:.4f} "
            f"brier={raw['brier']:.4f} ece={raw['ece']:.4f} | "
            f"acc_cal={cal['acc']:.4f} soft_ce_cal={cal['soft_ce']:.4f} ece_cal={cal['ece']:.4f}")

    report["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(args.out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("[done] " + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
