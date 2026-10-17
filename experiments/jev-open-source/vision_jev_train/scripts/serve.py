"""起推理 HTTP 服务。

输入一条请求：图像、上下文、判别问题、候选列表。
输出每个候选的校准后概率与最优候选。

刻意只用标准库和 transformers，不加 Web 框架，减少读者的环境负担。
"""
import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLACE = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_LEN = 320
MAX_PIXELS = 320 * 320


def build_engine(ckpt, model, temperature):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForSequenceClassification

    proc = AutoProcessor.from_pretrained(ckpt)
    base = Qwen3_5ForSequenceClassification.from_pretrained(
        model, num_labels=1, dtype=torch.bfloat16
    )
    tc = base.config.get_text_config()
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    base.num_labels = 1

    from peft import PeftModel

    net = PeftModel.from_pretrained(base, ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device).eval()
    return proc, tok, net, device, temperature, torch, F, Image


def decide(engine, payload):
    proc, tok, net, device, temperature, torch, F, Image = engine
    image = Image.open(io.BytesIO(base64.b64decode(payload["image_png_b64"]))).convert("RGB")
    # 与 train.py 同因：不能对含图像占位符的串启用 truncation
    state = (payload.get("state") or "")[-MAX_LEN * 4:]
    texts = [
        f"{PLACE}State:\n{state}\n\nQuestion:\n"
        f"{payload['question']}\n\nOption:\n{o}"
        for o in payload["options"]
    ]
    enc = proc(
        text=texts,
        images=[image] * len(texts),
        return_tensors="pt",
        padding=True,
        max_pixels=MAX_PIXELS,
    )
    keep = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"]
    enc = {k: enc[k] for k in keep if k in enc}
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    with torch.no_grad():
        logits = net(**enc).logits.float().squeeze(-1) / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1).cpu().tolist()
    best = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "options": payload["options"],
        "probs": probs,
        "prediction": best,
        "confidence": probs[best],
        "temperature": temperature,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA adapter 目录，如 runs/v2/final")
    ap.add_argument("--model", required=True, help="底座本地路径，如 models/Qwen3.5-2B")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    temp = args.temperature
    if temp is None:
        tp = Path(args.ckpt).parent / "temperatures.json"
        if tp.exists():
            temp = json.loads(tp.read_text())["temperature"]
        else:
            temp = 1.0

    engine = build_engine(args.ckpt, args.model, temp)
    print(f"[serve] ckpt={args.ckpt} T={temp} port={args.port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                out = decide(engine, payload)
                body = json.dumps(out, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": repr(e)}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
