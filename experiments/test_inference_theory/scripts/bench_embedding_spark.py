"""
Embedding 模型推理性能 Benchmark（在 dgx Spark 上运行）
================================================================================
测量 6 个 embedding 模型在 GB10 GPU 上的：
  1. 模型加载：加载时间、权重显存占用
  2. 单样本延迟：batch=1，不同序列长度（32/128/512/2048 tokens）
  3. 批处理吞吐：固定序列长度，扫 batch size（1/8/32/128/256/512）
  4. GPU 峰值算力：BF16 matmul benchmark
  5. 每次推理的 GPU 利用率（nvidia-smi 采样）

输出：JSON，每个模型一个文件，供后续理论模型标定使用。
"""
import json, time, os, sys, subprocess, threading
import torch

# ---- 模型清单（参数量从小到大）----
MODELS = [
    {"name": "all-MiniLM-L6-v2",       "id": "sentence-transformers/all-MiniLM-L6-v2",  "params_m": 22,  "dim": 384,  "layers": 6,  "ctx": 512},
    {"name": "bge-small-zh-v1.5",      "id": "BAAI/bge-small-zh-v1.5",                 "params_m": 23,  "dim": 512,  "layers": 4,  "ctx": 512},
    {"name": "jina-v2-small-en",       "id": "jinaai/jina-embeddings-v2-small-en",      "params_m": 28,  "dim": 512,  "layers": 4,  "ctx": 8192},
    {"name": "nomic-embed-text-v1.5",  "id": "nomic-ai/nomic-embed-text-v1.5",          "params_m": 108, "dim": 768,  "layers": 12, "ctx": 2048},
    {"name": "bge-m3",                 "id": "BAAI/bge-m3",                            "params_m": 558, "dim": 1024, "layers": 24, "ctx": 8194},
    {"name": "jina-embeddings-v4",     "id": "jinaai/jina-embeddings-v4",              "params_m": 1813,"dim": 2048, "layers": 36, "ctx": 128000},
]

BATCH_SIZES = [1, 8, 32, 128, 256, 512]
SEQ_LENGTHS  = [32, 128, 512, 2048]

def make_texts(n, approx_tokens):
    words = max(8, int(approx_tokens / 0.75))
    base = " ".join(["word"] * words)
    return [base] * n

def gpu_util_sampler(stop_event, samples):
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            parts = out.stdout.strip().split(", ")
            if len(parts) >= 3:
                samples.append({
                    "util_pct": float(parts[0]),
                    "mem_used_mb": float(parts[1]),
                    "mem_total_mb": float(parts[2]),
                })
        except Exception:
            pass
        stop_event.wait(0.1)

def bench_gpu_peak():
    if not torch.cuda.is_available():
        return {"error": "no cuda"}
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 50
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    flops = 2 * n**3
    tflops = flops / dt / 1e12
    del a, b, c
    torch.cuda.empty_cache()
    return {"bf16_peak_tflops": round(tflops, 1), "matrix": f"{n}x{n}"}

def bench_model(model_info):
    from sentence_transformers import SentenceTransformer
    name = model_info["name"]
    result = {"model": name, "model_id": model_info["id"],
              "params_m": model_info["params_m"], "dim": model_info["dim"],
              "layers": model_info["layers"], "ctx": model_info["ctx"]}

    print(f"\n{'='*60}")
    print(f"Benchmark: {name} ({model_info['params_m']}M params, dim={model_info['dim']})")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model = SentenceTransformer(model_info["id"], device="cuda",
                                     model_kwargs={"torch_dtype": torch.float16})
    except Exception as e:
        result["load_error"] = str(e)[:200]
        print(f"  load failed: {e}")
        return result
    load_time = time.time() - t0
    weight_mem_mb = torch.cuda.memory_allocated() / 1024**2
    result["load_time_s"] = round(load_time, 2)
    result["weight_mem_mb"] = round(weight_mem_mb, 1)
    print(f"  load: {load_time:.1f}s | weight mem: {weight_mem_mb:.0f}MB")

    single_latency = {}
    for seq in SEQ_LENGTHS:
        if seq > model_info["ctx"]:
            continue
        texts = make_texts(1, seq)
        for _ in range(3):
            model.encode(texts, batch_size=1, show_progress_bar=False)
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 20
        for _ in range(iters):
            model.encode(texts, batch_size=1, show_progress_bar=False)
        torch.cuda.synchronize()
        lat_ms = (time.time() - t0) / iters * 1000
        single_latency[f"seq{seq}"] = round(lat_ms, 2)
    result["single_latency_ms"] = single_latency
    print(f"  single latency: {single_latency}")

    seq_bench = 128
    batch_results = {}
    for bs in BATCH_SIZES:
        texts = make_texts(bs, seq_bench)
        samples = []
        stop = threading.Event()
        sampler = threading.Thread(target=gpu_util_sampler, args=(stop, samples))
        sampler.start()
        try:
            model.encode(texts[:min(bs,8)], batch_size=bs, show_progress_bar=False)
            torch.cuda.synchronize()
            t0 = time.time()
            emb = model.encode(texts, batch_size=bs, show_progress_bar=False)
            torch.cuda.synchronize()
            dt = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            batch_results[f"bs{bs}"] = "OOM"
            print(f"    bs={bs}: OOM")
            continue
        finally:
            stop.set()
            sampler.join()
        throughput = bs / dt
        latency_ms = dt / bs * 1000
        avg_util = sum(s["util_pct"] for s in samples) / len(samples) if samples else 0
        peak_mem = max(s["mem_used_mb"] for s in samples) if samples else 0
        batch_results[f"bs{bs}"] = {
            "throughput_sps": round(throughput, 1),
            "per_sentence_ms": round(latency_ms, 2),
            "gpu_util_avg": round(avg_util, 1),
            "gpu_util_max": round(max((s["util_pct"] for s in samples), default=0), 1),
            "peak_mem_mb": round(peak_mem, 0),
        }
        print(f"    bs={bs:>3}: {throughput:>7.1f} sent/s | {latency_ms:>6.2f} ms/sent | util {avg_util:.0f}% | mem {peak_mem:.0f}MB")
    result["batch_sweep_seq128"] = batch_results

    long_seq = {}
    for seq in [512, 2048]:
        if seq > model_info["ctx"]:
            continue
        texts = make_texts(32, seq)
        try:
            model.encode(texts[:8], batch_size=32, show_progress_bar=False)
            torch.cuda.synchronize()
            t0 = time.time()
            model.encode(texts, batch_size=32, show_progress_bar=False)
            torch.cuda.synchronize()
            dt = time.time() - t0
            long_seq[f"seq{seq}"] = round(dt / 32 * 1000, 2)
        except Exception:
            long_seq[f"seq{seq}"] = "OOM"
    result["bs32_long_seq_ms"] = long_seq

    del model
    torch.cuda.empty_cache()
    return result

def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    print("\n--- GPU peak compute ---")
    peak = bench_gpu_peak()
    print(f"  BF16 peak: {peak.get('bf16_peak_tflops', '?')} TFLOPS")

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    models = MODELS
    if only:
        models = [m for m in MODELS if m["name"] in only]
        print(f"testing only: {[m['name'] for m in models]}")

    all_results = {"gpu": torch.cuda.get_device_name(0), "peak": peak, "models": []}
    for m in models:
        r = bench_model(m)
        all_results["models"].append(r)
        out = os.path.expanduser(f"~/emb_bench_{m['name']}.json")
        with open(out, "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    with open(os.path.expanduser("~/emb_bench_all.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print("all done! results in ~/emb_bench_*.json")

if __name__ == "__main__":
    main()
