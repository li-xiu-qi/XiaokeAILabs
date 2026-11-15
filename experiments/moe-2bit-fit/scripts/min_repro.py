"""最小复现：不跑 bench，只验证「模型能加载 + 能出一次 token」。

用途是隔离问题，不是测性能。环境刚装好时先跑它，确认权重、插件、kernel 编译这条链
本身没问题，再上 bench。正式测量用 scripts/run_bench_pair.sh。

默认使用与历史基准一致的 gpu_memory_utilization=0.85 和 max_model_len=2048。
实际内存峰值还受系统和其他进程影响，应在有足够余量时运行。若需要复查高内存配置，
显式传入 --gpu-memory-utilization 0.92 --max-model-len 4096。

启动方式见 launch_repro2.sh：带上 CUDA_LAUNCH_BLOCKING 与详细日志，
挂住时先识别实际 Python 子进程，再用 `kill -USR1 <python-pid>` 让 faulthandler 打出全部线程栈。
"""
import argparse
import faulthandler
import os
import signal

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="models/DeepSeek-V4-Flash-2bit")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
parser.add_argument("--max-model-len", type=int, default=2048)
args = parser.parse_args()
if not 0 < args.gpu_memory_utilization <= 1:
    parser.error("--gpu-memory-utilization must be in (0, 1]")
if args.max_model_len < 1:
    parser.error("--max-model-len must be positive")

faulthandler.register(signal.SIGUSR1)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUTE_DSL_ARCH"] = "sm_121a"
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["VQ2_EXPERTS_DIR"] = args.model

from vllm import LLM, SamplingParams

print("[repro] constructing LLM ...", flush=True)
llm = LLM(
    model=args.model,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_model_len=args.max_model_len,
    enforce_eager=True,
    max_num_seqs=1,
    kv_cache_dtype="fp8",
)
print("[repro] LLM constructed OK", flush=True)
o = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(max_tokens=8, temperature=0.0),
    use_tqdm=False,
)
print("[repro] generate OK:", repr(o[0].outputs[0].text), flush=True)
