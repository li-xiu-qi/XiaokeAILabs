"""最小复现：不跑 bench，只验证「模型能加载 + 能出一次 token」。

用途是隔离问题，不是测性能。环境刚装好时先跑它，确认权重、插件、kernel 编译这条链
本身没问题，再上 bench。正式测量用 scripts/run_bench_pair.sh。

这里的 gpu_memory_utilization=0.92 与 max_model_len=4096 是放宽后的配置，用来复现
默认参数下的 OOM 与挂死。稳定跑 bench 请用 0.85 与 2048，原因见 README「已知限制」。

启动方式见 launch_repro2.sh：带上 CUDA_LAUNCH_BLOCKING 与详细日志，
挂住时 `kill -USR1 <pid>` 让 faulthandler 打出全部线程栈。
"""
import faulthandler
import os
import signal

faulthandler.register(signal.SIGUSR1)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUTE_DSL_ARCH"] = "sm_121a"
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["VQ2_EXPERTS_DIR"] = "models/DeepSeek-V4-Flash-2bit"

from vllm import LLM, SamplingParams

print("[repro] constructing LLM ...", flush=True)
llm = LLM(
    model="models/DeepSeek-V4-Flash-2bit",
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.92,
    max_model_len=4096,
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
