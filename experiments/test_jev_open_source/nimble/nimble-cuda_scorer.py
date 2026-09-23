"""MLX 版独立候选打分器的 CUDA 与 ROCm 对应实现。

prompt 构造与 MLX 版共用；只把候选输出行以 FP32 投影出来。
checkpoint 权重保持加载时的 dtype，默认 BF16；不生成文本，也不拟合概率。
checkpoint 放不进 GPU 时，可借助 accelerate device map 从主机内存流式取权重。
"""

import hashlib
import json
import math
import time
from collections import Counter

import torch

from nimble.scoring.parallel_schema import choice_key, prepare_prompts

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
HOST_MEMORY_BUDGET = "512GiB"


def physical_weight(module):
    """取出模块的真实权重，包括被 accelerate offload 到主机内存的情形。

    offload 掉的参数是 meta tensor，值存在 hook 的 weights map 里，由 forward hook 按需流到 GPU。
    候选投影若直接读 head 的权重，会绕过这些 hook。
    """
    weight = module.weight
    if weight.device.type != "meta":
        return weight
    weights = getattr(getattr(module, "_hf_hook", None), "weights_map", None)
    try:
        return weights["weight"]
    except (TypeError, KeyError):
        raise RuntimeError("Output head is on the meta device without an offload weights map") from None


def execution_device(module):
    """一个模块在哪运行：有 device map 时取其 hook 声明的设备，否则取其权重的设备。"""
    device = getattr(getattr(module, "_hf_hook", None), "execution_device", None)
    return torch.device(device) if device is not None else module.weight.device


def candidate_projection(hidden, weight, token_ids):
    """只把候选行以 FP32 投影出来，落在 hidden state 所在设备上。"""
    indices = torch.tensor(token_ids, device=weight.device)
    rows = weight.index_select(0, indices).to(hidden.device)
    return hidden.float() @ rows.float().T


class CudaCandidateScorer:
    def __init__(self, model_path, model_id, revision, max_input_tokens=4096, temperature=1.0,
                 device_map=None, max_gpu_memory=None, dtype="bfloat16", attention=None):
        if not torch.cuda.is_available():
            raise RuntimeError("This runner requires a CUDA or ROCm GPU")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(DTYPES)}")
        dtype = DTYPES[dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU reports no BF16 support; pass dtype=float16 to override")
        if not isinstance(max_input_tokens, int) or max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if attention not in (None, "sdpa", "eager"):
            raise ValueError("attention must be sdpa or eager")
        if max_gpu_memory is not None and device_map is None:
            raise ValueError("max_gpu_memory requires a device_map")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

        # 避免 TF32 在小规模 FP32 候选投影上引入舍入误差。
        torch.backends.cuda.matmul.allow_tf32 = False
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if config.model_type not in {"qwen3_5", "gemma3_text"}:
            raise ValueError(f"Unsupported architecture: {config.model_type}")
        cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else AutoModelForCausalLM
        # Torch 2.8 优化过的 CUDA SDPA 在 prompt 越过 512 token 滑窗后，
        # 曾对 Gemma 给出错误输出。Eager 与 CPU 及纯数学 SDPA 的检查结果一致，
        # Gemma 保留这条已验证的路径。
        attention = attention or ("eager" if config.model_type == "gemma3_text" else "sdpa")
        loading_options = dict(local_files_only=True, dtype=dtype, attn_implementation=attention,
                               output_loading_info=True)
        if device_map is not None:
            loading_options["device_map"] = device_map
            if max_gpu_memory is not None:
                # 超出 GPU 预算的权重留在主机内存，通过 hook 流入。
                loading_options["max_memory"] = {index: max_gpu_memory for index in range(torch.cuda.device_count())}
                loading_options["max_memory"]["cpu"] = HOST_MEMORY_BUDGET
        self.model, loading = cls.from_pretrained(model_path, **loading_options)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint did not load exactly: {loading}")
        self.model = self.model.eval()
        if device_map is None:
            self.model = self.model.to("cuda")
        elif not any(p.device.type == "cuda" for p in self.model.parameters() if p.device.type != "meta"):
            raise ValueError("device_map placed no weights on a CUDA device")
        self.backbone = self.model.model
        self.head_weight = physical_weight(self.model.get_output_embeddings())
        self.device = execution_device(self.model.get_input_embeddings())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.system_role = config.model_type != "gemma3_text"
        self.model_id, self.revision = model_id, revision
        text_config = getattr(config, "text_config", config)
        self.max_input_tokens = min(max_input_tokens, text_config.max_position_embeddings - 1)
        self.temperature = temperature
        placement = Counter(str(d) for d in getattr(self.model, "hf_device_map", {}).values())
        properties = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
        self.runtime = {
            "dtype": str(dtype).removeprefix("torch."), "attention": attention,
            "device_map": device_map if device_map is None or isinstance(device_map, str) else "custom",
            "max_gpu_memory": max_gpu_memory, "placement": dict(placement) or None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
            "gpus": [p.name for p in properties],
        }

    def prepare(self, context, schema):
        return prepare_prompts(self.tokenizer, context, schema, self.max_input_tokens,
                               system_role=self.system_role)

    @torch.inference_mode()
    def score(self, context, schema, mode="independent"):
        if mode != "independent":
            raise ValueError("CUDA runner currently supports independent full-prompt prefill only")
        prepared = self.prepare(context, schema)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fields, output = {}, {}
        for name, choices, ids, candidates in zip(prepared.names, prepared.choices,
                                                   prepared.full_ids, prepared.candidate_ids):
            tokens = torch.tensor([ids], device=self.device)
            hidden = self.backbone(input_ids=tokens, use_cache=False).last_hidden_state[:, -1, :]
            logits = candidate_projection(hidden, self.head_weight, candidates)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite candidate logits")
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            best = logits.argmax().item()
            keys = [choice_key(value) for value in choices]
            output[name] = choices[best]
            fields[name] = {
                "value": choices[best], "scores": dict(zip(keys, probabilities.tolist())),
                "logits": dict(zip(keys, logits.tolist())), "candidate_token_ids": candidates,
                "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices)),
                "prompt_token_count": len(ids),
                "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
            del hidden, logits, probabilities
        torch.cuda.synchronize()
        return {"model": self.model_id, "revision": self.revision, "backend": "cuda",
                "temperature": self.temperature, "temperature_fitted": False,
                "runtime": self.runtime,
                "context": context, "output": output, "fields": fields,
                "metrics": {"mode": mode, "fields": len(fields),
                            "total_seconds": time.perf_counter() - started,
                            "cuda_peak_active_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "full_vocabulary_projection": False}}
