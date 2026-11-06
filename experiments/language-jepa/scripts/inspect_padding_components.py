"""Compare valid-token hidden states for two lengths of the same padded batch.

Run after preparing the patched author trainer and toy data as in README.
"""
import argparse
import importlib.util
import json
import os
import threading
import time

import torch
from transformers import default_data_collator, set_seed


def available_memory():
    with open("/proc/meminfo", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def start_guard():
    def guard():
        while True:
            try:
                if available_memory() < 50 * 1024**3:
                    os._exit(85)
            except Exception:
                os._exit(86)
            time.sleep(2)
    if available_memory() < 50 * 1024**3:
        raise RuntimeError("Available system memory is below 50 GiB")
    threading.Thread(target=guard, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reduction-check", action="store_true", help="Run the local CUDA BF16 reduction comparison")
    args = parser.parse_args()
    if args.device == "cuda":
        start_guard()
    elif not torch.cuda.is_available():
        # The author's preprocessing queries the current CUDA rank.
        torch.cuda.current_device = lambda: 0
    torch.set_num_threads(4)
    set_seed(42)
    spec = importlib.util.spec_from_file_location("jepa_author", args.trainer)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    model, tokenizer = author.setup_model_and_tokenizer(
        args.model, use_lora=True, lora_rank=8, seed=42
    )
    model = model.to(args.device).eval()
    dataset = author.load_and_prepare_dataset(
        args.data, tokenizer, args.model, max_length=128, regular=True,
        unmask_assistant_special_tokens=True,
    )
    batch = {
        key: value.to(args.device)
        for key, value in default_data_collator([dataset[34], dataset[85]]).items()
    }
    lengths = batch["attention_mask"].sum(-1).tolist()
    if lengths != [32, 32]:
        raise ValueError(f"Unexpected effective lengths: {lengths}")
    layer = model.base_model.model.model.layers[0]
    modules = {
        "input_norm": layer.input_layernorm,
        "attention": layer.self_attn,
        "post_norm": layer.post_attention_layernorm,
        "mlp": layer.mlp,
        "layer": layer,
        "k_base": layer.self_attn.k_proj.base_layer,
        "k_lora_a": layer.self_attn.k_proj.lora_A["default"],
        "k_lora_b": layer.self_attn.k_proj.lora_B["default"],
        "v_base": layer.self_attn.v_proj.base_layer,
        "v_lora_a": layer.self_attn.v_proj.lora_A["default"],
        "v_lora_b": layer.self_attn.v_proj.lora_B["default"],
    }
    for attention in ("sdpa", "eager"):
        model.set_attn_implementation(attention)
        runs = {}
        for name, length in (("padded", 128), ("trimmed", 32)):
            captured, positions, handles = {}, {}, []
            for key, module in modules.items():
                def record(_module, _args, output, key=key):
                    value = output[0] if isinstance(output, tuple) else output
                    captured[key] = value[:, :32].detach().float().cpu()
                handles.append(module.register_forward_hook(record))

            for projection in ("k", "v"):
                proj = getattr(layer.self_attn, projection + "_proj")

                def reference(module, inputs, projection=projection):
                    x = inputs[0][:, :32].detach().float()
                    base = module.base_layer
                    bias = base.bias.detach().float() if base.bias is not None else None
                    captured[projection + "_input"] = x.cpu()
                    captured[projection + "_fp32_reference"] = torch.nn.functional.linear(
                        x, base.weight.detach().float(), bias
                    ).cpu()

                handles.append(proj.register_forward_pre_hook(reference))

            def record_positions(_module, _args, kwargs):
                cos, sin = kwargs["position_embeddings"]
                positions["cos"] = cos[:, :32].detach().float().cpu()
                positions["sin"] = sin[:, :32].detach().float().cpu()

            handles.append(layer.self_attn.register_forward_pre_hook(
                record_positions, with_kwargs=True
            ))
            try:
                with torch.inference_mode():
                    output = model(
                        **{key: value[:, :length] for key, value in batch.items()},
                        output_hidden_states=True,
                    )
                runs[name] = {
                    "loss": output.loss.item(),
                    "logits": output.logits[:, :32].detach().float().cpu(),
                    "layers": [x[:, :32].detach().float().cpu()
                               for x in output.hidden_states],
                    "modules": captured,
                    "positions": positions,
                }
            finally:
                for handle in handles:
                    handle.remove()
        print(json.dumps({"device": args.device, "attention": attention,
                          "loss_padded": runs["padded"]["loss"],
                          "loss_trimmed": runs["trimmed"]["loss"]}), flush=True)
        for name in ("padded", "trimmed"):
            for projection in ("k", "v"):
                captured = runs[name]["modules"]
                diff = (captured[projection + "_base"] -
                        captured[projection + "_fp32_reference"]).abs()
                print(json.dumps({"attention": attention, "run": name,
                                  "projection_reference": projection,
                                  "max": diff.max().item(),
                                  "mean": diff.mean().item()}), flush=True)
        for category in ("modules", "positions", "layers", "logits"):
            lhs, rhs = runs["padded"][category], runs["trimmed"][category]
            if category == "logits":
                pairs = [("logits", lhs, rhs)]
            elif category == "layers":
                pairs = [(str(i), x, y) for i, (x, y) in enumerate(zip(lhs, rhs))]
            else:
                pairs = [(key, lhs[key], rhs[key]) for key in lhs]
            for key, x, y in pairs:
                diff = (x - y).abs()
                print(json.dumps({"attention": attention, "category": category,
                                  "name": key, "max": diff.max().item(),
                                  "mean": diff.mean().item()}), flush=True)


    if args.reduction_check:
        if args.device != "cuda":
            raise ValueError("--reduction-check requires --device cuda")
        projection = layer.self_attn.k_proj
        saved = {}

        def capture_k(module, inputs):
            saved["input"] = inputs[0].detach().clone()
            saved["weight"] = module.base_layer.weight.detach().clone()
            saved["bias"] = module.base_layer.bias.detach().clone()

        handle = projection.register_forward_pre_hook(capture_k)
        try:
            with torch.inference_mode():
                model(**batch)
        finally:
            handle.remove()
        x, weight, bias = saved["input"], saved["weight"], saved["bias"]
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                reference = torch.nn.functional.linear(
                    x[:, :32].float(), weight.float(), bias.float()
                )
                print(json.dumps({"reduction_check": "environment",
                                  "torch": torch.__version__,
                                  "device": torch.cuda.get_device_name(),
                                  "tf32": False}), flush=True)
                for reduced in (True, False):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
                    results = {}
                    for length in (128, 32):
                        result = torch.nn.functional.linear(
                            x[:, :length], weight, bias
                        )[:, :32].float()
                        results[length] = result
                        error = (result - reference).abs()
                        print(json.dumps({"reduced_precision_reduction": reduced,
                                          "length": length, "mean_error": error.mean().item(),
                                          "max_error": error.max().item()}), flush=True)
                    difference = (results[128] - results[32]).abs()
                    print(json.dumps({"reduced_precision_reduction": reduced,
                                      "pair_mean": difference.mean().item(),
                                      "pair_max": difference.max().item()}), flush=True)
                for length in (128, 32):
                    result = torch.nn.functional.linear(
                        x[:, :length].float(), weight.float(), bias.float()
                    )[:, :32]
                    difference = (result - reference).abs()
                    print(json.dumps({"fp32_length": length,
                                      "pair_mean": difference.mean().item(),
                                      "pair_max": difference.max().item()}), flush=True)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction

if __name__ == "__main__":
    main()
