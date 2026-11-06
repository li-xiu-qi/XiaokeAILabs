"""Recheck padded versus trimmed batches without training or writing results.

Requires the patched author training script produced by run_single_gpu.sh.
"""
import argparse
import hashlib
import importlib.util
import json

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, help="Patched author script")
    parser.add_argument("--model", required=True, help="Local base model directory")
    parser.add_argument("--data", required=True, help="Generated train.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--first-layer-check", action="store_true", help="Compare first-layer attention to FP32 math")
    args = parser.parse_args()

    if args.device == "cpu" and not torch.cuda.is_available():
        # The author's preprocessing checks the CUDA rank even on CPU.
        torch.cuda.current_device = lambda: 0
    torch.set_num_threads(4)
    set_seed(42)
    spec = importlib.util.spec_from_file_location("jepa_author", args.trainer)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    special = [f"<|predictor_{i}|>" for i in range(1, 11)] + [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": [s for s in special if s not in tokenizer.vocab]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, use_cache=False,
        device_map={"": 0} if args.device == "cuda" else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config).eval()
    dataset = author.load_and_prepare_dataset(
        args.data, tokenizer, args.model, max_length=128, regular=True,
        unmask_assistant_special_tokens=True,
    )
    for attention in ("sdpa", "eager"):
        model.set_attn_implementation(attention)
        for indices in ((132, 7), (132,), (7,), (34, 85)):
            batch = default_data_collator([dataset[i] for i in indices])
            for trim in (False, True):
                length = int(batch["attention_mask"].sum(-1).max()) if trim else 128
                inputs = {k: v[:, :length].to(args.device) for k, v in batch.items()}
                with torch.no_grad():
                    loss = model(**inputs).loss.item()
                print(json.dumps({
                    "device": args.device, "attention": attention, "indices": indices,
                    "length": length, "loss": loss,
                    "input_sha256": hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes()).hexdigest(),
                    "labels_sha256": hashlib.sha256(inputs["labels"].cpu().numpy().tobytes()).hexdigest(),
                }), flush=True)

    if args.first_layer_check:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        import torch.nn.functional as F

        model.set_attn_implementation("sdpa")
        layer = model.base_model.model.model.layers[0].self_attn
        original, captured = layer.forward, {}

        def capture(*positional, **kwargs):
            if not captured:
                captured.update(
                    hidden=kwargs["hidden_states"].detach(),
                    positions=tuple(p.detach() for p in kwargs["position_embeddings"]),
                    mask=kwargs["attention_mask"].detach(),
                )
            return original(*positional, **kwargs)

        layer.forward = capture
        batch = default_data_collator([dataset[34], dataset[85]])
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            model(**inputs)
            hidden, mask = captured["hidden"], captured["mask"]
            shape = (*hidden.shape[:-1], -1, layer.head_dim)
            q = layer.q_proj(hidden).view(shape).transpose(1, 2)
            k = layer.k_proj(hidden).view(shape).transpose(1, 2)
            v = layer.v_proj(hidden).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *captured["positions"])
            k = repeat_kv(k, layer.num_key_value_groups)
            v = repeat_kv(v, layer.num_key_value_groups)
            scores = (q.float() @ k.float().transpose(-2, -1)) * layer.scaling
            scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
            reference = F.softmax(scores, dim=-1) @ v.float()
            sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.scaling, dropout_p=0.0)
            scores_bf16 = (q @ k.transpose(-2, -1)) * layer.scaling
            scores_bf16 = scores_bf16.masked_fill(~mask, torch.finfo(q.dtype).min)
            manual_bf16 = F.softmax(scores_bf16, dim=-1, dtype=torch.float32).to(q.dtype) @ v
            for name, result in (("sdpa_bf16", sdpa), ("manual_bf16", manual_bf16)):
                error = (result[:, :, :32].float() - reference[:, :, :32]).abs()
                print(json.dumps({"first_layer": name, "mean_abs_error": error.mean().item(),
                                  "max_abs_error": error.max().item()}), flush=True)


if __name__ == "__main__":
    main()
