"""
Real-Hardware Speculative Decoding Benchmark for MiniCPM-OCR 1.0 + MiniCPM-OCR-1.0-DSpark
=========================================================================================

Compares:
1. Baseline Mode: Standard autoregressive greedy decoding with MiniCPM-OCR-1.0 (1.3B)
2. DSpark Mode:   Block speculative decoding with MiniCPM-OCR-1.0 (1.3B target) +
                  MiniCPM-OCR-1.0-DSpark (0.3B 5-layer bidirectional block draft model +
                  low-rank Markov transition head + d2t vocabulary mapping)

Includes exact hybrid state rollback for Qwen3_5GatedDeltaNet (18 linear attention layers +
6 full attention layers), guaranteeing 100% lossless greedy token parity.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.muse_glimmer_assistant.configuration_muse_glimmer_assistant import (
    MuseGlimmerAssistantConfig,
)
from transformers.models.muse_glimmer_assistant.modeling_muse_glimmer_assistant import (
    DFlashCache,
    MuseGlimmerAssistantModel,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, l2norm


def load_dspark_drafter(dspark_dir: str, device: str = "cuda"):
    with open(Path(dspark_dir) / "config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    mg_cfg = MuseGlimmerAssistantConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_size=cfg_dict["hidden_size"],
        intermediate_size=cfg_dict["intermediate_size"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        head_dim=cfg_dict["head_dim"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        rope_parameters=cfg_dict.get("rope_scaling")
        or {"rope_theta": cfg_dict.get("rope_theta", 1000000.0), "rope_type": "default"},
        block_size=cfg_dict["block_size"],
        dflash_config=cfg_dict["dflash_config"],
        dtype=torch.bfloat16,
    )
    draft_core = MuseGlimmerAssistantModel(mg_cfg).to(device=device, dtype=torch.bfloat16).eval()
    sd = load_file(str(Path(dspark_dir) / "model.safetensors"), device=device)

    mapped_sd = {}
    for k, v in sd.items():
        if k == "fc.weight":
            mapped_sd["encoder.fc.weight"] = v
        elif k == "hidden_norm.weight":
            mapped_sd["encoder.output_norm_enc.weight"] = v
        elif k.startswith("layers.") or k == "norm.weight":
            mapped_sd[k] = v
    draft_core.load_state_dict(mapped_sd, strict=True)

    drafter_heads = {
        "lm_head_weight": sd["lm_head.weight"],
        "markov_w1": sd["markov_head.markov_w1.weight"],
        "markov_w2": sd["markov_head.markov_w2.weight"],
        "d2t": sd["d2t"],
        "mask_token_id": cfg_dict["dflash_config"]["mask_token_id"],
        "target_layer_ids": cfg_dict["dflash_config"]["target_layer_ids"],
        "block_size": cfg_dict["block_size"],
    }
    return draft_core, drafter_heads


def patch_target_for_speculative_rollback(target_model):
    """Enable per-token recurrent state snapshots on Qwen3_5GatedDeltaNet during block verification."""
    for layer in target_model.model.language_model.layers:
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            orig_forward = la.forward

            def make_spec_forward(mod, orig_fn):
                def spec_forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
                    bsz, seq_len, _ = hidden_states.shape
                    if (
                        cache_params is not None
                        and cache_params.get_recurrent_state(mod.layer_idx, 0) is not None
                        and 1 < seq_len <= 16
                    ):
                        conv_prev = cache_params.get_conv_state(mod.layer_idx, 0).clone()
                        rec_prev = cache_params.get_recurrent_state(mod.layer_idx, 0).clone()
                        mixed_qkv = mod.in_proj_qkv(hidden_states).transpose(1, 2)
                        z = mod.in_proj_z(hidden_states).reshape(bsz, seq_len, -1, mod.head_v_dim)
                        b = mod.in_proj_b(hidden_states)
                        a = mod.in_proj_a(hidden_states)
                        full_conv = torch.cat([conv_prev, mixed_qkv], dim=-1)
                        mod._spec_full_conv = full_conv
                        mixed_qkv_out = causal_conv1d_fn(
                            full_conv,
                            mod.conv1d.weight.squeeze(1),
                            mod.conv1d.bias,
                            activation=mod.activation,
                        )[:, :, -seq_len:].transpose(1, 2)
                        q, k, v = torch.split(
                            mixed_qkv_out, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1
                        )
                        q = q.reshape(bsz, seq_len, -1, mod.head_k_dim)
                        k = k.reshape(bsz, seq_len, -1, mod.head_k_dim)
                        v = v.reshape(bsz, seq_len, -1, mod.head_v_dim)
                        beta = b.sigmoid()
                        g = -mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias)
                        if mod.num_v_heads // mod.num_k_heads > 1:
                            q = q.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
                            k = k.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
                        init_dtype = q.dtype
                        q, k, v, beta, decay = [
                            x.transpose(1, 2).to(torch.float32).contiguous()
                            for x in (q, k, v, beta, g)
                        ]
                        q = l2norm(q, dim=-1, eps=1e-6) / (q.shape[-1] ** 0.5)
                        k = l2norm(k, dim=-1, eps=1e-6)
                        last_rec = rec_prev.to(v)
                        core_out = torch.zeros_like(v)
                        rec_hist = []
                        for i in range(seq_len):
                            q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
                            last_rec = last_rec * decay[:, :, i].exp()[..., None, None]
                            delta = (
                                v_t - (last_rec * k_t.unsqueeze(-1)).sum(dim=-2)
                            ) * beta[:, :, i].unsqueeze(-1)
                            last_rec = last_rec + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                            core_out[:, :, i] = (last_rec * q_t.unsqueeze(-1)).sum(dim=-2)
                            rec_hist.append(last_rec.clone())
                        mod._spec_rec_hist = rec_hist
                        core_out = core_out.transpose(1, 2).contiguous().to(init_dtype)
                        core_out = mod.norm(
                            core_out.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)
                        ).reshape(bsz, seq_len, -1)
                        return mod.out_proj(core_out)
                    return orig_fn(
                        hidden_states,
                        cache_params=cache_params,
                        attention_mask=attention_mask,
                        **kwargs,
                    )

                return spec_forward

            la.forward = make_spec_forward(la, orig_forward)


def rollback_target_cache(target_model, cache, num_kept_in_block: int, block_len: int):
    remove_n = block_len - num_kept_in_block
    for layer_idx, layer_cache in enumerate(cache.layers):
        if hasattr(layer_cache, "keys") and layer_cache.keys is not None and layer_cache.keys.numel() > 0:
            if remove_n > 0:
                layer_cache.keys = layer_cache.keys[:, :, :-remove_n, :]
                layer_cache.values = layer_cache.values[:, :, :-remove_n, :]
        if hasattr(layer_cache, "conv_states") and layer_cache.is_conv_states_initialized.get(0, False):
            la_mod = target_model.model.language_model.layers[layer_idx].linear_attn
            ksize = layer_cache.conv_kernel_size[0]
            full_conv = la_mod._spec_full_conv
            kept_conv = full_conv[:, :, : full_conv.shape[-1] - remove_n]
            layer_cache.conv_states[0].copy_(kept_conv[:, :, -ksize:])
            layer_cache.recurrent_states[0].copy_(la_mod._spec_rec_hist[num_kept_in_block - 1])


@torch.no_grad()
def run_baseline_greedy(target_model, inputs, max_new_tokens: int, eos_id: int):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = target_model(**inputs, use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    tokens = [next_tok.item()]
    target_steps = 1

    for _ in range(max_new_tokens - 1):
        if tokens[-1] == eos_id:
            break
        out = target_model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        tokens.append(next_tok.item())
        target_steps += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "elapsed_s": elapsed,
        "tokens_per_s": len(tokens) / elapsed,
        "ms_per_token": (elapsed * 1000.0) / len(tokens),
        "target_forwards": target_steps,
    }


@torch.no_grad()
def run_dspark_speculative(
    target_model, draft_core, drafter_heads, inputs, max_new_tokens: int, eos_id: int
):
    lm_head_weight = drafter_heads["lm_head_weight"]
    markov_w1 = drafter_heads["markov_w1"]
    markov_w2 = drafter_heads["markov_w2"]
    d2t = drafter_heads["d2t"]
    mask_token_id = drafter_heads["mask_token_id"]
    target_layer_ids = drafter_heads["target_layer_ids"]
    block_size = drafter_heads["block_size"]
    device = inputs["input_ids"].device

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_spec = target_model(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out_spec.past_key_values
    anchor_tok = out_spec.logits[:, -1, :].argmax(dim=-1).item()
    spec_tokens = [anchor_tok]

    d_cache = DFlashCache()
    ctx_len = inputs["input_ids"].shape[1]
    ctx_hs = torch.cat([out_spec.hidden_states[i + 1] for i in target_layer_ids], dim=-1)
    cur_pos = ctx_len
    target_steps = 1
    draft_steps = 0
    accepted_counts = []

    while len(spec_tokens) < max_new_tokens and spec_tokens[-1] != eos_id:
        noise_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        noise_ids[0, 0] = anchor_tok
        noise_embeds = target_model.model.language_model.embed_tokens(noise_ids)
        pos_ids = torch.arange(
            cur_pos - ctx_hs.shape[1], cur_pos + block_size, device=device
        ).unsqueeze(0)
        d_out = draft_core(
            noise_embeds=noise_embeds,
            context_hidden_states=ctx_hs,
            position_ids=pos_ids,
            past_key_values=d_cache,
            use_cache=True,
        )
        for d_layer in d_cache.layers:
            d_layer.keys = d_layer.keys[:, :, :-block_size, :]
            d_layer.values = d_layer.values[:, :, :-block_size, :]

        base_logits = F.linear(d_out.last_hidden_state[0], lm_head_weight)
        draft_cands = []
        prev_t = anchor_tok
        for j in range(block_size):
            m_bias = F.linear(markov_w1[prev_t], markov_w2)
            d_id = (base_logits[j] + m_bias).argmax(dim=-1)
            t_id = (d_id + d2t[d_id]).item()
            draft_cands.append(t_id)
            prev_t = t_id
        draft_steps += 1

        verify_list = [anchor_tok] + draft_cands[:-1]
        verify_ids = torch.tensor([verify_list], dtype=torch.long, device=device)
        v_out = target_model(
            input_ids=verify_ids,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        target_steps += 1
        v_preds = v_out.logits[0].argmax(dim=-1).tolist()

        n_accept = 0
        hit_eos = False
        for j in range(len(verify_list) - 1):
            if v_preds[j] == verify_list[j + 1]:
                n_accept += 1
                spec_tokens.append(v_preds[j])
                if v_preds[j] == eos_id:
                    hit_eos = True
                    break
                if len(spec_tokens) >= max_new_tokens:
                    break
            else:
                break

        bonus_tok = v_preds[n_accept]
        if not hit_eos and len(spec_tokens) < max_new_tokens:
            spec_tokens.append(bonus_tok)
        accepted_counts.append(n_accept + 1)

        if hit_eos or len(spec_tokens) >= max_new_tokens or spec_tokens[-1] == eos_id:
            break

        num_kept = n_accept + 1
        rollback_target_cache(target_model, past_kv, num_kept, len(verify_list))
        ctx_hs = torch.cat(
            [v_out.hidden_states[i + 1][:, :num_kept, :] for i in target_layer_ids], dim=-1
        )
        cur_pos += num_kept
        anchor_tok = bonus_tok

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "tokens": spec_tokens,
        "num_tokens": len(spec_tokens),
        "elapsed_s": elapsed,
        "tokens_per_s": len(spec_tokens) / elapsed,
        "ms_per_token": (elapsed * 1000.0) / len(spec_tokens),
        "target_forwards": target_steps,
        "draft_forwards": draft_steps,
        "avg_accept_length": sum(accepted_counts) / max(len(accepted_counts), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", default="models/MiniCPM-OCR-1.0")
    parser.add_argument("--dspark-dir", default="models/MiniCPM-OCR-1.0-DSpark")
    parser.add_argument("--testdata-dir", default="testdata")
    parser.add_argument("--examples-dir", default="repo/scripts/minicpm-ocr-jev/examples")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-json", default="docs/evidence/acceleration_benchmark.json")
    args = parser.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.target_dir, trust_remote_code=True)
    target_model = AutoModelForImageTextToText.from_pretrained(
        args.target_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
        trust_remote_code=True,
    ).eval()
    patch_target_for_speculative_rollback(target_model)

    draft_core, drafter_heads = load_dspark_drafter(args.dspark_dir, device=device)
    eos_id = processor.tokenizer.eos_token_id

    test_cases = [
        {
            "id": "case1_cn_document",
            "name": "Chinese Document Page (Text + Table + Formula)",
            "image_path": str(Path(args.testdata_dir) / "testpage_cn.jpg"),
            "crop": (0, 0, 1200, 950),
            "prompt": "Text Recognition:",
            "max_new_tokens": args.max_new_tokens,
        },
        {
            "id": "case2_text_putnam",
            "name": "Mathematical Competition Proof Text (01_text_putnam)",
            "image_path": str(Path(args.examples_dir) / "01_text_putnam" / "block.png"),
            "crop": None,
            "prompt": "Text Recognition:",
            "max_new_tokens": args.max_new_tokens,
        },
        {
            "id": "case3_formula_putnam",
            "name": "Multi-line Mathematical Formula Block (02_formula_putnam)",
            "image_path": str(Path(args.examples_dir) / "02_formula_putnam" / "block.png"),
            "crop": None,
            "prompt": "Formula Recognition:",
            "max_new_tokens": args.max_new_tokens,
        },
        {
            "id": "case4_table_gb_standard",
            "name": "Chinese National Standard Structured Table (03_table_gb_standard)",
            "image_path": str(Path(args.examples_dir) / "03_table_gb_standard" / "block.png"),
            "crop": None,
            "prompt": "Table Recognition:",
            "max_new_tokens": args.max_new_tokens,
        },
    ]

    warm_img = Image.open(test_cases[0]["image_path"]).convert("RGB").crop((0, 0, 512, 512))
    warm_msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": warm_img},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    warm_txt = processor.apply_chat_template(warm_msg, tokenize=False, add_generation_prompt=True)
    warm_in = processor(text=[warm_txt], images=[warm_img], return_tensors="pt").to(device)
    _ = run_baseline_greedy(target_model, warm_in, 16, eos_id)
    _ = run_dspark_speculative(target_model, draft_core, drafter_heads, warm_in, 16, eos_id)
    torch.cuda.reset_peak_memory_stats()

    results = []
    tot_base_tok = 0
    tot_base_time = 0.0
    tot_base_steps = 0
    tot_spec_tok = 0
    tot_spec_time = 0.0
    tot_spec_steps = 0
    tot_draft_steps = 0
    all_accept_lens = []

    for tc in test_cases:
        img = Image.open(tc["image_path"]).convert("RGB")
        if tc["crop"] is not None:
            img = img.crop(tc["crop"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": tc["prompt"]},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

        base_res = run_baseline_greedy(target_model, inputs, tc["max_new_tokens"], eos_id)
        spec_res = run_dspark_speculative(
            target_model, draft_core, drafter_heads, inputs, len(base_res["tokens"]), eos_id
        )
        spec_tokens = spec_res["tokens"][: len(base_res["tokens"])]
        parity = spec_tokens == base_res["tokens"]
        speedup = base_res["elapsed_s"] / spec_res["elapsed_s"]

        tot_base_tok += base_res["num_tokens"]
        tot_base_time += base_res["elapsed_s"]
        tot_base_steps += base_res["target_forwards"]
        tot_spec_tok += len(spec_tokens)
        tot_spec_time += spec_res["elapsed_s"]
        tot_spec_steps += spec_res["target_forwards"]
        tot_draft_steps += spec_res["draft_forwards"]
        all_accept_lens.append(spec_res["avg_accept_length"])

        case_summary = {
            "id": tc["id"],
            "name": tc["name"],
            "prompt": tc["prompt"],
            "num_tokens": base_res["num_tokens"],
            "baseline_elapsed_s": round(base_res["elapsed_s"], 4),
            "baseline_tok_per_s": round(base_res["tokens_per_s"], 2),
            "baseline_ms_per_tok": round(base_res["ms_per_token"], 2),
            "baseline_target_forwards": base_res["target_forwards"],
            "dspark_elapsed_s": round(spec_res["elapsed_s"], 4),
            "dspark_tok_per_s": round(len(spec_tokens) / spec_res["elapsed_s"], 2),
            "dspark_ms_per_tok": round((spec_res["elapsed_s"] * 1000.0) / len(spec_tokens), 2),
            "dspark_target_forwards": spec_res["target_forwards"],
            "dspark_draft_forwards": spec_res["draft_forwards"],
            "dspark_avg_accept_length": round(spec_res["avg_accept_length"], 2),
            "speedup": round(speedup, 2),
            "token_parity": parity,
            "decoded_preview": processor.tokenizer.decode(spec_tokens[:40]),
        }
        results.append(case_summary)
        print(
            f"[{tc['id']}] {base_res['num_tokens']} tok | "
            f"Base: {base_res['elapsed_s']:.3f}s ({base_res['tokens_per_s']:.2f} tok/s) | "
            f"DSpark: {spec_res['elapsed_s']:.3f}s ({case_summary['dspark_tok_per_s']:.2f} tok/s) | "
            f"Speedup: {speedup:.2f}x | AvgAccept: {spec_res['avg_accept_length']:.2f} | Parity: {parity}"
        )

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    overall = {
        "hardware": torch.cuda.get_device_name(0),
        "arch": platform.machine(),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "block_size": drafter_heads["block_size"],
        "target_layer_ids": drafter_heads["target_layer_ids"],
        "total_tokens": tot_base_tok,
        "baseline_total_s": round(tot_base_time, 4),
        "baseline_avg_tok_per_s": round(tot_base_tok / tot_base_time, 2),
        "baseline_avg_ms_per_tok": round((tot_base_time * 1000.0) / tot_base_tok, 2),
        "baseline_total_target_forwards": tot_base_steps,
        "dspark_total_s": round(tot_spec_time, 4),
        "dspark_avg_tok_per_s": round(tot_spec_tok / tot_spec_time, 2),
        "dspark_avg_ms_per_tok": round((tot_spec_time * 1000.0) / tot_spec_tok, 2),
        "dspark_total_target_forwards": tot_spec_steps,
        "dspark_total_draft_forwards": tot_draft_steps,
        "dspark_mean_accept_length": round(sum(all_accept_lens) / len(all_accept_lens), 2),
        "overall_speedup": round(tot_base_time / tot_spec_time, 2),
        "all_cases_token_parity": all(r["token_parity"] for r in results),
        "peak_vram_gib": round(peak_vram_gib, 2),
        "cases": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print("\n=== OVERALL SUMMARY ===")
    print(json.dumps({k: v for k, v in overall.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
