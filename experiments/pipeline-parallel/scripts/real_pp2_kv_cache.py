"""未验证的双机推理原型：每个节点先加载完整模型，再尝试切分。

加载的模型可执行其自带的远程代码；只使用可信来源。该脚本不能证明
单节点无法容纳的模型可以分片加载，运行效果和正确性均待验证。
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="High-Speed Real Dual-GPU Pipeline Parallel Inference with KV Cache (PP=2)")
    parser.add_argument("--role", type=str, required=True, choices=["stage1", "stage2"])
    parser.add_argument("--model-path", type=str, required=True, help="Path to local HuggingFace/ModelScope model")
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--master-port", type=str, default="29520")
    parser.add_argument("--prompt", type=str, default="跨机流水线推理可以将模型的计算划分为")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # 1. 进程组与网络配置
    # 留给用户通过 GLOO_SOCKET_IFNAME 指定网卡；未设置时由 Gloo 选择。
    rank = 0 if args.role == "stage1" else 1

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    world_size = 2

    print(f"[{args.role.upper()}] Initializing Gloo process group (rank={rank}, master={args.master_addr}:{args.master_port})...")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"[{args.role.upper()}] Connected successfully!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{args.role.upper()}] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. 模型加载
    print(f"[{args.role.upper()}] Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    full_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    full_model.eval()

    total_layers = len(full_model.model.layers)
    half_layers = total_layers // 2
    hidden_dim = full_model.config.hidden_size

    # 3. 流水线阶段切分
    if args.role == "stage1":
        full_model.model.layers = full_model.model.layers[:half_layers]
        full_model.model.norm = nn.Identity()
        del full_model.lm_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[STAGE 1] Stage 1 initialized (Layers 0..{half_layers - 1}). Posterior layers dropped.")
    else:
        full_model.model.layers = full_model.model.layers[half_layers:]
        del full_model.model.embed_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[STAGE 2] Stage 2 initialized (Layers {half_layers}..{total_layers - 1}). Anterior layers dropped.")

    # 4. 执行带 KV-Cache 的高速流水线推理
    if args.role == "stage1":
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        print(f"\n=======================================================")
        print(f"[STAGE 1] Prompt: \"{args.prompt}\" ({prompt_len} tokens)")
        print(f"[STAGE 1] KV-Cache Pipeline Autoregressive Generation Started:")
        print(f"=======================================================\n")
        sys.stdout.write(args.prompt)
        sys.stdout.flush()

        generated_ids = []
        total_gen_time = 0.0

        # --- A. Prefill 阶段 (首字计算) ---
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            out_stage1 = full_model.model(input_ids, use_cache=True)
            h_prompt = out_stage1.last_hidden_state # [1, prompt_len, hidden_dim]
            past_kv = out_stage1.past_key_values

        # 发送 Prefill 长度与全部 prompt 激活
        header = torch.tensor([prompt_len], dtype=torch.long)
        dist.send(header, dst=1)
        dist.send(h_prompt.contiguous().cpu().float(), dst=1)

        # 接收第一个生成的 token
        token_buf = torch.zeros(1, dtype=torch.long)
        dist.recv(token_buf, src=1)
        first_token = token_buf.item()
        generated_ids.append(first_token)
        sys.stdout.write(tokenizer.decode([first_token], skip_special_tokens=True))
        sys.stdout.flush()
        t_prefill = time.perf_counter() - t_prefill_start

        # --- B. Decode 阶段 (KV-Cache 极速逐字生成) ---
        # 预分配固定尺寸传输 Buffer: [1, 1, hidden_dim]
        curr_token = first_token
        send_act_buf = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

        t_decode_start = time.perf_counter()
        decode_steps = 0

        for step in range(args.max_new_tokens - 1):
            step_t0 = time.perf_counter()
            single_input = torch.tensor([[curr_token]], device=device)

            with torch.no_grad():
                out_stage1 = full_model.model(single_input, past_key_values=past_kv, use_cache=True)
                h_step = out_stage1.last_hidden_state # [1, 1, hidden_dim]
                past_kv = out_stage1.past_key_values

            # 单消息极速定长发送 (仅 1.79 KB)
            send_act_buf.copy_(h_step)
            dist.send(send_act_buf, dst=1)

            # 接收 Stage 2 采样的 next token
            dist.recv(token_buf, src=1)
            next_token = token_buf.item()

            generated_ids.append(next_token)
            curr_token = next_token
            decode_steps += 1

            word = tokenizer.decode([next_token], skip_special_tokens=True)
            sys.stdout.write(word)
            sys.stdout.flush()

            if next_token == tokenizer.eos_token_id:
                break

        t_decode_total = time.perf_counter() - t_decode_start

        print(f"\n\n=======================================================")
        print(f"[STAGE 1] Performance Breakdown:")
        print(f"  Prefill Latency (1st token): {t_prefill * 1000:.2f} ms")
        print(f"  Decode Throughput ({decode_steps} tokens): {decode_steps / t_decode_total:.2f} tokens/s")
        print(f"  Avg Latency per token: {t_decode_total / decode_steps * 1000:.2f} ms/tok")
        print(f"  Total Generated Tokens: {len(generated_ids)}")
        print(f"=======================================================")

    else:
        # Stage 2 接收与后向推理
        print(f"[STAGE 2] Waiting for prefill activation from Stage 1...")
        
        # --- A. Prefill 阶段 ---
        header = torch.zeros(1, dtype=torch.long)
        dist.recv(header, src=0)
        prompt_len = header.item()

        act_prompt_buf = torch.zeros(1, prompt_len, hidden_dim, dtype=torch.float32)
        dist.recv(act_prompt_buf, src=0)

        with torch.no_grad():
            out_stage2 = full_model.model(inputs_embeds=act_prompt_buf.to(device).bfloat16(), use_cache=True)
            past_kv = out_stage2.past_key_values
            logits = full_model.lm_head(out_stage2.last_hidden_state)
            
            last_logits = logits[0, -1, :].float()
            if args.temperature == 0.0:
                next_token = torch.argmax(last_logits).unsqueeze(0)
            else:
                probs = torch.softmax(last_logits / args.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        token_send = next_token.cpu()
        dist.send(token_send, dst=0)

        # --- B. Decode 阶段 ---
        recv_act_buf = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
        
        for step in range(args.max_new_tokens - 1):
            dist.recv(recv_act_buf, src=0)
            h_in = recv_act_buf.to(device).bfloat16()

            with torch.no_grad():
                out_stage2 = full_model.model(inputs_embeds=h_in, past_key_values=past_kv, use_cache=True)
                past_kv = out_stage2.past_key_values
                logits = full_model.lm_head(out_stage2.last_hidden_state)

                last_logits = logits[0, -1, :].float()
                if args.temperature == 0.0:
                    next_token = torch.argmax(last_logits).unsqueeze(0)
                else:
                    probs = torch.softmax(last_logits / args.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            token_send = next_token.cpu()
            dist.send(token_send, dst=0)

            if next_token.item() == tokenizer.eos_token_id:
                break

        print(f"[STAGE 2] Finished serving pipeline generation.")

    dist.destroy_process_group()
    print(f"[{args.role.upper()}] Done!")

if __name__ == "__main__":
    main()
