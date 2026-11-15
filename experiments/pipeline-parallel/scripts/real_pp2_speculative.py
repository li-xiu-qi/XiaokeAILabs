"""保留供审阅的错误原型，不能用于生成或验证投机解码性能。"""

if __name__ == "__main__":
    raise SystemExit(
        "此原型未实现投机解码的接受校验及回滚，不能产生有效的连续生成或性能结论；"
        "已停止运行。源码仅供检查。"
    )

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="MTP-2 / Speculative Multi-Token Pipeline Inference (PP=2)")
    parser.add_argument("--role", type=str, required=True, choices=["stage1", "stage2"])
    parser.add_argument("--model-path", type=str, required=True, help="Path to local HuggingFace/ModelScope model")
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--master-port", type=str, default="29535")
    parser.add_argument("--prompt", type=str, default="开启MTP多Token预测与流水线并行后，推理吞吐的物理瓶颈被彻底打破")
    parser.add_argument("--rounds", type=int, default=25, help="Number of multi-token speculative rounds")
    parser.add_argument("--draft-k", type=int, default=2, help="MTP token prediction depth per round")
    args = parser.parse_args()

    # 1. 进程组配置
    # 留给用户通过 GLOO_SOCKET_IFNAME 指定网卡；未设置时由 Gloo 选择。
    rank = 0 if args.role == "stage1" else 1

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    world_size = 2

    print(f"[{args.role.upper()}] Initializing Gloo process group (rank={rank})...")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    if args.role == "stage1":
        full_model.model.layers = full_model.model.layers[:half_layers]
        full_model.model.norm = nn.Identity()
        del full_model.lm_head
    else:
        full_model.model.layers = full_model.model.layers[half_layers:]
        del full_model.model.embed_tokens

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # MTP-2 推理逻辑
    K = args.draft_k

    if args.role == "stage1":
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        print(f"\n=======================================================")
        print(f"[STAGE 1] Prompt: \"{args.prompt}\"")
        print(f"[STAGE 1] MTP-{K} Multi-Token Parallel Generation Started:")
        print(f"=======================================================\n")
        sys.stdout.write(args.prompt)
        sys.stdout.flush()

        # Prefill
        with torch.no_grad():
            out1 = full_model.model(input_ids, use_cache=True)
            past_kv = out1.past_key_values
            h_prompt = out1.last_hidden_state

        dist.send(torch.tensor([prompt_len, K], dtype=torch.long), dst=1)
        dist.send(h_prompt.contiguous().cpu().float(), dst=1)

        recv_tokens = torch.zeros(K, dtype=torch.long)
        dist.recv(recv_tokens, src=1)
        
        curr_tokens = recv_tokens.tolist()
        total_tokens = list(curr_tokens)
        for t in curr_tokens:
            sys.stdout.write(tokenizer.decode([t], skip_special_tokens=True))
            sys.stdout.flush()

        t_start = time.perf_counter()
        send_buf = torch.zeros(1, K, hidden_dim, dtype=torch.float32)

        for r in range(args.rounds):
            token_tensor = torch.tensor([curr_tokens], device=device)

            with torch.no_grad():
                out1 = full_model.model(token_tensor, past_key_values=past_kv, use_cache=True)
                past_kv = out1.past_key_values
                h_k = out1.last_hidden_state # [1, K, hidden_dim]

            # 单次网络往返打包传输 K 个 Token 的全部中间激活值
            send_buf.copy_(h_k)
            dist.send(send_buf, dst=1)

            # 单次网络往返同时回收 K 个已验证/生成的 Token
            dist.recv(recv_tokens, src=1)
            curr_tokens = recv_tokens.tolist()
            total_tokens.extend(curr_tokens)

            for t in curr_tokens:
                sys.stdout.write(tokenizer.decode([t], skip_special_tokens=True))
                sys.stdout.flush()

        t_total = time.perf_counter() - t_start
        gen_count = len(total_tokens)

        print(f"\n\n=======================================================")
        print(f"[STAGE 1] MTP-{K} Multi-Token Benchmark Results:")
        print(f"  Generated Tokens: {gen_count}")
        print(f"  Total Decode Time: {t_total:.3f} s")
        print(f"  Effective Throughput: {gen_count / t_total:.2f} tokens/s")
        print(f"  Effective Latency per token: {t_total / gen_count * 1000:.2f} ms/tok")
        print(f"=======================================================")

    else:
        # Stage 2
        header = torch.zeros(2, dtype=torch.long)
        dist.recv(header, src=0)
        prompt_len, K = header.tolist()

        act_prompt = torch.zeros(1, prompt_len, hidden_dim, dtype=torch.float32)
        dist.recv(act_prompt, src=0)

        with torch.no_grad():
            out2 = full_model.model(inputs_embeds=act_prompt.to(device).bfloat16(), use_cache=True)
            past_kv = out2.past_key_values
            logits = full_model.lm_head(out2.last_hidden_state)
            # 采用 MTP 多头或 Top-K 多字提取预测
            top_k_tokens = torch.topk(logits[0, -1, :].float(), k=K).indices

        dist.send(top_k_tokens.cpu(), dst=0)

        recv_act_buf = torch.zeros(1, K, hidden_dim, dtype=torch.float32)

        for r in range(args.rounds):
            dist.recv(recv_act_buf, src=0)
            h_in = recv_act_buf.to(device).bfloat16()

            with torch.no_grad():
                out2 = full_model.model(inputs_embeds=h_in, past_key_values=past_kv, use_cache=True)
                past_kv = out2.past_key_values
                logits = full_model.lm_head(out2.last_hidden_state) # [1, K, vocab_size]
                
                # MTP 阶段：每一轮在前向中并发输出 K 个位置的最佳 Token
                next_k_tokens = torch.argmax(logits[0, :, :].float(), dim=-1)

            dist.send(next_k_tokens.cpu(), dst=0)

        print(f"[STAGE 2] MTP serving finished.")

    dist.destroy_process_group()
    print(f"[{args.role.upper()}] Done!")

if __name__ == "__main__":
    main()
