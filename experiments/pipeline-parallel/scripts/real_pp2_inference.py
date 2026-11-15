"""双机推理原型：每个节点先加载完整模型，再尝试切分。

加载的模型可执行其自带的远程代码；只使用可信来源。该脚本不能证明
单节点无法容纳的模型可以分片加载；其他模型的运行效果和正确性需要单独验证。
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Real Cross-Machine Dual-GPU Pipeline Parallel Inference (PP=2)")
    parser.add_argument("--role", type=str, required=True, choices=["baseline", "stage1", "stage2"])
    parser.add_argument("--model-path", type=str, required=True, help="Path to local HuggingFace/ModelScope model")
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--master-port", type=str, default="29505")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--prompt", type=str, default="在跨机分布式流水线并行推理中，双卡协同的优势在于")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="JSON result path; fails if the file already exists")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.role != "baseline" and not args.master_addr:
        parser.error("--master-addr is required for stage1 and stage2")
    if args.role == "baseline" and args.temperature != 0:
        parser.error("baseline comparison requires --temperature 0")
    if args.out and Path(args.out).exists():
        parser.error("--out already exists; choose a new result path")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def save_result(input_ids, generated_ids):
        result = {
            "role": args.role,
            "backend": args.backend if args.role != "baseline" else None,
            "prompt": args.prompt,
            "input_ids": input_ids,
            "generated_ids": generated_ids,
            "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "torch_version": torch.__version__,
        }
        print("\nRESULT " + json.dumps(result, ensure_ascii=False))
        if args.out:
            with open(args.out, "x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")

    if args.role == "baseline":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device).eval()
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        prompt_ids = input_ids[0].tolist()
        generated_ids = []
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids=input_ids, use_cache=False).logits
                token = int(torch.argmax(logits[0, -1, :].float()).item())
            generated_ids.append(token)
            input_ids = torch.cat((input_ids, input_ids.new_tensor([[token]])), dim=1)
            if token == tokenizer.eos_token_id:
                break
        save_result(prompt_ids, generated_ids)
        return

    # 1. 网络与进程组初始化；NCCL 使用 GPU 张量，Gloo 使用 CPU 张量。
    rank = 0 if args.role == "stage1" else 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.backend == "nccl" and device.type != "cuda":
        parser.error("NCCL requires CUDA on both stages")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    comm_device = device if args.backend == "nccl" else torch.device("cpu")

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    world_size = 2

    print(f"[{args.role.upper()}] Initializing {args.backend} distributed process group (rank={rank}, master={args.master_addr}:{args.master_port})...")
    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)
    print(f"[{args.role.upper()}] Process group connected successfully!")

    print(f"[{args.role.upper()}] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. 分词器与完整模型加载
    print(f"[{args.role.upper()}] Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    full_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)
    full_model.eval()

    total_layers = len(full_model.model.layers)
    half_layers = total_layers // 2
    hidden_dim = full_model.config.hidden_size
    print(f"[{args.role.upper()}] Total layers: {total_layers}. Split point: {half_layers}. Hidden dim: {hidden_dim}")

    # 3. 物理切分流水线 Stage
    if args.role == "stage1":
        # Stage 1: 保留 layers 0 .. half-1，将 norm 替换为 Identity 从而直接输出中间层隐状态
        full_model.model.layers = full_model.model.layers[:half_layers]
        full_model.model.norm = nn.Identity()
        del full_model.lm_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[STAGE 1] Active layers: 0 .. {half_layers - 1}. Posterior layers released.")
    else:
        # Stage 2: 保留 layers half .. N-1，保持 norm 与 lm_head
        full_model.model.layers = full_model.model.layers[half_layers:]
        del full_model.model.embed_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[STAGE 2] Active layers: {half_layers} .. {total_layers - 1}, Norm, and LM Head.")

    # 4. 执行真实端到端推理循环
    if args.role == "stage1":
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        print(f"\n=======================================================")
        print(f"[STAGE 1] Prompt: \"{args.prompt}\" ({prompt_len} tokens)")
        print(f"[STAGE 1] Real Dual-Machine PP=2 Generation Started:")
        print(f"=======================================================\n")
        sys.stdout.write(args.prompt)
        sys.stdout.flush()

        generated_ids = input_ids[0].tolist()
        total_gen_time = 0.0

        for step in range(args.max_new_tokens):
            step_start = time.perf_counter()
            current_input = torch.tensor([generated_ids], device=device)
            seq_len = current_input.shape[1]

            # Stage 1 前半段 Transformer 推理
            with torch.no_grad():
                out_stage1 = full_model.model(current_input, use_cache=False)
                h_mid = out_stage1.last_hidden_state # [1, seq_len, hidden_dim]

            # 发送序列长度与激活张量到 Stage 2
            h_comm = h_mid.to(device=comm_device, dtype=torch.float32).contiguous()
            
            # 1. 发送头信息 [seq_len]
            header = torch.tensor([seq_len], dtype=torch.long, device=comm_device)
            dist.send(header, dst=1)
            
            # 2. 发送中间激活张量
            dist.send(h_comm, dst=1)

            # 3. 等待 Stage 2 传回下一个生成的 token_id
            recv_token_buf = torch.zeros(1, dtype=torch.long, device=comm_device)
            dist.recv(recv_token_buf, src=1)

            next_token = recv_token_buf.item()
            generated_ids.append(next_token)
            step_time = time.perf_counter() - step_start
            total_gen_time += step_time

            word = tokenizer.decode([next_token], skip_special_tokens=True)
            sys.stdout.write(word)
            sys.stdout.flush()

            if next_token == tokenizer.eos_token_id:
                break

        gen_count = len(generated_ids) - prompt_len
        print(f"\n\n=======================================================")
        print(f"[STAGE 1] Real Inference Completed!")
        print(f"[STAGE 1] Generated: {gen_count} tokens in {total_gen_time:.3f} s")
        print(f"[STAGE 1] Real Dual-Machine End-to-End Throughput: {gen_count / total_gen_time:.2f} tokens/s")
        print(f"=======================================================")
        save_result(generated_ids[:prompt_len], generated_ids[prompt_len:])

    else:
        # Stage 2: 接收激活值并执行后半段 Transformer 与采样
        print(f"[STAGE 2] Waiting for activation tensors from Stage 1...")
        for step in range(args.max_new_tokens):
            # 1. 接收头信息 [seq_len]
            header = torch.zeros(1, dtype=torch.long, device=comm_device)
            dist.recv(header, src=0)
            seq_len = header.item()

            # 2. 接收中间激活张量 [1, seq_len, hidden_dim]
            act_buf = torch.zeros(1, seq_len, hidden_dim, dtype=torch.float32, device=comm_device)
            dist.recv(act_buf, src=0)
            
            h_in = act_buf.to(device).bfloat16()

            with torch.no_grad():
                # 后半段 Transformer 推理 (inputs_embeds 直接接入)
                out_stage2 = full_model.model(inputs_embeds=h_in, use_cache=False)
                h_norm = out_stage2.last_hidden_state
                logits = full_model.lm_head(h_norm) # [1, seq_len, vocab_size]

                # 采样最后一个 token
                last_logits = logits[0, -1, :].float()
                if args.temperature == 0.0:
                    next_token = torch.argmax(last_logits).unsqueeze(0)
                else:
                    probs = torch.softmax(last_logits / args.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            # 3. 将采样的 token_id 回传 Stage 1
            send_buf = next_token.to(comm_device)
            dist.send(send_buf, dst=0)

            token_id = next_token.item()
            word = tokenizer.decode([token_id], skip_special_tokens=True)
            sys.stdout.write(word)
            sys.stdout.flush()

            if token_id == tokenizer.eos_token_id:
                break

        print(f"\n[STAGE 2] Finished serving pipeline generation.")

    dist.destroy_process_group()
    print(f"[{args.role.upper()}] Done!")

if __name__ == "__main__":
    main()
