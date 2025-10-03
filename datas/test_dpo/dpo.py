from dataclasses import dataclass
import math
import os
import random
from typing import List, Dict, Any, Optional

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm


@dataclass
class Config:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    output_dir: str = "smol_dpo_output"
    seed: int = 42
    beta: float = 0.1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.05
    logging_steps: int = 1
    max_prompt_length: int = 1024
    max_total_length: int = 1536
    use_bf16: bool = True
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_final: bool = True
    push_to_hub: bool = False
    dataset_sample_size: Optional[int] = 5000
    dataset_batch_size: Optional[int] = 8
    reference_eval_batch_size: int = 8


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_triple(example: Dict[str, Any]) -> Dict[str, str]:
    chosen_conv = example.get("chosen")
    rejected_conv = example.get("rejected")
    if not chosen_conv or not rejected_conv:
        raise ValueError("数据条目缺少 chosen 或 rejected 字段")
    try:
        prompt = chosen_conv[0]["content"].strip()
        chosen_resp = chosen_conv[1]["content"].strip()
        rejected_resp = rejected_conv[1]["content"].strip()
    except Exception as e:
        raise ValueError(f"解析对话格式失败: {e}\n原始数据: {example}")
    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp}


def tokenize_batch(tokenizer: AutoTokenizer, examples: List[Dict[str, str]], max_prompt_length: int, max_total_length: int, device: torch.device):
    input_id_lists = []
    attention_mask_lists = []
    response_mask_lists = []

    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
        resp_ids = tokenizer(ex["response"], add_special_tokens=False).input_ids

        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]

        full_ids = prompt_ids + resp_ids
        if len(full_ids) > max_total_length:
            keep_resp = max_total_length - len(prompt_ids)
            if keep_resp <= 0:
                prompt_cut = max_total_length // 2
                resp_cut = max_total_length - prompt_cut
                prompt_ids = prompt_ids[-prompt_cut:]
                resp_ids = resp_ids[:resp_cut]
            else:
                resp_ids = resp_ids[:keep_resp]
            full_ids = prompt_ids + resp_ids

        response_start = len(prompt_ids)
        seq_len = len(full_ids)
        response_mask = [False] * seq_len
        for i in range(response_start, seq_len):
            response_mask[i] = True

        input_id_lists.append(full_ids)
        response_mask_lists.append(response_mask)

    max_len = max(len(x) for x in input_id_lists)
    for i in range(len(input_id_lists)):
        pad_len = max_len - len(input_id_lists[i])
        if pad_len > 0:
            input_id_lists[i] = input_id_lists[i] + [tokenizer.pad_token_id] * pad_len
            response_mask_lists[i] = response_mask_lists[i] + [False] * pad_len
    attention_mask_lists = [[1 if tok != tokenizer.pad_token_id else 0 for tok in seq] for seq in input_id_lists]

    batch = {
        "input_ids": torch.tensor(input_id_lists, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask_lists, dtype=torch.long, device=device),
        "response_mask": torch.tensor(response_mask_lists, dtype=torch.bool, device=device),
    }
    return batch


def batch_logps(model, batch) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    input_ids_next = input_ids[:, 1:]
    response_mask_next = response_mask[:, 1:]
    token_logp = log_probs.gather(dim=-1, index=input_ids_next.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp.masked_fill(~response_mask_next, 0.0)
    seq_logp = token_logp.sum(dim=-1)
    return seq_logp


def build_pair_batches(tokenizer, pairs: List[Dict[str, str]], cfg: Config, device: torch.device):
    chosen_examples = []
    rejected_examples = []
    for p in pairs:
        chosen_examples.append({"prompt": p["prompt"], "response": p["chosen"]})
        rejected_examples.append({"prompt": p["prompt"], "response": p["rejected"]})

    chosen_batch = tokenize_batch(tokenizer, chosen_examples, cfg.max_prompt_length, cfg.max_total_length, device)
    rejected_batch = tokenize_batch(tokenizer, rejected_examples, cfg.max_prompt_length, cfg.max_total_length, device)
    return chosen_batch, rejected_batch


def plot_history(history: Dict[str, List[float]], cfg: Config):
    if not history["update"]:
        print("暂无训练数据可视化")
        return
    os.makedirs(cfg.output_dir, exist_ok=True)
    update_values = history["update"]
    metric_configs = {
        "loss": {
            "title": "Loss over Updates",
            "ylabel": "Loss",
            "filename": "training_loss.png",
            "label": "Loss",
        },
        "adv": {
            "title": "Advantage over Updates",
            "ylabel": "Advantage",
            "filename": "training_advantage.png",
            "label": "Advantage",
        },
        "pref_prob": {
            "title": "Preference Probability over Updates",
            "ylabel": "Preference Probability",
            "filename": "training_pref_prob.png",
            "label": "Preference Prob.",
        },
    }
    for key, cfg_item in metric_configs.items():
        values = history.get(key)
        if not values:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(update_values, values, label=cfg_item["label"])
        plt.title(cfg_item["title"])
        plt.xlabel("Update Step")
        plt.ylabel(cfg_item["ylabel"])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(cfg.output_dir, cfg_item["filename"])
        plt.savefig(plot_path)
        plt.close()
        print(f"已保存 {cfg_item['label']} 曲线至 {plot_path}")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(cfg: Config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(cfg: Config, device: torch.device):
    print("加载 policy 模型 ...")
    policy = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32).to(device)
    policy.config.use_cache = False
    return policy


def load_reference_model(cfg: Config, device: torch.device):
    print("加载 reference 模型 (冻结参数) ...")
    reference = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False
    return reference


def precompute_reference_logps(reference, tokenizer: AutoTokenizer, triples: List[Dict[str, Any]], cfg: Config, device: torch.device):
    if not triples:
        return
    print("使用参考模型预计算得分 ...")
    eval_bs = cfg.reference_eval_batch_size if cfg.reference_eval_batch_size > 0 else get_train_batch_size(cfg)
    ref_logp_chosen_list: List[torch.Tensor] = []
    ref_logp_rejected_list: List[torch.Tensor] = []
    reference.eval()
    with torch.no_grad(), tqdm(total=len(triples), desc="参考模型预计算", unit="sample") as pbar:
        for start in range(0, len(triples), eval_bs):
            end = min(start + eval_bs, len(triples))
            batch_pairs = triples[start:end]
            chosen_batch, rejected_batch = build_pair_batches(tokenizer, batch_pairs, cfg, device)
            ref_logp_chosen = batch_logps(reference, chosen_batch)
            ref_logp_rejected = batch_logps(reference, rejected_batch)
            ref_logp_chosen_list.append(ref_logp_chosen.cpu())
            ref_logp_rejected_list.append(ref_logp_rejected.cpu())
            pbar.update(len(batch_pairs))
    all_ref_logp_chosen = torch.cat(ref_logp_chosen_list)
    all_ref_logp_rejected = torch.cat(ref_logp_rejected_list)
    for idx in range(len(triples)):
        triples[idx]["ref_logp_chosen"] = float(all_ref_logp_chosen[idx].item())
        triples[idx]["ref_logp_rejected"] = float(all_ref_logp_rejected[idx].item())
    print("参考模型预计算完成")


def get_train_batch_size(cfg: Config) -> int:
    if cfg.dataset_batch_size is not None and cfg.dataset_batch_size > 0:
        return cfg.dataset_batch_size
    return 1


def compute_total_update_steps(num_samples: int, cfg: Config) -> int:
    if num_samples <= 0:
        raise ValueError("筛选后的样本数量为 0, 无法训练")
    train_batch_size = get_train_batch_size(cfg)
    batches_per_epoch = math.ceil(num_samples / train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / cfg.gradient_accumulation_steps)
    return updates_per_epoch * cfg.num_epochs


def prepare_optimizer(policy, cfg: Config, device: torch.device, total_training_steps: int):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_steps = max(int(total_training_steps * cfg.warmup_ratio), 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    scaler_enabled = cfg.use_bf16 and (device.type == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if scaler_enabled else None
    if scaler_enabled:
        print("启用 bfloat16 自动混合精度")
    else:
        print("未启用 bfloat16 自动混合精度")
    return optimizer, scheduler, scaler_enabled, amp_dtype


def forward_dpo(policy, tokenizer, batch_pairs, cfg: Config, device: torch.device, scaler_enabled: bool, amp_dtype):
    chosen_batch, rejected_batch = build_pair_batches(tokenizer, batch_pairs, cfg, device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler_enabled):
        policy_logp_chosen = batch_logps(policy, chosen_batch)
        policy_logp_rejected = batch_logps(policy, rejected_batch)
        ref_logp_chosen = torch.tensor([pair["ref_logp_chosen"] for pair in batch_pairs], device=device)
        ref_logp_rejected = torch.tensor([pair["ref_logp_rejected"] for pair in batch_pairs], device=device)
        diff_policy = policy_logp_chosen - policy_logp_rejected
        diff_ref = ref_logp_chosen - ref_logp_rejected
        dpo_term = diff_policy - diff_ref
        losses = -F.logsigmoid(cfg.beta * dpo_term)
        loss = losses.mean() / cfg.gradient_accumulation_steps
        pref_prob = torch.sigmoid(cfg.beta * dpo_term).mean().item()
    metrics = {
        "losses": losses.detach(),
        "diff_policy": diff_policy.detach(),
        "diff_ref": diff_ref.detach(),
        "dpo_term": dpo_term.detach(),
        "pref_prob": pref_prob,
    }
    return loss, metrics


def maybe_log_metrics(cfg: Config, metrics: Dict[str, Any], history: Dict[str, List[float]], epoch: int, batch_idx: int, global_update: int, progress_bar: Optional[tqdm] = None):
    if cfg.logging_steps <= 0 or global_update % cfg.logging_steps != 0:
        return
    loss_avg = metrics["losses"].mean().item()
    diff_policy_avg = metrics["diff_policy"].mean().item()
    diff_ref_avg = metrics["diff_ref"].mean().item()
    adv_avg = metrics["dpo_term"].mean().item()
    pref_prob = metrics["pref_prob"]
    history["update"].append(global_update)
    history["loss"].append(loss_avg)
    history["adv"].append(adv_avg)
    history["pref_prob"].append(pref_prob)
    if progress_bar is not None:
        progress_bar.set_postfix({"epoch": f"{epoch + 1}/{cfg.num_epochs}", "loss": f"{loss_avg:.4f}", "adv": f"{adv_avg:.4f}", "pref": f"{pref_prob:.4f}"})
    else:
        print(f"epoch={epoch + 1}/{cfg.num_epochs} | batch={batch_idx + 1} | update={global_update} " f"loss={loss_avg:.4f} diff_policy={diff_policy_avg:.4f} diff_ref={diff_ref_avg:.4f} adv={adv_avg:.4f} pref_prob={pref_prob:.4f}")


def run_training_loop(cfg: Config, policy, tokenizer, triples: List[Dict[str, str]], optimizer, scheduler, device: torch.device, scaler_enabled: bool, amp_dtype):
    history = {"update": [], "loss": [], "adv": [], "pref_prob": []}
    policy.train()
    global_update = 0
    num_samples = len(triples)
    train_batch_size = get_train_batch_size(cfg)
    batches_per_epoch = math.ceil(num_samples / train_batch_size)

    total_batches = cfg.num_epochs * batches_per_epoch
    with tqdm(total=total_batches, desc="训练进度", unit="batch") as pbar:
        for epoch in range(cfg.num_epochs):
            random.shuffle(triples)
            optimizer.zero_grad()
            for batch_idx in range(batches_per_epoch):
                start = batch_idx * train_batch_size
                end = min(start + train_batch_size, num_samples)
                batch_pairs = triples[start:end]
                if not batch_pairs:
                    pbar.update(1)
                    continue
                loss, metrics = forward_dpo(policy, tokenizer, batch_pairs, cfg, device, scaler_enabled, amp_dtype)
                loss.backward()
                accu_idx = batch_idx % cfg.gradient_accumulation_steps
                is_update_step = (accu_idx + 1 == cfg.gradient_accumulation_steps) or (batch_idx == batches_per_epoch - 1)
                if is_update_step:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_update += 1
                    maybe_log_metrics(cfg, metrics, history, epoch, batch_idx, global_update, pbar)
                pbar.update(1)

    return history


def select_and_extract_dataset(cfg: Config):
    print("加载数据集 ...")
    raw_dataset = load_dataset(path="trl-lib/ultrafeedback_binarized", split="train")
    print(f"原始数据集大小: {len(raw_dataset)}")
    if cfg.dataset_sample_size is not None and cfg.dataset_sample_size > 0:
        select_num = min(cfg.dataset_sample_size, len(raw_dataset))
        print(f"抽取 {select_num} 条样本用于训练")
        sampled_dataset = raw_dataset.shuffle(seed=cfg.seed).select(range(select_num))
    else:
        sampled_dataset = raw_dataset
        print("使用全部样本")
    print("逐条解析样本")
    triples = [extract_triple(sampled_dataset[i]) for i in range(len(sampled_dataset))]
    return triples


def train(cfg: Config):
    set_seed(cfg.seed)
    device = select_device()
    print(f"使用设备: {device}")

    tokenizer = load_tokenizer(cfg)

    triples = select_and_extract_dataset(cfg)
    num_samples = len(triples)
    total_training_steps = compute_total_update_steps(num_samples, cfg)
    train_batch_size = get_train_batch_size(cfg)
    print(f"筛选后样本数: {num_samples}, 总更新步数: {total_training_steps}")
    print(f"训练批大小: {train_batch_size}, 梯度累积步数: {cfg.gradient_accumulation_steps}")

    reference = load_reference_model(cfg, device)
    precompute_reference_logps(reference, tokenizer, triples, cfg, device)
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("已释放参考模型显存")

    policy = load_policy_model(cfg, device)
    optimizer, scheduler, scaler_enabled, amp_dtype = prepare_optimizer(policy, cfg, device, total_training_steps)

    history = run_training_loop(cfg, policy, tokenizer, triples, optimizer, scheduler, device, scaler_enabled, amp_dtype)

    plot_history(history, cfg)

    if cfg.save_final:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"保存模型到 {cfg.output_dir}")
        policy.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
    if cfg.push_to_hub and os.getenv("HF_TOKEN"):
        try:
            print("推送到 Hugging Face Hub ...")
            policy.push_to_hub(cfg.output_dir, tags=["manual-dpo", "epoch"])
            tokenizer.push_to_hub(cfg.output_dir)
        except Exception as e:
            print(f"推送失败: {e}")

    print("训练完成！")


def main():
    cfg = Config()
    train(cfg)


if __name__ == "__main__":
    main()
