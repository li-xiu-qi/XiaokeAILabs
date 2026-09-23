from __future__ import annotations

import math
import os
import random
import json
from typing import List, Tuple, Dict, Any, Optional, TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
	CLIPModel,
	CLIPProcessor,
	get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm

from data_common import (
	PairRecord,
	CaptionPairDataset,
	EvalResult,
	evaluate_fast_limited,
	Flickr30kDataBuilder,
	plot_loss_curve,
	plot_recall_comparison,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"检测到设备: {DEVICE}")


class TrainConfig(TypedDict):
	model_name: str
	output_dir: str
	seed: int
	train_batch_size: int
	eval_batch_size: int
	grad_accum_steps: int
	lr: float
	weight_decay: float
	num_epochs: int
	warmup_ratio: float
	unfreeze_vision_last_n: int
	unfreeze_text_last_n: int
	fp16: bool
	bf16: bool
	max_train_samples: Optional[int]
	max_eval_samples: Optional[int]
	fast_dev_run: bool
	eval_every_steps: int
	early_stopping_patience: int
	save_every_eval: bool
	max_grad_norm: float
	skip_test_in_fast_dev: bool
	fast_dev_eval_samples: int
	print_trainable_params: bool


CONFIG: TrainConfig = {
    "model_name": "./CLIP-ViT-B-32-laion2B-s34B-b79K",
	# "model_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # 若本地已下载可替换为本地路径
	"output_dir": "./clip_output",
	"seed": 42,
	"train_batch_size": 300,
	"eval_batch_size": 640,
	"grad_accum_steps": 1,
	"lr": 5e-5,
	"weight_decay": 0.01,
	"num_epochs": 5,
	"warmup_ratio": 0.05,
	"unfreeze_vision_last_n": 2,
	"unfreeze_text_last_n": 4,
	"fp16": True,
	"bf16": True,
	"max_train_samples": None,
	"max_eval_samples": None,
	"fast_dev_run": False,
	"eval_every_steps": 0,
	"early_stopping_patience": 0,
	"save_every_eval": False,
	"max_grad_norm": 0.0,
	"skip_test_in_fast_dev": False,
	"fast_dev_eval_samples": 64,
	"print_trainable_params": True,
}

FAST_DEV_MAX_TRAIN = 256
FAST_DEV_MAX_EVAL = 128


def set_seed(seed: int) -> None:
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
	if not os.path.isdir(path): 
		os.makedirs(path, exist_ok=True)


def dump_train_config(cfg: TrainConfig, output_dir: str) -> None:
	ensure_dir(output_dir)
	config_dict = {k: cfg[k] for k in cfg}
	out_path = os.path.join(output_dir, "train_config.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(config_dict, f, ensure_ascii=False, indent=2)
	print(f"训练配置已保存: {out_path}")




def split_and_expand_dataset(max_train: Optional[int], max_eval: Optional[int], fast_dev: bool) -> Tuple[CaptionPairDataset, CaptionPairDataset, CaptionPairDataset]:
    print("加载数据集 flickr30k-v2 ...")
    builder = Flickr30kDataBuilder(seed=CONFIG["seed"])
    return builder.build(
        max_train=max_train,
        max_eval=max_eval,
        fast_dev=fast_dev,
        fast_dev_max_train=FAST_DEV_MAX_TRAIN,
        fast_dev_max_eval=FAST_DEV_MAX_EVAL,
    )


def build_dataloader(ds: CaptionPairDataset, processor: CLIPProcessor, batch_size: int, shuffle: bool) -> DataLoader:

	def collate(batch: List[PairRecord]) -> Dict[str, torch.Tensor]:
		images = [b["image"] for b in batch]
		texts = [b["text"] for b in batch]
		enc = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
		return enc

	return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=4, pin_memory=True)


def freeze_for_light_tuning(model: CLIPModel, vision_last_n: int, text_last_n: int) -> None:
	for p in model.parameters():
		p.requires_grad = False

	if vision_last_n > 0:
		vision_layers = model.vision_model.encoder.layers
		for layer in vision_layers[-vision_last_n:]:
			for p in layer.parameters():
				p.requires_grad = True

	if text_last_n > 0:
		text_layers = model.text_model.encoder.layers
		for layer in text_layers[-text_last_n:]:
			for p in layer.parameters():
				p.requires_grad = True

	for name in ["visual_projection", "text_projection", "logit_scale"]:
		attr = getattr(model, name, None)
		if attr is None:
			continue
		if isinstance(attr, torch.nn.Parameter):
			attr.requires_grad = True
		else:
			for p in attr.parameters():
				p.requires_grad = True

	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	total = sum(p.numel() for p in model.parameters())
	print(f"解冻参数量: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

	if CONFIG.get("print_trainable_params", False):
		print("可训练参数名称列表 (前若干):")
		shown = 0
		for n, p in model.named_parameters():
			if p.requires_grad:
				print(f"  {n}: {tuple(p.shape)}")
				shown += 1
				if shown >= 50:
					print("  ... (已截断)")
					break


def train() -> None:
	cfg = CONFIG
	set_seed(cfg["seed"])
	device = DEVICE
	print(f"使用设备: {device}")

	train_ds, val_ds, test_ds = split_and_expand_dataset(
		cfg["max_train_samples"], cfg["max_eval_samples"], cfg["fast_dev_run"]
	)

	print("加载模型与处理器 ...")
	processor = CLIPProcessor.from_pretrained(cfg["model_name"])
	model = CLIPModel.from_pretrained(cfg["model_name"])
	freeze_for_light_tuning(model, cfg["unfreeze_vision_last_n"], cfg["unfreeze_text_last_n"])
	model.to(device)

	train_loader = build_dataloader(train_ds, processor, cfg["train_batch_size"], shuffle=True)

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

	total_steps = math.ceil(len(train_loader) / cfg["grad_accum_steps"]) * cfg["num_epochs"]
	warmup_steps = int(total_steps * cfg["warmup_ratio"])
	scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	use_fp16 = cfg["fp16"] and not cfg["bf16"] and torch.cuda.is_available()
	try:
		if use_fp16:
			try:
				scaler = torch.amp.GradScaler("cuda", enabled=True)
			except TypeError:
				scaler = torch.amp.GradScaler(enabled=True)
		else:
			scaler = torch.amp.GradScaler(enabled=False)
	except AttributeError:
		scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
	autocast_dtype = torch.bfloat16 if (cfg["bf16"] and torch.cuda.is_available()) else torch.float16

	ce_loss = nn.CrossEntropyLoss()

	ensure_dir(cfg["output_dir"])

	global_step = 0
	best_metric = -1.0
	best_state: Optional[Dict[str, torch.Tensor]] = None
	no_improve_count = 0
	epoch_loss_history: List[float] = []

	dump_train_config(cfg, cfg["output_dir"])

	def compute_select_metric(res: EvalResult) -> float:
		return (res.recall_i2t_r1 + res.recall_t2i_r1) / 2.0

	def do_validation(tag: str, force_limit: bool) -> float:
		limit_samples = None
		if cfg["fast_dev_run"] and force_limit:
			limit_samples = cfg.get("fast_dev_eval_samples") or None
			print(f"[验证] fast_dev_run 裁剪验证样本至 {limit_samples}")
		res = evaluate_fast_limited(
			model,
			processor,
			val_ds,
			cfg["eval_batch_size"],
			device,
			amp=(cfg["fp16"] or cfg["bf16"]),
			autocast_dtype=autocast_dtype,
			limit_samples=limit_samples,
		)
		metric_local = compute_select_metric(res)
		print(
			f"验证[{tag}] -> I2T@1={res.recall_i2t_r1:.4f} T2I@1={res.recall_t2i_r1:.4f} 选取指标={metric_local:.4f}"
		)
		return metric_local

	def maybe_save_checkpoint(metric_value: float, tag: str):
		nonlocal best_metric, best_state
		improved = metric_value > best_metric
		if improved:
			best_metric = metric_value
			best_state = {k: v.cpu() for k, v in model.state_dict().items()}
			path = os.path.join(cfg["output_dir"], "best.pt")
			torch.save(best_state, path)
			print(f"更新最佳模型 ({tag}), 指标={best_metric:.4f} 已保存 best.pt")
		if cfg["save_every_eval"]:
			all_path = os.path.join(cfg["output_dir"], f"checkpoint_{tag.replace('/', '_')}.pt")
			torch.save({k: v.cpu() for k, v in model.state_dict().items()}, all_path)
			print(f"已保存评估点 checkpoint: {all_path}")
		return improved

	for epoch in range(cfg["num_epochs"]):
		model.train()
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")
		running_loss = 0.0
		steps_in_epoch = 0
		optimizer.zero_grad(set_to_none=True)

		for step, batch in enumerate(pbar):
			steps_in_epoch += 1
			batch = {k: v.to(device) for k, v in batch.items()}
			with torch.amp.autocast(
				device_type='cuda',
				enabled=(cfg["fp16"] or cfg["bf16"]) and torch.cuda.is_available(),
				dtype=autocast_dtype,
			):
				out = model(**batch)
				logits_img = out.logits_per_image
				logits_txt = out.logits_per_text
				labels = torch.arange(logits_img.size(0), device=device)
				loss_i = ce_loss(logits_img, labels)
				loss_t = ce_loss(logits_txt, labels)
				loss = (loss_i + loss_t) / 2.0

			if scaler.is_enabled():
				scaler.scale(loss).backward()
			else:
				loss.backward()

			if (step + 1) % cfg["grad_accum_steps"] == 0:
				if cfg["max_grad_norm"] > 0:
					if scaler.is_enabled():
						scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
					else:
						torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])

				if scaler.is_enabled():
					scaler.step(optimizer)
					scaler.update()
				else:
					optimizer.step()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)
				global_step += 1

			running_loss += loss.item()
			pbar.set_postfix({"loss": f"{running_loss/(step+1):.4f}"})

			if cfg["fast_dev_run"] and global_step >= 10:
				print("FAST_DEV_RUN: 提前结束训练循环")
				break

			if (
				cfg["eval_every_steps"] > 0
				and global_step % cfg["eval_every_steps"] == 0
				and not cfg["fast_dev_run"]
			):
				metric_now = do_validation(tag=f"step_{global_step}", force_limit=False)
				improved = maybe_save_checkpoint(metric_now, tag=f"step_{global_step}")
				if not improved and cfg["early_stopping_patience"] > 0:
					no_improve_count += 1
					print(f"早停计数 +1 -> {no_improve_count}")
					if no_improve_count >= cfg["early_stopping_patience"]:
						print("触发早停 (step 评估阶段)")
						break
				else:
					no_improve_count = 0

		if steps_in_epoch > 0:
			avg_epoch_loss = running_loss / steps_in_epoch
			epoch_loss_history.append(avg_epoch_loss)
			print(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}")

		if cfg["early_stopping_patience"] > 0 and no_improve_count >= cfg["early_stopping_patience"]:
			break

		metric_epoch = do_validation(tag=f"epoch_{epoch+1}", force_limit=True)
		improved_epoch = maybe_save_checkpoint(metric_epoch, tag=f"epoch_{epoch+1}")
		if not improved_epoch and cfg["early_stopping_patience"] > 0:
			no_improve_count += 1
			print(f"早停计数 +1 -> {no_improve_count}")
			if no_improve_count >= cfg["early_stopping_patience"]:
				print("触发早停 (epoch 评估阶段)")
				break
		else:
			no_improve_count = 0

		if cfg["fast_dev_run"]:
			print("FAST_DEV_RUN: 已完成 1 个 epoch")
			break

	plot_loss_curve(epoch_loss_history, cfg["output_dir"])

	if best_state is not None:
		model.load_state_dict(best_state)
	if cfg["fast_dev_run"] and cfg.get("skip_test_in_fast_dev", False):
		print("FAST_DEV_RUN: 跳过测试集评估 (已配置 skip_test_in_fast_dev=True)")
	else:
		print("测试集评估 ...")
		limit_samples_test = None
		if cfg["fast_dev_run"]:
			limit_samples_test = cfg.get("fast_dev_eval_samples") or None
		fine_tuned_test_res = evaluate_fast_limited(
			model,
			processor,
			test_ds,
			cfg["eval_batch_size"],
			device,
			amp=(cfg["fp16"] or cfg["bf16"]),
			autocast_dtype=autocast_dtype,
			limit_samples=limit_samples_test,
		)
		print(
			"Test: "
			f"I2T@1={fine_tuned_test_res.recall_i2t_r1:.4f} I2T@5={fine_tuned_test_res.recall_i2t_r5:.4f} I2T@10={fine_tuned_test_res.recall_i2t_r10:.4f} | "
			f"T2I@1={fine_tuned_test_res.recall_t2i_r1:.4f} T2I@5={fine_tuned_test_res.recall_t2i_r5:.4f} T2I@10={fine_tuned_test_res.recall_t2i_r10:.4f}"
		)

		print("加载原始未微调 (Pretrained) 模型进行同一测试集评估以对比 ...")
		baseline_model = CLIPModel.from_pretrained(cfg["model_name"]).to(device)
		baseline_model.eval()
		baseline_test_res = evaluate_fast_limited(
			baseline_model,
			processor,
			test_ds,
			cfg["eval_batch_size"],
			device,
			amp=(cfg["fp16"] or cfg["bf16"]),
			autocast_dtype=autocast_dtype,
			limit_samples=limit_samples_test,
		)
		print(
			"Baseline(Test): "
			f"I2T@1={baseline_test_res.recall_i2t_r1:.4f} I2T@5={baseline_test_res.recall_i2t_r5:.4f} I2T@10={baseline_test_res.recall_i2t_r10:.4f} | "
			f"T2I@1={baseline_test_res.recall_t2i_r1:.4f} T2I@5={baseline_test_res.recall_t2i_r5:.4f} T2I@10={baseline_test_res.recall_t2i_r10:.4f}"
		)

		def _improve(new: float, old: float) -> str:
			abs_gain = new - old
			rel = (abs_gain / (old + 1e-8)) * 100 if old > 0 else float('inf')
			return f"+{abs_gain:.4f} ({rel:+.2f}% )"

		print("对比提升 (FineTuned - Baseline):")
		print(
			"  I2T@1: " + _improve(fine_tuned_test_res.recall_i2t_r1, baseline_test_res.recall_i2t_r1)
			+ " | I2T@5: " + _improve(fine_tuned_test_res.recall_i2t_r5, baseline_test_res.recall_i2t_r5)
			+ " | I2T@10: " + _improve(fine_tuned_test_res.recall_i2t_r10, baseline_test_res.recall_i2t_r10)
		)
		print(
			"  T2I@1: " + _improve(fine_tuned_test_res.recall_t2i_r1, baseline_test_res.recall_t2i_r1)
			+ " | T2I@5: " + _improve(fine_tuned_test_res.recall_t2i_r5, baseline_test_res.recall_t2i_r5)
			+ " | T2I@10: " + _improve(fine_tuned_test_res.recall_t2i_r10, baseline_test_res.recall_t2i_r10)
		)
		avg_baseline = (baseline_test_res.recall_i2t_r1 + baseline_test_res.recall_t2i_r1) / 2
		avg_finetune = (fine_tuned_test_res.recall_i2t_r1 + fine_tuned_test_res.recall_t2i_r1) / 2
		print(
			f"  Avg@1: {_improve(avg_finetune, avg_baseline)}  (fine_tuned={avg_finetune:.4f}, baseline={avg_baseline:.4f})"
		)
		plot_recall_comparison(baseline_test_res, fine_tuned_test_res, cfg["output_dir"])

	model.save_pretrained(os.path.join(cfg["output_dir"], "final_model"))
	processor.save_pretrained(os.path.join(cfg["output_dir"], "final_model"))
	print("已保存 final_model")


def main() -> None:
	train()


if __name__ == "__main__":
	main()