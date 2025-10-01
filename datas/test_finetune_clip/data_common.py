"""公共数据处理与评估工具

集中放置:
1. PairRecord / CaptionPairDataset 结构
2. 通用评估逻辑 (evaluate / evaluate_fast_limited)
3. 数据集构建基类 + 两个具体实现:
   - ArxivCapDataBuilder (本地单 caption 数据集, 可按比例切分)
   - Flickr30kDataBuilder (远程 flickr30k-v2, 多 caption 展开)

后续如果再增加新的数据来源, 只需继承 BaseCaptionDataBuilder 覆盖若干方法。

注意: 尽量保持最少依赖, 只放“数据与评估”相关内容, 训练循环仍由各脚本自行控制。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TypedDict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor
from datasets import load_from_disk, load_dataset, Dataset as HFDataset, DatasetDict


class PairRecord(TypedDict):
    image: Any
    text: str
    image_id: int


class CaptionPairDataset(Dataset):

    def __init__(self, pairs: List[PairRecord]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairRecord:
        return self.pairs[idx]


@dataclass
class EvalResult:
    recall_i2t_r1: float
    recall_i2t_r5: float
    recall_i2t_r10: float
    recall_t2i_r1: float
    recall_t2i_r5: float
    recall_t2i_r10: float


def evaluate(
    model: CLIPModel,
    processor: CLIPProcessor,
    ds: CaptionPairDataset,
    batch_size: int,
    device: torch.device,
    amp: bool,
    autocast_dtype: torch.dtype,
) -> EvalResult:
    return _evaluate_internal(model, processor, ds, batch_size, device, amp, autocast_dtype, limit_samples=None)


def evaluate_fast_limited(
    model: CLIPModel,
    processor: CLIPProcessor,
    ds: CaptionPairDataset,
    batch_size: int,
    device: torch.device,
    amp: bool,
    autocast_dtype: torch.dtype,
    limit_samples: Optional[int],
) -> EvalResult:
    return _evaluate_internal(model, processor, ds, batch_size, device, amp, autocast_dtype, limit_samples=limit_samples)


def _evaluate_internal(
    model: CLIPModel,
    processor: CLIPProcessor,
    ds: CaptionPairDataset,
    batch_size: int,
    device: torch.device,
    amp: bool,
    autocast_dtype: torch.dtype,
    limit_samples: Optional[int],
) -> EvalResult:
    model.eval()
    if limit_samples is not None:
        pairs = ds.pairs[: limit_samples]
    else:
        pairs = ds.pairs

    texts: List[str] = []
    image_ids: List[int] = []
    for rec in pairs:
        texts.append(rec["text"])
        image_ids.append(rec["image_id"])

    uniq_image_map: Dict[int, Any] = {}
    for rec in pairs:
        if rec["image_id"] not in uniq_image_map:
            uniq_image_map[rec["image_id"]] = rec["image"]
    uniq_image_ids = list(uniq_image_map.keys())
    uniq_images = [uniq_image_map[i] for i in uniq_image_ids]
    image_id_to_index = {iid: idx for idx, iid in enumerate(uniq_image_ids)}

    img_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(uniq_images), batch_size):
            batch_imgs = uniq_images[i : i + batch_size]
            enc = processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp and torch.cuda.is_available(),
                dtype=autocast_dtype,
            ):
                out = model.vision_model(
                    **{k: v for k, v in enc.items() if k.startswith("pixel_values") or k == "pixel_values"}
                )
                pooled = out["pooler_output"]
                proj = model.visual_projection(pooled)
                img_feats.append(proj)
    img_feats_t = torch.cat(img_feats, dim=0)
    img_feats_t = img_feats_t / img_feats_t.norm(dim=-1, keepdim=True)

    txt_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_txt = texts[i : i + batch_size]
            enc = processor(text=batch_txt, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp and torch.cuda.is_available(),
                dtype=autocast_dtype,
            ):
                out = model.text_model(
                    **{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                )
                pooled = out["pooler_output"]
                proj = model.text_projection(pooled)
                txt_feats.append(proj)
    txt_feats_t = torch.cat(txt_feats, dim=0)
    txt_feats_t = txt_feats_t / txt_feats_t.norm(dim=-1, keepdim=True)

    sim = img_feats_t @ txt_feats_t.t()

    image_to_text_indices: Dict[int, List[int]] = {}
    text_to_image_index: List[int] = []
    for txt_idx, img_id in enumerate(image_ids):
        img_index = image_id_to_index[img_id]
        text_to_image_index.append(img_index)
        image_to_text_indices.setdefault(img_index, []).append(txt_idx)

    ks = [1, 5, 10]
    recall_i2t: Dict[int, float] = {}
    for k in ks:
        hit = 0
        for img_index, pos_txt_indices in image_to_text_indices.items():
            scores = sim[img_index]
            topk = scores.topk(k).indices.tolist()
            if any(t in topk for t in pos_txt_indices):
                hit += 1
        recall_i2t[k] = hit / len(image_to_text_indices)

    recall_t2i: Dict[int, float] = {}
    sim_t = sim.t()
    for k in ks:
        hit = 0
        for txt_idx, img_index in enumerate(text_to_image_index):
            scores = sim_t[txt_idx]
            topk = scores.topk(k).indices.tolist()
            if img_index in topk:
                hit += 1
        recall_t2i[k] = hit / len(text_to_image_index)

    return EvalResult(
        recall_i2t_r1=recall_i2t[1],
        recall_i2t_r5=recall_i2t[5],
        recall_i2t_r10=recall_i2t[10],
        recall_t2i_r1=recall_t2i[1],
        recall_t2i_r5=recall_t2i[5],
        recall_t2i_r10=recall_t2i[10],
    )


class BaseCaptionDataBuilder:

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _load_raw(self):
        raise NotImplementedError

    def _split(self, raw) -> Tuple[HFDataset, HFDataset, HFDataset]:
        raise NotImplementedError

    def _expand(self, ds: HFDataset) -> List[PairRecord]:
        raise NotImplementedError

    def build(
        self,
        max_train: Optional[int],
        max_eval: Optional[int],
        fast_dev: bool,
        fast_dev_max_train: int = 256,
        fast_dev_max_eval: int = 128,
    ) -> Tuple[CaptionPairDataset, CaptionPairDataset, CaptionPairDataset]:
        raw = self._load_raw()
        train_raw, val_raw, test_raw = self._split(raw)

        train_pairs = self._expand(train_raw)
        val_pairs = self._expand(val_raw)
        test_pairs = self._expand(test_raw)

        if fast_dev:
            max_train = fast_dev_max_train if max_train is None else min(max_train, fast_dev_max_train)
            max_eval = fast_dev_max_eval if max_eval is None else min(max_eval, fast_dev_max_eval)

        def _limit(lst: List[PairRecord], m: Optional[int]) -> List[PairRecord]:
            if m is not None and len(lst) > m:
                return lst[:m]
            return lst

        train_pairs = _limit(train_pairs, max_train)
        val_pairs = _limit(val_pairs, max_eval)
        test_pairs = _limit(test_pairs, max_eval)

        print(
            f"展开后样本数: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}"
        )
        return (
            CaptionPairDataset(train_pairs),
            CaptionPairDataset(val_pairs),
            CaptionPairDataset(test_pairs),
        )


class ArxivCapDataBuilder(BaseCaptionDataBuilder):
    def __init__(self, dataset_path: str, val_ratio: float, test_ratio: float, seed: int = 42):
        super().__init__(seed=seed)
        self.dataset_path = dataset_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def _load_raw(self):
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"未找到数据集目录: {self.dataset_path}")
        raw = load_from_disk(self.dataset_path)
        if isinstance(raw, HFDataset):
            raw = DatasetDict({"train": raw})
        return raw

    def _split(self, raw) -> Tuple[HFDataset, HFDataset, HFDataset]:
        dataset_train = raw.get("train")
        dataset_val = raw.get("validation")
        dataset_test = raw.get("test")
        if dataset_train is None:
            raise ValueError("数据集中缺少 train split")
        if dataset_val is not None and dataset_test is not None:
            return dataset_train, dataset_val, dataset_test

        if self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("val/test 比例需非负")
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio 必须 < 1")
        if self.val_ratio + self.test_ratio == 0:
            empty = dataset_train.select([])
            return dataset_train, empty, empty
        mix = dataset_train.train_test_split(test_size=self.val_ratio + self.test_ratio, seed=self.seed)
        val_test = mix["test"].train_test_split(
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio), seed=self.seed
        )
        return mix["train"], val_test["train"], val_test["test"]

    def _expand(self, ds: HFDataset) -> List[PairRecord]:
        out: List[PairRecord] = []
        image_id_map: Dict[str, int] = {}
        for idx, rec in enumerate(ds):
            caption = (rec.get("caption") or "").strip()
            if not caption:
                continue
            image = rec.get("image")
            if image is None:
                continue
            raw_key = rec.get("arxiv_id") or rec.get("image_file") or f"idx_{idx}"
            key = str(raw_key)
            if key not in image_id_map:
                image_id_map[key] = len(image_id_map)
            image_id = image_id_map[key]
            out.append(PairRecord(image=image, text=caption, image_id=image_id))
        return out


class Flickr30kDataBuilder(BaseCaptionDataBuilder):
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def _load_raw(self):
        return load_dataset("WhereIsAI/flickr30k-v2")

    def _split(self, raw) -> Tuple[HFDataset, HFDataset, HFDataset]:
        if "validation" not in raw or "test" not in raw:
            tt = raw["train"].train_test_split(test_size=0.2, seed=self.seed)
            tv = tt["test"].train_test_split(test_size=0.5, seed=self.seed)
            return tt["train"], tv["train"], tv["test"]
        return raw["train"], raw["validation"], raw["test"]

    def _expand(self, ds: HFDataset) -> List[PairRecord]:
        out: List[PairRecord] = []
        for rec in ds:
            caps = rec.get("caption")
            if not caps:
                continue
            image = rec.get("image")
            img_id = int(rec.get("img_id", rec.get("image_id", 0)))
            for c in caps:
                if not c:
                    continue
                out.append(PairRecord(image=image, text=c.strip(), image_id=img_id))
        return out


def plot_loss_curve(loss_history: List[float], output_dir: str, title: str = "Training Loss Curve") -> Optional[str]:
    if not loss_history:
        print("[plot_loss_curve] 无 loss 数据, 跳过绘制")
        return None
    os.makedirs(output_dir, exist_ok=True)
    steps = list(range(1, len(loss_history) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, loss_history, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_loss_curve] 已保存: {out_path}")
    return out_path


def plot_recall_comparison(
    baseline: EvalResult,
    finetuned: EvalResult,
    output_dir: str,
    filename: str = "recall_comparison.png",
    title: str = "Recall Comparison: Fine-tuned vs. Baseline",
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    metrics: List[Tuple[str, float, float]] = [
        ("I2T@1", baseline.recall_i2t_r1, finetuned.recall_i2t_r1),
        ("I2T@5", baseline.recall_i2t_r5, finetuned.recall_i2t_r5),
        ("I2T@10", baseline.recall_i2t_r10, finetuned.recall_i2t_r10),
        ("T2I@1", baseline.recall_t2i_r1, finetuned.recall_t2i_r1),
        ("T2I@5", baseline.recall_t2i_r5, finetuned.recall_t2i_r5),
        ("T2I@10", baseline.recall_t2i_r10, finetuned.recall_t2i_r10),
    ]
    labels = [m[0] for m in metrics]
    baseline_vals = [m[1] for m in metrics]
    finetune_vals = [m[2] for m in metrics]
    xs = list(range(len(labels)))
    bw = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([x - bw / 2 for x in xs], baseline_vals, width=bw, label="Baseline", color="#6c8ebf")
    plt.bar([x + bw / 2 for x in xs], finetune_vals, width=bw, label="Fine-tuned", color="#c0504d")
    plt.xticks(xs, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recall")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_recall_comparison] 已保存: {out_path}")
    return out_path


__all__ = [
    "PairRecord",
    "CaptionPairDataset",
    "EvalResult",
    "evaluate",
    "evaluate_fast_limited",
    "BaseCaptionDataBuilder",
    "ArxivCapDataBuilder",
    "Flickr30kDataBuilder",
    "plot_loss_curve",
    "plot_recall_comparison",
]