"""Run the real training loop on CUDA using tiny synthetic data and a random CLIP.

This is a plumbing check, not a reproduction of the Flickr30k experiment.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finetune_clip_with_freeze as training
from data_common import CaptionPairDataset


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class SyntheticProcessor:
    def __call__(self, images=None, text=None, **kwargs):
        batch = Batch()
        if images is not None:
            batch["pixel_values"] = torch.stack(images)
        if text is not None:
            batch["input_ids"] = torch.tensor([[1, 3 + int(s) % 20, 2] for s in text])
            batch["attention_mask"] = torch.ones((len(text), 3), dtype=torch.long)
        return batch

    def save_pretrained(self, path):
        Path(path, "synthetic_processor.json").write_text('{}', encoding="utf-8")


def tiny_model():
    text = CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                          bos_token_id=1, eos_token_id=2, pad_token_id=0)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, image_size=16, patch_size=8)
    return CLIPModel(CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                                projection_dim=8))


def dataset(offset):
    generator = torch.Generator().manual_seed(offset)
    return CaptionPairDataset([{"image": torch.rand(3, 16, 16, generator=generator),
                                "text": str(i), "image_id": offset + i}
                               for i in range(4)])


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    out = Path(os.environ["XL_OUT_DIR"]) / "synthetic-clip-training"
    model = tiny_model()
    before = model.visual_projection.weight.detach().clone()
    training.CONFIG.update(output_dir=str(out), train_batch_size=2, eval_batch_size=2,
                           num_epochs=1, unfreeze_vision_last_n=1, unfreeze_text_last_n=1,
                           fp16=False, bf16=False, fast_dev_run=True,
                           skip_test_in_fast_dev=False, print_trainable_params=False)
    with patch.object(training, "split_and_expand_dataset", return_value=(dataset(1), dataset(10), dataset(20))), \
         patch.object(training.CLIPProcessor, "from_pretrained", return_value=SyntheticProcessor()), \
         patch.object(training.CLIPModel, "from_pretrained", side_effect=[model, tiny_model()]):
        training.train()
    assert not torch.equal(before, model.visual_projection.weight.detach().cpu()), "projection weights did not change"
    assert (out / "best.pt").is_file() and (out / "final_model" / "config.json").is_file()
    print(json.dumps({"cuda_device": torch.cuda.get_device_name(), "training_steps": 2,
                      "weights_changed": True, "validation_test_and_save": True}))


if __name__ == "__main__":
    main()
