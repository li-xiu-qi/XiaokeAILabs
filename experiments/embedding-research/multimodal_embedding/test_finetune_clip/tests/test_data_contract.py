"""CPU-only tests for the split and contrastive batch contracts."""

import sys
import unittest
from pathlib import Path

from datasets import Dataset
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_common import Flickr30kDataBuilder, UniqueImageBatchSampler


def make_raw(splits, ids):
    return {"train": Dataset.from_dict({"split": splits, "img_id": ids})}


class FlickrPartitionTest(unittest.TestCase):
    def test_uses_image_level_split_fields(self):
        raw = make_raw(["train", "test", "val", "train"], ["10", "20", "30", "40"])
        train, val, test = Flickr30kDataBuilder()._split(raw)
        self.assertEqual(list(train["img_id"]), ["10", "40"])
        self.assertEqual(list(val["img_id"]), ["30"])
        self.assertEqual(list(test["img_id"]), ["20"])

    def test_rejects_id_leakage_even_if_string_format_differs(self):
        raw = make_raw(["train", "val", "test"], ["01", "1", "3"])
        with self.assertRaisesRegex(ValueError, "跨数据划分"):
            Flickr30kDataBuilder()._split(raw)

    def test_rejects_unknown_split_and_missing_id(self):
        with self.assertRaisesRegex(ValueError, "未知数据划分"):
            Flickr30kDataBuilder()._split(make_raw(["train", "val", "unknown"], ["1", "2", "3"]))
        with self.assertRaisesRegex(ValueError, "缺少 img_id"):
            Flickr30kDataBuilder()._split(make_raw(["train", "val", "test"], ["1", "", "3"]))


class UniqueImageBatchSamplerTest(unittest.TestCase):
    def setUp(self):
        self.pairs = [{"image_id": image_id, "text": str(caption), "image": None}
                      for image_id in range(7) for caption in range(5)]

    def test_each_batch_uses_distinct_images(self):
        sampler = UniqueImageBatchSampler(self.pairs, batch_size=3, seed=17)
        for _ in range(3):
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            self.assertEqual([len(batch) for batch in batches], [3, 3])
            ids = [self.pairs[index]["image_id"] for batch in batches for index in batch]
            self.assertEqual(len(ids), len(set(ids)))

    def test_seed_is_reproducible_and_epochs_change(self):
        first = UniqueImageBatchSampler(self.pairs, batch_size=3, seed=17)
        second = UniqueImageBatchSampler(self.pairs, batch_size=3, seed=17)
        first_epoch = list(first)
        self.assertEqual(first_epoch, list(second))
        self.assertEqual(list(first), list(second))
        self.assertNotEqual(first_epoch, list(UniqueImageBatchSampler(self.pairs, 3, 18)))

    def test_rejects_degenerate_batches(self):
        with self.assertRaisesRegex(ValueError, "至少需要两张"):
            UniqueImageBatchSampler([{"image_id": 1}], batch_size=2)
        with self.assertRaisesRegex(ValueError, "至少需要两张"):
            UniqueImageBatchSampler(self.pairs, batch_size=1)
        with self.assertRaisesRegex(ValueError, "缺少 image_id"):
            UniqueImageBatchSampler([{}], batch_size=2)

    def test_tiny_clip_training_step_on_cpu(self):
        text = CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                              bos_token_id=1, eos_token_id=2, pad_token_id=0)
        vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, image_size=16, patch_size=8)
        model = CLIPModel(CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                                     projection_dim=8)).cpu()
        batch = next(iter(UniqueImageBatchSampler(self.pairs, batch_size=3, seed=17)))
        self.assertEqual(len({self.pairs[i]["image_id"] for i in batch}), len(batch))
        out = model(input_ids=torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]]),
                    attention_mask=torch.ones(3, 3, dtype=torch.long),
                    pixel_values=torch.rand(3, 3, 16, 16))
        labels = torch.arange(len(batch))
        loss = (torch.nn.functional.cross_entropy(out.logits_per_image, labels)
                + torch.nn.functional.cross_entropy(out.logits_per_text, labels)) / 2
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.visual_projection.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
