"""Check BGE-VL's image and image-plus-text inference on CUDA with local weights."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local BGE-VL model directory")
    parser.add_argument("--output-dir", required=True, help="New directory for generated inputs and results")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    image_paths = [out / "red.png", out / "blue.png"]
    for path, color in zip(image_paths, ("red", "blue")):
        Image.new("RGB", (224, 224), color).save(path)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).to("cuda")
    model.set_processor(args.model)
    model.eval()
    assert next(model.parameters()).device.type == "cuda"
    with torch.no_grad():
        candidates = model.encode(images=[str(path) for path in image_paths])
        image_query = model.encode(images=str(image_paths[0]))
        composed_query = model.encode(images=str(image_paths[0]), text="Make the background darker")
        image_scores = image_query @ candidates.T
        composed_scores = composed_query @ candidates.T
    assert image_scores.shape == composed_scores.shape == (1, 2)
    assert torch.isfinite(image_scores).all() and torch.isfinite(composed_scores).all()
    result = {"device": torch.cuda.get_device_name(), "candidate_shape": list(candidates.shape),
              "image_scores": image_scores.cpu().tolist(), "composed_scores": composed_scores.cpu().tolist()}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
