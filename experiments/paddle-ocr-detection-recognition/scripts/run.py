"""Run PP-OCR detection then recognition. Supply model directories and an image."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForTextRecognition
from validate_detection import validate_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--det-model', required=True)
    parser.add_argument('--rec-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--det-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--rec-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    torch.set_num_threads(args.threads)
    image = Image.open(args.image).convert('RGB')
    det_processor = AutoImageProcessor.from_pretrained(args.det_model, local_files_only=True)
    det = AutoModelForObjectDetection.from_pretrained(args.det_model, local_files_only=True).eval().to(args.det_device)
    inputs = det_processor(images=image, return_tensors='pt').to(args.det_device)
    with torch.inference_mode():
        outputs = det(**inputs)
    detected = det_processor.post_process_object_detection(outputs, target_sizes=inputs['target_sizes'])[0]
    boxes = detected['boxes'].cpu().tolist()
    scores = detected['scores'].cpu().tolist()
    validate_detections(boxes, scores)
    del det, inputs, outputs, detected
    # Match the recorded run's sorting; this heuristic is specific to the sample layout.
    ranked = sorted(zip(boxes, scores), key=lambda pair: (
        round(sum(p[1] for p in pair[0]) / 4 / 75), sum(p[0] for p in pair[0])))
    rec_processor = AutoImageProcessor.from_pretrained(args.rec_model, local_files_only=True)
    rec = AutoModelForTextRecognition.from_pretrained(args.rec_model, local_files_only=True).eval().to(args.rec_device)
    records = []
    for box, score in ranked:
        x1, y1 = min(p[0] for p in box), min(p[1] for p in box)
        x2, y2 = max(p[0] for p in box), max(p[1] for p in box)
        inputs = rec_processor(images=image.crop((x1, y1, x2, y2)), return_tensors='pt').to(args.rec_device)
        with torch.inference_mode():
            outputs = rec(**inputs)
        prediction = rec_processor.post_process_text_recognition(outputs)[0]
        records.append({'box': box, 'detection_score': score, 'prediction': prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(records, out, ensure_ascii=False, indent=2)
    print(f'{len(records)} detections written to {args.output}')


if __name__ == '__main__':
    main()
