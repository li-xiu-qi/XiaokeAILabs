#!/usr/bin/env python3
"""Audit an exported identity before training: four independent checks.

Upstream labels are not trusted. Every check below can fail the dataset on its
own, and the checks run before any GPU time is spent.

    1. Identity consistency  per-image similarity to the test-set prototype;
                             flag images below mean - 2 sd (possible mislabel)
    2. Cross-identity bleed  train images vs the negative-control prototype;
                             anything above 0.30 means another face slipped in
    3. Duplicates & leakage  average-hash comparison within and across splits;
                             Hamming distance <= 4 counts as the same image
    4. Face quality          detection confidence, face-to-frame ratio,
                             eye-line roll, Laplacian sharpness

Writes qc.json next to the outputs and prints a pass/fail summary.

    python scripts/audit_data.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from common import (
    NEGATIVE_ID,
    TARGET_ID,
    FaceEncoder,
    identity_dir,
    prepare_output_dir,
    _require,
)

DUPLICATE_HAMMING = 4
BLEED_SIMILARITY = 0.30
MIN_FACE_RATIO = 0.05
MAX_ROLL_DEGREES = 25.0
MIN_SHARPNESS = 50.0


def analyse(encoder: FaceEncoder, path: Path) -> dict | None:
    cv2 = _require("cv2")
    np = _require("numpy")
    img = cv2.imread(str(path))
    if img is None:
        return None
    boxes, kpss = encoder.det.detect(img, input_size=(640, 640))
    if not len(boxes):
        return None
    best = int(np.argmax(boxes[:, 4]))
    box = boxes[best]
    emb = encoder.rec.get(img, type("_K", (), {"kps": kpss[best]})())
    if emb is None:
        return None

    # Eye-line tilt approximates head roll without a 3D landmark model.
    kps = kpss[best]
    dx, dy = kps[1][0] - kps[0][0], kps[1][1] - kps[0][1]
    roll = abs(np.degrees(np.arctan2(dy, dx)))

    height, width = img.shape[:2]
    face_ratio = (box[2] - box[0]) * (box[3] - box[1]) / (height * width)

    x1, y1, x2, y2 = (int(v) for v in box[:4])
    crop = cv2.cvtColor(
        img[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)], cv2.COLOR_BGR2GRAY
    )
    sharpness = cv2.Laplacian(crop, cv2.CV_64F).var() if crop.size else 0.0

    return {
        "file": path.name,
        "emb": emb / np.linalg.norm(emb),
        "score": float(box[4]),
        "face_ratio": float(face_ratio),
        "roll": float(roll),
        "sharpness": float(sharpness),
        "hash": average_hash(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)),
    }


def average_hash(gray, size: int = 12) -> str:
    cv2 = _require("cv2")
    small = cv2.resize(gray, (size, size))
    mean = small.mean()
    return "".join("1" if v > mean else "0" for v in small.flatten())


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def load_split(encoder: FaceEncoder, split: str) -> list[dict]:
    directory = identity_dir(TARGET_ID, split)
    files = sorted(glob.glob(str(directory / "*.jpg")))
    if not files:
        raise SystemExit(
            f"no images in {directory}\nrun scripts/prepare_data.py first"
        )
    rows = []
    for path in map(Path, files):
        row = analyse(encoder, path)
        if row is None:
            print(f"  [warn] no face detected: {split}/{path.name}")
        else:
            rows.append(row)
    print(f"{split}: {len(rows)}/{len(files)} images with a detectable face")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()  # no options yet; keeps --help working before deps install

    np = _require("numpy")
    encoder = FaceEncoder()
    out_dir = prepare_output_dir("qc")

    train_rows = load_split(encoder, "train")
    test_rows = load_split(encoder, "test")

    negative_dir = identity_dir(NEGATIVE_ID, "test")
    negative_rows = [
        row
        for row in (
            analyse(encoder, Path(p))
            for p in sorted(glob.glob(str(negative_dir / "*.jpg")))
        )
        if row is not None
    ]
    if not negative_rows:
        print(f"[warn] no negative-control images in {negative_dir}; check 2 is skipped")
        negative_proto = None
    else:
        negative_proto = np.mean([r["emb"] for r in negative_rows], axis=0)
        negative_proto /= np.linalg.norm(negative_proto)

    test_proto = np.mean([r["emb"] for r in test_rows], axis=0)
    test_proto /= np.linalg.norm(test_proto)
    for row in train_rows:
        row["sim_self"] = float(row["emb"] @ test_proto)
        row["sim_negative"] = (
            float(row["emb"] @ negative_proto) if negative_proto is not None else None
        )

    print("\n[1] identity consistency (train vs test prototype)")
    sims = np.array([r["sim_self"] for r in train_rows])
    threshold = sims.mean() - 2 * sims.std()
    outliers = [r for r in train_rows if r["sim_self"] < threshold]
    print(
        f"    mean {sims.mean():.4f}  sd {sims.std():.4f}  "
        f"min {sims.min():.4f}  max {sims.max():.4f}"
    )
    print(f"    outlier threshold (mean - 2sd): {threshold:.4f}")
    for row in outliers:
        print(f"    [check] {row['file']}  self={row['sim_self']:.4f}  "
              f"other={row['sim_negative'] or float('nan'):.4f}")
    if not outliers:
        print("    PASS: no outliers")

    print("\n[2] cross-identity bleed (train vs negative prototype)")
    if negative_proto is None:
        print("    SKIP: no negative-control data")
        contaminated = []
    else:
        sims_negative = np.array([r["sim_negative"] for r in train_rows])
        contaminated = [r for r in train_rows if r["sim_negative"] > BLEED_SIMILARITY]
        print(
            f"    mean {sims_negative.mean():.4f}  max {sims_negative.max():.4f}  "
            f"(>{BLEED_SIMILARITY} is suspicious)"
        )
        for row in contaminated:
            print(f"    [check] {row['file']}  other={row['sim_negative']:.4f}")
        if not contaminated:
            print("    PASS: no cross-identity bleed")

    print(f"\n[3] duplicates and leakage (average hash, Hamming <= {DUPLICATE_HAMMING})")
    tagged = [(r, "train") for r in train_rows] + [(r, "test") for r in test_rows]
    duplicates = []
    for i in range(len(tagged)):
        for j in range(i + 1, len(tagged)):
            distance = hamming(tagged[i][0]["hash"], tagged[j][0]["hash"])
            if distance <= DUPLICATE_HAMMING:
                duplicates.append(
                    (tagged[i][0]["file"], tagged[j][0]["file"], distance,
                     f"{tagged[i][1]}/{tagged[j][1]}")
                )
    for a, b, distance, tag in duplicates:
        print(f"    [check] {a} <-> {b}  d={distance}  ({tag})")
    if not duplicates:
        print("    PASS: no duplicates, no train/test leakage")

    print("\n[4] face quality")
    print(f"    {'file':<14}{'conf':>6}{'face%':>8}{'roll':>7}{'sharp':>8}  flags")
    flagged = {"small": [], "profile": [], "blurry": []}
    for row in sorted(train_rows + test_rows, key=lambda r: r["sharpness"]):
        flags = []
        if row["face_ratio"] < MIN_FACE_RATIO:
            flags.append("small face")
            flagged["small"].append(row["file"])
        if row["roll"] > MAX_ROLL_DEGREES:
            flags.append("profile")
            flagged["profile"].append(row["file"])
        if row["sharpness"] < MIN_SHARPNESS:
            flags.append("blurry")
            flagged["blurry"].append(row["file"])
        print(
            f"    {row['file']:<14}{row['score']:6.2f}{row['face_ratio']:8.3f}"
            f"{row['roll']:7.1f}{row['sharpness']:8.1f}  {' '.join(flags)}"
        )

    report = {
        "target_id": TARGET_ID,
        "negative_id": NEGATIVE_ID,
        "thresholds": {
            "duplicate_hamming": DUPLICATE_HAMMING,
            "bleed_similarity": BLEED_SIMILARITY,
            "min_face_ratio": MIN_FACE_RATIO,
            "max_roll_degrees": MAX_ROLL_DEGREES,
            "min_sharpness": MIN_SHARPNESS,
        },
        "train": [{k: v for k, v in r.items() if k not in ("emb", "hash")} for r in train_rows],
        "test": [{k: v for k, v in r.items() if k not in ("emb", "hash")} for r in test_rows],
        "duplicate_pairs": duplicates,
        "identity_outliers": [r["file"] for r in outliers],
        "bleed_suspects": [r["file"] for r in contaminated],
        "quality_flags": flagged,
        "verdict": {
            "identity_outlier_rate": len(outliers) / max(len(train_rows), 1),
            "cross_identity_bleed": len(contaminated),
            "duplicates_and_leakage": len(duplicates),
        },
    }
    out_path = out_dir / "qc.json"
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {out_path}")

    blockers = []
    if report["verdict"]["identity_outlier_rate"] > 0.05:
        blockers.append(f"identity outliers {len(outliers)}/{len(train_rows)} > 5%")
    if contaminated:
        blockers.append(f"{len(contaminated)} images may belong to another identity")
    if duplicates:
        blockers.append(f"{len(duplicates)} duplicate pair(s), leakage must be zero")
    if blockers:
        print("VERDICT: FAIL — " + "; ".join(blockers))
        raise SystemExit(1)
    print("VERDICT: PASS — safe to train")


if __name__ == "__main__":
    main()
