#!/usr/bin/env python3
"""Download the experiment packages at a fixed revision and verify every required file."""
import argparse
import hashlib
import re
from pathlib import Path, PurePosixPath

MODEL = (
    "Merkyor/Qwen3.8-27B-Coder390-EfficientThink-"
    "Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2"
)
MODEL_REVISION = "196cc2ce161d7239976fcd5dc07adf06ae31a35c"
NINFER_NAME = "Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer"
NINFER_SHA256 = "3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808"
PATTERNS = {
    "ninfer": [
        f"NVFP4-NInfer/W4A4/{NINFER_NAME}",
        "NVFP4-NInfer/W4A4/SHA256SUMS",
    ],
    "nvfp4": ["NVFP4/W4A4/*"],
    "fp8": ["FP8/*"],
}
CHECKSUM_MANIFESTS = {
    "ninfer": [Path("NVFP4-NInfer/W4A4/SHA256SUMS")],
    "nvfp4": [
        Path("NVFP4/W4A4/SHA256SUMS"),
        Path("NVFP4/W4A4/DFlash2-FP8/SHA256SUMS"),
    ],
    "fp8": [
        Path("FP8/SHA256SUMS"),
        Path("FP8/DFlash2-FP8/SHA256SUMS"),
    ],
}


class VerificationError(RuntimeError):
    """Raised when the local package does not match the pinned source."""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_manifest(path):
    entries = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", stripped)
        if not match:
            raise VerificationError(f"invalid checksum line: {path}:{line_number}")
        expected, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise VerificationError(f"unsafe checksum path: {path}:{line_number}")
        key = str(relative)
        if key in entries and entries[key] != expected.lower():
            raise VerificationError(f"conflicting checksum entries: {path}:{line_number}")
        entries[key] = expected.lower()
    if not entries:
        raise VerificationError(f"empty checksum manifest: {path}")
    return entries


def required_entries(variant, entries):
    if variant != "ninfer":
        return entries
    expected = entries.get(NINFER_NAME)
    if expected != NINFER_SHA256:
        raise VerificationError("the NInfer artifact checksum differs from the experiment record")
    return {NINFER_NAME: expected}


def verify_variant(root, variant):
    verified = {}
    for manifest_relative in CHECKSUM_MANIFESTS[variant]:
        manifest = root / manifest_relative
        if not manifest.is_file():
            raise VerificationError(f"missing checksum manifest: {manifest}")
        entries = required_entries(variant, parse_checksum_manifest(manifest))
        for name, expected in entries.items():
            path = manifest.parent / Path(*PurePosixPath(name).parts)
            if not path.is_file():
                raise VerificationError(f"missing model file: {path}")
            actual = sha256(path)
            if actual != expected:
                raise VerificationError(
                    f"SHA256 mismatch: {path}\nexpected: {expected}\nactual:   {actual}"
                )
            relative = path.relative_to(root).as_posix()
            if relative in verified:
                raise VerificationError(f"duplicate verified path: {relative}")
            verified[relative] = path.stat().st_size
            print(f"verified {relative}", flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(
        description="Download one model variant, or verify an existing local copy."
    )
    parser.add_argument("variant", choices=sorted(PATTERNS))
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip network access and verify the existing files with SHA256SUMS",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    root = Path(args.model_root)
    if not args.verify_only:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            message = "modelscope is required for downloading; install requirements.txt"
            raise SystemExit(message) from exc
        root = Path(
            snapshot_download(
                model_id=MODEL,
                revision=MODEL_REVISION,
                local_dir=args.model_root,
                allow_patterns=PATTERNS[args.variant],
                max_workers=args.workers,
            )
        )

    try:
        verified = verify_variant(root, args.variant)
    except (OSError, VerificationError) as exc:
        raise SystemExit(f"verification failed: {exc}") from exc
    total_bytes = sum(verified.values())
    print(
        f"verified {len(verified)} files, {total_bytes} bytes, "
        f"variant={args.variant}, revision={MODEL_REVISION}"
    )


if __name__ == "__main__":
    main()
