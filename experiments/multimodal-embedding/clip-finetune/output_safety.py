"""Prevent a training run from reusing files from an earlier run."""

import os


def require_empty_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        raise FileExistsError(f"Output directory is not empty: {path}; choose a new directory")
    # Claim the directory atomically so two training processes cannot both start there.
    claim = os.path.join(path, ".training-started")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Output directory already claimed: {path}; choose a new directory") from exc
    os.close(fd)
