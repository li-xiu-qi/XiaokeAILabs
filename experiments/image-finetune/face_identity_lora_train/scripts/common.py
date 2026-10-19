"""Shared utilities for the face-identity LoRA experiment.

All paths are resolved from environment variables with sane defaults, so the
scripts run unchanged on any machine that has the data and models in place.

    FACE_EXP_ROOT    experiment directory          (default: this repo checkout)
    FACE_DATA_DIR    CelebA-derived train/test dir (default: $FACE_EXP_ROOT/data)
    FACE_MODELS_DIR  insightface model dir         (default: ~/.insightface/models/buffalo_l)
    FACE_OUT_DIR     evaluation outputs            (default: $FACE_EXP_ROOT/output)
    SDXL_BASE        base model path or HF repo id (default: stabilityai/stable-diffusion-xl-base-1.0)

Target identity is `celeb_id=14`; the negative control is `celeb_id=15`.
Both are read from FACE_DATA_DIR as `<data_dir>/celeb_14/{train,test}`.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from types import SimpleNamespace

EXP_ROOT = Path(os.environ.get("FACE_EXP_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("FACE_DATA_DIR", EXP_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("FACE_MODELS_DIR", Path.home() / ".insightface/models/buffalo_l"))
OUT_DIR = Path(os.environ.get("FACE_OUT_DIR", EXP_ROOT / "output"))
SDXL_BASE = os.environ.get("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")

TARGET_ID = int(os.environ.get("FACE_TARGET_ID", 14))
NEGATIVE_ID = int(os.environ.get("FACE_NEGATIVE_ID", 15))
TRIGGER = os.environ.get("FACE_TRIGGER", "vkqzm")
PROMPT = f"{TRIGGER}, a photo of a person, front facing, sharp focus"


def identity_dir(celeb_id: int, split: str = "test") -> Path:
    """Directory holding one identity's images for a split."""
    return DATA_DIR / f"celeb_{celeb_id}" / split


REQUIREMENTS = (
    "numpy, opencv-python, onnxruntime, insightface — install with "
    "`bash scripts/setup_env.sh`"
)


def _require(module: str):
    """Import lazily so `--help` works before the environment is provisioned."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            f"missing dependency: {module}\nrequired: {REQUIREMENTS}"
        ) from exc


class FaceEncoder:
    """ArcFace embedding extractor backed by RetinaFace detection."""

    def __init__(self, models_dir: Path | None = None) -> None:
        np = _require("numpy")
        from insightface.model_zoo import ArcFaceONNX, RetinaFace

        self.np = np
        mdir = Path(models_dir or MODELS_DIR)
        det_path, rec_path = mdir / "det_10g.onnx", mdir / "w600k_r50.onnx"
        for p in (det_path, rec_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"missing model file: {p}\n"
                    f"run scripts/setup_env.sh first, or set FACE_MODELS_DIR"
                )
        self.det = RetinaFace(model_file=str(det_path))
        self.rec = ArcFaceONNX(str(rec_path))
        self.det.prepare(ctx_id=0, input_size=(640, 640))

    def embed(self, img_bgr) -> object:
        """Unit-normalised 512-d embedding of the most confident face, or None."""
        np = self.np
        boxes, kpss = self.det.detect(img_bgr, input_size=(640, 640))
        if not len(boxes):
            return None
        i = int(np.argmax(boxes[:, 4]))
        emb = self.rec.get(img_bgr, SimpleNamespace(kps=kpss[i]))
        return None if emb is None else emb / np.linalg.norm(emb)

    def embed_file(self, path: str | Path):
        cv2 = _require("cv2")
        img = cv2.imread(str(path))
        return None if img is None else self.embed(img)

    def detect_count(self, img_bgr) -> int:
        boxes, _ = self.det.detect(img_bgr, input_size=(640, 640))
        return len(boxes)


def collect_embeddings(encoder: FaceEncoder, directory: Path):
    """Embeddings for every .jpg in a directory, skipping images with no face."""
    np = _require("numpy")
    vectors = []
    for path in sorted(glob.glob(str(Path(directory) / "*.jpg"))):
        emb = encoder.embed_file(path)
        if emb is not None:
            vectors.append(emb)
    if not vectors:
        raise RuntimeError(f"no usable faces in {directory}")
    return np.array(vectors)


def prototype(vectors):
    """Unit-normalised mean embedding, used as an identity reference."""
    np = _require("numpy")
    mean = vectors.mean(axis=0)
    return mean / np.linalg.norm(mean)


def to_rgb_array(pil_image):
    """PIL image -> BGR array, the convention cv2 expects."""
    cv2 = _require("cv2")
    np = _require("numpy")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def prepare_output_dir(*parts: str) -> Path:
    path = OUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
