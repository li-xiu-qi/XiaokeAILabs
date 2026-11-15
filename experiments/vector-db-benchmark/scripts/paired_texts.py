"""Check that vector and text archives use the same ordered input texts."""

import hashlib
import struct

import numpy as np


def text_digest(texts):
    """Hash UTF-8 texts with length framing, including order and empty strings."""
    digest = hashlib.sha256(b"20news-texts-v1\0")
    for text in texts:
        if not isinstance(text, str):
            raise ValueError("Texts must be Unicode strings")
        raw = text.encode("utf-8")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def load_aligned_texts(embedding_path, text_path):
    """Return train/test texts only after checking provenance and row counts.

    Old archives lack content hashes. Matching names or lengths cannot prove that
    the current corpus contains the texts used to generate historical vectors.
    """
    with np.load(embedding_path, allow_pickle=False) as embeddings, np.load(
        text_path, allow_pickle=False
    ) as text_file:
        result = {}
        for split in ("train", "test"):
            digest_key = f"{split}_text_sha256"
            text_key = f"{split}_texts"
            vector_key = f"{split}_emb"
            if digest_key not in embeddings.files or digest_key not in text_file.files:
                raise ValueError(
                    "Archives have no paired text hashes; generate both together with "
                    "scripts/gen_emb.py --texts-output. Historical alignment is unknown."
                )
            vectors = embeddings[vector_key]
            texts = text_file[text_key]
            if vectors.ndim != 2 or texts.ndim != 1 or texts.dtype.kind != "U":
                raise ValueError(f"Invalid {split} vector or text array")
            if len(vectors) != len(texts):
                raise ValueError(f"{split} vector/text row counts differ")
            actual = text_digest(texts.tolist())
            if actual != str(embeddings[digest_key].item()) or actual != str(
                text_file[digest_key].item()
            ):
                raise ValueError(f"{split} text content or ordering does not match embeddings")
            result[split] = texts.tolist()
        return result["train"], result["test"]
