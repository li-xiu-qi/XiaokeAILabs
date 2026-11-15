#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "用法: bash scripts/read_gguf_meta.sh <model.gguf>" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"
"$PYTHON_BIN" - "$1" <<'PY'
import json
import struct
import sys

SCALAR_FORMATS = {
    0: ("B", 1),   # uint8
    1: ("b", 1),   # int8
    2: ("H", 2),   # uint16
    3: ("h", 2),   # int16
    4: ("I", 4),   # uint32
    5: ("i", 4),   # int32
    6: ("f", 4),   # float32
    7: ("?", 1),   # bool
    10: ("Q", 8),  # uint64
    11: ("q", 8),  # int64
    12: ("d", 8),  # float64
}


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


def read_scalar(handle, fmt, size):
    return struct.unpack("<" + fmt, read_exact(handle, size))[0]


def read_string(handle):
    length = read_scalar(handle, "Q", 8)
    return read_exact(handle, length).decode("utf-8")


def read_value(handle, value_type):
    if value_type == 8:
        return read_string(handle)
    if value_type == 9:
        element_type = read_scalar(handle, "I", 4)
        count = read_scalar(handle, "Q", 8)
        return [read_value(handle, element_type) for _ in range(count)]
    if value_type in SCALAR_FORMATS:
        fmt, size = SCALAR_FORMATS[value_type]
        return read_scalar(handle, fmt, size)
    raise ValueError(f"Unsupported GGUF metadata type {value_type}")


def read_gguf_metadata(path):
    with open(path, "rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = read_scalar(handle, "I", 4)
        tensor_count = read_scalar(handle, "Q", 8)
        metadata_kv_count = read_scalar(handle, "Q", 8)
        metadata = {
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_kv_count,
        }
        for _ in range(metadata_kv_count):
            key = read_string(handle)
            value_type = read_scalar(handle, "I", 4)
            metadata[key] = read_value(handle, value_type)
        return metadata


metadata = read_gguf_metadata(sys.argv[1])
print(json.dumps(metadata, indent=2, ensure_ascii=False))
PY
