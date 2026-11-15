#!/usr/bin/env bash
# validate_realtime_environment.sh — build this experiment's realtime Python
# environment from an empty venv and self-check it.
#
# Steps (any failure exits immediately; no version is retried or changed):
#   1. create an empty venv
#   2. pip install pip==24.0
#   3. pip install torch==2.11.0 torchaudio==2.11.0 (PyTorch cu130 index)
#   4. pip install torchcodec==0.15.0 (same index)
#   5. pip install -r requirements-realtime.txt
#   6. pip check
#   7. import av, torch, torchaudio, torchcodec, transformers, fastapi,
#      websockets, stepaudio2_minicpmo (actual module names are recorded; a
#      differing distribution module name is resolved, never guessed)
#   8. torch.cuda availability
#   9. generate a 1-second WAV with the Python standard library and decode it
#      with TorchCodec AudioDecoder (FFmpeg backend check; samples > 0)
#
# Scope: environment only. No model weights, no training data, no model
# service, no GPU inference beyond the CUDA availability check. Cleanup is
# limited to the generated WAV inside the new venv.
#
# Platform: the pinned wheels come from the PyTorch cu130 index, verified on
# Linux ARM64 with CUDA 13.0. Other platforms need their own matching wheels.
#
# Env:
#   PYTHON_BIN   interpreter used to create the venv (default python3.12)
#   VENV_DIR     venv location; default is $XL_DATA_DIR/venv-realtime-$XL_RUN_ID
#                when both are set (remote runs), else <experiment>/.work/
#                venv-realtime (git-ignored). Must be empty or absent.
#   FFMPEG_LIB_DIR optional FFmpeg shared-library directory. The script adds
#                it and optional sibling blas/ and lapack/ directories to
#                LD_LIBRARY_PATH before importing TorchCodec.
#   XL_OUT_DIR   when set, summary.json is written here (injected by remote runs)
#
# Output: machine-readable summary.json (public information only: system
# architecture/OS, Python and package versions, pip check, CUDA, decode
# result, and the failing stage/error). No hostname, username, addresses or
# private paths are recorded.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-$EXPERIMENT_DIR/requirements-realtime.txt}
PYTORCH_INDEX=${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu130}

PYTHON_BIN=${PYTHON_BIN:-python3.12}
FFMPEG_LIB_DIR=${FFMPEG_LIB_DIR:-}
if [ -z "${VENV_DIR:-}" ]; then
  if [ -n "${XL_DATA_DIR:-}" ] && [ -n "${XL_RUN_ID:-}" ]; then
    VENV_DIR="$XL_DATA_DIR/venv-realtime-$XL_RUN_ID"
  else
    VENV_DIR="$EXPERIMENT_DIR/.work/venv-realtime"
  fi
fi
VENV_PY="$VENV_DIR/bin/python"
TMP_DIR="$VENV_DIR/validation-tmp"
WAV_PATH="$TMP_DIR/tone-1s.wav"
STAGE_ERR_FILE="$TMP_DIR/stage.err"
if [ -n "${XL_OUT_DIR:-}" ]; then
  SUMMARY_PATH="$XL_OUT_DIR/summary.json"
else
  SUMMARY_PATH="$VENV_DIR/validation-summary.json"
fi

STAGE=start

finalize() {
  local rc=$?
  local fin_py
  fin_py=$(command -v python3 || command -v python || echo "$PYTHON_BIN")
  "$fin_py" - "$TMP_DIR" "$SUMMARY_PATH" "$STAGE" "$VENV_DIR" "${XL_DATA_DIR:-}" \
    "$rc" "$STAGE_ERR_FILE" "$PYTHON_BIN" <<'PYEOF' || echo "[validate] WARNING: summary finalize failed" >&2
import json
import os
import sys

tmp, out_path, stage, venv_dir, data_dir, rc, err_file, python_bin = sys.argv[1:9]
rc = int(rc)


def load(name):
    path = os.path.join(tmp, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def scrub(value):
    if isinstance(value, str):
        for secret in (venv_dir, data_dir, os.path.expanduser("~")):
            if secret:
                value = value.replace(secret, "<private>")
        return value[-1500:]
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


system = load("system.json") or {}
pip_info = load("pip.json") or {}
pip_check = load("pip_check.json") or {"ok": None, "output": None}
validate = load("validate.json") or {}

failed_stage = None
error = None
if validate and validate.get("ok") is not True:
    failed_stage = validate.get("stage") or "validate"
    error = validate.get("error")
elif rc != 0:
    failed_stage = stage
    try:
        with open(err_file, encoding="utf-8", errors="replace") as f:
            tail = f.read().strip()
    except OSError:
        tail = ""
    error = tail or f"stage {stage} failed"

ok = rc == 0 and failed_stage is None

summary = {
    "stage": failed_stage or "complete",
    "ok": ok,
    "error": error,
    "system": {
        "system": system.get("system"),
        "machine": system.get("machine"),
        "platform": system.get("platform"),
    },
    "python": {
        "interpreter": python_bin,
        "version": system.get("python"),
        "pip": pip_info.get("pip"),
    },
    "pip_check": {"ok": pip_check.get("ok"), "output": pip_check.get("output")},
    "imports": validate.get("imports"),
    "packages": validate.get("packages"),
    "cuda": validate.get("cuda"),
    "decode": validate.get("decode"),
    "venv": os.path.basename(venv_dir.rstrip("/")) if venv_dir else None,
}
summary = scrub(summary)
out_dir = os.path.dirname(os.path.abspath(out_path))
os.makedirs(out_dir, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[validate] summary: stage={summary['stage']} ok={summary['ok']} -> {out_path}")
PYEOF
}

trap 'finalize; rm -f "$WAV_PATH" 2>/dev/null || true' EXIT

if [ -e "$VENV_DIR" ] && [ -n "$(ls -A "$VENV_DIR" 2>/dev/null)" ]; then
  echo "FATAL: VENV_DIR already exists and is not empty: $(basename "$VENV_DIR")" >&2
  STAGE=venv_create
  exit 1
fi
mkdir -p "$VENV_DIR" "$TMP_DIR"

run_stage() {
  local stage=$1
  shift
  STAGE=$stage
  : > "$STAGE_ERR_FILE"
  echo "[validate] === stage: $stage ==="
  "$@" 2>>"$STAGE_ERR_FILE"
}

run_stage venv_create "$PYTHON_BIN" -m venv "$VENV_DIR"
[ -x "$VENV_PY" ] || { echo "FATAL: venv python missing after creation" >&2; exit 1; }

"$VENV_PY" - "$TMP_DIR" <<'PYEOF'
import json
import os
import platform
import sys

tmp = sys.argv[1]
with open(os.path.join(tmp, "system.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        f,
        indent=2,
    )
PYEOF

run_stage pip_upgrade "$VENV_PY" -m pip install pip==24.0
"$VENV_PY" - "$TMP_DIR" <<'PYEOF'
import json
import os
import subprocess
import sys

tmp = sys.argv[1]
out = subprocess.run(
    [sys.executable, "-m", "pip", "--version"], capture_output=True, text=True
)
parts = out.stdout.strip().split()
version = parts[1] if len(parts) > 1 else None
with open(os.path.join(tmp, "pip.json"), "w", encoding="utf-8") as f:
    json.dump({"pip": version}, f, indent=2)
PYEOF

run_stage install_torch "$VENV_PY" -m pip install \
  torch==2.11.0 torchaudio==2.11.0 --index-url "$PYTORCH_INDEX"
run_stage install_torchcodec "$VENV_PY" -m pip install \
  torchcodec==0.15.0 --index-url "$PYTORCH_INDEX"
run_stage install_requirements "$VENV_PY" -m pip install -r "$REQUIREMENTS_FILE"

STAGE=pip_check
: > "$STAGE_ERR_FILE"
echo "[validate] === stage: pip_check ==="
PIP_CHECK_RC=0
"$VENV_PY" -m pip check > "$TMP_DIR/pip_check.out" 2>>"$STAGE_ERR_FILE" || PIP_CHECK_RC=1
"$VENV_PY" - "$TMP_DIR" "$PIP_CHECK_RC" <<'PYEOF'
import json
import os
import sys

tmp, rc = sys.argv[1], int(sys.argv[2])
with open(os.path.join(tmp, "pip_check.out"), encoding="utf-8", errors="replace") as f:
    output = f.read().strip()
with open(os.path.join(tmp, "pip_check.json"), "w", encoding="utf-8") as f:
    json.dump({"ok": rc == 0, "output": output[-1200:]}, f, indent=2)
PYEOF
[ "$PIP_CHECK_RC" -eq 0 ] || exit 1

if [ -z "$FFMPEG_LIB_DIR" ]; then
  # PyAV Linux wheels may bundle an ABI-compatible FFmpeg runtime in a
  # sibling av.libs directory. Prefer an explicit caller override;
  # otherwise expose the bundled libraries to TorchCodec when present.
  AV_LIB_DIR=$("$VENV_PY" -c 'from pathlib import Path; import av; print(Path(av.__file__).resolve().parent.parent / "av.libs")' 2>/dev/null || true)
  if [ -d "$AV_LIB_DIR" ]; then
    FFMPEG_LIB_DIR=$AV_LIB_DIR
    echo "[validate] using FFmpeg libraries bundled with PyAV"
  fi
fi

if [ -n "$FFMPEG_LIB_DIR" ]; then
  STAGE=ffmpeg_libs
  if [ ! -d "$FFMPEG_LIB_DIR" ]; then
    echo "FATAL: FFMPEG_LIB_DIR does not name an existing directory" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  for dependency_dir in "$FFMPEG_LIB_DIR/blas" "$FFMPEG_LIB_DIR/lapack"; do
    if [ -d "$dependency_dir" ]; then
      export LD_LIBRARY_PATH="$dependency_dir:$LD_LIBRARY_PATH"
    fi
  done
fi

STAGE=validate
: > "$STAGE_ERR_FILE"
echo "[validate] === stage: validate (imports, cuda, decoder) ==="
"$VENV_PY" - "$TMP_DIR" "$WAV_PATH" <<'PYEOF'
import importlib
import importlib.metadata as im
import json
import math
import os
import struct
import sys
import wave

tmp, wav_path = sys.argv[1], sys.argv[2]

state = {
    "stage": "imports",
    "ok": False,
    "error": None,
    "imports": [],
    "packages": {},
    "cuda": None,
    "decode": None,
}


def save():
    with open(os.path.join(tmp, "validate.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fail(stage, exc):
    state["stage"] = stage
    state["error"] = f"{type(exc).__name__}: {exc}"
    save()
    sys.exit(1)


# (requested module, distribution). av is imported first so that any FFmpeg
# libraries bundled with its wheel are loaded before torchcodec initializes.
MODULES = [
    ("av", "av"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("torchcodec", "torchcodec"),
    ("transformers", "transformers"),
    ("fastapi", "fastapi"),
    ("websockets", "websockets"),
    ("stepaudio2_minicpmo", "stepaudio2-minicpmo"),
]


def dist_version(dist):
    try:
        return im.version(dist)
    except Exception:
        return None


def module_version(module):
    for attr in ("__version__", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    return None


def alternative_modules(dist):
    names = []
    try:
        text = im.distribution(dist).read_text("top_level.txt") or ""
        names.extend(n.strip() for n in text.splitlines() if n.strip())
    except Exception:
        pass
    try:
        for modules, dists in im.packages_distributions().items():
            if dist in dists:
                names.append(modules)
    except Exception:
        pass
    unique = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


try:
    for module_name, dist in MODULES:
        try:
            module = importlib.import_module(module_name)
            actual = module_name
        except ModuleNotFoundError as exc:
            actual = None
            for candidate in alternative_modules(dist):
                try:
                    importlib.import_module(candidate)
                except ImportError:
                    continue
                actual = candidate
                break
            if actual is None:
                fail(f"imports:{module_name}", exc)
            module = importlib.import_module(actual)
        state["imports"].append(
            {
                "requested": module_name,
                "module": actual,
                "status": "ok",
                "version": module_version(module) or dist_version(dist),
            }
        )
except SystemExit:
    raise
except Exception as exc:
    fail("imports", exc)

state["stage"] = "packages"
try:
    state["packages"] = {
        dist: version
        for _, dist in MODULES
        if (version := dist_version(dist))
    }
except Exception as exc:
    fail("packages", exc)

state["stage"] = "cuda"
try:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            devices.append(torch.cuda.get_device_name(index))
        except Exception:
            devices.append(None)
    state["cuda"] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }
except SystemExit:
    raise
except Exception as exc:
    fail("cuda", exc)

state["stage"] = "wav_generate"
try:
    sample_rate = 44100
    duration_s = 1.0
    frames = int(sample_rate * duration_s)
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        payload = bytearray()
        for i in range(frames):
            value = int(0.5 * 32767 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            payload += struct.pack("<h", value)
        wav.writeframes(bytes(payload))
except Exception as exc:
    fail("wav_generate", exc)

state["stage"] = "torchcodec_decode"
try:
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(wav_path)
    samples = decoder.get_all_samples()
    data = samples.data
    num_frames = int(data.shape[-1]) if data.dim() >= 1 else 0
    if num_frames <= 0:
        raise RuntimeError("AudioDecoder decoded zero samples")
    sample_rate_out = None
    channels = None
    for source in (samples, getattr(decoder, "metadata", None)):
        if sample_rate_out is None:
            sample_rate_out = getattr(source, "sample_rate", None)
        if channels is None:
            channels = getattr(source, "num_channels", None)
    if channels is None and data.dim() >= 2:
        channels = int(data.shape[0])
    state["decode"] = {
        "ok": True,
        "sample_rate": int(sample_rate_out) if sample_rate_out else None,
        "channels": int(channels) if channels else None,
        "num_frames": num_frames,
        "duration_s": round(num_frames / float(sample_rate_out), 3)
        if sample_rate_out
        else None,
    }
except Exception as exc:
    fail("torchcodec_decode", exc)

state["stage"] = "complete"
state["ok"] = True
save()
print("[validate] imports ok:", ", ".join(item["module"] for item in state["imports"]))
print("[validate] cuda:", json.dumps(state["cuda"]))
print("[validate] decode:", json.dumps(state["decode"]))
PYEOF

STAGE=complete
echo "[validate] all stages passed"
exit 0
