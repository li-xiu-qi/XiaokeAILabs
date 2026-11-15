#!/usr/bin/env bash
# prepare_ffmpeg_prefix.sh — reproducible FFmpeg 6.x runtime prefix for torchcodec
#
# Why this repair exists: torchcodec ships one backend per FFmpeg major version
# (libtorchcodec_core4..core8) that dlopen the sonames libavutil.so.56 .. .so.60.
# PyAV wheels bundle FFmpeg under hashed file names (e.g.
# libavutil-<hash>.so.60.8.100), so a directory of PyAV libraries can never
# satisfy those sonames; that is why pointing LD_LIBRARY_PATH at av.libs failed.
# The reproducible repair is a private prefix built from Ubuntu arm64 FFmpeg
# 6.1.1 runtime packages, whose libavutil.so.58 satisfies the core6 backend.
#
# Steps (any failure exits immediately; nothing is retried or re-versioned):
#   1. verify the .deb source directories exist and contain packages
#   2. create a fresh prefix (refuse an existing non-empty one)
#   3. dpkg-deb -x every .deb into the prefix (no maintainer scripts run)
#   4. verify the required sonames and the blas/ lapack/ subdirectories
#   5. write ld-path.txt / libdir.txt for callers
#   6. gate: import torchcodec and open a recorded video with AudioDecoder
#      under the prepared LD_LIBRARY_PATH
#
# Scope: shared libraries only. No system directories, no model weights, no
# training, no network downloads.
#
# Env:
#   FFMPEG_PREFIX   target prefix; default $XL_DATA_DIR/ffmpeg6-prefix-$XL_RUN_ID
#                   when both are set (remote runs), else <experiment>/.work/
#                   ffmpeg6-prefix. Must be absent or empty.
#   FFMPEG_DEB_DIRS space-separated directories holding the FFmpeg runtime
#                   .deb files (required; see README for how to obtain them).
#   GATE_PYTHON     interpreter with torchcodec installed (required).
#   GATE_VIDEO      optional recorded video; the gate opens it with
#                   AudioDecoder and requires more than zero samples.
#   XL_OUT_DIR      when set, summary.json is written here, else into the prefix.
#
# Output: summary.json with public information only (unpacked package names and
# versions, library directory, soname verification, gate result, failing
# stage/error). Hostnames, usernames, addresses and private paths are scrubbed.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

if [ -z "${FFMPEG_DEB_DIRS:-}" ]; then
  echo "FATAL: FFMPEG_DEB_DIRS is required (space-separated directories with the FFmpeg runtime .deb files)" >&2
  exit 2
fi
GATE_PYTHON=${GATE_PYTHON:?GATE_PYTHON required (interpreter with torchcodec installed)}
if [ -z "${FFMPEG_PREFIX:-}" ]; then
  if [ -n "${XL_DATA_DIR:-}" ] && [ -n "${XL_RUN_ID:-}" ]; then
    FFMPEG_PREFIX="$XL_DATA_DIR/ffmpeg6-prefix-$XL_RUN_ID"
  else
    FFMPEG_PREFIX="$EXPERIMENT_DIR/.work/ffmpeg6-prefix"
  fi
fi

STAGE=start
TMP_DIR="$FFMPEG_PREFIX/.prepare-tmp"
if [ -n "${XL_OUT_DIR:-}" ]; then
  SUMMARY_PATH="$XL_OUT_DIR/summary.json"
  mkdir -p "$XL_OUT_DIR"
else
  SUMMARY_PATH="$FFMPEG_PREFIX/prepare-summary.json"
fi

# FFmpeg 6.1.1 sonames satisfied by the core6 backend of torchcodec 0.15.
REQUIRED_SONAMES="libavutil.so.58 libavcodec.so.60 libavformat.so.60 libavdevice.so.60 libavfilter.so.9 libswscale.so.7 libswresample.so.4"

arch_lib_dir() {
  case "$(uname -m)" in
    aarch64) echo "aarch64-linux-gnu" ;;
    x86_64) echo "x86_64-linux-gnu" ;;
    *) echo "" ;;
  esac
}
ARCH_DIR=$(arch_lib_dir)
if [ -z "$ARCH_DIR" ]; then
  echo "FATAL: unsupported architecture $(uname -m); extend arch_lib_dir" >&2
  exit 2
fi
LIB_DIR="$FFMPEG_PREFIX/usr/lib/$ARCH_DIR"

finalize() {
  local rc=$?
  local fin_py
  fin_py=$(command -v python3 || command -v python || echo "$GATE_PYTHON")
  "$fin_py" - "$TMP_DIR" "$SUMMARY_PATH" "$STAGE" "$FFMPEG_PREFIX" "${XL_DATA_DIR:-}" \
    "$GATE_PYTHON" "$rc" "$ARCH_DIR" <<'PYEOF' || echo "[prepare] WARNING: summary finalize failed" >&2
import json
import os
import sys

tmp, out_path, stage, prefix, data_dir, gate_python, rc, arch_dir = sys.argv[1:9]
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
        for secret in (prefix, data_dir, os.path.expanduser("~"), gate_python):
            if secret:
                value = value.replace(secret, "<private>")
        return value[-1500:]
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


debs = load("debs.json") or {"count": 0, "packages": []}
sonames = load("sonames.json") or {"ok": None, "missing": None}
gate = load("gate.json") or {"ok": None, "error": None}

ok = rc == 0 and stage == "complete"
summary = {
    "stage": "complete" if ok else stage,
    "ok": ok,
    "error": gate.get("error") if not ok else None,
    "architecture": arch_dir,
    "deb_sources": {
        "count": debs.get("count"),
        "packages": debs.get("packages"),
    },
    "library_dir": "usr/lib/" + arch_dir,
    "required_sonames": sonames,
    "gate": gate,
}
summary = scrub(summary)
out_dir = os.path.dirname(os.path.abspath(out_path))
os.makedirs(out_dir, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[prepare] summary: stage={summary['stage']} ok={summary['ok']} -> {out_path}")
PYEOF
  rm -rf "$TMP_DIR" 2>/dev/null || true
}
trap 'finalize' EXIT

if [ -e "$FFMPEG_PREFIX" ] && [ -n "$(ls -A "$FFMPEG_PREFIX" 2>/dev/null)" ]; then
  echo "FATAL: FFMPEG_PREFIX already exists and is not empty: $(basename "$FFMPEG_PREFIX")" >&2
  STAGE=prefix_create
  exit 1
fi
mkdir -p "$TMP_DIR" "$LIB_DIR"

# ---------------------------------------------------------------- 1. deb sources
STAGE=deb_sources
DEB_FILES=()
for dir in $FFMPEG_DEB_DIRS; do
  if [ ! -d "$dir" ]; then
    echo "FATAL: FFmpeg .deb directory does not exist: $(basename "$dir")" >&2
    exit 1
  fi
  while IFS= read -r f; do
    DEB_FILES+=("$f")
  done < <(find "$dir" -maxdepth 1 -name '*.deb' -type f | sort)
done
if [ "${#DEB_FILES[@]}" -eq 0 ]; then
  echo "FATAL: no .deb files found in FFMPEG_DEB_DIRS" >&2
  exit 1
fi
echo "[prepare] found ${#DEB_FILES[@]} .deb files"

command -v dpkg-deb >/dev/null || { echo "FATAL: dpkg-deb not available" >&2; exit 1; }

# ---------------------------------------------------------------- 2. unpack
STAGE=unpack
UNPACKED=0
for deb in "${DEB_FILES[@]}"; do
  if ! dpkg-deb -x "$deb" "$FFMPEG_PREFIX" >>"$TMP_DIR/unpack.log" 2>&1; then
    echo "FATAL: dpkg-deb -x failed for $(basename "$deb")" >&2
    tail -5 "$TMP_DIR/unpack.log" >&2 || true
    exit 1
  fi
  UNPACKED=$((UNPACKED + 1))
done
echo "[prepare] unpacked $UNPACKED packages into the prefix"

python3 - "$TMP_DIR" "${DEB_FILES[@]}" <<'PYEOF'
import json
import subprocess
import sys

tmp = sys.argv[1]
debs = sys.argv[2:]
packages = []
for deb in debs:
    try:
        out = subprocess.run(
            ["dpkg-deb", "-f", deb, "Package", "Version", "Architecture"],
            capture_output=True, text=True, check=True).stdout
        fields = dict(
            line.split(": ", 1)
            for line in out.strip().splitlines()
            if ": " in line
        )
        packages.append(
            {
                "package": fields.get("Package"),
                "version": fields.get("Version"),
                "architecture": fields.get("Architecture"),
            }
        )
    except Exception:
        packages.append({"package": deb.rsplit("/", 1)[-1], "version": None, "architecture": None})
packages.sort(key=lambda p: (p.get("package") or ""))
with open(f"{tmp}/debs.json", "w", encoding="utf-8") as f:
    json.dump({"count": len(packages), "packages": packages}, f, indent=2)
PYEOF

# ---------------------------------------------------------------- 3. verify ABI
STAGE=verify_sonames
MISSING=()
for soname in $REQUIRED_SONAMES; do
  if [ ! -e "$LIB_DIR/$soname" ]; then
    MISSING+=("$soname")
  fi
done
BLAS_OK=false
LAPACK_OK=false
if [ -d "$LIB_DIR/blas" ] && ls "$LIB_DIR/blas"/*.so* >/dev/null 2>&1; then
  BLAS_OK=true
fi
if [ -d "$LIB_DIR/lapack" ] && ls "$LIB_DIR/lapack"/*.so* >/dev/null 2>&1; then
  LAPACK_OK=true
fi
python3 - "$TMP_DIR" "$BLAS_OK" "$LAPACK_OK" "${MISSING[@]:-}" <<'PYEOF'
import json
import sys

tmp, blas_ok, lapack_ok = sys.argv[1], sys.argv[2], sys.argv[3]
missing = [m for m in sys.argv[4:] if m]
state = {
    "ok": not missing and blas_ok == "true" and lapack_ok == "true",
    "missing_sonames": missing,
    "blas_subdir": blas_ok == "true",
    "lapack_subdir": lapack_ok == "true",
}
with open(f"{tmp}/sonames.json", "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
PYEOF
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "FATAL: required FFmpeg sonames missing from the prefix: ${MISSING[*]}" >&2
  exit 1
fi
if [ "$BLAS_OK" != true ] || [ "$LAPACK_OK" != true ]; then
  echo "FATAL: blas/ or lapack/ subdirectory missing from the prefix (blas=$BLAS_OK lapack=$LAPACK_OK)" >&2
  exit 1
fi
echo "[prepare] required sonames present; blas/ and lapack/ subdirectories present"

# ---------------------------------------------------------------- 4. paths for callers
STAGE=write_paths
LD_PATH="$LIB_DIR:$LIB_DIR/blas:$LIB_DIR/lapack"
printf '%s\n' "$LD_PATH" > "$FFMPEG_PREFIX/ld-path.txt"
printf '%s\n' "$LIB_DIR" > "$FFMPEG_PREFIX/libdir.txt"
if [ -n "${XL_OUT_DIR:-}" ]; then
  printf '%s\n' "$LD_PATH" > "$XL_OUT_DIR/ld-path.txt"
  printf '%s\n' "$LIB_DIR" > "$XL_OUT_DIR/libdir.txt"
fi
echo "[prepare] LD_LIBRARY_PATH for torchcodec: (prefix usr/lib/$ARCH_DIR plus blas/ and lapack/)"

# ---------------------------------------------------------------- 5. load gate
STAGE=gate
GATE_RC=0
LD_LIBRARY_PATH="$LD_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  GATE_VIDEO="${GATE_VIDEO:-}" "$GATE_PYTHON" - > "$TMP_DIR/gate.out" 2>&1 <<'PYEOF' || GATE_RC=$?
import os

import torchcodec
from torchcodec.decoders import AudioDecoder

print("torchcodec", torchcodec.__version__)
video = os.environ.get("GATE_VIDEO") or ""
if video:
    decoder = AudioDecoder(video)
    samples = decoder.get_all_samples()
    frames = int(samples.data.shape[0]) if samples.data.dim() >= 1 else 0
    if frames <= 0:
        raise RuntimeError("AudioDecoder decoded zero samples")
    print("gate_decode_ok frames", frames)
print("GATE_OK")
PYEOF
if [ "$GATE_RC" -ne 0 ]; then
  echo "FATAL: torchcodec load gate failed under the prepared LD_LIBRARY_PATH" >&2
  tail -15 "$TMP_DIR/gate.out" >&2 || true
fi
cat "$TMP_DIR/gate.out" || true
python3 - "$TMP_DIR" "$GATE_RC" <<'PYEOF'
import json
import sys

tmp, rc = sys.argv[1], int(sys.argv[2])
try:
    with open(f"{tmp}/gate.out", encoding="utf-8", errors="replace") as f:
        tail = f.read().strip()
except OSError:
    tail = ""
state = {
    "ok": rc == 0,
    "torchcodec_import": "ok" if rc == 0 else "failed",
    "video_decode": "ok" if rc == 0 and "gate_decode_ok" in tail else ("not_run" if not tail or "GATE_OK" in tail else "failed"),
    "error": None if rc == 0 else tail[-1200:],
}
with open(f"{tmp}/gate.json", "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
PYEOF
[ "$GATE_RC" -eq 0 ] || exit 1

STAGE=complete
echo "[prepare] FFmpeg prefix ready and torchcodec gate passed"
exit 0
