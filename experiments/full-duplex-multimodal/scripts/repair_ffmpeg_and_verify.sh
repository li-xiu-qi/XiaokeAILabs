#!/usr/bin/env bash
# repair_ffmpeg_and_verify.sh — one bounded FFmpeg/TorchCodec ABI repair run.
#
# Stages, in order, stopping at the first failure (no retry, no re-versioning):
#   A. ffmpeg-prefix   : prepare an FFmpeg 6.1.1 runtime prefix from cached
#                        Ubuntu arm64 .deb files (scripts/prepare_ffmpeg_prefix.sh)
#                        and gate it by importing torchcodec and decoding the
#                        recorded video with AudioDecoder.
#   B. env-validation  : create an empty venv, install the pinned realtime
#                        dependencies, run pip check, imports, CUDA and WAV
#                        decode with FFMPEG_LIB_DIR pointing at the prefix
#                        (scripts/validate_realtime_environment.sh).
#   C. demo-probe      : start the official Demo stack and run exactly one
#                        video-mode GPU probe on the fixed recorded material,
#                        capped at 60 seconds (scripts/run_demo_verify.sh).
#
# Bounded by design: at most one probe, session cap <= 60 s, debs and wheels
# came from caches in the successful run (one index metadata fetch), no model weights are fetched,
# and caches are written only under XL_DATA_DIR. Any stage failure stops the
# run and leaves failure evidence in XL_OUT_DIR/summary.json.
#
# Required env (no private defaults; the caller supplies device paths):
#   XL_OUT_DIR   output root; stage results go to subdirectories
#   XL_DATA_DIR  shared data root on the device (venv, prefix, caches)
#   XL_RUN_ID    run identifier (venv and prefix names include it)
#   DEMO_DIR     MiniCPM-o-Demo source tree (gateway.py, worker.py, py_backend/)
#   MODEL_DIR    local MiniCPM-o-4.5 snapshot directory
#   DEMO_PYTHON  python interpreter of the demo environment (has torchcodec)
#   VIDEO        fixed recorded input video
#   FFMPEG_DEB_DIRS  space-separated directories with the FFmpeg .deb files
# Optional env:
#   PYTHON_BIN           venv creator (default python3.12)
#   PROBE_MAX_SESSION_S  probe session cap (default 60; values above 60 are
#                        rejected to keep the run bounded)
#   PROBE_KIND=barge-in  run the recorded interruption probe after environment validation
#   INTERRUPT_VIDEO      second recorded input clip (required for barge-in)
#   HF_HOME              HuggingFace cache (default $XL_DATA_DIR/huggingface)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREPARE_SCRIPT=$SCRIPT_DIR/prepare_ffmpeg_prefix.sh
VALIDATE_SCRIPT=$SCRIPT_DIR/validate_realtime_environment.sh
DEMO_SCRIPT=$SCRIPT_DIR/run_demo_verify.sh

: "${XL_OUT_DIR:?XL_OUT_DIR required}"
: "${XL_DATA_DIR:?XL_DATA_DIR required}"
: "${XL_RUN_ID:?XL_RUN_ID required}"
: "${DEMO_DIR:?DEMO_DIR required}"
: "${MODEL_DIR:?MODEL_DIR required}"
: "${DEMO_PYTHON:?DEMO_PYTHON required}"
: "${VIDEO:?VIDEO required}"
: "${FFMPEG_DEB_DIRS:?FFMPEG_DEB_DIRS required}"

PYTHON_BIN=${PYTHON_BIN:-python3.12}
PROBE_MAX_SESSION_S=${PROBE_MAX_SESSION_S:-60}
PROBE_KIND=${PROBE_KIND:-stream}
HF_HOME=${HF_HOME:-$XL_DATA_DIR/huggingface}
if [ "$PROBE_MAX_SESSION_S" -gt 60 ]; then
  echo "FATAL: PROBE_MAX_SESSION_S must not exceed 60 for this bounded repair run" >&2
  exit 2
fi

# Preconditions: fixed inputs and tools must exist before anything is prepared.
for path in "$VIDEO" "$DEMO_DIR" "$MODEL_DIR"; do
  [ -e "$path" ] || { echo "FATAL: required path missing: $(basename "$path")" >&2; exit 2; }
done
if [ "$PROBE_KIND" = barge-in ]; then
  : "${INTERRUPT_VIDEO:?INTERRUPT_VIDEO required for PROBE_KIND=barge-in}"
  [ -f "$INTERRUPT_VIDEO" ] || { echo "FATAL: INTERRUPT_VIDEO is missing" >&2; exit 2; }
  if [ "$PROBE_MAX_SESSION_S" -gt 45 ]; then
    echo "FATAL: PROBE_MAX_SESSION_S must not exceed 45 for barge-in" >&2
    exit 2
  fi
fi
[ -x "$DEMO_PYTHON" ] || { echo "FATAL: DEMO_PYTHON is not executable" >&2; exit 2; }
for f in gateway.py worker.py py_backend; do
  [ -e "$DEMO_DIR/$f" ] || { echo "FATAL: DEMO_DIR missing $f" >&2; exit 2; }
done
command -v "$PYTHON_BIN" >/dev/null || { echo "FATAL: PYTHON_BIN not found: $PYTHON_BIN" >&2; exit 2; }

STAGE_OUT=$XL_OUT_DIR
STAGES_FILE=$STAGE_OUT/stages.tsv
: > "$STAGES_FILE"

log() { echo "[repair] $(date -u +%H:%M:%S) $*"; }

record_stage() {
  # name ok exit_code error
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(printf '%s' "${4:-}" | tr '\n\t' '  ' | cut -c1-400)" >> "$STAGES_FILE"
}

write_summary() {
  python3 - "$STAGE_OUT" "$STAGES_FILE" "$XL_DATA_DIR" "$DEMO_DIR" "$MODEL_DIR" \
    "$DEMO_PYTHON" "$VIDEO" "$FFMPEG_DEB_DIRS" <<'PYEOF'
import json
import os
import sys

out, stages_file = sys.argv[1], sys.argv[2]
secrets = [s for s in sys.argv[3:8] if s] + [os.path.expanduser("~")]


def scrub(value):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "<private>")
        return value[-1200:]
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def load(name):
    path = os.path.join(out, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


stages = []
with open(stages_file, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        while len(parts) < 4:
            parts.append("")
        stages.append(
            {"stage": parts[0], "ok": parts[1] == "true", "exit_code": int(parts[2] or 0), "error": parts[3] or None}
        )

env = load("env-validation/summary.json") or {}
probe_summary = load("demo-probe/summary.json") or {}
probe = probe_summary.get("probe") or {}
prefix = load("ffmpeg-prefix/summary.json") or {}

overall_ok = all(s["ok"] for s in stages) and bool(stages)
summary = {
    "run": (
        "full-duplex-barge-in-probe"
        if probe_summary.get("probe_kind") == "barge-in"
        else "ffmpeg-abi-repair-and-verify"
    ),
    "ok": overall_ok,
    "stages": stages,
    "ffmpeg_prefix": {
        "ok": prefix.get("ok"),
        "library_dir": prefix.get("library_dir"),
        "deb_count": (prefix.get("deb_sources") or {}).get("count"),
        "ffmpeg_packages": [
            p
            for p in (prefix.get("deb_sources") or {}).get("packages") or []
            if (p.get("package") or "").startswith(("libav", "libsw", "libpostproc"))
        ],
        "gate": prefix.get("gate"),
    },
    "environment": {
        "python": (env.get("python") or {}).get("version"),
        "pip": (env.get("python") or {}).get("pip"),
        "pip_check": env.get("pip_check"),
        "imports": env.get("imports"),
        "packages": env.get("packages"),
        "cuda": env.get("cuda"),
        "decode": env.get("decode"),
    },
    "probe": {
        "mode": probe_summary.get("mode"),
        "ok": probe_summary.get("ok"),
        "exit_code": probe_summary.get("probe_exit"),
        "success": probe.get("success"),
        "end_reason": probe.get("end_reason"),
        "input_chunks_sent": probe.get("input_chunks_sent"),
        "input_frames_sent": probe.get("input_frames_sent"),
        "vision_metrics_deltas": probe.get("vision_metrics_deltas"),
        "output_audio_chunks": probe.get("output_audio_chunks"),
        "output_audio_ms": probe.get("output_audio_ms"),
        "output_audio_rms": probe.get("output_audio_rms"),
        "first_audio_ms": probe.get("first_audio_ms"),
        "overlap_ms": probe.get("overlap_ms"),
        "listen_events": probe.get("listen_events"),
        "probe_kind": probe_summary.get("probe_kind"),
        "initial_input_chunks": probe.get("initial_input_chunks"),
        "interrupt_input_chunks": probe.get("interrupt_input_chunks"),
        "assistant_audio_chunks_before_interrupt": probe.get("assistant_audio_chunks_before_interrupt"),
        "assistant_audio_chunks_after_interrupt_started": probe.get("assistant_audio_chunks_after_interrupt_started"),
        "first_assistant_audio_ms": probe.get("first_assistant_audio_ms"),
        "interruption_started_ms": probe.get("interruption_started_ms"),
        "last_assistant_audio_delta_after_interrupt_ms": probe.get("last_assistant_audio_delta_after_interrupt_ms"),
        "listen_events_after_interrupt_ms": probe.get("listen_events_after_interrupt_ms"),
        "error": probe.get("error"),
    },
    "boundaries": {
        "single_probe": True,
        "max_session_s": 60,
        "sources": "package wheels and FFmpeg .deb files from device caches; one PyTorch index metadata fetch after cache expiry with run-scoped proxy variables cleared; no model fetch",
        "caches": "written under XL_DATA_DIR only",
        "input": "fixed recorded material supplied by the caller",
    },
}
summary = scrub(summary)
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[repair] summary: ok={summary['ok']} stages={[s['stage'] for s in stages]}")
PYEOF
}

log "=== stage A: prepare FFmpeg prefix ==="
STAGE_A_RC=0
XL_OUT_DIR=$STAGE_OUT/ffmpeg-prefix \
  FFMPEG_PREFIX=$XL_DATA_DIR/ffmpeg6-prefix-$XL_RUN_ID \
  FFMPEG_DEB_DIRS="$FFMPEG_DEB_DIRS" \
  GATE_PYTHON=$DEMO_PYTHON \
  GATE_VIDEO=$VIDEO \
  bash "$PREPARE_SCRIPT" || STAGE_A_RC=$?
if [ "$STAGE_A_RC" -ne 0 ]; then
  record_stage ffmpeg_prefix false "$STAGE_A_RC" "prepare_ffmpeg_prefix.sh failed"
  write_summary
  log "FATAL: FFmpeg prefix preparation failed; stopping before environment validation"
  exit 1
fi
LIB_DIR=$(cat "$STAGE_OUT/ffmpeg-prefix/libdir.txt")
record_stage ffmpeg_prefix true 0 ""
log "FFmpeg prefix ready; torchcodec gate passed"

log "=== stage B: empty-venv environment validation against the prefix ==="
STAGE_B_RC=0
XL_OUT_DIR=$STAGE_OUT/env-validation \
  XL_DATA_DIR=$XL_DATA_DIR \
  XL_RUN_ID=$XL_RUN_ID \
  PYTHON_BIN=$PYTHON_BIN \
  VENV_DIR=$XL_DATA_DIR/venv-realtime-$XL_RUN_ID \
  FFMPEG_LIB_DIR=$LIB_DIR \
  bash "$VALIDATE_SCRIPT" || STAGE_B_RC=$?
if [ "$STAGE_B_RC" -ne 0 ]; then
  record_stage env_validation false "$STAGE_B_RC" "validate_realtime_environment.sh failed"
  write_summary
  log "FATAL: environment validation failed; stopping before the GPU probe"
  exit 1
fi
record_stage env_validation true 0 ""
log "environment validation passed (pip check, imports, CUDA, decode)"

log "=== stage C: one $PROBE_KIND video GPU probe (<= ${PROBE_MAX_SESSION_S}s) ==="
STAGE_C_RC=0
XL_OUT_DIR=$STAGE_OUT/demo-probe \
  PROBE_MODE=video \
  PROBE_KIND="$PROBE_KIND" \
  PROBE_MAX_SESSION_S=$PROBE_MAX_SESSION_S \
  DEMO_DIR=$DEMO_DIR \
  MODEL_DIR=$MODEL_DIR \
  PYTHON=$DEMO_PYTHON \
  VIDEO=$VIDEO \
  INITIAL_VIDEO="$VIDEO" \
  INTERRUPT_VIDEO="${INTERRUPT_VIDEO:-}" \
  PROBE_CHUNK_MS="${PROBE_CHUNK_MS:-160}" \
  PROBE_INITIAL_SECONDS="${PROBE_INITIAL_SECONDS:-4}" \
  PROBE_INTERRUPT_SECONDS="${PROBE_INTERRUPT_SECONDS:-4}" \
  PROBE_OBSERVE_AFTER_S="${PROBE_OBSERVE_AFTER_S:-8}" \
  HF_HOME=$HF_HOME \
  FFMPEG_LIB_DIR=$LIB_DIR \
  bash "$DEMO_SCRIPT" || STAGE_C_RC=$?
PROBE_OK=false
if [ -f "$STAGE_OUT/demo-probe/summary.json" ]; then
  PROBE_OK=$(python3 -c "import json;print(str(json.load(open('$STAGE_OUT/demo-probe/summary.json')).get('ok')).lower())" 2>/dev/null || echo false)
fi
if [ "$STAGE_C_RC" -ne 0 ] || [ "$PROBE_OK" != true ]; then
  record_stage demo_probe false "$STAGE_C_RC" "run_demo_verify.sh/probe failed"
  write_summary
  log "FATAL: GPU probe failed; evidence retained, no retry"
  exit 1
fi
record_stage demo_probe true 0 ""
write_summary
log "=== repair run complete: all stages passed ==="
exit 0
