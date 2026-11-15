#!/usr/bin/env bash
# Start the official MiniCPM-o-Demo stack (py_backend server -> worker ->
# gateway) on this device, run the realtime duplex probe against it, then stop
# every service started here and verify the ports are released.
#
# Machine-specific paths are injected through environment variables by the
# caller; nothing device-specific is hardcoded.
#
# Required env:
#   DEMO_DIR   MiniCPM-o-Demo source tree (contains gateway.py, worker.py, py_backend/)
#   MODEL_DIR  local MiniCPM-o-4.5 snapshot directory
#   PYTHON     python interpreter of the isolated demo environment
#   VIDEO      recorded input video (fixed material, e.g. assets/omni_duplex1.mp4)
#   XL_OUT_DIR output directory for logs and probe results (set by the caller)
# Optional env:
#   PROBE_KIND stream (default) or barge-in; each invocation runs one probe
#   PROBE_MODE video (default) or audio for the stream probe
#   PROBE_CHUNK_MS audio chunk size sent per input.append, 1-5000 (default 1000)
#   INITIAL_VIDEO / INTERRUPT_VIDEO for PROBE_KIND=barge-in
#   HF_HOME    HuggingFace cache directory
#   REF_AUDIO  reference audio for the assistant voice (demo default when unset)

set -uo pipefail

DEMO_DIR=${DEMO_DIR:?DEMO_DIR required}
MODEL_DIR=${MODEL_DIR:?MODEL_DIR required}
PYTHON=${PYTHON:?PYTHON required}
VIDEO=${VIDEO:?VIDEO required}
PROBE_MODE=${PROBE_MODE:-video}
PROBE_KIND=${PROBE_KIND:-stream}
case "$PROBE_MODE" in
  video|audio) ;;
  *) echo "FATAL: PROBE_MODE must be video or audio" >&2; exit 2 ;;
esac
case "$PROBE_KIND" in
  stream|barge-in) ;;
  *) echo "FATAL: PROBE_KIND must be stream or barge-in" >&2; exit 2 ;;
esac
if [ "$PROBE_KIND" = barge-in ]; then
  PROBE_CHUNK_MS=${PROBE_CHUNK_MS:-160}
else
  PROBE_CHUNK_MS=${PROBE_CHUNK_MS:-1000}
fi
case "$PROBE_CHUNK_MS" in
  ''|*[!0-9]*) echo "FATAL: PROBE_CHUNK_MS must be an integer in 1-5000" >&2; exit 2 ;;
esac
if [ "$PROBE_CHUNK_MS" -lt 1 ] || [ "$PROBE_CHUNK_MS" -gt 5000 ]; then
  echo "FATAL: PROBE_CHUNK_MS must be an integer in 1-5000" >&2
  exit 2
fi
OUT_DIR=${XL_OUT_DIR:-out}
export HF_HOME=${HF_HOME:-}
# torchcodec in the demo backend needs FFmpeg shared libraries; point
# LD_LIBRARY_PATH at a directory providing them when the device has no system
# FFmpeg (optional; leave unset when the system already provides them).
FFMPEG_LIB_DIR=${FFMPEG_LIB_DIR:-}
if [ -n "$FFMPEG_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # Ubuntu's extracted FFmpeg dependencies may put BLAS and LAPACK libraries
  # in sibling subdirectories; the dynamic linker does not search recursively.
  for dependency_dir in "$FFMPEG_LIB_DIR/blas" "$FFMPEG_LIB_DIR/lapack"; do
    if [ -d "$dependency_dir" ]; then
      export LD_LIBRARY_PATH="$dependency_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  done
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ "$PROBE_KIND" = barge-in ]; then
  [ "$PROBE_MODE" = video ] || { echo "FATAL: barge-in probe requires PROBE_MODE=video" >&2; exit 2; }
  INITIAL_VIDEO=${INITIAL_VIDEO:?INITIAL_VIDEO required for PROBE_KIND=barge-in}
  INTERRUPT_VIDEO=${INTERRUPT_VIDEO:?INTERRUPT_VIDEO required for PROBE_KIND=barge-in}
  [ -f "$INITIAL_VIDEO" ] || { echo "FATAL: INITIAL_VIDEO not found" >&2; exit 2; }
  [ -f "$INTERRUPT_VIDEO" ] || { echo "FATAL: INTERRUPT_VIDEO not found" >&2; exit 2; }
  if [ "${PROBE_MAX_SESSION_S:-45}" -gt 45 ]; then
    echo "FATAL: PROBE_MAX_SESSION_S must not exceed 45 for the barge-in probe" >&2
    exit 2
  fi
  PROBE_SCRIPT=${PROBE_SCRIPT:-$SCRIPT_DIR/demo_barge_in_probe.py}
  PROBE_OUTPUT_NAME=probe_barge_in_metrics.json
  PROBE_LOG_NAME=probe_barge_in.log
else
  PROBE_SCRIPT=${PROBE_SCRIPT:-$SCRIPT_DIR/demo_realtime_probe.py}
  PROBE_OUTPUT_NAME=probe_${PROBE_MODE}_metrics.json
  PROBE_LOG_NAME=probe_${PROBE_MODE}.log
fi

BACKEND_PORT=${BACKEND_PORT:-22500}
WORKER_PORT=${WORKER_PORT:-22400}
GATEWAY_PORT=${GATEWAY_PORT:-8006}
INTERNAL_PORT=${INTERNAL_PORT:-8007}

LOG_DIR=$OUT_DIR/logs
mkdir -p "$LOG_DIR"
SUMMARY=$OUT_DIR/summary.json

BACKEND_PID=""
WORKER_PID=""
GATEWAY_PID=""

# Refuse to touch ports already owned by another process. Cleanup below only
# signals the PIDs launched by this script.
for port in $BACKEND_PORT $WORKER_PORT $GATEWAY_PORT $INTERNAL_PORT; do
  if ss -tln 2>/dev/null | grep -qE ":${port}([[:space:]]|$)"; then
    echo "FATAL: port $port is already in use; refusing to start services" >&2
    exit 1
  fi
done

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

cleanup() {
  log "cleanup: stopping services started by this run"
  for pid in $GATEWAY_PID $WORKER_PID $BACKEND_PID; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
  for pid in $GATEWAY_PID $WORKER_PID $BACKEND_PID; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  local listening
  listening=$(ss -tln 2>/dev/null | grep -E ":($GATEWAY_PORT|$WORKER_PORT|$BACKEND_PORT) " || true)
  if [ -n "$listening" ]; then
    log "cleanup WARNING: ports still listening:"
    echo "$listening"
    echo "{\"services_stopped\": false}" > "$OUT_DIR/cleanup_status.json"
  else
    log "cleanup OK: gateway/worker/backend ports released"
    echo "{\"services_stopped\": true}" > "$OUT_DIR/cleanup_status.json"
  fi
}
trap cleanup EXIT

wait_health() {
  local url=$1 name=$2 deadline=$3
  local waited=0
  while [ $waited -lt $deadline ]; do
    if curl -sf -m 5 "$url" > /dev/null 2>&1; then
      log "health OK: $name ($url)"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  log "health FAILED: $name ($url) after ${deadline}s"
  return 1
}

log "=== demo realtime verification start ==="
log "out_dir=$OUT_DIR"
log "python=$PYTHON"
log "demo_dir=$DEMO_DIR"
log "model_dir=$MODEL_DIR"
log "video=$VIDEO"
log "pre: $(free -g | head -2 | tail -1)"

cd "$DEMO_DIR" || exit 1

REF_ARGS=()
if [ -n "${REF_AUDIO:-}" ]; then
  REF_ARGS=(--ref-audio-path "$REF_AUDIO")
fi

log "starting py_backend server on :$BACKEND_PORT"
PYTHONPATH=. HF_HOME=${HF_HOME:-} nohup "$PYTHON" -m py_backend.server \
  --host 127.0.0.1 --port "$BACKEND_PORT" --gpu-id 0 \
  --model-path "$MODEL_DIR" "${REF_ARGS[@]}" \
  > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
log "backend pid=$BACKEND_PID"

if ! wait_health "http://127.0.0.1:$BACKEND_PORT/health" backend 300; then
  log "FATAL: backend did not become healthy; see logs/backend.log"
  tail -50 "$LOG_DIR/backend.log" || true
  echo "{\"stage\": \"backend_health\", \"ok\": false}" > "$SUMMARY"
  exit 1
fi

log "starting worker on :$WORKER_PORT"
PYTHONPATH=. HF_HOME=${HF_HOME:-} nohup "$PYTHON" worker.py \
  --host 127.0.0.1 --port "$WORKER_PORT" --gpu-id 0 \
  --backend-server-url "http://127.0.0.1:$BACKEND_PORT" \
  > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
log "worker pid=$WORKER_PID"

if ! wait_health "http://127.0.0.1:$WORKER_PORT/health" worker 120; then
  log "FATAL: worker did not become healthy; see logs/worker.log"
  tail -50 "$LOG_DIR/worker.log" || true
  echo "{\"stage\": \"worker_health\", \"ok\": false}" > "$SUMMARY"
  exit 1
fi

log "starting gateway on :$GATEWAY_PORT (http)"
PYTHONPATH=. HF_HOME=${HF_HOME:-} nohup "$PYTHON" gateway.py \
  --host 127.0.0.1 --port "$GATEWAY_PORT" --internal-port "$INTERNAL_PORT" --http \
  > "$LOG_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!
log "gateway pid=$GATEWAY_PID"

if ! wait_health "http://127.0.0.1:$GATEWAY_PORT/health" gateway 120; then
  log "FATAL: gateway did not become healthy; see logs/gateway.log"
  tail -50 "$LOG_DIR/gateway.log" || true
  echo "{\"stage\": \"gateway_health\", \"ok\": false}" > "$SUMMARY"
  exit 1
fi

log "registering worker with gateway"
if ! curl -sf -m 10 -X PUT "http://127.0.0.1:$INTERNAL_PORT/internal/workers/local-worker" \
  -H 'content-type: application/json' \
  --data "{\"endpoint\":\"127.0.0.1:$WORKER_PORT\",\"gpu_group\":\"gpu-0\"}"; then
  log "FATAL: worker registration failed"
  echo "{\"stage\": \"worker_register\", \"ok\": false}" > "$SUMMARY"
  exit 1
fi
log "worker registered"

PROBE_RC=0
if [ "$PROBE_KIND" = barge-in ]; then
  MAX_SESSION=${PROBE_MAX_SESSION_S:-45}
elif [ "$PROBE_MODE" = audio ]; then
  MAX_SESSION=${PROBE_AUDIO_MAX_SESSION_S:-70}
else
  MAX_SESSION=${PROBE_MAX_SESSION_S:-100}
fi
log "=== probe: $PROBE_MODE duplex (recorded material, paced realtime) ==="
PROBE_RC=0
if [ "$PROBE_KIND" = barge-in ]; then
  "$PYTHON" "$PROBE_SCRIPT" \
    --url "http://127.0.0.1:$GATEWAY_PORT" \
    --initial-video "$INITIAL_VIDEO" \
    --interrupt-video "$INTERRUPT_VIDEO" \
    --chunk-ms "${PROBE_CHUNK_MS:-160}" \
    --max-session-s "$MAX_SESSION" \
    --observe-after-s "${PROBE_OBSERVE_AFTER_S:-8}" \
    --initial-seconds "${PROBE_INITIAL_SECONDS:-4}" \
    --interrupt-seconds "${PROBE_INTERRUPT_SECONDS:-4}" \
    --output "$OUT_DIR/$PROBE_OUTPUT_NAME" \
    --audio-output "$OUT_DIR/probe_barge_in_audio.wav" \
    > "$LOG_DIR/$PROBE_LOG_NAME" 2>&1
else
  PROBE_ARGS=()
  if [ "$PROBE_MODE" = video ]; then
    PROBE_ARGS+=(--include-events)
  fi
  "$PYTHON" "$PROBE_SCRIPT" \
    --url "http://127.0.0.1:$GATEWAY_PORT" \
    --mode "$PROBE_MODE" \
    --video "$VIDEO" \
    --chunk-ms "$PROBE_CHUNK_MS" \
    --max-session-s "$MAX_SESSION" \
    --output "$OUT_DIR/$PROBE_OUTPUT_NAME" \
    --audio-output "$OUT_DIR/probe_${PROBE_MODE}_audio.wav" \
    "${PROBE_ARGS[@]}" \
    > "$LOG_DIR/$PROBE_LOG_NAME" 2>&1
fi
PROBE_RC=$?
log "probe $PROBE_KIND/$PROBE_MODE exit=$PROBE_RC"
tail -30 "$LOG_DIR/$PROBE_LOG_NAME" || true

log "post: $(free -g | head -2 | tail -1)"

"$PYTHON" - "$OUT_DIR" "$PROBE_MODE" "$PROBE_KIND" "$PROBE_RC" <<'PYEOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
mode, probe_kind, probe_rc = sys.argv[2], sys.argv[3], int(sys.argv[4])


def load(name):
    p = out_dir / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


probe = load("probe_barge_in_metrics.json" if probe_kind == "barge-in" else f"probe_{mode}_metrics.json")
summary = {
    "stage": "complete",
    "mode": mode,
    "probe_kind": probe_kind,
    "ok": probe_rc == 0 and bool(probe and probe.get("success")),
    "probe_exit": probe_rc,
    "probe": probe,
}
(out_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def line(tag, r):
    if not r:
        return f"- {tag}: no result (probe failed)"
    if not r.get("success"):
        reason = r.get("failure_reason") or r.get("error") or "acceptance criteria not met"
        return f"- {tag}: failed — {reason}"
    return (
        f"- {tag}: session={r.get('session_id')} end={r.get('end_reason')} "
        f"input_chunks={r.get('input_chunks_sent')} frames={r.get('input_frames_sent')} "
        f"listen={r.get('listen_events')} text_chunks={r.get('text_delta_chunks')} "
        f"audio_chunks={r.get('output_audio_chunks')} audio_ms={r.get('output_audio_ms')} "
        f"audio_rms={r.get('output_audio_rms')} first_audio_ms={r.get('first_audio_ms')} "
        f"overlap_ms={r.get('overlap_ms')} vision_metric_deltas={r.get('vision_metrics_deltas')}"
    )


report = [
    "# MiniCPM-o-Demo realtime duplex verification run",
    "",
    "Generated by scripts/run_demo_verify.sh on the compute device.",
    "Stack: py_backend server (GPU) -> worker -> gateway; client: scripts/demo_realtime_probe.py",
    "(adapted from the official examples/realtime/video_probe.py; media extraction via PyAV).",
    "",
    "## Probe results",
    "",
    line(f"{mode} duplex", probe),
    "",
    f"## Probe results ({probe_kind}/{mode})",
    "",
    "```",
    json.dumps(probe, ensure_ascii=False, indent=2) if probe else "(no result)",
    "```",
    "",
]
(out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
PYEOF

log "services stopping next (trap)"

log "=== demo realtime verification done ==="
exit "$PROBE_RC"
