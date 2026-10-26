#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-start}
SRC=${NINFER_SRC:-}
ART=${NINFER_ARTIFACT:-}
RUN_DIR=${NINFER_RUN_DIR:-$PWD/artifacts}
FFMPEG_PREFIX=${NINFER_FFMPEG_PREFIX:-}
LIBCURL_PREFIX=${NINFER_LIBCURL_PREFIX:-}
[[ -z "$FFMPEG_PREFIX" && -d "$HOME/ffmpeg-env" ]] && FFMPEG_PREFIX="$HOME/ffmpeg-env"
[[ -z "$LIBCURL_PREFIX" && -d "$HOME/mamba-ninfer" ]] && LIBCURL_PREFIX="$HOME/mamba-ninfer"
PORT=${NINFER_PORT:-19410}
MAX_CONTEXT=${NINFER_MAX_CONTEXT:-32768}
KV_CAPACITY=${NINFER_KV_CAPACITY:-262144}
MAX_CONCURRENCY=${NINFER_MAX_CONCURRENCY:-8}
DRAFT_TOKENS=${NINFER_DRAFT_TOKENS:-7}
MODEL_ID=${NINFER_MODEL_ID:-qwen3.8-27b-nvfp4-dflash2}
PID_FILE="$RUN_DIR/run/ninfer-serve.pid"
TAG_FILE="$RUN_DIR/run/ninfer-serve-model-tag"
LOG_FILE="$RUN_DIR/artifacts/ninfer-serve.log"
MODEL_TAG=$(basename "${ART:-model.ninfer}")
if [[ -z "$ART" && -f "$TAG_FILE" ]]; then
    MODEL_TAG=$(cat "$TAG_FILE")
fi
mkdir -p "$RUN_DIR/run" "$RUN_DIR/artifacts"

running() {
    pgrep -f "[n]infer-serve.*${MODEL_TAG}" >/dev/null 2>&1
}

case "$ACTION" in
    stop)
        if [[ -z "$ART" && ! -f "$TAG_FILE" ]]; then
            echo "no current NInfer record; set NINFER_ARTIFACT to identify the server" >&2
            exit 1
        fi
        [[ -f "$PID_FILE" ]] && kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
        if running; then
            pkill -TERM -f "[n]infer-serve.*${MODEL_TAG}" || true
            for _ in $(seq 1 20); do
                running || break
                sleep 2
            done
        fi
        running && pkill -KILL -f "[n]infer-serve.*${MODEL_TAG}" || true
        rm -f "$PID_FILE" "$TAG_FILE"
        echo "NInfer server stopped"
        ;;
    status)
        curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models"
        ;;
    start)
        [[ -n "$SRC" ]] || { echo "set NINFER_SRC to the GB10 fork directory" >&2; exit 1; }
        [[ -n "$ART" ]] || { echo "set NINFER_ARTIFACT to the V3 .ninfer file" >&2; exit 1; }
        [[ -x "$SRC/build-gb10-system-pkg/apps/ninfer-serve" ]] || {
            echo "missing ninfer-serve; configure and build first" >&2
            exit 1
        }
        [[ -f "$ART" ]] || { echo "missing artifact: $ART" >&2; exit 1; }
        running && { echo "NInfer server is already running" >&2; exit 1; }
        if [[ -n "$LIBCURL_PREFIX" || -n "$FFMPEG_PREFIX" ]]; then
            LD_PARTS=()
            [[ -n "$LIBCURL_PREFIX" ]] && LD_PARTS+=("$LIBCURL_PREFIX/lib")
            [[ -n "$FFMPEG_PREFIX" ]] && LD_PARTS+=("$FFMPEG_PREFIX/lib")
            export LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PARTS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        : > "$LOG_FILE"
        CMD=(
            "$SRC/build-gb10-system-pkg/apps/ninfer-serve" "$ART"
            --host 127.0.0.1 --port "$PORT" --device 0
            --model-id "$MODEL_ID"
            --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY"
            --max-concurrency "$MAX_CONCURRENCY"
            --spec dflash2 --draft-tokens "$DRAFT_TOKENS" --lm-head-draft
            --vision
            --request-log-jsonl "$RUN_DIR/artifacts/ninfer-server-requests.jsonl"
            --log-level info
        )
        if [[ -x "$HOME/run_guarded.sh" ]]; then
            setsid nohup bash "$HOME/run_guarded.sh" ninfer-gb10-inference 50 \
                "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        else
            setsid nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        fi
        echo $! > "$PID_FILE"
        printf '%s\n' "$MODEL_TAG" > "$TAG_FILE"
        for _ in $(seq 1 120); do
            if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
                echo "NInfer server ready on ${PORT}"
                curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models"
                exit 0
            fi
            if ! running; then
                echo "NInfer server exited before ready" >&2
                tail -n 100 "$LOG_FILE" >&2
                exit 1
            fi
            sleep 2
        done
        echo "NInfer server did not become ready" >&2
        tail -n 100 "$LOG_FILE" >&2
        exit 1
        ;;
    *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
