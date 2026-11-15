#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-start}
SRC=${NINFER_SRC:-}
ART=${NINFER_ARTIFACT:-}
RUN_DIR=${NINFER_RUN_DIR:-${XL_OUT_DIR:-$PWD/.work}}
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
TOKEN_FILE="$RUN_DIR/run/ninfer-serve-token"
PORT_FILE="$RUN_DIR/run/ninfer-serve-port"
LOG_FILE="$RUN_DIR/artifacts/ninfer-serve.log"
MODEL_TAG=$(basename "${ART:-model.ninfer}")
if [[ -z "$ART" && -f "$TAG_FILE" ]]; then
    MODEL_TAG=$(cat "$TAG_FILE")
fi
mkdir -p "$RUN_DIR/run" "$RUN_DIR/artifacts"

owned_pids() {
    [[ -f "$PID_FILE" && -f "$TOKEN_FILE" ]] || return 1
    local recorded_pid token pid
    read -r recorded_pid < "$PID_FILE"
    read -r token < "$TOKEN_FILE"
    [[ "$recorded_pid" =~ ^[0-9]+$ && "$token" =~ ^[0-9a-f-]{36}$ ]] || return 1
    # The guard can create a new session, so check each inherited token rather
    # than assuming that all descendants stay in one process group.
    while read -r pid; do
        if grep -zFxq "XL_NINFER_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null; then
            printf '%s\n' "$pid"
        fi
    done < <(ps -u "$(id -u)" -o pid=)
}

owned_running() {
    [[ -n "$(owned_pids)" ]]
}

port_owned() {
    [[ -f "$TOKEN_FILE" ]] || return 1
    local token pid sockets
    read -r token < "$TOKEN_FILE"
    [[ "$token" =~ ^[0-9a-f-]{36}$ ]] || return 1
    sockets=$(ss -H -ltnp "sport = :$PORT") || return 1
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        if grep -zFxq "XL_NINFER_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null; then
            return 0
        fi
    done < <(printf '%s\n' "$sockets" | grep -oE 'pid=[0-9]+' | cut -d= -f2)
    return 1
}

stop_owned() {
    owned_running || { echo "cannot verify ownership of the recorded server; refusing to signal it" >&2; return 1; }
    local pid token
    read -r token < "$TOKEN_FILE"
    while read -r pid; do
        grep -zFxq "XL_NINFER_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
    done < <(owned_pids)
    for _ in $(seq 1 20); do
        owned_running || break
        sleep 2
    done
    if owned_running; then
        while read -r pid; do
            grep -zFxq "XL_NINFER_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        done < <(owned_pids)
        for _ in $(seq 1 5); do
            owned_running || break
            sleep 1
        done
    fi
    if owned_running; then
        echo "could not stop all verified processes; keeping the server record" >&2
        return 1
    fi
    rm -f "$PID_FILE" "$TAG_FILE" "$TOKEN_FILE" "$PORT_FILE"
}

case "$ACTION" in
    stop)
        stop_owned
        echo "NInfer server stopped"
        ;;
    status)
        owned_running || { echo "cannot verify ownership of the recorded server" >&2; exit 1; }
        [[ -f "$PORT_FILE" ]] || { echo "server port record is missing" >&2; exit 1; }
        read -r PORT < "$PORT_FILE"
        [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo "invalid recorded port" >&2; exit 1; }
        port_owned || { echo "cannot verify ownership of the listening port" >&2; exit 1; }
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
        if [[ -e "$PID_FILE" || -e "$TOKEN_FILE" || -e "$PORT_FILE" ]]; then
            echo "server record already exists; check it before starting a new instance" >&2
            exit 1
        fi
        [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo "invalid port" >&2; exit 1; }
        sockets=$(ss -H -ltn "sport = :$PORT") || { echo "could not check port $PORT" >&2; exit 1; }
        [[ -z "$sockets" ]] || { echo "port $PORT is already in use" >&2; exit 1; }
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
        INSTANCE_TOKEN=$(cat /proc/sys/kernel/random/uuid)
        printf '%s\n' "$INSTANCE_TOKEN" > "$TOKEN_FILE"
        if [[ -x "$HOME/run_guarded.sh" ]]; then
            env XL_NINFER_INSTANCE="$INSTANCE_TOKEN" setsid nohup bash "$HOME/run_guarded.sh" ninfer-gb10-inference 50 \
                "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        else
            env XL_NINFER_INSTANCE="$INSTANCE_TOKEN" setsid nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        fi
        echo $! > "$PID_FILE"
        printf '%s\n' "$MODEL_TAG" > "$TAG_FILE"
        printf '%s\n' "$PORT" > "$PORT_FILE"
        for _ in $(seq 1 120); do
            if owned_running && port_owned && curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
                echo "NInfer server ready on ${PORT}"
                curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models"
                exit 0
            fi
            if ! owned_running && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
                echo "NInfer server exited before ready" >&2
                tail -n 100 "$LOG_FILE" >&2
                echo "server records retained for inspection" >&2
                exit 1
            fi
            sleep 2
        done
        echo "NInfer server did not become ready" >&2
        tail -n 100 "$LOG_FILE" >&2
        stop_owned || true
        exit 1
        ;;
    *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
