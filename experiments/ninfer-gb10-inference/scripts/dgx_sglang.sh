#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:?usage: dgx_sglang.sh start|stop|status|logs ...}
SGLANG_MODEL_ROOT=${SGLANG_MODEL_ROOT:-}
SGLANG_PYTHON=${SGLANG_PYTHON:-}
RUN_DIR=${SGLANG_RUN_DIR:-$PWD/artifacts}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PORT=${PORT:-19410}
DECODE_GRAPH=${DECODE_GRAPH:-disabled}
MODEL_TAG=$(basename "${SGLANG_MODEL_ROOT:-model}")
mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/run"
CURRENT="$RUN_DIR/run/current.env"
if [[ -z "$SGLANG_MODEL_ROOT" && -f "$CURRENT" ]]; then
    # shellcheck disable=SC1090
    source "$CURRENT"
    MODEL_TAG="$SERVED_MODEL"
fi
export PATH="$(dirname "${SGLANG_PYTHON:-python}"):$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

running() {
    pgrep -f "[s]glang.launch_server.*${MODEL_TAG}" >/dev/null 2>&1
}

stop_server() {
    [[ -f "$CURRENT" ]] || {
        echo "no current SGLang record; refusing to stop an unidentified process" >&2
        return 1
    }
    # shellcheck disable=SC1090
    source "$CURRENT"
    [[ -n "${SERVED_MODEL:-}" ]] || {
        echo "current SGLang record is missing SERVED_MODEL" >&2
        return 1
    }
    MODEL_TAG="$SERVED_MODEL"
    [[ -n "${WRAPPER_PID:-}" ]] && kill -TERM "$WRAPPER_PID" 2>/dev/null || true
    if running; then
        pkill -TERM -f "[s]glang.launch_server.*${MODEL_TAG}" || true
        for _ in $(seq 1 30); do
            running || break
            sleep 2
        done
    fi
    if running; then
        pkill -KILL -f "[s]glang.launch_server.*${MODEL_TAG}" || true
        for _ in $(seq 1 5); do
            running || break
            sleep 1
        done
    fi
    if running; then
        echo "failed to stop SGLang for ${MODEL_TAG}; keeping the current record" >&2
        return 1
    fi
    rm -f "$CURRENT"
}

fail_start() {
    echo "$1" >&2
    if [[ -f "${LOG_FILE:-}" ]]; then
        tail -n 100 "$LOG_FILE" >&2
    fi
    stop_server || echo "failed to stop the partially started SGLang server" >&2
    exit 1
}

case "$ACTION" in
    stop)
        stop_server
        echo "Coder390 SGLang server stopped"
        ;;
    status)
        curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/get_model_info"
        ;;
    logs)
        [[ -f "$CURRENT" ]] || { echo "no current server record" >&2; exit 1; }
        # shellcheck disable=SC1090
        source "$CURRENT"
        tail -n 100 "$LOG_FILE"
        ;;
    start)
        [[ -n "$SGLANG_MODEL_ROOT" ]] || { echo "set SGLANG_MODEL_ROOT to the Coder390 root" >&2; exit 1; }
        [[ -n "$SGLANG_PYTHON" ]] || { echo "set SGLANG_PYTHON to the env with SGLang installed" >&2; exit 1; }
        VARIANT=${2:?choose nvfp4 or fp8}
        SPEC=${3:-dflash2}
        case "$DECODE_GRAPH" in
            disabled|full|breakable|tc_piecewise) ;;
            *) echo "unknown decode graph backend: $DECODE_GRAPH" >&2; exit 2 ;;
        esac
        case "$VARIANT" in
            nvfp4) MODEL_PATH="$SGLANG_MODEL_ROOT/NVFP4/W4A4"; SHORT=nvfp4-w4a4 ;;
            fp8) MODEL_PATH="$SGLANG_MODEL_ROOT/FP8"; SHORT=fp8-w8a8 ;;
            *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
        esac
        case "$SPEC" in
            dflash2|none) ;;
            *) echo "unknown speculative mode: $SPEC" >&2; exit 2 ;;
        esac
        [[ -f "$MODEL_PATH/config.json" ]] || { echo "missing model: $MODEL_PATH" >&2; exit 1; }
        running && { echo "a Coder390 server is already running" >&2; exit 1; }
        if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
            echo "port ${PORT} is occupied" >&2
            exit 1
        fi
        SERVED_MODEL="qwen3.8-27b-coder390-${SHORT}-${SPEC}"
        LOG_FILE="$RUN_DIR/artifacts/${VARIANT}-${SPEC}-${DECODE_GRAPH}-server.log"
        CMD=(
            "$SGLANG_PYTHON" -m sglang.launch_server
            --model-path "$MODEL_PATH"
            --served-model-name "$SERVED_MODEL"
            --host 127.0.0.1 --port "$PORT"
            --trust-remote-code
            --dtype bfloat16
            --context-length 32768
            --max-running-requests 8
            --max-total-tokens 65536
            --max-mamba-cache-size 48
            --mem-fraction-static 0.50
            --attention-backend flashinfer
            --disable-flashinfer-autotune
            --chunked-prefill-size 8192
            --kv-cache-dtype fp8_e4m3
            --mamba-radix-cache-strategy extra_buffer
            --mamba-ssm-dtype bfloat16
            --cuda-graph-backend-decode "$DECODE_GRAPH"
            --cuda-graph-max-bs-decode 8
            --cuda-graph-backend-prefill disabled
            --reasoning-parser qwen3
            --tool-call-parser qwen3_coder
        )
        if [[ "$SPEC" == dflash2 ]]; then
            CMD+=(
                --speculative-algorithm DFLASH
                --speculative-draft-model-path "$MODEL_PATH/DFlash2-FP8"
                --speculative-draft-model-quantization fp8
                --speculative-num-draft-tokens 8
            )
        fi
        : > "$LOG_FILE"
        if [[ -x "$HOME/run_guarded.sh" ]]; then
            setsid nohup bash "$HOME/run_guarded.sh" "coder390-${VARIANT}-${SPEC}-${DECODE_GRAPH}" 50 \
                "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        else
            setsid nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        fi
        WRAPPER_PID=$!
        printf 'WRAPPER_PID=%s\nVARIANT=%s\nSPEC=%s\nMODEL_PATH=%s\nSERVED_MODEL=%s\nLOG_FILE=%s\nPORT=%s\n' \
            "$WRAPPER_PID" "$VARIANT" "$SPEC" "$MODEL_PATH" "$SERVED_MODEL" "$LOG_FILE" "$PORT" > "$CURRENT"
        for i in $(seq 1 240); do
            if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                echo "SGLang ready on ${PORT}"
                if ! curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/get_model_info"; then
                    fail_start "SGLang became ready but the model info query failed"
                fi
                exit 0
            fi
            if ! kill -0 "$WRAPPER_PID" 2>/dev/null && ! running; then
                fail_start "SGLang exited before ready"
            fi
            sleep 5
        done
        fail_start "SGLang did not become ready"
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
