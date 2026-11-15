#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:?usage: dgx_sglang.sh start|stop|status|logs ...}
SGLANG_MODEL_ROOT=${SGLANG_MODEL_ROOT:-}
SGLANG_PYTHON=${SGLANG_PYTHON:-}
RUN_DIR=${SGLANG_RUN_DIR:-${XL_OUT_DIR:-$PWD/.work}}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PORT=${PORT:-19410}
DECODE_GRAPH=${DECODE_GRAPH:-disabled}
MODEL_TAG=$(basename "${SGLANG_MODEL_ROOT:-model}")
mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/run"
CURRENT="$RUN_DIR/run/current.env"
TOKEN_FILE="$RUN_DIR/run/sglang-instance-token"
BASELINE_FILE="$RUN_DIR/run/sglang-process-baseline"
if [[ -z "$SGLANG_MODEL_ROOT" && -f "$CURRENT" ]]; then
    # shellcheck disable=SC1090
    source "$CURRENT"
    MODEL_TAG="$SERVED_MODEL"
fi
export PATH="$(dirname "${SGLANG_PYTHON:-python}"):$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

owned_pids() {
    [[ -f "$CURRENT" && -f "$TOKEN_FILE" ]] || return 1
    local token pid
    read -r token < "$TOKEN_FILE"
    [[ "$token" =~ ^[0-9a-f-]{36}$ ]] || return 1
    # The optional guard starts a separate session; the token is inherited
    # by both the guard and server even when their process groups differ.
    while read -r pid; do
        if grep -zFxq "XL_SGLANG_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null; then
            printf '%s\n' "$pid"
        fi
    done < <(ps -u "$(id -u)" -o pid=)
}

running() { [[ -n "$(owned_pids)" ]]; }

pid_identity() {
    local pid="$1" stat
    IFS= read -r stat < "/proc/$pid/stat" || return 1
    stat=${stat##*) }
    # Fields after the parenthesized comm start at field 3; starttime is 22.
    set -- $stat
    [[ $# -ge 20 ]] || return 1
    printf '%s:%s\n' "$pid" "${20}"
}

save_process_baseline() {
    local pids pid identity
    pids=$(ps -u "$(id -u)" -o pid=) || return 1
    while read -r pid; do
        [[ -n "$pid" && -d "/proc/$pid" ]] || continue
        if identity=$(pid_identity "$pid" 2>/dev/null); then
            printf '%s\n' "$identity"
        elif [[ -d "/proc/$pid" ]]; then
            return 1
        fi
    done <<< "$pids"
}

instance_absent() {
    [[ -f "$TOKEN_FILE" && -f "$BASELINE_FILE" ]] || return 1
    local token pid pids result identity
    read -r token < "$TOKEN_FILE"
    [[ "$token" =~ ^[0-9a-f-]{36}$ ]] || return 1
    pids=$(ps -u "$(id -u)" -o pid=) || return 1
    while read -r pid; do
        [[ -n "$pid" && -d "/proc/$pid" ]] || continue
        if [[ ! -r "/proc/$pid/environ" ]]; then
            # An unreadable process observed before this launch cannot be a
            # descendant. Include starttime so PID reuse cannot bypass this.
            identity=$(pid_identity "$pid" 2>/dev/null) || { [[ ! -d "/proc/$pid" ]] && continue; return 1; }
            grep -Fxq "$identity" "$BASELINE_FILE" || return 1
            continue
        fi
        if grep -zFxq "XL_SGLANG_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null; then
            return 1
        else
            result=$?
            if [[ "$result" != 1 && -d "/proc/$pid" ]]; then
                identity=$(pid_identity "$pid" 2>/dev/null) || { [[ ! -d "/proc/$pid" ]] && continue; return 1; }
                grep -Fxq "$identity" "$BASELINE_FILE" || return 1
            fi
        fi
    done <<< "$pids"
    return 0
}

stop_server() {
    [[ -f "$CURRENT" ]] || {
        echo "no current SGLang record; refusing to stop an unidentified process" >&2
        return 1
    }
    running || { echo "cannot verify ownership of the recorded server; refusing to signal it" >&2; return 1; }
    local pid token
    read -r token < "$TOKEN_FILE"
    while read -r pid; do
        grep -zFxq "XL_SGLANG_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
    done < <(owned_pids)
    for _ in $(seq 1 30); do
        running || break
        sleep 2
    done
    if running; then
        while read -r pid; do
            grep -zFxq "XL_SGLANG_INSTANCE=$token" "/proc/$pid/environ" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        done < <(owned_pids)
        for _ in $(seq 1 5); do
            running || break
            sleep 1
        done
    fi
    if running; then
        echo "failed to stop the recorded SGLang instance; keeping the current record" >&2
        return 1
    fi
    if ! instance_absent; then
        echo "cannot prove the instance has exited; keeping its record for inspection" >&2
        return 1
    fi
    rm -f "$CURRENT" "$TOKEN_FILE" "$BASELINE_FILE"
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
        running || { echo "cannot verify ownership of the recorded server" >&2; exit 1; }
        source "$CURRENT"
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
        [[ ! -e "$CURRENT" && ! -e "$TOKEN_FILE" && ! -e "$BASELINE_FILE" ]] || { echo "server record already exists; check it before starting a new instance" >&2; exit 1; }
        [[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo "invalid port" >&2; exit 1; }
        if ! LISTENERS=$(ss -H -ltn "sport = :$PORT"); then
            echo "cannot check port ${PORT}; refusing to start" >&2
            exit 1
        fi
        if [[ -n "$LISTENERS" ]]; then
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
        if ! save_process_baseline > "$BASELINE_FILE"; then
            rm -f "$BASELINE_FILE"
            echo "cannot capture process identities; refusing to start" >&2
            exit 1
        fi
        INSTANCE_TOKEN=$(cat /proc/sys/kernel/random/uuid)
        printf '%s\n' "$INSTANCE_TOKEN" > "$TOKEN_FILE"
        if [[ -x "$HOME/run_guarded.sh" ]]; then
            env XL_SGLANG_INSTANCE="$INSTANCE_TOKEN" setsid nohup bash "$HOME/run_guarded.sh" "coder390-${VARIANT}-${SPEC}-${DECODE_GRAPH}" 50 \
                "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        else
            env XL_SGLANG_INSTANCE="$INSTANCE_TOKEN" setsid nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 </dev/null &
        fi
        WRAPPER_PID=$!
        printf 'WRAPPER_PID=%q\nVARIANT=%q\nSPEC=%q\nMODEL_PATH=%q\nSERVED_MODEL=%q\nLOG_FILE=%q\nPORT=%q\n' \
            "$WRAPPER_PID" "$VARIANT" "$SPEC" "$MODEL_PATH" "$SERVED_MODEL" "$LOG_FILE" "$PORT" > "$CURRENT"
        for i in $(seq 1 240); do
            if running && curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                echo "SGLang ready on ${PORT}"
                if ! curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/get_model_info"; then
                    fail_start "SGLang became ready but the model info query failed"
                fi
                exit 0
            fi
            if ! kill -0 "$WRAPPER_PID" 2>/dev/null && ! running; then
                echo "SGLang exited before ready" >&2
                tail -n 100 "$LOG_FILE" >&2
                if instance_absent; then
                    rm -f "$CURRENT" "$TOKEN_FILE" "$BASELINE_FILE"
                else
                    echo "cannot prove the instance has exited; keeping its record" >&2
                fi
                exit 1
            fi
            sleep 5
        done
        fail_start "SGLang did not become ready"
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
