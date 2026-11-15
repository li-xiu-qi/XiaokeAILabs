#!/usr/bin/env bash
# Start Strata, make one bounded semantic request, save evidence, and stop it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${STRATA_WORK_ROOT:-$REPO_ROOT/.work/strata-dgx-inference}"
UPSTREAM="$WORK_ROOT/upstream"
PORT="${STRATA_PORT:-8890}"
OUT_DIR="${XL_OUT_DIR:-$WORK_ROOT/artifacts/gpu-smoke}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$UPSTREAM/run-iq3_s.sh" ]]; then
  echo "Missing prepared Strata launcher: $UPSTREAM/run-iq3_s.sh" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to record the GPU state for this smoke run." >&2
  exit 2
fi
if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "Port $PORT already serves an API; refusing to reuse an unowned server." >&2
  exit 2
fi

SERVER_LOG="$OUT_DIR/server.log"
(cd "$UPSTREAM" && exec "./run-iq3_s.sh") >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

READY=0
for _ in $(seq 1 300); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Strata exited before the API became ready; see $SERVER_LOG" >&2
    exit 1
  fi
  if curl --silent --fail --max-time 3 "http://127.0.0.1:$PORT/v1/models" -o "$OUT_DIR/models.json"; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "Strata API did not become ready within 10 minutes; see $SERVER_LOG" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv,noheader >"$OUT_DIR/gpu-before.csv"
curl --silent --show-error --fail --max-time 180 \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"17乘以23等于多少？请只在最终答案中输出数字 391。"}],"max_tokens":128,"temperature":0}' \
  >"$OUT_DIR/response.json"
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv,noheader >"$OUT_DIR/gpu-after.csv"

python3 - "$OUT_DIR/response.json" <<'PY'
import json
import re
import sys
from pathlib import Path

response_path = Path(sys.argv[1])
response = json.loads(response_path.read_text(encoding="utf-8"))
content = response["choices"][0]["message"].get("content")
if not isinstance(content, str) or not content.strip():
    choice = response["choices"][0]
    raise SystemExit(
        "No final answer text; finish_reason="
        f"{choice.get('finish_reason')!r}. Increase the token limit or adjust the prompt."
    )
if not re.search(r"(?<!\d)391(?!\d)|三百九十一", content):
    raise SystemExit(f"Arithmetic semantic check failed: {content!r}")
print(f"Arithmetic semantic check passed: {content.strip()}")
timings = response.get("timings", {})
print("Timings:", json.dumps(timings, ensure_ascii=False, sort_keys=True))
summary = {
    "model": response.get("model"),
    "prompt": "17乘以23等于多少？请只在最终答案中输出数字 391。",
    "answer": content.strip(),
    "finish_reason": response["choices"][0].get("finish_reason"),
    "timings": timings,
    "gpu_before": Path(sys.argv[1]).with_name("gpu-before.csv").read_text(encoding="utf-8").strip(),
    "gpu_after": Path(sys.argv[1]).with_name("gpu-after.csv").read_text(encoding="utf-8").strip(),
    "semantic_check": "passed",
}
response_path.with_name("summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

echo "Evidence written to $OUT_DIR"
