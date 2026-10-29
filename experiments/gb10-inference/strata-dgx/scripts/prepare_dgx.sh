#!/usr/bin/env bash
# Prepare Strata on DGX Spark (aarch64, CUDA 13.0, GB10)
# Usage: bash scripts/prepare_dgx.sh
set -euo pipefail

ROOT="${STRATA_WORK_ROOT:-$HOME/strata-dgx-inference}"
UPSTREAM="$ROOT/upstream"
ARTIFACTS="$ROOT/artifacts"
DATA_DIR="$ROOT/data"
MODEL_DIR="$ROOT/models"
PINNED_REPO="https://github.com/ferstar/Strata.git"
PINNED_BRANCH="gb10-022"
PINNED_COMMIT="d891f006aaec64024513a1e55856afd992baaa94"

mkdir -p "$ARTIFACTS" "$DATA_DIR" "$MODEL_DIR"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This experiment requires aarch64, found $(uname -m)." >&2
  exit 2
fi
[[ -x /usr/local/cuda/bin/nvcc ]] || { echo "CUDA nvcc is missing." >&2; exit 2; }

if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone --depth 1 --branch "$PINNED_BRANCH" "$PINNED_REPO" "$UPSTREAM"
fi
cd "$UPSTREAM"
if [[ "$(git rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
  echo "HEAD differs from the pinned commit; refusing to replace or update the checkout." >&2
  exit 2
fi

# Cap build concurrency for GB10 (default uses all cores, may OOM)
python3 - <<'PY'
from pathlib import Path
p = Path("setup.py")
data = p.read_bytes()
old = b'str(max(2, (os.cpu_count() or 4) // 2))'
new = b'str(int(os.environ.get("STRATA_BUILD_JOBS", max(2, (os.cpu_count() or 4) // 2))))'
if data.count(new) == 1:
    print("build concurrency override already present")
elif data.count(old) == 1:
    p.write_bytes(data.replace(old, new))
    print("build concurrency override installed")
else:
    raise SystemExit("expected exactly one setup.py build concurrency expression")
PY

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export STRATA_BUILD_JOBS=4
export STRATA_NVCC=/usr/local/cuda/bin/nvcc
export PATH="/usr/local/cuda/bin:$PATH"

timeout --signal=TERM --kill-after=30s 2h ./setup.sh \
  --build --no-start \
  --family qwen --model IQ3_S \
  --context 8192 --vision no \
  --data-dir "$DATA_DIR" --models-dir "$MODEL_DIR" \
  --port 8890 --yes 2>&1 | tee "$ARTIFACTS/prepare.log"

test -x "$UPSTREAM/engine/strata"
"$UPSTREAM/engine/strata" --version | tee "$ARTIFACTS/engine-version.txt"

echo ""
echo "Preparation complete. Next steps:"
echo "1. Edit $UPSTREAM/strata-iq3_s.json: set expert-cache to 4096 (auto causes OOM on GB10)"
echo "2. Start server: cd $UPSTREAM && ./run-iq3_s.sh"
echo "3. Run benchmark: bash $ROOT/scripts/run_benchmark.sh"
