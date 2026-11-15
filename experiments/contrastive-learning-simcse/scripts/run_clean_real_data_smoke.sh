#!/usr/bin/env bash
# Rebuild a small, revision-pinned real-data SimCSE run in an isolated CUDA venv.
# The run is a pipeline check, not a quality benchmark or paper reproduction.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

if [[ "${XL_REQUIRE_EXTERNAL_ENV:-0}" == "1" ]]; then
  for name in XL_RUN_ID XL_DATA_DIR XL_OUT_DIR; do
    [[ -n "${!name:-}" ]] || { echo "FATAL: $name required in strict mode" >&2; exit 2; }
  done
fi
XL_RUN_ID=${XL_RUN_ID:-simcse-$(date -u +%Y%m%dT%H%M%SZ)}
XL_DATA_DIR=${XL_DATA_DIR:-$EXP_DIR/.work/data/$XL_RUN_ID}
XL_OUT_DIR=${XL_OUT_DIR:-$EXP_DIR/.work/out/$XL_RUN_ID}
HF_HOME=${HF_HOME:-$XL_DATA_DIR/huggingface}
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/simcse-$XL_RUN_ID}
DATA_DIR=$XL_DATA_DIR/datasets/simcse-$XL_RUN_ID

export XL_RUN_ID XL_DATA_DIR XL_OUT_DIR HF_HOME HF_ENDPOINT
mkdir -p "$XL_OUT_DIR" "$XL_DATA_DIR" "$HF_HOME" "$DATA_DIR" "$(dirname "$VENV_DIR")"
if [[ -e "$XL_OUT_DIR/summary.json" || -e "$VENV_DIR" || -e "$DATA_DIR/unsup_wiki.jsonl" ]]; then
  echo "FATAL: refusing to overwrite an existing run path; choose a new XL_RUN_ID/VENV_DIR" >&2
  exit 2
fi

STAGE=initialization
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
trap_rc=0
write_failure_summary() {
  local rc=$?
  if [[ $rc -ne 0 && ! -e "$XL_OUT_DIR/summary.json" ]]; then
    local failure_python=${PYTHON_BIN:-python3}
    command -v "$failure_python" >/dev/null 2>&1 || failure_python=python3
    "$failure_python" - "$XL_OUT_DIR" "$STAGE" "$rc" "$STARTED" <<'PY'
import json, os, sys
out, stage, rc, started = sys.argv[1:]
summary = {
    "date": started[:10], "experiment": "SimCSE real-data pipeline smoke",
    "status": "failed", "failed_stage": stage, "exit_code": int(rc),
    "purpose": "small real-data pipeline check; not a quality benchmark or paper reproduction",
}
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2); f.write("\n")
PY
  fi
}
trap write_failure_summary EXIT

STAGE=create_venv
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
[[ -x "$PY" ]] || { echo "FATAL: venv Python missing" >&2; exit 1; }

STAGE=install_dependencies
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
"$PY" -m pip install --disable-pip-version-check -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check | tee "$XL_OUT_DIR/pip-check.txt"
"$PY" - "$XL_OUT_DIR/environment.json" <<'PY'
import json, platform, sys
import datasets, numpy, torch, transformers
state = {
    "python": platform.python_version(), "platform": platform.platform(),
    "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__, "datasets": datasets.__version__,
    "numpy": numpy.__version__,
}
if not state["cuda_available"]:
    raise SystemExit("CUDA is required for this run")
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(state, f, ensure_ascii=False, indent=2); f.write("\n")
PY

STAGE=download_and_prepare_real_data
export DATA_DIR
"$PY" - "$XL_OUT_DIR" <<'PY'
import hashlib, itertools, json, os, sys
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

out = sys.argv[1]
endpoint = os.environ["HF_ENDPOINT"]
api = HfApi(endpoint=endpoint)
model_id = "google-bert/bert-base-uncased"
wiki_id = "wikimedia/wikipedia"
snli_id = "stanfordnlp/snli"
glue_id = "nyu-mll/glue"
model_sha = api.model_info(model_id).sha
wiki_sha = api.dataset_info(wiki_id).sha
snli_sha = api.dataset_info(snli_id).sha
glue_sha = api.dataset_info(glue_id).sha

model_path = snapshot_download(
    repo_id=model_id, revision=model_sha,
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
)

wiki = load_dataset(wiki_id, "20231101.en", split="train", streaming=True, revision=wiki_sha)
sentences = []
for row in itertools.islice(wiki, 200):
    for sentence in row["text"].split(". "):
        sentence = sentence.strip()
        if 20 < len(sentence) < 200:
            sentences.append(sentence)
            if len(sentences) == 32:
                break
    if len(sentences) == 32:
        break
if len(sentences) < 32:
    raise RuntimeError(f"Wikipedia yielded only {len(sentences)} usable sentences in the bounded sample")

snli = load_dataset(snli_id, split="train", streaming=True, revision=snli_sha)
by_premise = defaultdict(lambda: {0: [], 2: []})
for row in itertools.islice(snli, 20000):
    if row["label"] in (0, 2) and row["premise"] and row["hypothesis"]:
        by_premise[row["premise"]][row["label"]].append(row["hypothesis"])
triplets = []
for premise, labels in by_premise.items():
    positives = sorted(set(labels[0]))
    negatives = sorted(set(labels[2]))
    if positives and negatives:
        for positive in positives:
            negative = next((n for n in negatives if n != positive and n != premise), None)
            if negative:
                triplets.append({"anchor": premise, "positive": positive, "negative": negative})
                break
    if len(triplets) == 32:
        break
if len(triplets) < 32:
    raise RuntimeError(f"SNLI yielded only {len(triplets)} usable triplets in the bounded 20,000-row scan")

sts = load_dataset(glue_id, "stsb", split="validation", streaming=True, revision=glue_sha)
sts_rows = []
for row in itertools.islice(sts, 100):
    sts_rows.append({"sentence1": row["sentence1"], "sentence2": row["sentence2"], "score": row["label"]})
if len(sts_rows) != 100:
    raise RuntimeError(f"GLUE STS-B yielded only {len(sts_rows)} evaluation pairs")

files = {
    "unsup_wiki.jsonl": [{"sentence": x} for x in sentences],
    "snli_triplets.jsonl": triplets,
    "stsb_val.jsonl": sts_rows,
}
manifest_files = {}
for name, rows in files.items():
    path = os.path.join(os.environ["DATA_DIR"], name)
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    manifest_files[name] = {"rows": len(rows), "sha256": h.hexdigest()}

manifest = {
    "datasets": {
        wiki_id: {"config": "20231101.en", "split": "train (streaming prefix)", "revision": wiki_sha},
        snli_id: {"split": "train (first 20,000 streamed rows searched)", "revision": snli_sha},
        glue_id: {"config": "stsb", "split": "validation (first 100 rows)", "revision": glue_sha},
    },
    "model": {"repo_id": model_id, "revision": model_sha},
    "files": manifest_files,
    "selection": {"wikipedia_sentences": 32, "snli_triplets": 32, "stsb_validation_pairs": 100},
    "endpoint": endpoint,
}
with open(os.path.join(out, "data-manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2); f.write("\n")
with open(os.path.join(os.environ["DATA_DIR"], "model-path.txt"), "x", encoding="utf-8") as f:
    f.write(model_path + "\n")
print(json.dumps({"data": manifest["selection"], "model_revision": model_sha,
                  "dataset_revisions": {wiki_id: wiki_sha, snli_id: snli_sha, glue_id: glue_sha}}, ensure_ascii=False))
PY

MODEL_PATH=$(cat "$DATA_DIR/model-path.txt")
STAGE=train_unsupervised
"$PY" "$SCRIPT_DIR/train_unsup_simcse.py" \
  --model "$MODEL_PATH" --data "$DATA_DIR/unsup_wiki.jsonl" --out "$XL_OUT_DIR/unsup" \
  --epochs 1 --batch-size 8 --lr 3e-5 --temperature 0.05 --max-length 32 \
  --limit 32 --device cuda --seed 42 2>&1 | tee "$XL_OUT_DIR/train-unsup.log"
STAGE=train_supervised
"$PY" "$SCRIPT_DIR/train_sup_simcse.py" \
  --model "$MODEL_PATH" --data "$DATA_DIR/snli_triplets.jsonl" --out "$XL_OUT_DIR/sup" \
  --epochs 1 --batch-size 8 --lr 3e-5 --temperature 0.05 --max-length 64 \
  --limit 32 --device cuda --seed 42 2>&1 | tee "$XL_OUT_DIR/train-sup.log"
STAGE=evaluate
"$PY" "$SCRIPT_DIR/evaluate_sts.py" \
  --model "$MODEL_PATH" --sts-files "$DATA_DIR/stsb_val.jsonl" --device cuda \
  --json-out "$XL_OUT_DIR/base-eval.json" 2>&1 | tee "$XL_OUT_DIR/eval-base.log"
"$PY" "$SCRIPT_DIR/evaluate_sts.py" \
  --model "$XL_OUT_DIR/unsup" --sts-files "$DATA_DIR/stsb_val.jsonl" --device cuda \
  --json-out "$XL_OUT_DIR/unsup-eval.json" 2>&1 | tee "$XL_OUT_DIR/eval-unsup.log"
"$PY" "$SCRIPT_DIR/evaluate_sts.py" \
  --model "$XL_OUT_DIR/sup" --sts-files "$DATA_DIR/stsb_val.jsonl" --device cuda \
  --json-out "$XL_OUT_DIR/sup-eval.json" 2>&1 | tee "$XL_OUT_DIR/eval-sup.log"

STAGE=finalize
"$PY" - "$XL_OUT_DIR" "$STARTED" <<'PY'
import json, os, sys
out, started = sys.argv[1:]
env=json.load(open(os.path.join(out,"environment.json"),encoding="utf-8"))
data=json.load(open(os.path.join(out,"data-manifest.json"),encoding="utf-8"))

def metric(name):
    obj=json.load(open(os.path.join(out,name),encoding="utf-8"))
    return obj
summary={
  "date": started[:10], "experiment":"SimCSE small real-data pipeline smoke", "status":"completed",
  "purpose":"environment and train/save/evaluate pipeline check; not a quality benchmark or paper reproduction",
  "platform": {k:env.get(k) for k in ("python","platform","torch","cuda_runtime","cuda_available","gpu","transformers","datasets","numpy")},
  "model": data["model"], "datasets": data["datasets"], "data_files": data["files"], "selection": data["selection"],
  "training": {"epochs":1,"batch_size":8,"learning_rate":3e-5,"temperature":0.05,"device":"CUDA","seed":42,"steps_per_variant":4},
  "evaluation": {"base":metric("base-eval.json"),"unsupervised":metric("unsup-eval.json"),"supervised":metric("sup-eval.json")},
  "pip_check":"passed", "limitations":["Only four optimizer steps per variant and a 100-pair STS-B prefix; scores are not evidence of model quality.","This is not a full reproduction of the SimCSE paper settings or results.","Data and model were fetched through the configured Hub endpoint; cold-cache behavior may vary."]
}
with open(os.path.join(out,"summary.json"),"w",encoding="utf-8") as f:
  json.dump(summary,f,ensure_ascii=False,indent=2); f.write("\n")
print("summary written", summary["status"])
PY
STAGE=complete
