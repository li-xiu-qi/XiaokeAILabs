#!/usr/bin/env bash
# Single GPU training, requires a GPU slot. Reads no existing run directory.
set -euo pipefail
: "${AUTHOR_REPO:?set AUTHOR_REPO to the pinned author checkout}"
: "${MODEL:?set MODEL to a local chat-model directory}"
: "${DATA:?set DATA to train.jsonl}"
: "${RUN_ROOT:?set RUN_ROOT to a new directory outside any checkout}"
: "${PYTHON:?set PYTHON to a prepared interpreter}"
: "${LAST_TOKEN:?set LAST_TOKEN after inspecting the model chat template}"

expected=ea0017c654ad917066ff32afc88276bea8ca5f7e
actual=$(git -C "$AUTHOR_REPO" rev-parse HEAD)
[[ "$actual" == "$expected" ]] || { echo 'author revision mismatch' >&2; exit 1; }
git -C "$AUTHOR_REPO" diff --quiet HEAD -- finetune.py || { echo 'author trainer has local modifications' >&2; exit 1; }
[[ -f "$DATA" && -d "$MODEL" ]] || { echo 'data/model missing' >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo 'run root already exists' >&2; exit 1; }
mkdir -p "$RUN_ROOT"
root=$(cd "$RUN_ROOT" && pwd -P)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
tools_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
"$PYTHON" "$tools_dir/audit_tokenizer.py" --model "$MODEL" --data "$DATA" --max-length 128
"$PYTHON" "$tools_dir/patch_author.py" --source "$AUTHOR_REPO/finetune.py" --output "$root/finetune_mask_fixed.py"
common=(--train_file "$DATA" --model_name "$MODEL" --num_epochs 1 --batch_size 2 --grad_accum 4 --max_length 128 --learning_rate 2e-5 --finetune_seed 42 --use_default_data_collator --unmask_assistant_special_tokens --lora --lora_rank 8)
# Author finetune.py deletes output_dir unconditionally; these targets must not exist.
for mode in baseline jepa; do
  dest="$root/$mode"
  [[ ! -e "$dest" ]] || { echo 'output unexpectedly exists' >&2; exit 1; }
  args=(--regular)
  if [[ "$mode" == jepa ]]; then args=(--last_token "$LAST_TOKEN" --predictors 0 --lbd 0.1); fi
  "$PYTHON" "$root/finetune_mask_fixed.py" "${common[@]}" --output_dir "$dest" "${args[@]}" 2>&1 | tee "$root/$mode.log"
done
