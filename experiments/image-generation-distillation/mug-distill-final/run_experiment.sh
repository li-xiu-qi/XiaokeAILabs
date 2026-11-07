#!/usr/bin/env bash
# Configure environment variables explicitly; run in a new, empty WORK directory.
set -euo pipefail
: "${PYTHON:?set PYTHON}" "${WORK:?set WORK}" "${STUDENT_MODEL:?set STUDENT_MODEL}" "${TEACHER_MODEL:?set TEACHER_MODEL}" "${TRAIN_SCRIPT:?set TRAIN_SCRIPT}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
cd "$WORK"
for d in self-original teacher-original self-data teacher-data self-run teacher-run eval-base eval-self eval-teacher eval-teacher-model; do [[ ! -e $d ]] || { echo "Refusing existing output: $d" >&2; exit 1; }; done
for arm in self teacher; do for suffix in generate train; do for log in "$arm-$suffix.log" "$arm-$suffix-memory.log"; do [[ ! -e $log ]] || { echo "Refusing existing log: $log" >&2; exit 1; }; done; done; done
for arm in base self teacher teacher-model; do for log in "eval-$arm.log" "eval-$arm-memory.log"; do [[ ! -e $log ]] || { echo "Refusing existing log: $log" >&2; exit 1; }; done; done
set -o noclobber
run_guarded() { local log=$1; shift; "$PYTHON" "$HERE/guarded_run.py" --minimum-gib 50 --log "$log" -- "$@"; }
run_guarded self-generate-memory.log "$PYTHON" "$HERE/generate_student.py" --model "$STUDENT_MODEL" --prompts "$HERE/train.jsonl" --out self-original --size 512 --steps 30 --guidance 7.5 > self-generate.log 2>&1
run_guarded teacher-generate-memory.log "$PYTHON" "$HERE/generate_teacher.py" --model "$TEACHER_MODEL" --prompts "$HERE/train.jsonl" --out teacher-original > teacher-generate.log 2>&1
"$PYTHON" "$HERE/prepare_mug_training_data.py" --prompts "$HERE/train.jsonl" --source self-original --out self-data --size 512
"$PYTHON" "$HERE/prepare_mug_training_data.py" --prompts "$HERE/train.jsonl" --source teacher-original --out teacher-data --size 1024
for arm in self teacher; do
 run_guarded "$arm-train-memory.log" "$PYTHON" "$TRAIN_SCRIPT" --pretrained_model_name_or_path "$STUDENT_MODEL" --train_data_dir "$arm-data" --image_column image --caption_column text --resolution 512 --train_batch_size 1 --gradient_accumulation_steps 4 --gradient_checkpointing --max_train_steps 400 --learning_rate 1e-4 --seed 42 --dataloader_num_workers 0 --mixed_precision fp16 --output_dir "$arm-run" > "$arm-train.log" 2>&1
done
for arm in base self teacher; do
 args=(); [[ $arm == base ]] || args+=(--lora "$WORK/$arm-run")
 run_guarded "eval-$arm-memory.log" "$PYTHON" "$HERE/generate_student.py" --model "$STUDENT_MODEL" --prompts "$HERE/test.jsonl" --out "eval-$arm" --size 512 --steps 30 --guidance 7.5 "${args[@]}" > "eval-$arm.log" 2>&1
done
run_guarded eval-teacher-model-memory.log "$PYTHON" "$HERE/generate_teacher.py" --model "$TEACHER_MODEL" --prompts "$HERE/test.jsonl" --out eval-teacher-model > eval-teacher-model.log 2>&1
echo 'Finished generation and training; independent blind review still required.'
