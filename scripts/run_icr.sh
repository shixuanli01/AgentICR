#!/usr/bin/env bash
# Full ICR run for one benchmark: independent beliefs (phase 1), all six
# directed revisions per retained item (phase 2), merge, and summary.
#
# Usage (from the repository root):
#   bash scripts/run_icr.sh TASK [ARTIFACT_ROOT]
#
# TASK is one of: medqa, arc_challenge, gsm8k, gpqa, humanevalplus.
#
# Environment variables (defaults reproduce the reported runs):
#   MODEL             Hugging Face model id          (default Qwen/Qwen3-4B)
#   MAX_NEW_TOKENS    token budget, both phases      (default 16384)
#   REPLICATION_ID    revision-seed replication      (default seed_pair_00)
#   RECEIVER_POLICY   revise (Critical Evaluation) or verify (Structured
#                     Verification)                   (default revise)
#   CONDITIONS        comma-separated revision conditions
#                     (default none,true_text,true_statebridge,true_latentmas)
#   CUDA_DEVICES      space-separated GPU ids         (default "0")
#   WORKERS_PER_GPU   worker processes per GPU        (default 1)
#   LATENT_STEPS      LatentMAS latent steps          (default 10)
#   BENCHMARK_LIMIT   first N items only, for smoke tests
#   PHASE1_ONLY       stop after phase 1
#   PYTHON            interpreter                     (default python)
#
# Phase 2 skips items that all three agents answered correctly
# (--skip-all-correct-items) and keeps every direction of every other item
# (--both-correct-sample 1), which is the setting of the reported runs.
#
# Every step is resumable: re-running the same command skips finished records.
set -euo pipefail

TASK="${1:?usage: run_icr.sh TASK [ARTIFACT_ROOT]}"
case "$TASK" in
  medqa|arc_challenge|gsm8k|gpqa|humanevalplus) ;;
  *) echo "unsupported task: $TASK" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
ARTIFACT_ROOT="${2:-artifacts/${TASK}_${MODEL_TAG}_seed42}"
PYTHON="${PYTHON:-python}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
REPLICATION_ID="${REPLICATION_ID:-seed_pair_00}"
RECEIVER_POLICY="${RECEIVER_POLICY:-revise}"
CONDITIONS="${CONDITIONS:-none,true_text,true_statebridge,true_latentmas}"
LATENT_STEPS="${LATENT_STEPS:-10}"
read -r -a GPUS <<< "${CUDA_DEVICES:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
WORLD=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

SELECTION_ARGS=(--model "$MODEL" --max-new-tokens "$MAX_NEW_TOKENS")
if [[ -n "${BENCHMARK_LIMIT:-}" ]]; then
  SELECTION_ARGS+=(--limit "$BENCHMARK_LIMIT")
fi

mkdir -p "$ARTIFACT_ROOT/logs"
pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap terminate_children INT TERM EXIT

# One process per (GPU, worker) slot; each process sees a single GPU.
launch_phase() {
  local module="$1"; shift
  local rank=0 gpu worker failed=0 pid
  for gpu in "${GPUS[@]}"; do
    for ((worker = 0; worker < WORKERS_PER_GPU; worker++)); do
      CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=0 "$PYTHON" -u -m "$module" \
        --artifact-root "$ARTIFACT_ROOT" --rank "$rank" --world-size "$WORLD" "$@" \
        > "$ARTIFACT_ROOT/logs/${module##*.}_rank${rank}_gpu${gpu}.log" 2>&1 &
      pids+=("$!")
      rank=$((rank + 1))
    done
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  [[ "$failed" -eq 0 ]]
}

echo "[$TASK] model=$MODEL root=$ARTIFACT_ROOT gpus=${GPUS[*]} workers/gpu=$WORKERS_PER_GPU"

echo "[$TASK] phase 1: independent beliefs"
launch_phase icr.prebeliefs --task "$TASK" --replication-id "$REPLICATION_ID" "${SELECTION_ARGS[@]}"
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" --phase prebeliefs --require-complete
"$PYTHON" -m icr.verify --artifact-root "$ARTIFACT_ROOT" > "$ARTIFACT_ROOT/logs/verification.log"

if [[ -n "${PHASE1_ONLY:-}" ]]; then
  trap - INT TERM EXIT
  echo "[$TASK] phase 1 complete (PHASE1_ONLY set)"
  exit 0
fi

echo "[$TASK] phase 2: revisions ($CONDITIONS, receiver policy $RECEIVER_POLICY)"
launch_phase icr.revisions --conditions "$CONDITIONS" --receiver-policy "$RECEIVER_POLICY" \
  --latent-steps "$LATENT_STEPS" --skip-all-correct-items --both-correct-sample 1 \
  --all-correct-sample 0 --global-resume --output-tag main
"$PYTHON" -m icr.merge --artifact-root "$ARTIFACT_ROOT" --phase revisions --require-complete

echo "[$TASK] summary"
"$PYTHON" -m icr.analyze --run "$TASK=$ARTIFACT_ROOT" --out "$ARTIFACT_ROOT/analysis"

trap - INT TERM EXIT
echo "[$TASK] done: $ARTIFACT_ROOT/analysis/summary.csv"
