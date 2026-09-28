#!/usr/bin/env bash
# Revisions on the mixed-correctness directions only (sender right / receiver
# wrong, and sender wrong / receiver right), reusing an existing phase-1 root.
# CR, PR and SI are measured; SR and SCR are not.
#
# Used for supplementary conditions (Answer Only), the Structured Verification
# receiver policy, and additional models.
#
# Usage (from the repository root):
#   bash scripts/run_icr_mixed.sh TASK ARTIFACT_ROOT
#
# ARTIFACT_ROOT must hold a complete phase 1 (run scripts/run_icr.sh with
# PHASE1_ONLY=1 first, or point it at a root created by run_icr.sh). To write a
# supplementary condition into a separate root, create the new directory and
# copy (or symlink) config.json, prebeliefs/ and messages/ from the
# phase-1 root into it; see README.md.
#
# Environment variables: CONDITIONS, RECEIVER_POLICY, CUDA_DEVICES,
# WORKERS_PER_GPU, LATENT_STEPS, OUTPUT_TAG (default "mixed"), PYTHON.
set -euo pipefail

TASK="${1:?usage: run_icr_mixed.sh TASK ARTIFACT_ROOT}"
ARTIFACT_ROOT="${2:?usage: run_icr_mixed.sh TASK ARTIFACT_ROOT}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
CONDITIONS="${CONDITIONS:-none,true_text,true_statebridge,true_latentmas}"
RECEIVER_POLICY="${RECEIVER_POLICY:-revise}"
OUTPUT_TAG="${OUTPUT_TAG:-mixed}"
read -r -a GPUS <<< "${CUDA_DEVICES:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
WORLD=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

mkdir -p "$ARTIFACT_ROOT/logs"
echo "[$TASK] mixed-direction revisions: $CONDITIONS | policy=$RECEIVER_POLICY root=$ARTIFACT_ROOT"
pids=()
trap 'kill -TERM "${pids[@]}" 2>/dev/null || true' INT TERM
rank=0
for gpu in "${GPUS[@]}"; do
  for ((worker = 0; worker < WORKERS_PER_GPU; worker++)); do
    CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=0 "$PYTHON" -u -m icr.revisions \
      --artifact-root "$ARTIFACT_ROOT" --conditions "$CONDITIONS" \
      --receiver-policy "$RECEIVER_POLICY" \
      --pair-classes correction_opportunity,destruction_risk \
      --latent-steps "${LATENT_STEPS:-10}" --global-resume --output-tag "$OUTPUT_TAG" \
      --rank "$rank" --world-size "$WORLD" \
      > "$ARTIFACT_ROOT/logs/revisions_${OUTPUT_TAG}_rank${rank}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    rank=$((rank + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
