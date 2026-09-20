#!/usr/bin/env bash
# Phase 0.6 — camera ablation for ACT: train top-only, wrist-only and top+wrist.
# Everything except the camera set is identical across the three runs.
# Docs: Details/phase06_camera_ablation_training.md
#
#   ./scripts/train_camera_ablation.sh            # all three, 100k steps
#   STEPS=500 SAVE_FREQ=500 ./scripts/train_camera_ablation.sh    # smoke test
#   SEED=1001 ./scripts/train_camera_ablation.sh wrist            # one condition, second seed
set -euo pipefail

DS=${DS:-HALDijkstraaa/so101_toolkit_cylinder_20260917_165544}
STEPS=${STEPS:-100000}
BATCH=${BATCH:-8}
SEED=${SEED:-1000}
WORKERS=${WORKERS:-8}
SAVE_FREQ=${SAVE_FREQ:-20000}
EVAL_SPLIT=${EVAL_SPLIT:-0.1}
OUT=${OUT:-outputs/train}

STATE="'observation.state': {'type': 'STATE', 'shape': [6]}"
TOP="'observation.images.top': {'type': 'VISUAL', 'shape': [3, 480, 640]}"
WRIST="'observation.images.wrist': {'type': 'VISUAL', 'shape': [3, 480, 640]}"

run_one () {
  local name=$1 feats=$2 job="act_${1}_s${SEED}"
  echo "=== $job : $STEPS steps, batch $BATCH, seed $SEED ==="
  # --policy.input_features is only auto-filled when empty (policies/factory.py), so this override sticks.
  lerobot-train \
    --dataset.repo_id="$DS" \
    --dataset.eval_split="$EVAL_SPLIT" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.input_features="{$feats}" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --num_workers="$WORKERS" \
    --seed="$SEED" \
    --save_freq="$SAVE_FREQ" \
    --output_dir="$OUT/$job" \
    --job_name="$job" \
    --wandb.enable=false \
    2>&1 | tee "$OUT/${job}.log"
}

mkdir -p "$OUT"
CONDS=("$@")
if [ ${#CONDS[@]} -eq 0 ]; then CONDS=(top wrist both); fi
for cond in "${CONDS[@]}"; do
  case $cond in
    top)   run_one top   "$STATE, $TOP" ;;
    wrist) run_one wrist "$STATE, $WRIST" ;;
    both)  run_one both  "$STATE, $TOP, $WRIST" ;;
    *)     echo "unknown condition: $cond (use top|wrist|both)" >&2; exit 1 ;;
  esac
done
echo "done. checkpoints under $OUT/act_*_s${SEED}/checkpoints/"
