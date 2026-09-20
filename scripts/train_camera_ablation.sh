#!/usr/bin/env bash
# Phase 0.6 — camera ablation for ACT: train top-only, wrist-only and top+wrist.
# Everything except the camera set is identical across the three runs.
# Docs: Details/phase06_camera_ablation_training.md
#
#   ./scripts/train_camera_ablation.sh                            # all three, 60k steps, ~4.6 h
#   STEPS=500 SAVE_FREQ=500 OUT=outputs/smoke ./scripts/train_camera_ablation.sh   # smoke test
#   STEPS=100000 ./scripts/train_camera_ablation.sh               # the full ACT recipe, ~7.7 h
#   EVAL_SPLIT=0 ./scripts/train_camera_ablation.sh               # train on all 50 episodes
#   SEED=1001 ./scripts/train_camera_ablation.sh wrist            # one condition, second seed
#
# Run the three SEQUENTIALLY, as this script does. Measured: training is GPU-bound, so three concurrent
# processes each slow to ~1/3 speed and finish 8% LATER than back to back (see the run book, section 5).
set -euo pipefail

DS=${DS:-HALDijkstraaa/so101_toolkit_cylinder_20260917_165544}
STEPS=${STEPS:-60000}
BATCH=${BATCH:-8}
SEED=${SEED:-1000}
WORKERS=${WORKERS:-8}
SAVE_FREQ=${SAVE_FREQ:-10000}
EVAL_SPLIT=${EVAL_SPLIT:-0.1}   # set to 0 to train on all 50 episodes
OUT=${OUT:-outputs/train}

# The 50 episodes were recorded position-block by position-block: P1 = ep 0-4, P2 = 5-9, ... P10 = 45-49.
# lerobot holds out the LAST ceil(n*eval_split) episodes of --dataset.episodes, so the default order would
# put all of P10 in validation and remove that position from training entirely. This reordering puts a
# balanced holdout last instead: the final episode of P2, P4, P6, P8 and P10. Every position stays in
# training. (Verified: LeRobotDataset preserves the given order, and factory.py slices the tail of it.)
HOLDOUT="9, 19, 29, 39, 49"
EPISODES=${EPISODES:-$(python3 - <<'PY'
h = [9, 19, 29, 39, 49]
print([e for e in range(50) if e not in h] + h)
PY
)}

STATE="'observation.state': {'type': 'STATE', 'shape': [6]}"
TOP="'observation.images.top': {'type': 'VISUAL', 'shape': [3, 480, 640]}"
WRIST="'observation.images.wrist': {'type': 'VISUAL', 'shape': [3, 480, 640]}"

run_one () {
  local name=$1 feats=$2 job="act_${1}_s${SEED}"
  echo "=== $job : $STEPS steps, batch $BATCH, seed $SEED ==="
  # --policy.input_features is only auto-filled when empty (policies/factory.py), so this override sticks.
  lerobot-train \
    --dataset.repo_id="$DS" \
    --dataset.episodes="$EPISODES" \
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
