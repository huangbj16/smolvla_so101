#!/usr/bin/env bash
# Phase 0.6 — camera ablation for ACT: train top-only, wrist-only and top+wrist.
# Everything except the camera set is identical across the three runs.
# Docs: Details/phase06_camera_ablation_training.md
#
#   SMOKE:  STEPS=500 SAVE_FREQ=500 EVAL_STEPS=250 OUT=outputs/smoke ./scripts/train_camera_ablation.sh
#   FINAL:  ./scripts/train_camera_ablation.sh                        # 60k steps, ~6.5 h, pushes to HF
#
#   ./scripts/train_camera_ablation.sh wrist       # one condition only
#   SEED=1001 ./scripts/train_camera_ablation.sh   # a second seed (new wandb runs + new HF repos)
#   WANDB=false PUSH_TO_HUB=false ./...            # local-only run
#
# Runs the three SEQUENTIALLY. Measured: training is GPU-bound, so three concurrent processes each slow
# to ~1/3 speed and the set finishes 8% LATER than back to back (see the run book, section 5).
# Runs in fp32, the lerobot default. --policy.use_amp=true is ~1.3x faster but changes the numerics.
set -euo pipefail

DS=${DS:-HALDijkstraaa/so101_toolkit_cylinder_20260917_165544}
STEPS=${STEPS:-60000}
BATCH=${BATCH:-8}
SEED=${SEED:-1000}
WORKERS=${WORKERS:-8}
SAVE_FREQ=${SAVE_FREQ:-10000}
OUT=${OUT:-outputs/train}

# --- validation ------------------------------------------------------------------------------------
EVAL_SPLIT=${EVAL_SPLIT:-0.1}   # set to 0 to train on all 50 episodes
# eval_steps=0 (lerobot's default) builds the eval dataloader and never uses it: the holdout episodes
# would just be dropped from training for nothing. Must be 0 when EVAL_SPLIT is 0, or validate() raises.
# Measured cost: ~30 s per pass on the full 5-episode holdout with two cameras, ~15 s with one. At 5000
# that is 12 points on the wandb curve for ~12 min across the three runs.
EVAL_STEPS=${EVAL_STEPS:-5000}
if [ "$EVAL_SPLIT" = "0" ] || [ "$EVAL_SPLIT" = "0.0" ]; then EVAL_STEPS=0; fi
# NOTE: do NOT set --max_eval_samples. It slices frames[:n] per task, i.e. the FIRST n frames of the
# holdout, which is the opening of one episode only — the reach phase of a single position.

# --- logging ---------------------------------------------------------------------------------------
LOG_FREQ=${LOG_FREQ:-200}
WANDB=${WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-phase06-camera-ablation}   # run name = job_name, so all 3 overlay

# --- hub -------------------------------------------------------------------------------------------
PUSH_TO_HUB=${PUSH_TO_HUB:-true}     # pushes the FINAL model of each run, private
HF_USER=${HF_USER:-HALDijkstraaa}
REPO_PREFIX=${REPO_PREFIX:-act_toolkit_cylinder}
# Guard: never push a smoke run to the Hub, even if PUSH_TO_HUB was left at its default.
if [ "$PUSH_TO_HUB" = "true" ] && [ "$STEPS" -lt 10000 ]; then
  echo "note: STEPS=$STEPS looks like a smoke run, so the Hub push is disabled for it." >&2
  PUSH_TO_HUB=false
fi

# --- episode order ---------------------------------------------------------------------------------
# The 50 episodes were recorded position-block by position-block: P1 = ep 0-4, P2 = 5-9, ... P10 = 45-49.
# lerobot holds out the LAST ceil(n*eval_split) episodes of --dataset.episodes, so the default order would
# put all of P10 in validation and remove that position from training entirely. This reordering puts a
# balanced holdout last instead: the final episode of P2, P4, P6, P8 and P10. Every position stays in
# training. (Verified: LeRobotDataset preserves the given order, and factory.py slices the tail of it.)
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
  local hub=()
  if [ "$PUSH_TO_HUB" = "true" ]; then
    hub=(--policy.push_to_hub=true "--policy.repo_id=${HF_USER}/${REPO_PREFIX}_${name}_s${SEED}"
         --policy.private=true)
  else
    hub=(--policy.push_to_hub=false)
  fi
  echo "=== $job : $STEPS steps, batch $BATCH, seed $SEED, push=$PUSH_TO_HUB, wandb=$WANDB ==="
  # --policy.input_features is only auto-filled when empty (policies/factory.py), so this override sticks.
  lerobot-train \
    --dataset.repo_id="$DS" \
    --dataset.episodes="$EPISODES" \
    --dataset.eval_split="$EVAL_SPLIT" \
    --eval_steps="$EVAL_STEPS" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.input_features="{$feats}" \
    "${hub[@]}" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --num_workers="$WORKERS" \
    --seed="$SEED" \
    --save_freq="$SAVE_FREQ" \
    --log_freq="$LOG_FREQ" \
    --output_dir="$OUT/$job" \
    --job_name="$job" \
    --wandb.enable="$WANDB" \
    --wandb.project="$WANDB_PROJECT" \
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
if [ "$PUSH_TO_HUB" = "true" ]; then
  echo "pushed: https://huggingface.co/${HF_USER}/${REPO_PREFIX}_<cond>_s${SEED} (private)"
fi
