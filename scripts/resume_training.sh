#!/usr/bin/env bash
# Resume a finished Phase 0.6 run to a higher step count.
# Docs: Details/phase06_camera_ablation_training.md section 8 (follow-ups).
#
#   ./scripts/resume_training.sh both 100000     # 60k -> 100k, ~2.2 h
#   ./scripts/resume_training.sh top  100000
#
# Everything is read back from the checkpoint's train_config.json, so the episode order, the 45/5
# holdout, the seed, save_freq, eval_steps and the wandb run id are all preserved — the wandb curve
# extends the SAME run rather than starting a new one. Only --steps is overridden.
# push_to_hub stays false; upload afterwards with ./scripts/push_models.sh
set -euo pipefail

COND=${1:-both}
STEPS=${2:-100000}
SEED=${SEED:-1000}
OUT=${OUT:-outputs/train}

RUN="$OUT/act_${COND}_s${SEED}"
CKPT="$RUN/checkpoints/last/pretrained_model"
STATE="$RUN/checkpoints/last/training_state/training_step.json"
[ -d "$CKPT" ]  || { echo "no checkpoint at $CKPT" >&2; exit 1; }
[ -f "$STATE" ] || { echo "no training state at $STATE — cannot resume" >&2; exit 1; }

DONE=$(python3 -c "import json;print(json.load(open('$STATE'))['step'])")
if [ "$DONE" -ge "$STEPS" ]; then
  echo "already at step $DONE; asked for $STEPS. Nothing to do." >&2; exit 1
fi
# measured throughput: ~5.3 it/s with two cameras, ~10.5 with one
RATE=$([ "$COND" = "both" ] && echo 5.3 || echo 10.5)
echo "=== resuming act_${COND}_s${SEED}: step $DONE -> $STEPS ($((STEPS - DONE)) more, ~$(python3 -c "print(f'{($STEPS-$DONE)/$RATE/3600:.1f}')") h) ==="

lerobot-train \
  --config_path="$CKPT/train_config.json" \
  --resume=true \
  --steps="$STEPS" \
  2>&1 | tee -a "$RUN.log"

echo "done. new checkpoints under $RUN/checkpoints/"
echo "next: ./scripts/push_models.sh $COND     # upload the extended model"
