#!/usr/bin/env bash
# Push completed Phase 0.6 models to private HF repos, after training.
# Kept separate from training on purpose: a network hiccup at push time must not destroy hours of
# finished GPU work (it did once — see the run book, section 7).
#
#   ./scripts/push_models.sh              # every condition that has a finished checkpoint
#   ./scripts/push_models.sh top          # just one
#   CKPT=040000 ./scripts/push_models.sh  # a specific checkpoint instead of the final one
set -euo pipefail

SEED=${SEED:-1000}
OUT=${OUT:-outputs/train}
HF_USER=${HF_USER:-HALDijkstraaa}
REPO_PREFIX=${REPO_PREFIX:-act_toolkit_cylinder}
CKPT=${CKPT:-last}

# huggingface_hub >= 1.0 talks through httpx, which accepts http://, https:// and socks5:// but NOT the
# bare socks:// that VPN clients often export. Clash's mixed port serves HTTP CONNECT on the same port,
# so rewriting the scheme is enough — no socksio install needed.
for v in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
  val=${!v:-}
  if [[ $val == socks://* ]]; then
    export "$v=http://${val#socks://}"
    echo "note: rewrote $v to ${!v} (httpx rejects the socks:// scheme)" >&2
  fi
done

CONDS=("$@")
if [ ${#CONDS[@]} -eq 0 ]; then CONDS=(top wrist both); fi
pushed=0
for cond in "${CONDS[@]}"; do
  dir="$OUT/act_${cond}_s${SEED}/checkpoints/$CKPT/pretrained_model"
  if [ ! -d "$dir" ]; then echo "skip $cond: no checkpoint at $dir"; continue; fi
  repo="${HF_USER}/${REPO_PREFIX}_${cond}_s${SEED}"
  echo "=== $cond -> $repo ($(du -sh "$dir" | cut -f1)) ==="
  hf upload "$repo" "$dir" . --repo-type model --private \
    --commit-message "ACT ${cond}-camera, seed ${SEED}, checkpoint ${CKPT}"
  pushed=$((pushed + 1))
done
echo "pushed $pushed model(s)."
