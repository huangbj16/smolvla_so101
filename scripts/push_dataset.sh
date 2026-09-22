#!/usr/bin/env bash
# Upload a locally built LeRobotDataset folder to a private Hub dataset repo.
#
#   ./scripts/push_dataset.sh ~/trimmed HALDijkstraaa/so101_toolkit_cylinder_20260917_165544_trimmed
#
# Handles the same proxy problem as scripts/push_models.sh: huggingface_hub >= 1.0 talks through httpx,
# which accepts http://, https:// and socks5:// but NOT the bare socks:// that VPN clients export.
set -euo pipefail

SRC=${1:?usage: push_dataset.sh <local_dir> <repo_id>}
REPO=${2:?usage: push_dataset.sh <local_dir> <repo_id>}

for v in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
  val=${!v:-}
  if [[ $val == socks://* ]]; then
    export "$v=http://${val#socks://}"
    echo "note: rewrote $v to ${!v} (httpx rejects the socks:// scheme)" >&2
  fi
done

for f in meta/info.json meta/stats.json meta/tasks.parquet; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f — is this a LeRobotDataset root?" >&2; exit 1; }
done
find "$SRC" -type d -empty -delete            # the image writer leaves empty staging dirs behind

echo "=== $SRC ($(du -sh "$SRC" | cut -f1)) -> $REPO ==="
hf upload "$REPO" "$SRC" . --repo-type dataset --private \
  --commit-message "Idle-trimmed copy: lead-in and tail cut with the dev > 2.0 rule"

# `hf upload` copies files but does NOT create the codebase-version tag, and lerobot refuses to load a
# dataset without it (get_safe_version -> "Your dataset must be tagged with a codebase version").
# LeRobotDataset.push_to_hub() does this for you; a plain upload does not.
python3 - "$SRC" "$REPO" <<'PY'
import json, sys
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]
ver = json.load(open(f"{src}/meta/info.json"))["codebase_version"]
api = HfApi()
tags = [t.name for t in api.list_repo_refs(repo, repo_type="dataset").tags]
if ver in tags:
    print(f"version tag {ver} already present")
else:
    api.create_tag(repo, tag=ver, repo_type="dataset")
    print(f"created version tag {ver}")
PY
echo "done: https://huggingface.co/datasets/$REPO"
