#!/usr/bin/env bash
# Phase 0.7 - build a Python 3.12 + lerobot 0.6.1 environment on the gpufree.cn box.
#
# Why this exists: the image ships lerobot 0.4.1 on Python 3.11.13, and 0.4.1's PI05Config has
# neither train_expert_only nor freeze_vision_encoder. Without those the 4090 would have to carry a
# full 4.14 B fine-tune (~50 GiB); with them the measured peak is 18.18 GiB. lerobot >=0.5 requires
# Python >=3.12, so the version we need cannot be installed into the image's interpreter at all.
#
# Run as root on the server. Idempotent - safe to re-run. Touches nothing in the base env.
#   bash gpufree_setup_py312.sh 2>&1 | tee /root/gpufree-data/setup_py312.log

set -uo pipefail

ENV_PREFIX=${ENV_PREFIX:-/root/gpufree-data/envs/lr312}
CACHE_ROOT=${CACHE_ROOT:-/root/gpufree-data/cache}
PYPI=${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}
ENV_FILE=${ENV_FILE:-/root/gpufree-data/env.sh}
CONDA=${CONDA:-/opt/conda/bin/conda}
BASE_PY=${BASE_PY:-/opt/conda/bin/python3.11}

# Confirmed on this box: torch 2.7.1 + torchcodec 0.5 ran the pi0.5 recipe at 2.66 s/step and
# 18.52 GiB peak. The image itself runs 2.7.1+cu128 on driver 575.57.08 (= CUDA 12.9), and lerobot
# 0.6.1 accepts torch>=2.7,<2.12, so there is no reason to reach for anything newer - and the
# laptop's 2.11.0 wheel is +cu130, which this driver cannot load at all.
PIN_TORCH="torch==2.7.1 torchvision==0.22.1"
PIN_CODEC="torchcodec==0.5"

PY="$ENV_PREFIX/bin/python"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mSTOP: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0 - facts
say "0 - facts"
mkdir -p "$CACHE_ROOT" "$(dirname "$ENV_PREFIX")" || die "cannot write under $(dirname "$ENV_PREFIX")"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/uv-python"
export HF_HUB_OFFLINE=1

df -h "$ENV_PREFIX" "$CACHE_ROOT" /root /data 2>/dev/null | sort -u

# rsync is NOT in the gpufree image, and it is how the 1.37 GB dataset gets in and the 9.35 GB
# checkpoints get out. Install it now rather than discovering it is missing mid-transfer.
if ! command -v rsync >/dev/null; then
    echo "--- installing rsync"
    (apt-get update -qq && apt-get install -y -qq rsync) \
        || echo "WARNING: rsync install failed - fall back to scp or the Jupyter file browser"
fi
echo "rsync    $(command -v rsync || echo '<absent>')"
echo "conda    $([ -x "$CONDA" ] && "$CONDA" --version || echo '<absent>')"
echo "base py  $("$BASE_PY" --version 2>&1)"

avail_kb=$(df -Pk "$(dirname "$ENV_PREFIX")" | awk 'NR==2{print $4}')
[ "${avail_kb:-0}" -lt 20000000 ] && echo "WARNING: under 20 GB free at $(dirname "$ENV_PREFIX") - the env needs ~12 GB"

# ---------------------------------------------------------------- 1 - reachability
say "1 - which package indexes are reachable"
"$BASE_PY" - <<'PYEOF'
import urllib.request, ssl, time
ssl._create_default_https_context = ssl._create_unverified_context
TARGETS = [
    ("pypi tsinghua",   "https://pypi.tuna.tsinghua.edu.cn/simple/torch/"),
    ("pypi.org",        "https://pypi.org/simple/torch/"),
    ("pytorch cu128",   "https://download.pytorch.org/whl/cu128/"),
    ("pytorch tsinghua","https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128/"),
    ("conda main",      "https://repo.anaconda.com/pkgs/main/noarch/repodata.json"),
    ("conda-forge",     "https://conda.anaconda.org/conda-forge/noarch/repodata.json"),
    ("conda tsinghua",  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch/repodata.json"),
    ("github",          "https://github.com"),
]
for name, url in TARGETS:
    t = time.time()
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "probe"})
        code = urllib.request.urlopen(req, timeout=8).status
        print(f"  {name:18s} OK    HTTP {code}  ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"  {name:18s} FAIL  {type(e).__name__}: {str(e)[:60]}  ({time.time()-t:.1f}s)")
PYEOF

# ---------------------------------------------------------------- 2 - the 3.12 interpreter
if [ -x "$PY" ]; then
    say "2 - interpreter already present: $("$PY" --version 2>&1)"
else
    say "2 - creating Python 3.12 at $ENV_PREFIX"
    made=""
    if [ -x "$CONDA" ]; then
        for ch in "" "-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --override-channels" \
                     "-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge --override-channels"; do
            echo "--- conda create ${ch:-<default channels>}"
            # shellcheck disable=SC2086
            if "$CONDA" create -y -p "$ENV_PREFIX" "python=3.12" $ch; then made="conda ${ch:-default}"; break; fi
            rm -rf "$ENV_PREFIX"
        done
    fi
    if [ -z "$made" ]; then
        echo "--- conda channels unusable, falling back to uv (CPython from GitHub, which is reachable)"
        "$BASE_PY" -m pip install -q --index-url "$PYPI" uv || die "cannot install uv"
        "$BASE_PY" -m uv venv --seed --python 3.12 "$ENV_PREFIX" || die "uv could not build a 3.12 venv"
        made="uv"
    fi
    echo "created by: $made"
fi
[ -x "$PY" ] || die "no interpreter at $PY"
"$PY" -m ensurepip --upgrade >/dev/null 2>&1
"$PY" -m pip install -q --index-url "$PYPI" --upgrade pip wheel || die "cannot bootstrap pip"
"$PY" --version | grep -q "3\.12" || die "interpreter is not 3.12: $("$PY" --version 2>&1)"

# ---------------------------------------------------------------- 3 - torch, verified before anything else
install_torch() {   # $1 = pin string
    for idx in "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128" \
               "https://download.pytorch.org/whl/cu128" "$PYPI"; do
        echo "--- trying $idx"
        # shellcheck disable=SC2086
        "$PY" -m pip install --index-url "$idx" --extra-index-url "$PYPI" $1 && return 0
    done
    return 1
}
cuda_ok() {
    "$PY" - <<'PYEOF'
import sys, torch
print(f"  torch {torch.__version__}  built for CUDA {torch.version.cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(1)
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print(f"  matmul OK on {torch.cuda.get_device_name(0)}  bf16 sum={(x@x).float().sum().item():.1f}")
PYEOF
}
say "3 - torch ($PIN_TORCH)"
install_torch "$PIN_TORCH" || die "could not install $PIN_TORCH from any index"
cuda_ok || die "torch cannot reach the GPU - stop and report the output above"

# ---------------------------------------------------------------- 4 - lerobot 0.6.1
say "4 - lerobot[pi,training]==0.6.1"
CONSTRAINTS="$CACHE_ROOT/constraints.txt"
{ tr ' ' '\n' <<<"$PIN_TORCH"; echo "$PIN_CODEC"; echo "numpy==2.2.6"; } | grep . > "$CONSTRAINTS"
echo "--- constraints (stop pip from swapping the verified torch stack):"; sed 's/^/    /' "$CONSTRAINTS"
"$PY" -m pip install --index-url "$PYPI" -c "$CONSTRAINTS" "lerobot[pi,training]==0.6.1" \
    || die "lerobot install failed - paste the resolver output back"
cuda_ok || die "the lerobot install broke torch's CUDA build; inspect $CONSTRAINTS"

# ---------------------------------------------------------------- 5 - make the hardcoded tokenizer id resolve offline
# processor_pi05.py asks for "google/paligemma-3b-pt-224" by hub id. The box has the files on disk but
# no route to huggingface.co, so seed the hub cache with a snapshot that points at them.
say "5 - seeding the paligemma tokenizer into the hub cache"
PALI=""
for c in /data/models/paligemma-3b-pt-224 /root/data/models/paligemma-3b-pt-224; do
    [ -f "$c/tokenizer.json" ] && PALI="$c" && break
done
[ -n "$PALI" ] || die "no local paligemma-3b-pt-224 with a tokenizer.json"
export HF_HOME=${HF_HOME:-$CACHE_ROOT/hf}
SHA=$(head -n1 "$PALI"/.cache/huggingface/download/config.json.metadata 2>/dev/null | tr -dc 'a-f0-9')
[ "${#SHA}" -eq 40 ] || SHA=$(printf 'paligemma-3b-pt-224-local' | sha1sum | cut -c1-40)
REPO="$HF_HOME/hub/models--google--paligemma-3b-pt-224"
mkdir -p "$REPO/refs" "$REPO/snapshots"
printf '%s' "$SHA" > "$REPO/refs/main"
rm -rf "$REPO/snapshots/$SHA"
ln -s "$PALI" "$REPO/snapshots/$SHA"
echo "  HF_HOME=$HF_HOME"
echo "  snapshot $SHA -> $PALI"

# ---------------------------------------------------------------- 6 - a Jupyter kernel for this env
# The notebook's cells shell out to $PY explicitly, so they work under any kernel - but having the
# 3.12 env as a selectable kernel makes ad-hoc inspection of checkpoints possible from the browser.
say "6 - registering a Jupyter kernel"
"$PY" -m pip install -q --index-url "$PYPI" ipykernel || die "cannot install ipykernel"
"$PY" -m ipykernel install --user --name lr312 \
    --display-name "lerobot 0.6.1 (py3.12)" || echo "WARNING: kernelspec install failed"
echo "  kernels now visible to Jupyter:"
ls -1 "$HOME/.local/share/jupyter/kernels" 2>/dev/null | sed 's/^/    /'

# ---------------------------------------------------------------- 7 - env file, and wandb
# Every later shell needs HF_HOME and HF_HUB_OFFLINE, and the training notebook decides whether to
# log to wandb from the environment. One file to source rather than four exports to remember.
say "7 - $ENV_FILE"
umask 077
cat > "$ENV_FILE" <<EOF
# source this before anything else: . $ENV_FILE
export PATH=$ENV_PREFIX/bin:\$PATH
export HF_HOME=$HF_HOME
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
EOF

if [ -n "${WANDB_API_KEY:-}" ]; then
    # Written to the env file (mode 600) and to ~/.netrc, because the notebook reads the variable
    # while the wandb client prefers the netrc. The key is never echoed.
    printf 'export WANDB_API_KEY=%s\n' "$WANDB_API_KEY" >> "$ENV_FILE"
    "$PY" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 \
        && echo "  wandb: logged in, key recorded in $ENV_FILE (mode 600)" \
        || echo "  wandb: login call failed, but the key is in $ENV_FILE - the notebook will use it"
else
    cat <<EOF
  wandb: NO KEY SET. wandb IS reachable from this box, so this is the only thing standing between
  you and live curves. Before running the training notebook:

      echo 'export WANDB_API_KEY=<your key from wandb.ai/authorize>' >> $ENV_FILE
      chmod 600 $ENV_FILE
      . $ENV_FILE && python -m wandb login --relogin "\$WANDB_API_KEY"

  Without it the run still logs the held-out loss to its own log file, and section 7 of the
  notebook turns that into a CSV - but you will not see it until you look.
EOF
fi
chmod 600 "$ENV_FILE"
umask 022

say "8 - done"
cat <<EOF
  interpreter   $PY
  torch pins    $PIN_TORCH  $PIN_CODEC
  env size      $(du -sh "$ENV_PREFIX" 2>/dev/null | cut -f1)
  free after    $(df -h "$ENV_PREFIX" | awk 'NR==2{print $4}') at $(df -h "$ENV_PREFIX" | awk 'NR==2{print $6}')
  rsync         $(command -v rsync || echo '<absent>')
  jupyter kernel  lr312

Start every later shell with:
  . $ENV_FILE

Now run the verifier and paste its report back:
  $PY $(dirname "$0")/gpufree_verify_py312.py --load
EOF
