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
CONDA=${CONDA:-/opt/conda/bin/conda}
BASE_PY=${BASE_PY:-/opt/conda/bin/python3.11}

# Pin set A - parity with this box's own proven stack. The image already runs torch 2.7.1+cu128 on
# driver 575.57.08 (= CUDA 12.9), and lerobot 0.6.1 accepts torch>=2.7,<2.12, so this is the pairing
# with no driver risk in it. torchcodec 0.5 is the version the image pairs with torch 2.7.1.
A_TORCH="torch==2.7.1 torchvision==0.22.1"
A_CODEC="torchcodec==0.5"
# Pin set B - parity with the laptop, tried only if A fails. Note the laptop's torch 2.11.0 wheel is
# +cu130, which driver 575 CANNOT load, so B must come from a cu128 index or it will not work here.
B_TORCH="torch==2.11.0 torchvision==0.26.0"
B_CODEC="torchcodec==0.11.1"

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
say "3 - torch (pin set A: $A_TORCH)"
PINNED_TORCH="$A_TORCH"; PINNED_CODEC="$A_CODEC"
install_torch "$A_TORCH" || die "could not install $A_TORCH from any index"
if ! cuda_ok; then
    say "3b - pin set A cannot talk to this driver; retrying with pin set B: $B_TORCH"
    "$PY" -m pip uninstall -y torch torchvision >/dev/null 2>&1
    PINNED_TORCH="$B_TORCH"; PINNED_CODEC="$B_CODEC"
    install_torch "$B_TORCH" || die "could not install $B_TORCH either"
    cuda_ok || die "neither pin set gives a working CUDA build - stop and report the output above"
fi

# ---------------------------------------------------------------- 4 - lerobot 0.6.1
say "4 - lerobot[pi,training]==0.6.1"
CONSTRAINTS="$CACHE_ROOT/constraints.txt"
{ tr ' ' '\n' <<<"$PINNED_TORCH"; echo "$PINNED_CODEC"; echo "numpy==2.2.6"; } | grep . > "$CONSTRAINTS"
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

say "6 - done"
cat <<EOF
  interpreter   $PY
  torch pins    $PINNED_TORCH  $PINNED_CODEC
  env size      $(du -sh "$ENV_PREFIX" 2>/dev/null | cut -f1)
  free after    $(df -h "$ENV_PREFIX" | awk 'NR==2{print $4}') at $(df -h "$ENV_PREFIX" | awk 'NR==2{print $6}')

Every later command needs these three lines:
  export HF_HOME=$HF_HOME
  export HF_HUB_OFFLINE=1
  export PATH=$ENV_PREFIX/bin:\$PATH

Now run the verifier and paste its report back:
  $PY $(dirname "$0")/gpufree_verify_py312.py
EOF
