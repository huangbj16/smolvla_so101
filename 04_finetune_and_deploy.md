# 04 — Fine-tune SmolVLA and deploy (if time)

> Reality check (repeat from README): fine-tuning a 450M VLA on **6 GB** is *tight*. Community
> configs assume 16–24 GB. Below are the 6 GB-survival settings; expect to fight OOM, and treat this
> as a proof-of-concept, not a full training run. If it won't fit, fine-tune in the cloud
> (Colab/A100, ~4 h for 20k steps) and copy the checkpoint back — deployment/inference *does* fit on 6 GB.
> `conda activate lerobot` first.

## Fine-tune with `lerobot-train`

> **Windows backslash bug — FIXED via [lerobot PR #2940](https://github.com/huggingface/lerobot/pull/2940).**
> The error `HFValidationError: Repo id must use alphanumeric chars … 'lerobot\smolvla_base'` came from
> lerobot coercing `pretrained_path` into a `Path`, which flips `/`→`\` on Windows. Patched in your local
> editable install: `pretrained_path` is now `str | Path` in `configs/policies.py`, and the `Path()`
> wrapper was removed in `configs/train.py` and `configs/eval.py`. **So `--policy.path=lerobot/smolvla_base`
> now works directly — no local-download workaround needed.**

Load the base model with **`--policy.path`** (a plain-string repo id):

```powershell
lerobot-train `
  --policy.path=lerobot/smolvla_base `
  --dataset.repo_id=HALDijkstraaa/so101_pick_place_pcb `
  --dataset.root="$env:USERPROFILE\.cache\huggingface\lerobot\HALDijkstraaa\so101_pick_place_pcb_20260721_183543" `
  --batch_size=4 `
  --steps=2000 `
  --output_dir=outputs/train/my_smolvla `
  --job_name=my_smolvla_training `
  --policy.device=cuda `
  --wandb.enable=false
  --policy.push_to_hub=false `
```

```powershell
lerobot-train `
  --policy.path=lerobot/smolvla_base `
  --dataset.repo_id=lerobot/svla_so100_stacking `
  --batch_size=4 `
  --steps=2000  `
  --policy.push_to_hub=false `
  --wandb.enable=false
```

### 6 GB memory-survival knobs (check exact names with `lerobot-train --help`)
- **`--batch_size=1`** (maybe 2). This is the biggest lever.
- **Freeze the VLM backbone** — train only the action expert. Look in `SmolVLAConfig`
  (`configuration_smolvla.py`) for a `freeze_vision_encoder` / `train_expert_only` (or similar) flag
  and enable it. This is the difference between fitting and OOM. Fewer trainable params → far less
  optimizer/gradient memory.
- **Mixed precision / bf16** — if there's an `--policy.dtype=bfloat16` or an AMP flag, use it.
- **Gradient accumulation** — if available (`--grad_accumulation_steps=N`), use it to get an effective
  larger batch without the memory (accumulate grads over N micro-batches, then step). This is the
  concept from your W1 training loop: `zero_grad → (backward × N) → step`.
- **Shorter chunk / fewer steps** if config allows, to shrink activations.
- Close other GPU apps (browsers included) — on 6 GB every 100 MB counts. Watch usage with
  `nvidia-smi` in another terminal.

If you OOM immediately: freeze the vision encoder first, then batch_size=1, then bf16. If still OOM,
go cloud for training.

## Actual Colab run — commands used (for record)

Copied verbatim from `colab_finetune_smolvla_updated.ipynb` (the run that produced the fine-tuned model:
lerobot 0.6.0, Colab A100, batch 64, 20k steps).

**Install:**
```python
!pip install -q "lerobot[smolvla]==0.6.0"
!pip install 'lerobot[dataset]'
```

**Hugging Face auth:**
```python
from huggingface_hub import login
try:
    from google.colab import userdata
    login(token=userdata.get("HF_TOKEN"))
    print("Logged in via Colab secret HF_TOKEN")
except Exception:
    print("No HF_TOKEN secret found - falling back to interactive login...")
    login()
```
```python
!hf auth whoami
```

**Weights & Biases auth:**
```python
!pip install wandb
```
```python
import wandb
try:
    from google.colab import userdata
    wandb.login(key=userdata.get("WANDB_API_KEY"))
    print("Logged into W&B via Colab secret WANDB_API_KEY")
except Exception:
    print("No WANDB_API_KEY secret found - falling back to interactive login...")
    wandb.login()
```

**Config:**
```python
HF_USER         = "HALDijkstraaa"
DATASET_REPO_ID = f"{HF_USER}/so101_pick_place_pcb_20260721_183543"         # <-- your dataset on the Hub (check exact name!)
BASE_MODEL      = "lerobot/smolvla_base"                     # forward slash is fine on Linux/Colab
MODEL_REPO_ID   = f"{HF_USER}/smolvla_so101_pcb_finetuned_v2"   # <-- output model repo (created in step 6)
OUTPUT_DIR      = "outputs/train/smolvla_so101_pcb_v2"

BATCH_SIZE = 64       # A100/L4: 8 is safe. Drop to 4 or 2 on a T4 if you hit OOM.
STEPS      = 20000   # a real fine-tune (~a few hours on A100). Lower (e.g. 2000) for a quick smoke test.
SAVE_FREQ  = 2000    # checkpoint every N steps (and always at the final step)

USE_WANDB     = True                 # set False to train without W&B logging
WANDB_PROJECT = "smolvla_so101_pcb_v2"  # your W&B project name (created on first run)

print("Dataset   :", DATASET_REPO_ID)
print("Base model:", BASE_MODEL)
print("Output ->  :", MODEL_REPO_ID)
print("W&B       :", f"{WANDB_PROJECT} (enabled={USE_WANDB})")
```

**Train:**
```python
TRAIN_CMD = (
    "lerobot-train"
    f" --dataset.repo_id={DATASET_REPO_ID}"
    f" --policy.path={BASE_MODEL}"
    f" --output_dir={OUTPUT_DIR}"
    " --job_name=smolvla_so101_pcb"
    " --policy.device=cuda"
    f" --batch_size={BATCH_SIZE}"
    f" --steps={STEPS}"
    f" --save_freq={SAVE_FREQ}"
    " --log_freq=100"
    " --policy.push_to_hub=false"
    f" --wandb.enable={'true' if USE_WANDB else 'false'}"
    f" --wandb.project={WANDB_PROJECT}"
    " --wandb.disable_artifact=true"
    " --rename_map='{\"observation.images.front\": \"observation.images.camera1\"}'"
)
print(TRAIN_CMD)
!{TRAIN_CMD}
```

**Locate checkpoint:**
```python
import os, glob
last = os.path.join(OUTPUT_DIR, "checkpoints", "last", "pretrained_model")
if os.path.exists(last):
    CKPT_DIR = last
else:
    CKPT_DIR = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoints", "*", "pretrained_model")))[-1]
print("Checkpoint dir:", CKPT_DIR)
print("Contents:", os.listdir(CKPT_DIR))
```

**Upload to the Hub:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(MODEL_REPO_ID, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path=CKPT_DIR,
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    commit_message="Fine-tuned SmolVLA on SO-101 PCB pick-and-place",
)
print(f"Uploaded -> https://huggingface.co/{MODEL_REPO_ID}")
```

### Run screenshots (record)

![Colab A100 running lerobot-train at batch 64 (GPU RAM ~10.4 / 40 GB).](results/Screenshot%202026-07-22%20125919.png)

![W&B dashboard (project smolvla_so101_pcb_v2): train/loss and component losses trend down over ~20k steps; lr shows cosine warmup→decay; samples-seen grows linearly.](results/Screenshot%202026-07-22%20144412.png)

## Monitor
- Loss should trend down (it's the flow-matching MSE from your W3/W4 — won't hit zero; it has the
  irreducible velocity-variance floor you already understand).
- Checkpoints land in `--output_dir`. Note the path to the best/last checkpoint.

## Deploy: run YOUR fine-tuned policy on the robot

`lerobot-record` doubles as the rollout runner — pass a `--policy.path` and it runs inference on the
real robot (recording the rollout so you can review it):
```powershell
lerobot-record ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" ^
  --dataset.repo_id=bingjian/so101_eval_pickplace ^
  --dataset.single_task="Pick up the cube and place it in the box" ^
  --policy.path=outputs/smolvla_so101_pickplace/checkpoints/last ^   # your fine-tuned checkpoint
  --dataset.num_episodes=5 ^
  --display_data=true
```
- **No `--teleop` here** — the policy drives the follower, not you.
- Use the **same camera name and instruction** as training, or the input keys won't match.
- Keep a hand near the e-stop / power for the first runs.

```powershell
# --- 1. pull one checkpoint from the Hub ---
# --- 1. pull one checkpoint from the Hub ---
$MODEL_REPO_ID = "HALDijkstraaa/smolvla_so101_pcb_finetuned_v3"    # your repo
$STEP   = "020000"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$POLICY = python -c "from huggingface_hub import snapshot_download; import os; r = snapshot_download('$MODEL_REPO_ID', allow_patterns='$STEP/*'); print(os.path.join(r, '$STEP'))"
$POLICY
Get-ChildItem $POLICY

lerobot-rollout `
  --policy.path="$POLICY" `
  --policy.device=cuda `
  --task="Pick up the blue PCB and place it in the white fixture pocket" `
  --robot.type=so101_follower `
  --robot.port=COM5 `
  --robot.id=my_follower `
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
  --display_data=true

```

### What "working" looks like
On a simple, consistent task with 20–50 clean demos, a fine-tuned SmolVLA should attempt the motion
and sometimes succeed. Don't expect robustness from a tiny dataset + 6 GB run — the goal here is the
**full loop working end-to-end**, which by itself teaches you an enormous amount about VLAs.

### Sanity checks if it flails
- **Same image key / instruction** as training? (mismatch = garbage input)
- **Normalization stats** loaded from the dataset? (raw vs normalized action scale — your T4 point)
- **Calibration** unchanged since data collection?
- Under-trained or over-tiny dataset → collect more consistent demos.

## The payoff
You'll have taken a vision-language-action policy from architecture understanding (W2–W4) → reading
the production code → collecting real robot data → training → deploying on hardware. That full arc is
exactly the story that makes your π₀ interview answers concrete instead of theoretical.
