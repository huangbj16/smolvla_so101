#!/usr/bin/env python3
"""Generate the Phase 0.7 round-2 training notebook from the Colab notebook that actually ran.

**Template: `colab_phase07_pi05_Colab.ipynb`.** That is the only one of the Phase 0.7 notebooks that
completed a real run on Colab; the `*_gpufree` ones target a rented box with its own data volume and
carry machinery Colab does not need or want - a `/root/gpufree-data` work-dir chooser, a PEP 668
`--break-system-packages` install path, and a `PHASE07_WORK` override. Deriving from the working one
keeps the install, auth, preflight, smoke-test and resume cells byte-identical to a configuration
that is known good, and confines the diff to the policy.

    python scripts/make_r2_notebook.py

Writes colab_phase07_r2_smolvla.ipynb.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "colab_phase07_pi05_Colab.ipynb"
OUT = REPO / "colab_phase07_r2_smolvla.ipynb"

D1 = "so101_tray_cylinder_varied_20261013_082911_trimmed"
START_FROM = "smolvla_tray_cylinder_s1000_trim"     # round 1's finished policy
JOB = "smolvla_tray_cylinder_s1000_d1"

TITLE = r"""# Phase 0.7 round 2 — continue SmolVLA on D1

**This continues round 1's SmolVLA on the new data.** It does not train from scratch and it does not
start from `smolvla_base`: the starting weights are `HALDijkstraaa/smolvla_tray_cylinder_s1000_trim`,
the policy that scored 6/18 in [round 1](Details/phase07_portable_rig.md).

| | |
|---|---|
| Dataset | `so101_tray_cylinder_varied_20261013_082911_trimmed` — 50 episodes, **head-trimmed, 2 s tail kept**, 33,337 frames |
| Starts from | `smolvla_tray_cylinder_s1000_trim` (round 1's final, 20k steps on D0) |
| Steps | **20,000** at batch 64 = 1,280,000 samples = **42.5 epochs of D1** |
| Learning rate | **1e-5** — lowered 10× from the preset, see below |
| Checkpoints | every 5,000 → 4, each pushed to the Hub as it is written |
| GPU | **L4 or better.** Round 1 peaked at **16.5 of 24 GiB** at batch 64 — a T4 is too small |
| Expected | **~14 h on an L4**, ~5 h on an A100 — Part B measures it in ~10 min and prints the real figure |
| Output | `HALDijkstraaa/smolvla_tray_cylinder_s1000_d1` (private) |

### The learning rate is the one thing that would silently ruin this

SmolVLA's preset is a cosine from peak **1e-4** down to 2.5e-6 over `scheduler_decay_steps`, after
1,000 warmup steps — and **a new run restarts that schedule from the top.** Round 1 *finished* at
2.5e-6, so accepting the default would hand the converged policy 1e-4 again: **40× its final rate**,
which would largely undo the warm start. The config cell therefore passes
`--policy.optimizer_lr=1e-5`, dropping the peak 10×. The cosine still runs, pinned to the real step
count, so the run anneals rather than sitting flat.

### The 5,000-step checkpoint is the one to compare against ACT

Steps are the wrong unit across policies with different batch sizes. At batch 64, **5,000 steps is
320,000 samples — exactly what ACT's whole 40,000-step run at batch 8 sees.** So the 5k checkpoint is
sample-matched to ACT and is where the clean comparison is read; 10k/15k/20k are the train-longer arm.

### What this costs the pre-registration

[§12.1](Details/phase07_portable_rig.md) pre-registered P1–P3 against *fresh* training on D1.
Continuing instead means this policy has seen D0 **and** D1, so a change in the ACT−SmolVLA gap can no
longer be attributed to D1's character alone — P1 and P2 are weakened, and **P3 is not measurable at
all**. The trade is deliberate: this is the faster route to a usable policy.

---

## 1 — Runtime check
"""

GPU_CELL = '''import subprocess, torch

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", GPU)
assert GPU, "No GPU visible. On Colab: Runtime > Change runtime type > GPU (L4 or A100)."
if "T4" in GPU:
    print("\\n!! T4 (16 GB): TOO SMALL. SmolVLA peaked at 16.5 GiB at batch 64 in round 1.")
    print("   Part B will OOM and step the batch down, which changes the recipe and breaks the")
    print("   sample-matched comparison with ACT. Switch to an L4.")
elif "L4" in GPU:
    print("\\nL4 (24 GB): fits - round 1 ran this exact recipe here at 16.5 GiB, no step-down.")
    print("            ~14 h for 20,000 steps. Speed is the constraint, not memory.")
elif "A100" in GPU:
    print("\\nA100: fits easily and is ~2.5-3x faster - roughly 5 h instead of 14.")
    print("      Note Colab bills A100 at several times the L4 rate, so the wall-clock saving")
    print("      is real but the compute-unit saving may not be.")
'''

INSTALL_CELL = '''!pip install -q "lerobot[smolvla,dataset,training]==0.6.1"
import lerobot, torch
print("lerobot", lerobot.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
'''

CONFIG_HEAD = f'''import collections, json, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

# Colab has one big disk and no separate data volume, so no redirection is needed.
WORK = Path("/content")
OUTPUT_ROOT = WORK / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
HF_USER = "HALDijkstraaa"
DATASET = f"{{HF_USER}}/{D1}"
PROJECT = "phase07-policy-transfer"
SEED    = 1000

# Held out for validation loss. ROUND 2 NOTE: D1 randomises the cylinder position continuously, so
# the 9-block structure that forced a hand-picked holdout in round 1 is gone - the verifier measures
# 1x block separation where D0 had 27x, and lerobot's default "last N" split would now be perfectly
# fair. The same indices are kept anyway so the split MECHANISM is identical to round 1 and the only
# thing differing between the two runs is the data. They are arbitrary here, which is the point.
HOLDOUT  = [9, 19, 29, 39, 49]
EPISODES = [e for e in range(50) if e not in HOLDOUT] + HOLDOUT

EVAL_SPLIT, LOG_FREQ = 0.1, 200

# A SmolVLA checkpoint is ~1.3 GB, so four of them plus the base model is a few GB on a Colab disk
# with ~68 GB free. Nothing needs reaping; keep them all.
KEEP_ALL_CHECKPOINTS = True
KEEP_CKPTS = 2

# ---- SmolVLA round 2 ----------------------------------------------------------------------------
JOB   = "{JOB}"
# ROUND 2 IS A CONTINUATION. BASE_MODEL is round 1's finished policy, NOT lerobot/smolvla_base.
BASE_MODEL = f"{{HF_USER}}/{START_FROM}"
# Round 1's recipe, unchanged: batch 64. STEPS is 20,000 further steps over D1.
BASE_BATCH, BASE_STEPS = 64, 20000
BATCH, STEPS = BASE_BATCH, BASE_STEPS
SAVE_FREQ = 5000      # 4 checkpoints; the 5,000 one is sample-matched to ACT's whole run
EVAL_STEPS = 5000     # held-out loss cadence; training is unaffected either way.
# SAVE_FREQ must be a multiple of EVAL_STEPS, so that every checkpoint has an eval at its own
# step: lerobot tests `step % eval_steps == 0` and `step % save_freq == 0` independently, and
# runs the eval first, so a mismatched pair leaves some checkpoints with no loss beside them.
assert SAVE_FREQ % EVAL_STEPS == 0, f"{{SAVE_FREQ}} is not a multiple of {{EVAL_STEPS}}"

# A WARM START NEEDS A LOWER PEAK LR. SmolVLA's preset cosine runs from 1e-4 down to 2.5e-6, and a
# new run restarts it from the top - 40x the rate round 1 ended on, which would undo the warm start.
OPTIM_LR = "1e-5"

# The checkpoint declares its own camera names (inherited from smolvla_base); the batch is renamed
# to match. Without this the run dies on a feature mismatch.
RENAME = {{
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
}}

NUM_WORKERS = min(8, os.cpu_count() or 2)
'''

TRAIN_CMD = '''

def train_cmd(job=None, steps=None, batch=None, extra=()):
    """Build the lerobot-train argv.

    Every default is resolved from the globals at CALL time, not bound at definition time, so a
    batch step-down in Part B reaches a later bare train_cmd() call. Checkpoints go to the Hub every
    SAVE_FREQ steps, so a disconnect costs at most that many.
    """
    job = JOB if job is None else job
    steps = STEPS if steps is None else steps
    batch = BATCH if batch is None else batch
    return ["lerobot-train",
        f"--dataset.repo_id={DATASET}", f"--dataset.episodes={EPISODES}",
        f"--dataset.eval_split={EVAL_SPLIT}", f"--eval_steps={EVAL_STEPS}",
        *( [f"--dataset.video_backend={globals()['VIDEO_BACKEND']}"]
           if globals().get("VIDEO_BACKEND") else [] ),
        # --policy.path is what makes this a CONTINUATION: it loads weights into a new run with a
        # fresh optimizer and step counter. (--resume=true is a different thing - it restores
        # optimizer state to continue an interrupted run, which is not what is wanted here.)
        f"--policy.path={BASE_MODEL}",
        f"--rename_map={json.dumps(RENAME)}",

        "--policy.device=cuda",
        "--policy.push_to_hub=true", f"--policy.repo_id={HF_USER}/{job}", "--policy.private=true",
        "--save_checkpoint_to_hub=true",
        # Pin the cosine to the real step count so the schedule completes, and lower its peak for
        # the warm start (see OPTIM_LR above).
        f"--policy.scheduler_decay_steps={steps}",
        f"--policy.optimizer_lr={OPTIM_LR}",
        f"--batch_size={batch}", f"--steps={steps}", f"--num_workers={NUM_WORKERS}",
        f"--seed={SEED}", f"--save_freq={SAVE_FREQ}", f"--log_freq={LOG_FREQ}",
        f"--output_dir={OUTPUT_ROOT}/{job}", f"--job_name={job}",
        "--wandb.enable=true", f"--wandb.project={PROJECT}", *extra]
'''


def main() -> int:
    nb = json.loads(TEMPLATE.read_text())
    cells = nb["cells"]

    # --- cell 0: title ---------------------------------------------------------------------
    cells[0]["source"] = TITLE.splitlines(keepends=True)
    # --- cell 1: GPU guidance (memory numbers are policy-specific) --------------------------
    cells[1]["source"] = GPU_CELL.splitlines(keepends=True)
    # --- cell 3: install extra ---------------------------------------------------------------
    cells[3]["source"] = INSTALL_CELL.splitlines(keepends=True)

    # --- cell 9: configuration + train_cmd ---------------------------------------------------
    cfg = "".join(cells[9]["source"])
    tail = cfg[cfg.index("\nKEEP = ("):]          # keep the log filter and helpers verbatim
    cells[9]["source"] = (CONFIG_HEAD + TRAIN_CMD + "\n" + tail).splitlines(keepends=True)

    # --- cell 8: the markdown above the config ----------------------------------------------
    cells[8]["source"] = (
        "## 4 — Configuration\n\n"
        "Everything the run needs, in one cell. The only differences from round 1 are the dataset, "
        "the starting weights and the learning rate.\n"
    ).splitlines(keepends=True)

    # --- cell 11: preflight. Drop the pi0.5 gated-backbone block; SmolVLA's is not gated. ----
    pf = "".join(cells[11]["source"])
    a = pf.index("# ------")
    b = pf.index("ok_ds = api.repo_exists")
    pf = (pf[:a]
          + "# SmolVLA's backbone is NOT gated, so the pi0.5 PaliGemma access check is gone from here.\n"
            "# What remains is the part that matters for any pretrained policy: that the dataset is\n"
            "# really there and tagged, and that the renamed camera keys line up with the checkpoint's.\n\n"
          + pf[b:])
    pf = pf.replace('checks.append(("idle-trimmed (fewer frames than the source\'s 35,141)", info["total_frames"] < 35141))',
                    'checks.append(("idle-trimmed (fewer frames than D1\'s raw 38,389)", info["total_frames"] < 38389))')
    pf = pf.replace('''    if k in ("freeze_vision_encoder", "train_expert_only", "optimizer_lr", "chunk_size",
             "n_action_steps", "dtype", "gradient_checkpointing", "resize_imgs_with_padding",
             "image_resolution"):''',
                    '''    if k in ("freeze_vision_encoder", "train_expert_only", "optimizer_lr", "chunk_size",
             "n_action_steps", "num_steps", "resize_imgs_with_padding", "empty_cameras"):''')
    cells[11]["source"] = pf.splitlines(keepends=True)

    # --- cell 13: the smoke test's throwaway repo --------------------------------------------
    run_cell = "".join(cells[13]["source"]).replace("pi05_smoke_delete_me", "smolvla_r2_smoke_delete_me")
    cells[13]["source"] = run_cell.splitlines(keepends=True)

    # --- trailing markdown: drop pi0.5 prose -------------------------------------------------
    cells[16]["source"] = (
        "---\n\n## Done\n\n"
        "The run pushes every checkpoint to "
        "`HALDijkstraaa/smolvla_tray_cylinder_s1000_d1` as it goes, so a dead Colab session costs at "
        "most `SAVE_FREQ` = 5,000 steps and Part C resumes from the Hub.\n\n"
        "**Next:** read the eval curve, then roll out. Compare the **5,000-step** checkpoint against "
        "ACT's 40,000-step run - those are sample-matched at 320,000 samples - and the 20,000-step "
        "one as the train-longer arm. Round 1's lesson is that the loss-optimal checkpoint was *not* "
        "the robot-optimal one, so do not pick on loss alone.\n"
    ).splitlines(keepends=True)

    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None

    # Nothing from the rented-box notebooks may survive into a Colab notebook.
    blob = json.dumps(nb)
    for banned in ("gpufree-data", "PHASE07_WORK", "break-system-packages", "paligemma", "pi05_base"):
        if banned in blob:
            sys.exit(f"rented-GPU or pi0.5 machinery survived into the output: {banned!r}")
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.name}  ({len(cells)} cells, from {TEMPLATE.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
