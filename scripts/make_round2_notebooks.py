#!/usr/bin/env python3
"""Generate the Phase 0.7 round-2 training notebooks from the round-1 ones.

Round 2 changes the **data** and nothing else that matters, so the notebooks are derived rather than
rewritten: the install, credential, preflight, resume and disk-watchdog machinery is copied verbatim
from the round-1 notebooks that already ran to completion, and only the configuration block and the
title are patched. [phase07_training.md §1](../Details/phase07_training.md) makes the same argument
for the round-1 set - the three were generated from one source rather than kept in step by review -
and the reason is stronger here, because round 2's whole claim is a comparison against round 1.

    python scripts/make_round2_notebooks.py

Writes colab_phase07_r2_act.ipynb and colab_phase07_r2_smolvla.ipynb.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
D1 = "so101_tray_cylinder_varied_20261013_082911_trimmed"

# (source notebook, output notebook, job name, round-1 steps -> round-2 steps)
SPECS = [
    ("colab_phase07_act.ipynb", "colab_phase07_r2_act.ipynb",
     "act_tray_cylinder_s1000_trim", "act_tray_cylinder_s1000_d1", 60000, 40000),
    ("colab_phase07_smolvla.ipynb", "colab_phase07_r2_smolvla.ipynb",
     "smolvla_tray_cylinder_s1000_trim", "smolvla_tray_cylinder_s1000_d1", 20000, 30000),
]

# The D0 holdout comment explains a block structure D1 does not have. Replaced rather than kept,
# because a stale rationale in a config cell is how the wrong thing gets copied forward.
OLD_HOLDOUT_COMMENT = """# Held out for validation loss. Episodes 0-44 are 9 tray-spot blocks of 5 CONSECUTIVE episodes and
# 45-49 are off-grid, so lerobot's default "last ceil(n*eval_split) episodes" split would hold out
# exactly the 5 off-grid episodes - removing the only unmarked placements from training. The split
# slices the tail of --dataset.episodes IN THE ORDER GIVEN, so putting the holdout last reorders it
# into a balanced one: the last episode of blocks 2, 4, 6, 8 plus one off-grid."""

NEW_HOLDOUT_COMMENT = """# Held out for validation loss. ROUND 2 NOTE: D1 randomises the cylinder position continuously, so
# the 9-block structure that forced a hand-picked holdout in round 1 is gone - the verifier measures
# 1x block separation where D0 had 27x, and lerobot's default "last N" split would now be perfectly
# fair. The same indices are kept anyway, so that the split MECHANISM is byte-identical to round 1
# and the only thing that differs between the two runs is the data itself. They are arbitrary here,
# which is the point: any 5 episodes of D1 are an equally fair holdout."""


def patch_config(src: str, spec) -> str:
    _, _, old_job, new_job, old_steps, new_steps = spec
    out = src
    out = out.replace('DATASET = f"{HF_USER}/so101_tray_cylinder_portable_20261003_211912_trimmed"',
                      f'DATASET = f"{{HF_USER}}/{D1}"')
    out = out.replace(f'JOB   = "{old_job}"', f'JOB   = "{new_job}"')
    out = out.replace(OLD_HOLDOUT_COMMENT, NEW_HOLDOUT_COMMENT)
    # BASE_STEPS appears as "BASE_BATCH, BASE_STEPS = <batch>, <steps>"
    import re
    out = re.sub(r"(BASE_BATCH, BASE_STEPS = \d+), " + str(old_steps),
                 r"\1, " + str(new_steps), out)
    if f"BASE_STEPS = " in out and str(new_steps) not in out:
        sys.exit(f"step patch failed for {new_job}")
    return out


def title_for(spec) -> str:
    name = "ACT" if "act" in spec[1] else "SmolVLA"
    other = "SmolVLA" if name == "ACT" else "ACT"
    other_nb = "colab_phase07_r2_smolvla.ipynb" if name == "ACT" else "colab_phase07_r2_act.ipynb"
    steps = spec[5]
    where = ("**local RTX 5080 Laptop** — measured 5.4 it/s at batch 8 with two cameras "
             "([phase06 §5](Details/phase06_camera_ablation_training.md)), which is *faster* than the "
             "L4's 4.47 it/s"
             if name == "ACT" else
             "**Colab L4** — batch 64 peaked at 16.5 of 24 GiB in round 1, which does not fit the "
             "5080's 15.5 GiB, and lerobot has no gradient accumulation to work around it")
    runtime = "**~2.1 h** for 40k steps locally" if name == "ACT" else "**~21 h** for 30k steps on an L4"
    matched = ("40k is the checkpoint round 1 compared at, and round 1 confirmed 40k ≈ 60k "
               "(prediction P1), so the extra 20k is dropped"
               if name == "ACT" else
               "30k extends past round 1's 20k because round 1 found the *later* checkpoint better on "
               "the robot despite a worse loss — **the 20k checkpoint is still written**, so the "
               "matched comparison and the train-longer arm both come out of one run")
    return f"""# Phase 0.7 round 2 — fine-tune {name} on D1, the varied dataset

One of **two notebooks**, one per policy. Round 2 trains only {name}; π0.5 is gated on the
frozen-feature probe in [§12.7](Details/phase07_portable_rig.md) and is not run here.

**This is a controlled comparison against round 1, not a fresh start.** Everything is held identical to
[the round-1 {name} notebook](colab_phase07_{'act' if name == 'ACT' else 'smolvla'}.ipynb) — recipe,
batch, seed, holdout mechanism, eval cadence — and **only the dataset changes**. D1 is the same size as
D0 (50 episodes), so the comparison is size-matched and the single variable is the data's character.
That is what [§12.1](Details/phase07_portable_rig.md) pre-registered.

| | |
|---|---|
| Dataset | `{D1}` — 50 episodes, **head-trimmed, 2 s tail kept** |
| Trained from | **`{'scratch' if name == 'ACT' else 'lerobot/smolvla_base'}`, fresh** — not warm-started from the round-1 checkpoint |
| Steps | **{steps:,}** — {matched} |
| GPU | {where} |
| Expected runtime | {runtime} — Part B prints the real figure |
| Output | `HALDijkstraaa/{spec[3]}` (private), checkpointed to the Hub as it goes |
| Sibling | [{other}]({other_nb}) |

**Why fresh rather than continued.** Warm-starting from the round-1 checkpoint would carry D0's
low-divergence structure into the run and confound the test; it would also make prediction **P3**
(ACT's KLD should no longer collapse) unmeasurable, because KLD behaviour on a warm start is not
comparable to KLD behaviour from scratch.

**What D1 changed, measured.** Reach-phase divergence between demonstrations rose **0.322 → 0.478**
(+48%, p < 0.0001; +0.113 and still p < 0.0001 after matching on position spread), the idle tail went
**45 → 95 frames**, and the 9-block position structure is **gone** (27× separation → 1×). The
pre-registered predictions that follow from it are in
[§12.1](Details/phase07_portable_rig.md) — read them before reading the results.
"""


def main() -> int:
    for spec in SPECS:
        src_nb, out_nb = spec[0], spec[1]
        nb = json.loads((REPO / src_nb).read_text())
        patched_cfg = False
        for cell in nb["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] == "code" and "BASE_BATCH, BASE_STEPS" in text:
                cell["source"] = patch_config(text, spec).splitlines(keepends=True)
                patched_cfg = True
            elif cell["cell_type"] == "code":
                # dataset name also appears in preflight / cleanup cells
                cell["source"] = text.replace(
                    "so101_tray_cylinder_portable_20261003_211912_trimmed", D1).replace(
                    spec[2], spec[3]).splitlines(keepends=True)
        if not patched_cfg:
            sys.exit(f"{src_nb}: never found the config cell")
        nb["cells"][0]["source"] = title_for(spec).splitlines(keepends=True)
        for cell in nb["cells"]:          # a committed notebook carries no stale output
            if cell["cell_type"] == "code":
                cell["outputs"], cell["execution_count"] = [], None
        (REPO / out_nb).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out_nb}  ({len(nb['cells'])} cells, from {src_nb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
