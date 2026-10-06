# Phase 0.7 — training three policies on the portable-rig dataset

Stage 3 of the Phase 0.7 programme: ACT, SmolVLA and π0.5 on the same 50 episodes, which is the **Q2
(policy transfer)** leg of [08 §A.1](../08_data_quality_research.md). The question these runs set up is
not "which policy wins" but *whether the data properties that matter hold across policy families*.

Dataset: [`..._20261003_211912_trimmed`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed)
(50 ep · 29,462 frames) · Verification: [phase07_data_verification.md](phase07_data_verification.md) ·
Rig: [phase07_portable_rig.md](phase07_portable_rig.md) · wandb project `phase07-policy-transfer`

| Policy | Notebook | Status |
|---|---|---|
| **ACT** | [colab_phase07_act.ipynb](../colab_phase07_act.ipynb) | ✅ **done 2026-10-05** |
| **SmolVLA** | [colab_phase07_smolvla.ipynb](../colab_phase07_smolvla.ipynb) | ⬜ not started — unblocked |
| **π0.5** | [colab_phase07_pi05.ipynb](../colab_phase07_pi05.ipynb) | ◐ smoke test passed 2026-10-06; full run ~16.7 h (§4) |

---

# 1 — Held fixed across all three

This is what makes the comparison mean anything, and it is enforced by generating the three notebooks
from one source rather than by review.

| | Value |
|---|---|
| Dataset | the trimmed copy — idle head/tail removed, which is **required** for the VLAs (§3 of the verification report) |
| Episodes / holdout | `[9, 19, 29, 39, 49]`, placed in the **tail** of `--dataset.episodes` |
| Split | `eval_split=0.1`, `eval_steps=5000` → 45 train / 5 eval |
| Seed | 1000 |
| Cameras | both (`top` + `wrist`) |
| Augmentation | off |
| Logging | one wandb project, so the eval curves overlay |

**Why the holdout is set by hand.** Episodes 0–44 are 9 tray-spot blocks of 5 *consecutive* episodes and
45–49 are off-grid placements. lerobot's default split takes the last `ceil(n × eval_split)` episodes of
the list, which would hold out exactly 45–49 — the only unmarked placements — removing them from training
and making validation measure nothing but off-grid generalisation. Reordering the episode list puts a
balanced holdout in the tail instead: the last episode of blocks 2, 4, 6, 8 plus one off-grid.

**Hyperparameters are per-policy on purpose.** Each uses its own published recipe; [08 §A.4](../08_data_quality_research.md)
holds hyperparameters fixed *after one clean run*, not identical across policy families. **Freeze what
these runs land on and reuse it for the dial grid.**

---

# 2 — ACT — done 2026-10-05

**Recipe.** From scratch, `resnet18`, `chunk_size=100`, `n_action_steps=100`, batch 8, 60,000 steps,
fp32, lr 1e-5 — the same recipe as the Phase 0.6 baselines.
[wandb run](https://wandb.ai/bj-huang-university-of-toronto/phase07-policy-transfer) ·
[model](https://huggingface.co/HALDijkstraaa/act_tray_cylinder_s1000_trim) (private, 6 checkpoints +
final, 3.92 GB).

| | |
|---|---|
| GPU / runtime | **L4, 3.85 h** (smoke test projected 3.7 h) |
| Throughput | 4.47 step/s · ~36 samples/s · peak **3.77 GiB** |
| Coverage | 480k samples = **17.9 epochs** over 29,462 frames |
| Parameters | 51,597,190 — identical to Phase 0.6, as expected |
| Train loss | 6.815 → **0.053** (L1 0.526 → 0.052; KLD 0.629 → 0.0002) |
| **Held-out loss** | 0.2197 @5k → **0.1792 @50k** (min) → 0.1848 @55k |

**Eval loss is flat from ~20k steps.** The curve reaches 0.1852 at 20k and ends at 0.1848 at 55k, with
the whole 20k–60k range inside a 0.179–0.192 band — i.e. the spread between late checkpoints is
checkpoint wobble, not progress.

- **For the rollouts:** compare the **40k and 60k** checkpoints on the robot, exactly as
  [phase06 §2](phase06_camera_ablation_training.md) prescribes. If they behave the same, 60k was more
  than enough.
- **For the dial grid:** if 40k ≈ 60k on hardware, dropping to ~30k steps halves the ACT compute for
  every condition in the grid, which matters at 19 conditions × 3 seeds.

**KLD collapsed to 0.0002.** Expected for ACT on a single-operator, single-grasp-style dataset: there is
almost no action multimodality for the VAE latent to encode, so it goes unused. Worth knowing before
reading anything into the latent, and a reminder that this dataset is deliberately low-variance.

### A tempting cross-phase comparison, and why it is not one

Phase 0.6 F3 (idle-trimmed, old rig) finished at `top` **0.1828** and `top+wrist` **0.1989**. Phase 0.7
`top+wrist` is **0.1848** — numerically better than the old two-camera run and level with the old
best single-camera run, which *looks* like the bird's-eye rig fixing the fusion degradation
[phase06 §6.5](phase06_camera_ablation_training.md) found.

**Do not report it that way.** The two datasets differ in rig, camera pose, calibration frame, task
string and object start distribution (tray vs taped table) all at once, so nothing attributes the
difference to the camera. On top of that, held-out loss is an action-prediction L1 that
[phase06 §2](phase06_camera_ablation_training.md) already showed correlates weakly with task success —
it separated the three Phase 0.6 conditions by less than one curve's wobble while the robot showed a
decisive double dissociation. Treat the number as a sanity check that training converged. **The answer is
rollouts.**

---

# 3 — SmolVLA — not started

Unblocked: its backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) is **not gated**, and the
preflight passes. Fine-tunes `lerobot/smolvla_base` with `--rename_map` turning `top`/`wrist` into
`camera1`/`camera2` (the checkpoint's own feature names win over the dataset's, so without the map the
run dies on a feature mismatch). Target batch 64 / 20k steps, stepped down on OOM with steps scaled to
hold samples-seen fixed. Inherits `freeze_vision_encoder=true` and `train_expert_only=true` from the base
config; Part A prints both so the run log records what was actually used.

---

# 4 — π0.5 — access granted 2026-10-06; smoke test passed

**Smoke test (A100 40 GB, batch 16, 300 steps):** all checks pass. 693,422,112 trainable parameters
(of 3.3 B — the frozen VLM is the rest), eval_loss 0.3419 → 0.1514 over 300 steps, peak **18.2 GiB**.

| | Measured | Full run (30,000 steps) |
|---|---|---|
| Throughput | **1.95 s/step** (~8 samples/s) | — |
| Wall clock | 13 min for 300 steps | **~16.7 h** (16.2 h train + ~0.5 h eval/pushes) |
| Coverage | — | 480k samples = **16.3 epochs** (ACT got 17.9) |
| Peak GPU | 18.2 GiB of 40 | ~18 GiB — room to spare |
| **Disk per checkpoint** | **11.55 GB** (9.35 model + 2.2 optimizer) | 6 checkpoints = **69 GB** |

**Two things this surfaced.**

1. **The run is ~16.7 h**, well past a comfortable Colab session. Part C resumes from the last Hub
   checkpoint, so a disconnect costs at most `save_freq` = 5,000 steps ≈ 2.7 h. Levers, in order of
   preference: try `--policy.gradient_checkpointing=false` (peak is 18.2 of 40 GiB, so there is headroom,
   and it trades memory back for speed — one 13-minute smoke test settles it); or cut `STEPS`, at the
   cost of epoch-matching to ACT (20k → 11.3 h / 10.9 epochs; 15k → 8.6 h / 8.1 epochs). Raising the
   batch does **not** help: the update dominates (`updt_s:1.949` vs `data_s:0.442`), so it is
   compute-bound and bigger batches buy proportionally longer steps.
2. **Disk would have ended the run.** Six unreaped checkpoints (69 GB) plus the 14.5 GB base model on a
   112 GB Colab disk does not fit — it filled to 76.7 GB during the *smoke test alone*. Part C now reaps
   superseded local checkpoints (keeping the newest 2 ≈ 23 GB); they are redundant locally the moment
   `save_checkpoint_to_hub` pushes them. The **Hub** repo still ends at ~79 GB, since every checkpoint is
   pushed — Part D has a pruning snippet.

> π0.5 checkpoints are ~19× an ACT checkpoint (11.55 GB vs 0.62 GB). Worth carrying into the
> [08 §A.4](../08_data_quality_research.md) grid: the π0.5 column costs far more storage than compute
> planning alone suggests.

## How it was blocked, and what fixed it



**The failure was.** The smoke test died ~10 minutes in, after the dataset downloads and the model is on the
GPU, while building the processor:

```
ValueError: Failed to instantiate processor step 'tokenizer_processor' with config:
{... 'tokenizer_name': 'google/paligemma-3b-pt-224'}
Error: You are trying to access a gated repo. 403 Client Error.
```

**Why there is no workaround.** π0.5's VLM is PaliGemma, and lerobot loads its tokenizer from
`google/paligemma-3b-pt-224`, hardcoded at `lerobot/policies/pi05/processor_pi05.py:152`. Checked on the
Hub: **no π0.5 checkpoint bundles a tokenizer** — not `lerobot/pi05_base`, not any of LeRobot's
`pi05_libero` / `pi05_droid` / `pi05_robocasa` variants, and not any community SO-100/SO-101 fine-tune
(`Shani123/...`, `pepijn223/...`, `SkieyFly/...`). Every one of them carries
`tokenizer_name: google/paligemma-3b-pt-224` in `policy_preprocessor.json` and fetches it at runtime. So
the answer to "how did everyone else fine-tune π0.5?" is simply: **they have access to the gated repo.**

Ungated third-party re-uploads of PaliGemma exist, but pointing at one would substitute a tokenizer that
may not match token-for-token, silently changing the language conditioning and making this run
incomparable with both the pretrained prior and everyone else's results. Not worth it to save a wait.

**Resolved 2026-10-06** by requesting access. For the record, what was needed:

1. Access at **https://huggingface.co/google/paligemma-3b-pt-224**. The gate is `manual`
   (`model_info(...).gated == "manual"`), so it is reviewed rather than granted on acceptance — request
   it early, since it is the long pole.
2. The token also needs **`canReadGatedRepos`** — *"Read access to contents of all public gated repos you
   can access"* in the fine-grained token settings. As of 2026-10-06 the token already has this, so only
   step 1 is outstanding.

The notebook now checks this in **Part A**, in about three seconds, and prints which of the two
requirements is missing. Previously it surfaced ten minutes and 9.4 GiB into the run.

> `model_info()` returns normally for a gated repo you cannot read — only fetching a file proves access.
> The preflight downloads `config.json` for exactly that reason.

**Note for [08 §A.4](../08_data_quality_research.md).** This is a scheduling dependency the compute plan
does not currently mention: π0.5 cannot run at all until a third party approves a licence request, so the
π0.5 column of the training grid is gated on something outside the project's control. It is also the
first item on the plan's own cut list, which makes the risk tolerable.

---

# 5 — Open

- **Rollouts.** Interleaved across checkpoints, blind-scored, on the robot. Nothing here substitutes for
  that, and §2 explains why the loss numbers cannot settle Q2.
- **Pre-registration.** [08 §A.2](../08_data_quality_research.md) asks for predicted orderings committed
  *before* the comparison is read. Worth writing down before the π0.5 run unblocks, so the full
  three-policy comparison is pre-registered rather than two-thirds post-hoc.
- **`phase07_log.csv`** still does not exist; per-episode `success` and `stage` are not recoverable from
  the data and the rollout protocol compares against them.
