# Phase 0.7 — training three policies on the portable-rig dataset

Stage 3 of the Phase 0.7 programme: ACT, SmolVLA and π0.5 on the same 50 episodes, which is the **Q2
(policy transfer)** leg of [08 §A.1](../08_data_quality_research.md). The question these runs set up is
not "which policy wins" but *whether the data properties that matter hold across policy families*.

Dataset: [`..._20261003_211912_trimmed`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed)
(50 ep · 29,462 frames) · Verification: [phase07_data_verification.md](phase07_data_verification.md) ·
Rig: [phase07_portable_rig.md](phase07_portable_rig.md) · wandb project `phase07-policy-transfer`

| Policy | Notebook | Status |
|---|---|---|
| **ACT** | [colab_phase07_act_gpufree.ipynb](../../notebooks/colab_phase07_act_gpufree.ipynb) | ✅ **done 2026-10-05** |
| **SmolVLA** | [colab_phase07_smolvla_gpufree.ipynb](../../notebooks/colab_phase07_smolvla_gpufree.ipynb) | ✅ **done 2026-10-08** — but **overfits**; use the 5k checkpoint (§3) |
| **π0.5** | [Colab](../../notebooks/colab_phase07_pi05_Colab.ipynb) · [4090](../../notebooks/phase07_gpufree_pi05_train.ipynb) | ✅ **done 2026-10-09** — 10k steps on a Colab **A100**, verified on the Hub; eval curve read 2026-10-12 — **does not overfit, still falling at 10k** (§4) |

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
| Logging | one wandb project (see the warning below about reading across policies) |

**Why the holdout is set by hand.** Episodes 0–44 are 9 tray-spot blocks of 5 *consecutive* episodes and
45–49 are off-grid placements. lerobot's default split takes the last `ceil(n × eval_split)` episodes of
the list, which would hold out exactly 45–49 — the only unmarked placements — removing them from training
and making validation measure nothing but off-grid generalisation. Reordering the episode list puts a
balanced holdout in the tail instead: the last episode of blocks 2, 4, 6, 8 plus one off-grid.

> ⚠ **Held-out loss does not compare ACT against either VLA.** They optimise different objectives — ACT
> is L1 + KLD over a VAE latent, SmolVLA and π0.5 are flow-matching — and the logs make it plain: ACT
> reports `l1_loss`/`kld_loss`, SmolVLA `losses_after_forward`, π0.5 a bare `loss`. An L1 on predicted
> actions and an MSE on a flow-matching velocity residual have no conversion between them, so reading
> ACT's 0.185 against either VLA's number is meaningless in both directions.
>
> **Refined 2026-10-12: SmolVLA and π0.5 *do* compare, up to one measurable factor.** This paragraph
> originally said all three were incomparable, and that was too strong. Checked in the source: the two
> VLAs share the objective, the reduction over the real 6 action dims, the `Beta(1.5, 1.0)` flow
> timestep distribution and the Gaussian noise. They differ only in the action normaliser — `MEAN_STD`
> against π0.5's `QUANTILES` — and that factor is computable from the dataset's own statistics rather
> than unknown: ≈1.3× deflation of π0.5's loss. Worked through in
> [phase07_portable_rig.md §10.7a](phase07_portable_rig.md), which is also where the π0.5 curve finally
> gets read.
>
> They share one wandb project so that **conditions and seeds within a policy** overlay, which is what
> the dial grid needs. **Across policy families, still only rollout success answers Q2** — a loss gap,
> even a comparable one, is the wrong units for the question.

**Hyperparameters are per-policy on purpose.** Each uses its own published recipe; [08 §A.4](../08_data_quality_research.md)
holds hyperparameters fixed *after one clean run*, not identical across policy families. **Freeze what
these runs land on and reuse it for the dial grid.**

## The three runs at a glance

Measured, not estimated — each row comes from that policy's own smoke test or completed run.

| | ACT | SmolVLA | π0.5 |
|---|---|---|---|
| GPU | L4 | **L4** | A100 40 GB |
| Trainable params | 51.6 M (all) | 99.9 M of 450 M | 693 M of 3.3 B |
| Batch × steps | 8 × 60,000 | 64 × 20,000 | 16 × **10,000** |
| Samples / **epochs** | 480 k / **18.0** | 1.28 M / **47.9** | 160 k / **6.0** |
| Best eval loss | 0.1792 @ 50k (+3% at end) | **0.1258 @ 5k (+61% at end)** | **≈0.0515 @ 10k — still falling** (§4) |
| Throughput | 4.47 step/s | 2.00 s/step | 2.06 s/step |
| **Wall clock** | **3.85 h** (done) | **14.2 h** (done) | **5.7 h** compute, 22 h elapsed (done) |
| Peak GPU | 3.8 GiB | 16.5 GiB | 18.2 GiB (smoke) |
| Checkpoint size | 0.62 GB | 1.3 GB | **11.55 GB** |

Epochs are over the 45-episode training split (26,721 frames), which is what lerobot reports. The three
ran on different GPU classes, so they never contended. **π0.5's step budget ended up a third of what
this table first planned** — 10,000 rather than 30,000, so 6.0 epochs rather than 18.0 — and its 22 h of
elapsed time is mostly a dead Colab session rather than compute. §4 has both stories.

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

# 3 — SmolVLA — done 2026-10-08, and it overfits

[model](https://huggingface.co/HALDijkstraaa/smolvla_tray_cylinder_s1000_trim) (private, 4 checkpoints +
final, 6.18 GB). 20,000 steps, batch 64, seed 1000, on an **L4** — the published recipe ran unmodified,
peaking at **16.5 of 24 GiB** with no step-down, so the sample budget is exactly the official one. 14.2 h
wall clock against an 11.4 h projection.

It fits a 24 GB card because `freeze_vision_encoder=true` and `train_expert_only=true` are inherited from
`smolvla_base`: only **99,880,992 of ~450 M** parameters train, so there is no optimizer state and no
backward activations for the VLM. [09 §D](../09_home_deployment_and_hardware.md)'s 16 GB figure is for a
*full* fine-tune, which this is not.

## Held-out loss rises monotonically from the first checkpoint

| step | epochs | train loss | **eval loss** | gap |
|---|---|---|---|---|
| 5,000 | 12.0 | 0.0543 | **0.1258** ← min | +0.072 |
| 10,000 | 24.0 | 0.0284 | 0.1695 | +0.141 |
| 15,000 | 35.9 | 0.0193 | 0.2029 | +0.184 |

Train loss falls the whole way (0.304 → 0.017); eval loss ends **+61% above its minimum** and the
generalisation gap triples. This is textbook overfitting, and it is the risk flagged when the epoch
budgets were first compared — **SmolVLA gets 47.9 epochs where ACT gets 18.0 and π0.5 only 6.0**, because each
policy keeps its own published recipe on a dataset far smaller than those recipes assume. 100 M trainable
parameters over 45 episodes for 48 epochs is simply a lot of passes.

**ACT, for contrast, did not overfit**: its eval loss bottomed at 0.1792 (50k) and finished at 0.1848,
+3%. Same dataset, same holdout, same seed. That difference is itself a Q2-relevant result — how much
data each policy family needs before it starts memorising — and it is worth stating in the write-up
rather than hiding behind "we used each policy's published recipe".

## What to do about it

1. **Roll out the 5,000-step checkpoint, not just the final one.** It is on the Hub. By held-out loss it
   is the best of the four by a wide margin, and 20k is the worst.
2. It also happens to be the **sample-matched** comparison: 5k = 12.0 epochs, bracketing ACT's 18.0
   alongside the 10k checkpoint's 24.0. A Q2 claim at matched epochs is much cleaner than one at matched
   recipes.
3. **Do not re-run.** The four checkpoints already span 12–48 epochs; the short run is already in hand.
4. **For the dial grid** ([08 §A.4](../08_data_quality_research.md)): SmolVLA should use ~5k steps, not
   20k. That is both the better model *and* ~2.9 h instead of 11.4 h per condition — a 4× saving on the
   SmolVLA column.

## Three caveats before reading too much into it

- **The minimum is at the earliest checkpoint measured.** `eval_steps=5000` gives no resolution below
  5k, so the true optimum may be earlier still. The smoke test's 300-step eval was 0.1511 — worse than
  5k's 0.1258 — so the optimum sits somewhere between roughly 1k and 10k.
- **Held-out loss is not task success.** [phase06 §2](phase06_camera_ablation_training.md) found it
  separated conditions by less than checkpoint wobble while the robot showed a decisive dissociation. A
  monotone +61% is a far stronger signal than the sub-wobble differences that paper dismissed, but it
  still has to be confirmed by rollouts, and it is possible the 20k policy acts better than its loss.
- **A specific, testable mechanism.** The holdout is `[9, 19, 29, 39, 49]`, and **episode 49 is one of
  the five off-grid placements**; the other four come from the 9-spot grid. If the policy is memorising
  nine discrete tray positions, it would get steadily worse at the unmarked one in exactly this shape.
  lerobot logs only a pooled `eval_loss`, so this is a hypothesis — but it connects directly to the block
  structure found in [the verification report §5](phase07_data_verification.md), and a per-episode eval
  loss would settle it.

Configuration notes: `--rename_map` turns `top`/`wrist` into `camera1`/`camera2`, because the
checkpoint's own feature names win over the dataset's and the run would otherwise die on a feature
mismatch. The cosine schedule was pinned to the real step count (`scheduler_decay_steps=20000`).

---

# 4 — π0.5 — done 2026-10-09 on a Colab A100

## What ran, and where — verified on the Hub 2026-10-11

**Both planned runs were abandoned.** The rented RTX 4090 and the Colab L4 were dropped without running;
what executed is the **Colab notebook unchanged except for the GPU** — its 10,000 steps at `save_freq`
2,500, on a Colab **A100** — pushed as
[`pi05_tray_cylinder_s1000_trim_A100`](https://huggingface.co/HALDijkstraaa/pi05_tray_cylinder_s1000_trim_A100)
(private, 4 checkpoints + final, **55.55 GB**, wandb run `sxpkx9st`). Nothing exists under `..._trim` or
`..._trim_L4` in the namespace, which is consistent: this is the only π0.5 run there was.

**The step budget is the thing to carry forward, not the card.** Because the notebook ran as written, the
A100 inherited the *L4's* 10,000 steps rather than the 30,000 the 4090 was sized for — so π0.5 got
**6.0 epochs where ACT got 18.0**, on a card that could have afforded the full budget in ~16 h. That is
the one respect in which this run is not the one §1 planned, and §3's overfitting result is why it may
not matter.

Every figure below is read back out of the repo itself — `train_config.json`, the `training_state` files
and the commit timestamps — rather than from what the notebook meant to do.

| | Planned (Colab notebook) | What ran | |
|---|---|---|---|
| GPU | L4 | **A100 40 GB** | the job name says so, and **2.06 s/step proves it** (below) |
| Steps × batch | 10,000 × 16 | **10,000 × 16** | ✅ `training_step.json` reads 2500 / 7500 / 10000 |
| Samples / epochs | 160 k / 6.0 | **160 k / 6.0** | ✅ over the 26,721-frame train split |
| Seed | 1000 | **1000** | ✅ |
| Holdout | `[9,19,29,39,49]` in the tail | **last 5 of the 50-element list, in that order** | ✅ the reordering survived into the saved config |
| `eval_split` / `eval_steps` | 0.1 / 2,500 | **0.1 / 2,500** | ✅ every checkpoint has an eval at its own step |
| `max_eval_samples` | — | **0** — uncapped | the whole held-out split each time, not a subsample |
| Augmentation | off | **`image_transforms.enable: false`** | ✅ as §1 requires |
| `gradient_checkpointing` | true | **true** | ✅ the one knob that was not to move |
| Trainable scope | action expert only | **`train_expert_only` + `freeze_vision_encoder`** | ✅ 693 M of 3.3 B |
| Precision | bf16 | **`dtype: bfloat16`**, `use_amp: false` | ✅ |
| Normalisation | `QUANTILES` | **state + action `QUANTILES`, visual `IDENTITY`** | ✅ as §1 records |
| Checkpoints | 4, at 2,500 | **002500 · 005000 · 007500 · 010000** | each **11.55 GB** |
| Video decode | torchcodec | **torchcodec** | the pyav fallback was never needed |

**The disk estimate was exact, not approximately right.** The smoke test projected 9.35 GB of model plus
2.2 GB of optimizer state per checkpoint; the Hub holds **9.354 + 2.195 GB**. Four checkpoints plus the
9.35 GB final model is 55.55 GB, against the 69 + 9.35 GB six would have cost. The
[08 §A.4](../08_data_quality_research.md) grid can budget π0.5 storage straight from this figure.

### The session died at 7,500 steps, and the resume worked

The commit timestamps make the shape of the run visible:

| checkpoint | pushed (UTC) | Δ from previous | s/step |
|---|---|---|---|
| 002500 | 2026-10-08 17:14:35 | — | — |
| 005000 | 2026-10-08 18:40:25 | 1 h 25 m 50 s | **2.060** |
| 007500 | 2026-10-08 20:06:08 | 1 h 25 m 43 s | **2.057** |
| 010000 | 2026-10-09 13:55:58 | **17 h 49 m 50 s** | — |
| final weights + processors | 2026-10-09 13:58:52 | 2 m 54 s | — |

Checkpoints 002500–007500 all carry `output_dir: /content/outputs/pi05_tray_cylinder_s1000_trim_A100` and
`pretrained_path: lerobot/pi05_base`. Checkpoint 010000 carries `output_dir: ..._A100_resumed`,
`resume: true` and `pretrained_path: ..._A100_resumed/checkpoints/007500/pretrained_model`. So the Colab
session dropped after 7,500 steps and **Part C's resume path rebuilt it from the Hub the next day** —
exactly the failure that cell exists for, at exactly the cost it predicted (at most `save_freq` = 2,500
steps of lost work).

**The resume did not restart the learning-rate schedule**, which is what a resume most often breaks
silently. `scheduler_state.json` reads `last_epoch` 2500 → 7500 → 10000 with `_last_lr`
2.1705e-05 → 5.7950e-06 → **2.5000e-06**, and each of those matches
`decay_lr + ½(1+cos(π·step/10,000))·(peak−decay)` to five digits. The cosine therefore ran once,
monotonically, straight across the break, and finished on **2.5e-06 — exactly the configured `decay_lr`
floor** at exactly the last step. Had `scheduler_decay_steps` been left at the published 30,000 rather
than pinned to the real count, step 10,000 would have landed at 1.94e-05 — 7.8× the intended floor, with
the run stopping halfway down its decay.

**Throughput, measured rather than projected.** The two interior intervals give 2.060 and 2.057 s/step,
each covering 2,500 training steps *plus* one held-out eval and one 11.55 GB push. Against the smoke
test's 1.95 s/step of pure training, that puts eval-and-push overhead at ~0.11 s/step, or ~4.5 min per
2,500 steps. So **10,000 steps is 5.7 h of compute**; the 22 h between the first push and the final
upload is almost entirely the dead session.

That rate also identifies the card independently of the job name: at 2.06 s/step it cannot have been the
L4, which was projected 2.5–3.5× slower, i.e. 5–7 s/step. **The L4 projection of 14–19 h stays
untested**, and so does the 4090 projection of 17–25 h.

### The held-out loss curve — read 2026-10-12: π0.5 does **not** overfit

**It is still falling at the last checkpoint.** Read off wandb run
[`sxpkx9st`](https://forge.coreweave.com/wandb/bj-huang-university-of-toronto/phase07-policy-transfer/runs/sxpkx9st):
eval loss decreases monotonically from ≈0.0532 at 2,500 to ≈0.0515 by 7,500, with **no minimum reached
anywhere in range**. Compare §3, where SmolVLA bottomed out at 5,000 on this same dataset and ended
+61% above it.

**So the two flow-matching VLAs behave oppositely on identical data, holdout and seed** — SmolVLA
overfitting by 48 epochs, π0.5 still underfit at 6.0. That is a Q2 result in itself, and it says the
10,000 steps the A100 inherited from the L4 notebook were **short rather than merely different**: the
run stopped while the curve was still improving, on a card that could have afforded 30,000 in ~16 h.

Two consequences:

- **Which π0.5 checkpoint to roll out is no longer a coin flip.** 010000 should be the better one, and
  the round-1 run book still rolls out both — 18 of 54 trials — because it is the only on-robot evidence
  either way, but as a confirmation rather than a gamble.
- **π0.5 has the lowest held-out loss of the three.** How far that carries is the subject of
  [phase07_portable_rig.md §10.7a](phase07_portable_rig.md): sound against SmolVLA after a ≈1.3×
  normalisation correction, unavailable against ACT, and in no case a statement about task success.

The Hub stores weights, optimizer state and configuration — **not metrics** — which is why this needed
wandb at all, and why [`eval_checkpoints_heldout.py`](../../scripts/eval_checkpoints_heldout.py) exists as
the credential-free alternative.

Two ways to fill it in, answering slightly different questions:

1. **Read the four `eval_loss` points off wandb.** What the run itself measured, directly comparable to
   §3's SmolVLA table, and free.
2. **Re-measure with [`scripts/eval_checkpoints_heldout.py`](../../scripts/eval_checkpoints_heldout.py).**
   All four checkpoints are on the Hub, so this runs anywhere with a GPU and does not need the Colab
   session back. It repeats each checkpoint under three seeds, which π0.5 specifically needs: the
   flow-matching loss draws a random timestep and a random noise sample per forward pass, so a single
   number per checkpoint cannot be told apart from its own sampling noise. The script prints the noise
   floor, and a gap smaller than that floor is not a result.

**Resolved differently, 2026-10-12.** Rather than wait on either, the round-1 run book rolls out **two**
π0.5 checkpoints — 005000 and 010000, i.e. 3.0 and 6.0 epochs — and lets the robot rank them
([phase07_portable_rig.md §10](phase07_portable_rig.md)). That costs one extra arm and removes the
dependency on credentials this machine does not have, and it ranks the checkpoints by the quantity that
actually matters. §3 stands as the reminder that made this the right call: the last checkpoint can be the
worst one, so 010000 is not the default.

### One thing to carry into the rollouts

The saved config declares **three** visual inputs — `base_0_rgb`, `left_wrist_0_rgb` and
`right_wrist_0_rgb` — because the feature names come from `pi05_base`, and its names win over the
dataset's. The `rename_map` feeds two of them (`top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`); the
third has no source in a two-camera dataset. `observation.state` is likewise declared `(32,)` against the
SO-101's 6 joints, while `action` comes out `(6,)`.

The consequence is for rollout, not training: the keys the robot produces have to match the keys the
policy was trained on, so the cameras must either be *named* `base_0_rgb` and `left_wrist_0_rgb` at
rollout time, or be renamed the same way training renamed them. The model card's auto-generated
`lerobot-rollout` command carries no rename map — it only warns that the names must match — so it will
not work as printed against cameras called `top` and `wrist`.

## The smoke test that sized all of this (2026-10-06)

Everything from here to the end of §4 is the **planning record**: the measurements that set the
configuration, the rented-4090 host that was prepared and then dropped, and the Colab variant whose
configuration became the run above. Kept because the sizing is reusable — and because the 4090 and L4
projections in it were never tested, so they remain projections.

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

## Running it on a rented RTX 4090 (2026-10-08) — abandoned

> Prepared, never run: the A100 came free first. Kept for the host-sizing work, which is reusable, and
> because the VRAM and disk budgets below are what proved π0.5 fits a 24 GB card at all.

Moved off Colab to a 4090 box — **24 GB VRAM, 50 GB RAM, 12 vCPU, 30 GB system disk + 100 GB data
disk**. Everything checks out against what the earlier runs measured:

| | Needed (measured) | Host has | |
|---|---|---|---|
| VRAM | **18.18 GiB** peak | 24 GB | ✅ ~6 GiB spare |
| RAM | ~26 GB peak, total | 50 GB | ✅ |
| Data disk | **84.5 GB** (below) | 98 GB at `/root/gpufree-data` | ✅ ~13 GB margin |
| vCPU | `num_workers=8` | 12 | ✅ |

> **`gradient_checkpointing` must stay `true` on a 24 GB card.** The 18.18 GiB figure is *with* it. It
> trades compute for memory, so disabling it to go faster — fine on a 40 GB A100 — would likely push
> past 24 GB here. This is the one knob not to touch.

**The host is bare** — CUDA 13.0 drivers and Python 3.12, nothing else — so Part A now installs the
stack itself and in a specific order:

| | Pinned | Why that version |
|---|---|---|
| `torch` | **2.11.0+cu130** | lerobot v0.6.1 requires `torch<2.12,>=2.7`, so 2.11 is the newest it accepts; 2.14 is on the cu130 index but would be rejected |
| `torchvision` | **0.26.0+cu130** | the matching release for torch 2.11, and inside `torchvision<0.27` |
| `torchaudio` | **2.11.0+cu130** | matching release, for a consistent trio |

The host's system Python is **PEP 668 "externally managed"** (Debian/Ubuntu), so pip refuses to install
into it: `error: externally-managed-environment`. The install cell detects this — a venv or conda prefix,
or the `EXTERNALLY-MANAGED` marker in the stdlib directory — and adds `--break-system-packages` only when
there is genuinely no environment to install into. That flag is the right trade on a disposable
single-purpose box, since what PEP 668 protects is an OS whose own tooling depends on the system Python.
The markdown gives the venv-plus-ipykernel route for anyone who would rather not. Tested across all four
cases: bare system Python, venv, conda, and no-marker.

**The install goes on `/`, not the data disk**, and that is deliberate rather than an oversight. Torch
plus the `nvidia-*` CUDA runtime wheels plus lerobot's dependencies are roughly **10–15 GB**: the 30 GB
root overlay has nothing else to do with it, while the data disk needs 84.5 GB for the base model and six
checkpoints. Installing there would leave about half a gigabyte of margin.

Torch is installed **from the CUDA 13.0 index before lerobot**, because installing lerobot first would
pull a default-CUDA wheel. The cell then verifies what actually landed rather than assuming: that torch
still reports CUDA 13.x after pip resolved lerobot's dependencies, that it sees the GPU, and that
`torchcodec` (pulled by `[dataset]`) imports against this torch. That last check matters because a
torchcodec compiled for a different torch still *exists* as a module, and lerobot selects its decoder on
`find_spec` alone — it would be chosen and then fail at decode time. If it does not import, the notebook
falls back to `--dataset.video_backend=pyav`, which costs nothing here since dataloading is not the
bottleneck (`data_s` 0.44 against `updt_s` 1.95).

**Credentials** are set in a cell before the install, since there is no Colab secret store: the two keys
go into the process environment (inherited by every `lerobot-train` subprocess) and the HF token file,
with an opt-in append to `~/.bashrc` for reconnected shells. Only a masked prefix is printed.

**Disk budget, keeping every checkpoint** (requested, so the whole eval curve stays inspectable — the
SmolVLA result in §3 is why that matters):

| | GB |
|---|---|
| `pi05_base` weights | 14.47 |
| PaliGemma tokenizer (weights never fetched) | 0.02 |
| trimmed dataset | 0.70 |
| 6 checkpoints × 11.55 | 69.30 |
| **total at end of run** | **84.5 of 100** |

The host mounts that disk at **`/root/gpufree-data`**, with a 30 GB overlay at `/` and a 16 GB
`/dev/shm`. The work-directory cell prefers an ordered list of candidate mounts rather than picking the
one with the most free space: this box also exposes a shared 1 TB `/data` that is 81% full, which *looks*
roomier and is the wrong place to write 85 GB. It prints every candidate it found so a bad guess is
visible, and `PHASE07_WORK` overrides. It also reports free space on `/` (pip puts 10–15 GB of torch and
dependencies there, and the overlay is only 30 GB) and the size of `/dev/shm`, since a container's 64 MB
default is the usual cause of "DataLoader worker killed by signal: Bus error" with `num_workers > 0`.

Two things make the disk fit rather than nearly fit. The notebook now points `HF_HOME` and
`HF_LEROBOT_HOME` at the **data disk** — left alone they default to `~/.cache`, i.e. the 30 GB system
disk, which the 14.5 GB base model plus pip and torch would fill. And Part B deletes its own smoke
checkpoint when it passes, returning 11.5 GB before the real run starts.

**A watchdog replaces the reaper.** Nothing is deleted during the run; instead, every 2 minutes it
measures the checkpoints actually written, extrapolates to those still to come, and compares against
free space — printing a status line normally and a loud warning, with options, if the projection stops
fitting. Part B prints the same projection *before* the run so a shortfall is known at minute 15 rather
than hour 14.

**`eval_steps` lowered to 2500** — 12 eval points instead of 6, and **every checkpoint gets an eval at
its own step**. SmolVLA's held-out minimum landed on the *earliest* checkpoint measured, so its true
optimum is unresolved; evaluating more often costs ~20 min across a ~20 h run and does not touch
training. `save_freq` stays at 5000, because checkpoints — not evals — are what the disk budget is made
of.

2500 rather than a rounder 2000 because lerobot tests `step % eval_steps == 0` and
`step % save_freq == 0` *independently* (`lerobot_train.py:593-595`). At 2000 the checkpoints at 5k, 15k
and 25k would have had no eval at their own step, leaving half the saved models without a loss beside
them — exactly the pairing needed to choose which checkpoint to roll out. The config cell now asserts
`SAVE_FREQ % EVAL_STEPS == 0` so the two cannot drift apart again.

**Expect roughly 17–25 h** on a 4090 (extrapolated from 1.95 s/step on an A100; the 4090 has more dense
bf16 compute but ~⅔ the memory bandwidth, and gradient checkpointing makes this compute-heavy). The
smoke test measures it exactly in ~15 min.

## A second, smaller run on a Colab L4 — became the A100 run

[colab_phase07_pi05_Colab.ipynb](../../notebooks/colab_phase07_pi05_Colab.ipynb) — and **this is the configuration
that actually ran**, on an A100 rather than the L4 it was sized for. Everything
from the runtime check onward is **copied verbatim** from the 4090 notebook — verified byte-identical for
the preflight, smoke, Part C and auth cells — so the two are comparable at matched steps. Only Part A
differs (Colab ships torch and has its own secret store) plus four knobs.

**Memory is fine; it is the other two limits that bite.**

| | 4090 host | Colab L4 |
|---|---|---|
| Peak VRAM | 18.18 GiB of 24 | same, of 24 — ~6 GiB spare |
| Steps | 30,000 (**18.0** epochs) | 10,000 (**6.0** epochs) |
| Projected | ~17–25 h | **~14–19 h** |
| `save_freq` | 5,000 → 6 checkpoints | 2,500 → 4 checkpoints |
| Local checkpoints | all kept | **newest 2** (reaped; all still pushed) |
| Hub repo | `..._trim` | `..._trim_L4` |

- **Speed.** The L4 is roughly 2.5–3.5× slower than the A100 that measured 1.95 s/step, so the full
  30,000 steps would be **41–57 h** — past any Colab session. 10,000 lands at 14–19 h.
- **Disk.** Measured across these runs, Colab has 112.6 GB with ~44 GB of base image, leaving **~68 GB**.
  Six π0.5 checkpoints (69.3 GB) plus the 14.5 GB base model is 84.5 GB and does not fit — the π0.5 smoke
  test alone once drove a Colab disk to 94.1 GB. Keeping the newest two is ~38 GB.
- **Is 6 epochs enough?** Unknown for π0.5, but not obviously short: SmolVLA on this same dataset bottomed
  at **12 epochs and was 61% worse by 48** (§3). If π0.5 overfits on a similar schedule, 6 epochs may be
  near its optimum. The 2,500-step checkpoints bracket it either way.

`gradient_checkpointing` stays `true` here too — the 18.18 GiB figure depends on it, and a 24 GB card has
no room to turn it off.

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

**The rollout run book is written:** [phase07_portable_rig.md §10](phase07_portable_rig.md) — six arms
(two checkpoints per policy), 54 trials, mirrored interleave, with the commands and the blind-scoring
protocol. It also records a measurement this doc could not make: **all six checkpoints load and run
inside the control-loop budget on the 5080**, π0.5 included, at 8.87 GiB peak and 316 ms per chunk
against an 833 ms budget. Not yet run.

- **Rollouts.** Interleaved across checkpoints, blind-scored, on the robot. Nothing here substitutes for
  that, and §2 explains why the loss numbers cannot settle Q2. → [§10](phase07_portable_rig.md)
- **π0.5's held-out curve — done 2026-10-12.** It does not overfit; it is still falling at 10k (§4), so
  π0.5 is underfit rather than over-trained and 10,000 steps was short. What remains open is the
  question that is now worth more than checkpoint selection ever was: **did held-out loss predict the
  robot's ordering**, across three policy families on one dataset?
  [§10.7a](phase07_portable_rig.md) sets out how far the cross-family comparison carries and proposes
  the policy-agnostic metric (un-normalised L1 in degrees) that would settle it.
- **Pre-registration.** [08 §A.2](../08_data_quality_research.md) asks for predicted orderings committed
  *before* the comparison is read. All three training runs are now done, so the *loss* comparison can no
  longer be pre-registered — but the **rollouts** have not started. A six-row predicted ordering with
  reasoning and confidence is drafted in [§10.7](phase07_portable_rig.md) and **needs confirming or
  editing before block 1 runs**, or it is worthless.
- **`phase07_log.csv`** still does not exist; per-episode `success` and `stage` are not recoverable from
  the data and the rollout protocol compares against them.
