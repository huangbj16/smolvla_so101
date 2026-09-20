# Phase 0.6 — camera ablation: train ACT on top / wrist / top+wrist

Run book for the training-and-evaluation half of question **C1**. Phase 0.5 measured whether the second
camera *should* help ([phase05_camera_test_results.md](phase05_camera_test_results.md)); this phase
measures what actually changes in the policy's behavior when you take a camera away.

**Pass 1 (this doc's default): one seed, three conditions, behavior-focused.** Not a success-rate
measurement — too few trials for that. The goal is to see *how* a top-only and a wrist-only policy move
differently, and whether those differences match what the divergence metric predicted. Extend to multiple
seeds and a statistical success comparison only if pass 1 looks promising (§4b).

Plan: [08 §A.4](../08_data_quality_research.md) · Task: [toolkit_task_card.md](toolkit_task_card.md) ·
Dataset: `HALDijkstraaa/so101_toolkit_cylinder_20260917_165544` (50 episodes, 37,690 frames, 30 fps).
Scripts: [`scripts/train_camera_ablation.sh`](../scripts/train_camera_ablation.sh) ·
[`scripts/make_eval_schedule.py`](../scripts/make_eval_schedule.py). Written 2026-09-20.

---

# Summary

- **Train three policies. Do not train one and mask a camera at inference** — in lerobot's ACT each camera
  contributes ~300 tokens to one encoder sequence, so removing a camera is an out-of-distribution input,
  not an ablation (§1).
- **60k steps, batch 8, checkpoints every 10k.** 100k is the ACT paper recipe, but for one simple task
  with 50 demos the 10k-spaced checkpoints tell you directly whether behavior was still changing at 60k
  (§2).
- **Run the three sequentially, not in parallel.** Measured: training is GPU-bound, so three concurrent
  processes each drop to ~1/3 speed and the set finishes **8% later** than back to back. VRAM was never
  the constraint (§5).
- **Total ~7 h at 60k steps** (top 1.6 h + wrist 1.6 h + top+wrist 3.1 h, plus ~30 min of validation
  passes) on the 5080 laptop, measured from the real trainer in fp32. No cloud GPU needed.
- **Tracked in wandb** (project `phase06-camera-ablation`, one run per condition so the curves overlay)
  and **pushed to the Hub as private repos** at the end of each run.
- **Larger batches buy nothing.** Throughput is flat at ~59 samples/s from batch 8 to 32 — the GPU is
  saturated at batch 8 (§5).
- **The default holdout had to be fixed.** Episodes were recorded position-block by position-block
  (P1 = ep 0–4 … P10 = ep 45–49), so lerobot's "last 5 episodes" split would have put *all of P10* in
  validation and removed that position from training. The scripts now hold out a balanced set (§2).
- **Nice property of lerobot's ACT: the ResNet18 backbone is shared across cameras**, so all three
  conditions have essentially the same parameter count. No capacity confound.

| Step | What | Time | Attended? |
|---|---|---|---|
| 0 | Preflight checks | 5 min | yes |
| 1 | Smoke test, 500 steps × 3 | ~5 min | yes |
| 2 | Three training runs, 60k steps, sequential | ~7 h | no |
| 3 | Checkpoint selection | 10 min | yes |
| 4 | Behavior pass: 30 rollouts, video recorded | ~45 min | yes |
| 5 | Watch the video, write up behavioral differences | ~1 h | yes |

---

# 1 — Why three policies, and not one with a masked camera

The tempting shortcut is to train `top+wrist` once and zero or drop a camera at inference. It does not
measure what we want. From `policies/act/modeling_act.py` (line ~475 in 0.6.1):

```python
for img in batch[OBS_IMAGES]:
    cam_features = self.backbone(img)["feature_map"]
    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
    ...
    encoder_in_tokens.extend(list(cam_features))
```

Each 480×640 image goes through ResNet18 (stride 32) to a 15×20 feature map = **300 tokens per camera**,
appended into one transformer encoder sequence alongside the latent and robot-state tokens.

| Shortcut | What actually happens |
|---|---|
| **Drop a camera** | Sequence length changes 602 → 302. The encoder never saw that length; the learned relationship between the state/latent tokens and the image block is gone. |
| **Zero the image** | Length is preserved, but a black frame is an input the model never saw. |

Either way you measure *graceful degradation under camera failure*, which is a robustness question, and
the error is biased against whichever camera the joint policy leaned on more — exactly the quantity under
test. Three separate trainings, three separate policies.

**Two architecture details worth knowing before reading the results:**

- **One shared backbone for all cameras.** Unlike the original ACT (a separate ResNet per camera), lerobot
  reuses `self.backbone` in the loop. So `top`, `wrist` and `top+wrist` have nearly identical parameter
  counts — measured 51.6M for the two-camera config. The only thing that differs between conditions is
  what the encoder sees, which makes this ablation cleaner than it would be upstream.
- **No camera-identity embedding.** `encoder_cam_feat_pos_embed` is a 2D sinusoidal embedding computed
  from the feature map's H×W, so **both cameras receive identical positional embeddings**. ACT tells the
  two views apart purely by appearance, not by any positional tag. (Relates to the temporal-encoding
  discussion in [phase05 results](phase05_camera_test_results.md) "Future studies".)

---

# 2 — Training recipe, and the two places the defaults are wrong

Everything except the camera set is held fixed. **Do not tune per condition** — a per-condition tune
invalidates the comparison.

| Knob | Value | Why |
|---|---|---|
| Policy | ACT, `resnet18`, `chunk_size=100`, `n_action_steps=100` | 0.6.1 defaults = the paper recipe |
| Batch / lr | 8 / 1e-5 | Original ACT recipe, and batch 8 already saturates the GPU (§5) |
| **Steps** | **60,000** (`save_freq=10000`) | See below |
| Seed | 1000, same for all three | One seed this pass; removes init/shuffle noise from the comparison |
| Dataset | `..._165544`, 50 episodes | The `_163836` (5 ep) and `_164823` (3 ep) folders are test runs — do not use |
| **Holdout** | **episodes 9, 19, 29, 39, 49** | See below |
| Normalization | Dataset stats, `use_imagenet_stats=true` | Default; identical inputs across runs |

### Why 60k and not 100k

100k × batch 8 = 800k samples ≈ **21 passes** over 37,690 frames. That is the ACT paper's recipe, sized
for harder bimanual tasks. This is one simple task with a fixed fixture and a single grasp style, so 100k
is likely past the point where behavior stops changing — but "likely" is not measured, and an undertrained
policy would look like a camera effect.

So: **60k steps with a checkpoint every 10k**, and let the checkpoints answer the question. Compare the
40k and 60k checkpoints on the robot. If they behave the same, you converged and 60k was enough. If they
differ, resume:

```bash
lerobot-train --config_path=outputs/train/act_both_s1000/checkpoints/last/pretrained_model/train_config.json --resume=true --steps=100000
```

Cost of the choice: 60k is ~6.3 h for all three, 100k is ~10.5 h. Both are one unattended session, so if
you would rather not think about it, run `STEPS=100000` and skip the question.

Disk: each ACT checkpoint is ~620 MB (51.6M params plus Adam state), so 6 checkpoints × 3 runs ≈ **11 GB**.

### Why the holdout had to be changed

Episodes were recorded **one position at a time**, five in a row before the cylinder moved:

```
ep 0-4 = P1 · 5-9 = P2 · 10-14 = P3 · … · 45-49 = P10
```

`datasets/factory.py` holds out *the last* `ceil(n × eval_split)` episodes of the episode list. With the
default ordering that is episodes 45–49 — **all five repetitions of P10**. Two things go wrong:

1. **P10 disappears from training.** All three policies would be evaluated at a tape mark they had never
   seen, at one of the ten positions you plan to test.
2. The validation loss would measure out-of-position generalization from a single position, not fit.

The fix uses the fact that the split slices the tail of `--dataset.episodes` **in the order given**
(verified: `LeRobotDataset` preserves the order, and the split takes the last `n_eval` of it). The scripts
pass the 45 training episodes first and a balanced holdout last:

```
holdout = 9, 19, 29, 39, 49      # the last episode of P2, P4, P6, P8, P10
```

Every position keeps at least 4 episodes in training. Verified end to end: 45 train / 5 eval, eval
episodes exactly `[9, 19, 29, 39, 49]`, and P10 retains episodes 45–48 in training.

**`eval_steps` must be set too.** lerobot defaults to `eval_steps=0`, which builds the eval dataloader and
then never runs it (`is_eval_step = cfg.eval_steps > 0 and ...`) — the holdout episodes would be dropped
from training and produce nothing in return. The script passes **`--eval_steps=2000`**: measured at ~30 s
per pass over the full 5-episode holdout with two cameras (~15 s with one), that is 30 points on the wandb
curve for about 8% extra wall clock.

> **Do not set `--max_eval_samples`.** It looks like a way to make validation cheaper, but it slices
> `frames[:per_task]` — the **first** n frames of the holdout, not a sample of it. With one task that means
> the opening of episode 9 and nothing else: you would be validating on the reach phase of a single
> position. Leave it at 0 and use the whole holdout.

To skip validation entirely and train on all 50, pass `EVAL_SPLIT=0` (the script then also forces
`eval_steps=0`, since `eval_steps > 0` with `eval_split == 0` raises). The validation loss is a sanity
check for divergence only — action-prediction L1 correlates weakly with task behavior, and it cannot
settle this ablation either way.

> **Knock-on caveat for Phase 0.5.** Position-block recording means **position is confounded with session
> time**: anything that drifted over the recording session (lighting, operator fatigue) varies with
> position index. Phase 0.5's conclusions are unaffected — it bootstrapped over episodes and never broke
> results down by position — but any future per-position analysis of this dataset carries the confound.

---

# 3 — Training, step by step

### Step 0 — preflight

```bash
conda activate lerobot && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv && du -sh ~/.cache/huggingface/lerobot/HALDijkstraaa/so101_toolkit_cylinder_20260917_165544 && df -h /home | tail -1
```

Checks: the env is active, the GPU is idle (close anything holding VRAM), the dataset is cached locally,
and there is disk room for ~11 GB of checkpoints.

### Step 1 — smoke test

Five minutes now beats discovering a typo after an overnight run. This runs 500 steps of each condition:

```bash
STEPS=500 SAVE_FREQ=500 EVAL_STEPS=250 OUT=outputs/smoke ./scripts/train_camera_ablation.sh
```

Check in the log that each run prints `Train/eval split: 45 train, 5 eval` and the right input features
(`observation.images.top` only, `observation.images.wrist` only, then both), and that loss is falling.
`num_learnable_params` should read **51,597,190 for all three** — the shared-backbone property from §1,
confirmed empirically. Then delete `outputs/smoke`.

Verified 2026-09-20: all three split 45/5, correct features, loss 6.89 → 2.97 over 500 steps, identical
parameter counts, 591 MB per checkpoint.

The smoke run also exercises wandb. It does **not** push to the Hub: the script refuses to push when
`STEPS < 10000`, so a forgotten `PUSH_TO_HUB` cannot litter your HF account with 500-step models.

### Step 2 — the three runs

```bash
./scripts/train_camera_ablation.sh
```

Runs `top`, `wrist`, `both` **sequentially** at 60k steps each, ~7 h total, logging to
`outputs/train/act_<cond>_s1000.log`, streaming to wandb, and pushing each final model to a private Hub
repo `HALDijkstraaa/act_toolkit_cylinder_<cond>_s1000`. Useful variants:

```bash
STEPS=100000 ./scripts/train_camera_ablation.sh
```

```bash
EVAL_SPLIT=0 ./scripts/train_camera_ablation.sh
```

```bash
SEED=1001 ./scripts/train_camera_ablation.sh wrist
```

```bash
WANDB=false PUSH_TO_HUB=false ./scripts/train_camera_ablation.sh
```

The script is a thin wrapper; the single command it issues for the top-only condition is:

```bash
lerobot-train --dataset.repo_id=HALDijkstraaa/so101_toolkit_cylinder_20260917_165544 --dataset.episodes="[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 9, 19, 29, 39, 49]" --dataset.eval_split=0.1 --eval_steps=2000 --policy.type=act --policy.device=cuda --policy.push_to_hub=true --policy.repo_id=HALDijkstraaa/act_toolkit_cylinder_top_s1000 --policy.private=true --policy.input_features="{'observation.state': {'type': 'STATE', 'shape': [6]}, 'observation.images.top': {'type': 'VISUAL', 'shape': [3, 480, 640]}}" --batch_size=8 --steps=60000 --num_workers=8 --seed=1000 --save_freq=10000 --log_freq=100 --output_dir=outputs/train/act_top_s1000 --job_name=act_top_s1000 --wandb.enable=true --wandb.project=phase06-camera-ablation
```

The `wrist` condition swaps the image key; the `both` condition **omits `--policy.input_features`
entirely**, since both cameras is the default.

> **How the camera selection works.** `policies/factory.py` fills `cfg.input_features` from the dataset
> only `if not cfg.input_features`, so a CLI override survives. There is no flag for "use a subset of
> cameras" — this is the mechanism.

### Step 3 — monitor and select

Watch it in wandb — all three runs land in project `phase06-camera-ablation`, named `act_top_s1000`,
`act_wrist_s1000`, `act_both_s1000`, so `train/loss` and `eval_loss` overlay on one chart. Or locally:

```bash
tail -f outputs/train/act_both_s1000.log
```

Expect ~10.6 it/s for the single-camera runs and ~5.4 it/s for `both`, with a smoothly falling L1 loss.
Take the **final checkpoint** of each run for the behavior pass, and keep the 40k one for the
convergence check in §2:

```
outputs/train/act_top_s1000/checkpoints/last/pretrained_model
```

---

# 4 — Evaluation

## 4a — Pass 1: the behavior pass (this phase)

30 rollouts, one per position per condition, ~45 minutes. **This is not a success-rate measurement** —
10 trials per condition gives a ±16-point standard error, so any difference under ~30 points is noise.
What it does give you is video of three policies attempting the same ten positions, which is enough to
see *behavioral* differences, and those are usually obvious to the eye long before they are statistically
significant.

```bash
python scripts/make_eval_schedule.py --trials-per-position 1 --seed 0 > Details/phase06_eval_log.csv
```

Columns: `trial, block, condition, position, success, fail_stage, time_s, notes`. Even in a behavior pass,
**fill in `fail_stage`** (`reach | grasp | transport | insert | retreat | none`) — that column is the
direct test of the Phase 0.5 predictions in §6.

**Load the camera presets first.** The policies were trained on images produced by preset 1 (C920 / top)
and preset 2 (C922 / wrist). An unloaded preset is a distribution shift that will look like a policy
failure. See [01 Step 6](../01_setup_robot.md).

Then, with `CKPT` set to whichever condition the schedule calls for:

```bash
CKPT=outputs/train/act_top_s1000/checkpoints/last/pretrained_model TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 lerobot-rollout --strategy.type=episodic --policy.path=$CKPT --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/phase06_eval_act_top --dataset.single_task="Pick up the white cylinder and place it in the hole of the black fixture" --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=15 --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

- **Always connect both cameras**, even for the single-camera policies. The policy consumes only the keys
  in its own `input_features`; the extra stream is ignored but keeps the physical scene identical (the
  wrist camera is mounted either way) and gives you both views on video for review. **Verify this on the
  first rollout** — if a single-camera policy errors on the extra key, drop that camera from
  `--robot.cameras` for its trials.
- **`reset_time_s=15`**, longer than the 3 s used for recording: you reposition the cylinder between trials.
- **→** ends a trial early, **←** discards and re-runs, **Esc** stops.
- `--strategy.type=episodic` saves video, which is the actual deliverable of this pass.

## 4b — Pass 2: the success-rate comparison (only if pass 1 looks promising)

Everything above is too small to compare success rates. When you want numbers:

| Decision | Value | Why |
|---|---|---|
| **Trials** | 3 per position × 10 positions × 3 policies = **90** (~2.5 h) | 30 per policy → ±9 points at 50%. At 20 trials it is ±11, enough to miss a real 15-point effect |
| **Seeds** | 3 per condition | With 50 episodes, seed variance is comparable to the effect size |
| **Ordering** | Interleaved blocks (`--trials-per-position 3`) | Blocked evaluation hands all session drift to whichever policy ran last |
| **Scoring** | From video, **blind to condition**, after the session | You will unconsciously favour the condition you expect to win |

Nine training runs (3 conditions × 3 seeds) is ~19 h sequentially here at 60k steps — two nights, or a
couple of hours on rented GPUs in parallel.

---

# 5 — Compute: measured on this machine

RTX 5080 Laptop (16 GB), driver 595.84, torch 2.11+cu130, 24 CPU cores. Two sets of numbers below:
**as-run** figures come from the real trainer in fp32 (lerobot's `use_amp` default), **bench** figures
come from an isolated harness under bf16 autocast, which is ~1.3× optimistic. Trust the as-run column for
planning; the bench rows are for the batch-size, worker and parallelism *comparisons*, where the ratio is
what matters and the precision is held constant.

### As run — plan from these

Measured from the smoke run, 2026-09-20, steady-state rate (the first ~200 steps are slower while the
dataloader fills):

| Condition | it/s | samples/s | VRAM | 60k steps | 100k steps |
|---|---|---|---|---|---|
| `top` | 10.6 | 85 | 2.1 GiB | **1.6 h** | 2.6 h |
| `wrist` | 10.4 | 84 | 2.1 GiB | **1.6 h** | 2.7 h |
| `top+wrist` | 5.4 | 43 | 3.7 GiB | **3.1 h** | 5.2 h |
| **Total, sequential** | | | | **~6.3 h** | **~10.5 h** |

Add a few minutes for periodic validation on the 5 held-out episodes every 5000 steps.

**`--policy.use_amp=true` is ~1.3× faster** (it drives Accelerate's `mixed_precision` to bf16), which
would bring the set to ~4.6 h. It is **not** enabled by default here: it changes the numerics of the
reference recipe, and nothing has verified ACT converges identically under it on this data. If you turn it
on, turn it on for all three conditions.

### Bench — batch size and workers (bf16 harness)

| Cameras | Batch | Workers | ms/step | **samples/s** | Peak VRAM |
|---|---|---|---|---|---|
| one (`top`) | 8 | 8 | 72 | **111** | 2.0 GiB |
| one (`top`) | 8 | 16 | 72 | 111 | 2.0 GiB |
| two | 8 | 8 | 134 | **60** | 3.5 GiB |
| two | 8 | 16 | 134 | 60 | 3.5 GiB |
| two | 16 | 8 | 269 | 59 | 6.3 GiB |
| two | 32 | 8 | 549 | 58 | 11.9 GiB |

**Two findings:**

1. **Larger batches buy nothing.** Throughput is flat at 58–60 samples/s across batch 8, 16 and 32 —
   step time scales exactly linearly with batch size. The GPU is already saturated at batch 8. Bigger
   batches only cost VRAM (11.9 of 16 GiB at batch 32) and change the optimization recipe. **Stay at 8.**
2. **The bottleneck is the GPU, not video decoding.** Doubling workers from 8 to 16 changes nothing, and
   a single-camera run is almost exactly **twice** as fast — which is the cost of one ResNet18 pass and
   300 tokens, not of decoding. Both videos are decoded either way (`LeRobotDataset` has no column
   selection), so if decoding were the limit the single-camera run would not have sped up.
   *This corrects an earlier note in this doc that said all three runs would cost the same wall clock.*

### Can the three run in parallel on 16 GB? Measured: yes, but don't

Three concurrent processes, batch 8, 6 workers each (bf16 harness, so compare the two rows to each other,
not to the as-run table):

| | ms/step each | samples/s each | 60k steps |
|---|---|---|---|
| Three in parallel | 295 / 295 / 299 | 27 | **5.0 h** (all finish together) |
| Sequential, back to back | 72 / 72 / 134 | 111 / 111 / 60 | **4.6 h** |

(Both rows exclude validation passes, which add ~30 min across the three runs at `eval_steps=2000`.)

- **Memory is not the problem**: the three allocator peaks sum to 7.5 GiB of 16 GiB.
- **Speed is.** Each process drops to roughly a third of its solo speed — the exact signature of three
  jobs time-slicing one saturated GPU — and the set finishes **8% later** than running them back to back,
  with the context-switching overhead as the loss. Note also that in parallel all three run at the *same*
  295 ms/step even though `both` does twice the work of `top`: the scheduler is dividing the GPU evenly,
  not doing more work.
- **Verdict: run them sequentially**, which is what the script does. Parallelism only pays on separate
  GPUs.

This supersedes the "ACT · ~5 GB @ bs 8 · train on cloud 4090" row in
[08 §D.1](../08_data_quality_research.md) for the small real-track datasets: measured, ACT trains
comfortably on the 5080, and the only reason to rent a GPU is running seeds concurrently.

---

# 6 — What to look for, and what Phase 0.5 predicts

The behavior pass produces video, not a p-value. Phase 0.5 found that, once divergence is normalized
within each phase, each camera wins exactly where its geometry says it should:

![Top vs wrist by phase](figs/04_top_vs_wrist_by_phase.png)

*Phase 0.5: top is better by 0.025 in reach and 0.093 in transport; wrist by 0.029 in grasp and 0.039 in
insert.*

**Pre-registered predictions** — write these down before watching the video, so the read isn't
retro-fitted:

| # | Prediction | Behavioral signature to look for | From |
|---|---|---|---|
| **P1** | `top+wrist` is the most competent overall | Fewest stalls, fewest retries | H-c: lowest divergence overall |
| **P2** | The `top+wrist` advantage over `wrist` alone is **small** | Hard to tell apart by eye | The two views are largely redundant: 0.499 vs 0.522 at k=10 |
| **P3** | `wrist`-only fails at **reach** | Arm sets off in a generic direction, or toward an average position, regardless of where the cylinder is. Should be worst at the extreme positions | H-b: top wins reach/transport |
| **P4** | `top`-only fails at **grasp/insert** | Reaches the right area confidently, then closes the gripper a few mm off, or hovers over the hole without seating | H-a: wrist wins grasp/insert |
| **P5** | All three struggle in grasp and insert | High retry/hover rate everywhere | In the fine phases *no* observation space beat the random baseline |

**Also worth watching for, because the metric cannot predict it:**

- **Does `wrist`-only exhibit search behavior?** Sweeping until the cylinder enters view would be the
  interesting positive result — a policy compensating for a missing view with motion.
- **Smoothness.** Higher action divergence in training data tends to show up as jitter or hesitation at
  the ambiguous moments, not only as failure.
- **Where each policy commits.** The moment a trajectory stops being generic and starts aiming at the
  actual cylinder position is a direct read on when the observation became informative.

**Write up:** for each condition, the failure-stage tally over the 10 trials, one or two representative
video clips, and a paragraph on the above. That, plus the held-out L1, is the first data point for Future
Study #1 in the [Phase 0.5 results](phase05_camera_test_results.md): does divergence predict performance?

---

# 7 — Gotchas

| Risk | Mitigation |
|---|---|
| Camera presets not loaded before eval | Check the rerun view first ([01 Step 6](../01_setup_robot.md)). A dark top image is a distribution shift, and it will look like policy failure |
| Using the default episode ordering | Would remove P10 from training entirely (§2). The script handles it; check the log says `45 train, 5 eval` |
| Running the three in parallel to "save time" | Measured 8% slower (§5). Sequential |
| Raising the batch size to "use the VRAM" | Measured zero throughput gain (§5), and it changes the recipe |
| Fixture moved between training data and eval | Re-check the taped outline against the top view before starting |
| Single-camera policy errors on the extra camera key | Verify on the first rollout (§4a); fall back to dropping the camera from `--robot.cameras` |
| Reading 10 trials per condition as a success rate | It is ±16 points. Pass 1 is for behavior; §4b is for numbers |
| Overnight run dies silently | The script `tee`s logs; check `outputs/train/*.log` before starting eval. The loop does not abort on a failed run |
| Holding out episodes with `eval_steps=0` | lerobot's default never evaluates them — you lose 5 episodes for nothing (§2). The script sets `--eval_steps=2000` |
| Capping validation with `--max_eval_samples` | It takes the first n frames, not a sample: you would validate on the reach phase of one position (§2) |
| A smoke run pushed to the Hub | The script refuses to push when `STEPS < 10000` |
