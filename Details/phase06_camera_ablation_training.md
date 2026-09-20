# Phase 0.6 — camera ablation: train ACT on top / wrist / top+wrist

Run book for the training-and-evaluation half of question **C1**. Phase 0.5 measured whether the second
camera *should* help ([phase05_camera_test_results.md](phase05_camera_test_results.md)); this phase
measures whether it *does*, by training three ACT policies and running them on the robot.

Plan: [08 §A.4](../08_data_quality_research.md) · Task: [toolkit_task_card.md](toolkit_task_card.md) ·
Dataset: `HALDijkstraaa/so101_toolkit_cylinder_20260917_165544` (50 episodes, 37,690 frames, 30 fps).
Scripts: [`scripts/train_camera_ablation.sh`](../scripts/train_camera_ablation.sh) ·
[`scripts/make_eval_schedule.py`](../scripts/make_eval_schedule.py). Written 2026-09-20.

---

# Summary

- **Train three policies. Do not train one and mask a camera at inference** — in lerobot's ACT each camera
  contributes ~300 tokens to one encoder sequence, so removing a camera is an out-of-distribution input,
  not an ablation (§1).
- **The 5080 laptop is enough.** Measured on this machine: 134 ms/step at batch 8 with both cameras,
  3.5 GiB of 16 GiB VRAM, **~3.7 h per 100k-step run**, ~11 h for all three. A cloud A100 buys nothing
  except parallelism for extra seeds (§5).
- **Evaluation, not training, is the bottleneck** — ~90 real-robot trials, ~2.5 h of hands-on time, and it
  has to be interleaved and blind-scored to be worth anything (§4).
- **Nice property of lerobot's ACT: the ResNet18 backbone is shared across cameras**, so all three
  conditions have essentially the same parameter count. No capacity confound.

| Step | What | Time | Attended? |
|---|---|---|---|
| 0 | Preflight checks | 5 min | yes |
| 1 | Smoke test, 500 steps × 3 | ~5 min | yes |
| 2 | Three training runs, 100k steps | ~11 h | no — run overnight |
| 3 | Checkpoint selection | 10 min | yes |
| 4 | Real-robot eval, 90 interleaved trials | ~2.5 h | yes |
| 5 | Blind scoring + analysis | ~2 h | yes |

---

# 1 — Why three policies, and not one with a masked camera

The tempting shortcut is to train `top+wrist` once and zero or drop a camera at inference. It does not
measure what we want. From [`policies/act/modeling_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py) (line ~475 in 0.6.1):

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
test. Three separate trainings, three separate numbers.

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

# 2 — What is held fixed

Everything except the camera set. Hyperparameters are the published ACT recipe; **do not tune per
condition** — a per-condition tune invalidates the comparison.

| Knob | Value | Why |
|---|---|---|
| Policy | ACT, `resnet18`, `chunk_size=100`, `n_action_steps=100` | 0.6.1 defaults = the paper recipe |
| Batch / lr / steps | 8 / 1e-5 / 100,000 | Original ACT recipe. 100k × 8 ≈ 21 passes over 37,690 frames |
| Seed | 1000 (same for all three) | Removes init/shuffle noise from the comparison |
| Dataset | `..._165544`, all 50 episodes | The `_163836` (5 ep) and `_164823` (3 ep) folders are test runs — do not use |
| Held-out split | `eval_split=0.1` → last 5 episodes | Deterministic, identical for all three runs |
| Normalization | Dataset stats, `use_imagenet_stats=true` | Default; identical inputs across runs |
| Eval checkpoint | Final step, same for all three | Cherry-picking a checkpoint per condition is a silent tune |

**Caveat on the held-out split.** `datasets/factory.py` holds out *the last* `ceil(n × eval_split)`
episodes per task — for the 5-rounds × 10-positions schedule that is the tail of round 5, i.e. the
end-of-session episodes, covering only 5 of the 10 positions. It is identical across the three runs so
the *comparison* is fair, but do not read the validation loss as an unbiased estimate of anything.
The real number is the robot success rate in §4.

---

# 3 — Training, step by step

### Step 0 — preflight

```bash
conda activate lerobot && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv && du -sh ~/.cache/huggingface/lerobot/HALDijkstraaa/so101_toolkit_cylinder_20260917_165544 && df -h /home | tail -1
```

Checks: the env is active, the GPU is idle (close anything holding VRAM), the dataset is cached locally,
and there is disk room. Budget **~2 GB of checkpoints per run** at `save_freq=20000` (each ACT checkpoint
is ~620 MB: 51.6M params plus Adam state).

### Step 1 — smoke test

Five minutes now beats discovering a typo after an overnight run. This runs 500 steps of each condition:

```bash
STEPS=500 SAVE_FREQ=500 OUT=outputs/smoke ./scripts/train_camera_ablation.sh
```

Check in the log that each run prints the right input features (`observation.images.top` only,
`observation.images.wrist` only, then both) and that loss is falling. Then delete `outputs/smoke`.

### Step 2 — the three runs

```bash
./scripts/train_camera_ablation.sh
```

Runs `top`, `wrist`, `both` sequentially at 100k steps each, ~11 h total, logging to
`outputs/train/act_<cond>_s1000.log`. Start it before bed. To run one condition only, or a second seed:

```bash
SEED=1001 ./scripts/train_camera_ablation.sh wrist
```

The script is a thin wrapper; the single command it issues for the top-only condition is:

```bash
lerobot-train --dataset.repo_id=HALDijkstraaa/so101_toolkit_cylinder_20260917_165544 --dataset.eval_split=0.1 --policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.input_features="{'observation.state': {'type': 'STATE', 'shape': [6]}, 'observation.images.top': {'type': 'VISUAL', 'shape': [3, 480, 640]}}" --batch_size=8 --steps=100000 --num_workers=8 --seed=1000 --save_freq=20000 --output_dir=outputs/train/act_top_s1000 --job_name=act_top_s1000 --wandb.enable=false
```

The `wrist` condition swaps the image key; the `both` condition **omits `--policy.input_features`
entirely**, since both cameras is the default.

> **How the camera selection works.** `policies/factory.py` fills `cfg.input_features` from the dataset
> only `if not cfg.input_features`, so a CLI override survives. There is no flag for "use a subset of
> cameras" — this is the mechanism. Note that `LeRobotDataset` has no column selection, so the
> single-camera runs still *decode* both videos: all three runs cost about the same wall clock.

### Step 3 — monitor and select

```bash
tail -f outputs/train/act_top_s1000.log
```

Expect ~7.5 it/s and a smoothly falling L1 loss. Then compare the held-out loss across conditions:

```bash
grep -h "eval" outputs/train/act_*_s1000.log | tail -20
```

Use this only as a **sanity check and for spotting divergence** — action-prediction L1 correlates weakly
with task success, so it cannot settle the ablation. Evaluate the **final checkpoint** of each run:

```
outputs/train/act_top_s1000/checkpoints/last/pretrained_model
```

---

# 4 — Real-robot evaluation

This is where the answer actually comes from, and where the experiment is easiest to ruin.

### Protocol

| Decision | Value | Why |
|---|---|---|
| **Trials** | 3 per position × 10 positions × 3 policies = **90** | 30 per policy. At 20 trials the binomial standard error on a 50% success rate is ±11 points — enough to miss a real 15-point effect |
| **Ordering** | **Interleaved**, balanced blocks | Blocked evaluation (30 top, then 30 wrist…) hands all session drift — lighting, fixture nudges, your own reset habits — to whichever policy ran last |
| **Positions** | The same taped P1–P10 as recording | Makes per-position breakdown possible |
| **Success** | Fully seated + released within 30 s ([task card](toolkit_task_card.md)) | Already frozen; do not redefine mid-session |
| **Scoring** | From video, **blind to condition**, after the session | You will unconsciously favour the condition you expect to win |
| **Also log** | The **stage** each failure happened at | This is the payoff — see §6 |

### Step 4a — generate the schedule

```bash
python scripts/make_eval_schedule.py --trials-per-position 3 --seed 0 > Details/phase06_eval_log.csv
```

Columns: `trial, block, condition, position, success, fail_stage, time_s, notes`. Fill `success` (0/1) and
`fail_stage` (`reach | grasp | transport | insert | retreat | none`) as you go, or afterwards from video.
Keep the file next to this doc — like `phase05_log.csv`, it cannot be reconstructed later.

At ~90 s per trial including reset, 90 trials is roughly **2.5 hours**. If that is too long, drop to
2 blocks (60 trials, 20 per policy) and accept that only differences above ~20 points will be detectable.

### Step 4b — load camera presets

Same as recording, and it matters more here than anywhere: the policies were trained on images produced by
preset 1 (C920 / top) and preset 2 (C922 / wrist). An unloaded preset is a distribution shift that will
look like a policy failure. See [01 Step 6](../01_setup_robot.md).

### Step 4c — run one trial

`--strategy.type=episodic` mirrors `lerobot-record`: it runs episodes with reset phases and saves video,
so you can score blind afterwards. Set `CKPT` to the condition the schedule calls for:

```bash
CKPT=outputs/train/act_top_s1000/checkpoints/last/pretrained_model TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 lerobot-rollout --strategy.type=episodic --policy.path=$CKPT --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/phase06_eval_act_top --dataset.single_task="Pick up the white cylinder and place it in the hole of the black fixture" --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=15 --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

- **Always connect both cameras**, even for the single-camera policies. The policy consumes only the keys
  in its own `input_features`; the extra stream is ignored but keeps the physical scene identical (the
  wrist camera is mounted either way) and gives you both views on video for scoring. **Verify this on the
  first rollout** — if a single-camera policy errors on the extra key, drop that camera from
  `--robot.cameras` for its trials.
- **`reset_time_s=15`**, longer than the 3 s used for recording: you have to reposition the cylinder on
  the next scheduled tape mark between trials.
- **→** ends a trial early (success or obvious failure), **←** discards and re-runs, **Esc** stops.
- Run in batches matching the schedule rather than 10-in-a-row per policy; the schedule tells you which
  condition and position comes next.

---

# 5 — Compute: measured on this machine

Benchmarked 2026-09-20 on the RTX 5080 Laptop (16 GB), driver 595.84, torch 2.11+cu130 — real dataloader,
real forward/backward, bf16 autocast, this dataset:

| Config | ms/step | it/s | Peak VRAM | 100k steps |
|---|---|---|---|---|
| top+wrist, batch 8, workers 8 | 134 | 7.5 | 3.5 / 16 GiB | **3.7 h** |

- **Three runs sequentially ≈ 11 h.** One overnight. **A cloud A100 is not needed.**
- **VRAM is not the constraint** (3.5 of 16 GiB), but raising the batch is still the wrong move: it
  deviates from the ACT recipe, and at 60 samples/s × 2 AV1 streams the pipeline may already be
  video-decode-bound rather than GPU-bound, in which case a bigger batch buys little wall clock.
  *(Untested — the larger-batch benchmark was not run.)*
- **Faster first pass:** 60k steps ≈ 2.2 h per run.
- **The one real argument for cloud is parallelism.** 3 conditions × 3 seeds = 9 runs is ~33 h here versus
  ~4 h on rented GPUs. With 50 episodes and 30 trials, seed-to-seed variance is comparable to the effect
  size, so multiple seeds matter more than raw speed does.

This supersedes the "ACT · ~5 GB @ bs 8 · train on cloud 4090" row in [08 §D.1](../08_data_quality_research.md)
for the small real-track datasets: measured, ACT trains comfortably on the 5080.

---

# 6 — What to report, and what Phase 0.5 predicts

The aggregate success rate per condition is the headline, but the **per-stage failure breakdown is where
the divergence metric gets tested**. Phase 0.5 found that, once divergence is normalized within each
phase, each camera wins exactly where its geometry says it should:

![Top vs wrist by phase](figs/04_top_vs_wrist_by_phase.png)

*Phase 0.5: top is better by 0.025 in reach and 0.093 in transport; wrist by 0.029 in grasp and 0.039 in
insert.*

**Pre-registered predictions** (write these down before scoring, so the analysis isn't retro-fitted):

| # | Prediction | From |
|---|---|---|
| **P1** | `top+wrist` ≥ both single-camera conditions | H-c confirmed: top+wrist has the lowest divergence overall |
| **P2** | The `top+wrist` margin over `wrist` alone is **small** | The two views are largely redundant: 0.499 vs 0.522 at k=10 |
| **P3** | `wrist`-only fails mostly at **reach** — it cannot see where the cylinder is | H-b: top wins reach/transport |
| **P4** | `top`-only fails mostly at **grasp/insert** — mm-scale offsets are a few pixels from above | H-a: wrist wins grasp/insert |
| **P5** | All three have a high absolute failure rate in grasp and insert | In the fine phases *no* observation space beat the random baseline |

**Report:**

1. Success rate per condition with a binomial CI (30 trials → ±~9 points at 50%).
2. Failure stage histogram per condition — the direct test of P3/P4.
3. Per-position success, 10 bars per condition — checks whether one awkward position dominates.
4. Held-out action L1 per condition, alongside success, as one data point for Future Study #1 in the
   [Phase 0.5 results](phase05_camera_test_results.md): does divergence predict performance?

That last item is the reason this phase is worth the robot time. Phase 0.5 produced a cheap offline metric;
these 90 trials are the first evidence that it means anything.

---

# 7 — Gotchas

| Risk | Mitigation |
|---|---|
| Camera presets not loaded before eval | Check the rerun view first ([01 Step 6](../01_setup_robot.md)). A dark top image is a distribution shift, and it will look like policy failure |
| Fixture moved between training data and eval | Re-check the taped outline against the top view before starting |
| Scoring drifts across a 2.5 h session | Score from video afterwards, blind to condition, in one sitting |
| Single-camera policy errors on the extra camera key | Verified on the first rollout (§4c); fall back to dropping the camera from `--robot.cameras` |
| Overnight run dies silently | `tee` logs are on by default in the script; check `outputs/train/*.log` before starting eval |
| Reading the held-out loss as the answer | It is a sanity check only — the split is the end-of-session round, and L1 correlates weakly with success |
