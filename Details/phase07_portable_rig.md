# Phase 0.7 — portable rig: bird's-eye top camera, and a new dataset

Run book for the rebuilt rig, the data collection that follows it (§1–§9), and the **round-1 policy
evaluation** (§10–§11). Everything — both arms, both cameras, the tray and the fixture — is now mounted
on a single acrylic board, and the top camera has been raised from a third-person view to a **bird's-eye
overview of the board**.

Plan: [08 §A.4](../08_data_quality_research.md) · Task: [toolkit_task_card.md](toolkit_task_card.md) ·
Calibration: [calibration/README.md](calibration/README.md) · Predecessors:
[phase05_camera_test_results.md](phase05_camera_test_results.md),
[phase06_camera_ablation_training.md](phase06_camera_ablation_training.md). Written 2026-10-03,
eval added 2026-10-12.

**Where this stands.** The 50 episodes were recorded and pushed 2026-10-03 as
`HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912` (§8) and verified 2026-10-04 — clean
recording, but the idle lead-in of §8 did not shrink, so all training uses the idle-trimmed copy
([verification](phase07_data_verification.md)). **All three policies are trained**
([training](phase07_training.md)). **Next: run §10** — 54 rollouts over six arms, which picks one
checkpoint per policy; §11 then measures success rates properly.

Items marked *Default:* are suggestions — fix them before the first episode, then do not change them
mid-collection.

---

# Summary

- **Three things changed at once:** the rig is on one portable acrylic board, the top camera is now
  bird's-eye instead of third-person, and **both arms were recalibrated on 2026-10-03**. All three break
  compatibility with Phase 0.5 / 0.6, so this is a fresh dataset, not an extension (§1).
- **The point of the taller camera is to delete background.** A third-person view frames the room behind
  the workspace: pixels that vary between sessions and carry no task information. A top-down view at
  height sees the board and little else — so the observation is almost entirely task-relevant, and it
  looks the same wherever the board is set down. That is the portability claim and the data-quality claim
  in one (§2).
- **It also fixes a definition.** "Top-left hole" used to mean top-left *as seen in the third-person top
  image*, which made the target a property of the camera pose. From directly above it is a property of
  the board (§5).
- **New task string, used verbatim in every episode:**
  `Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture`
- **Do not mix this data with Phase 0.5 / 0.6.** Different action frame *and* different observation
  space; concatenating them would train a policy on two incompatible mappings (§4).
- **Carry the three collection lessons forward:** shuffled position rounds, not position blocks; start
  moving the instant the episode starts; one grasp style (§6, §7).

| Step | What | Time | Attended? |
|---|---|---|---|
| 0 | Rig spec: measure and write down §3 | 20 min | yes |
| 1 | Calibration check + replay sanity | 15 min | yes |
| 2 | Camera preflight, both views | 15 min | yes |
| 3 | 5 throwaway episodes, inspect in rerun | 20 min | yes |
| 4 | Record 50 D0 episodes | ~1.5 h | yes |
| 5 | Verify the dataset, push, log | 20 min | yes |

Collection (steps 0–5) is done. The evaluation that follows training adds three more:

| Step | What | Time | Attended? |
|---|---|---|---|
| 6 | Eval preflight: revert the ACT patch, replay an episode, check the top view (§10.1, §10.6) | 20 min | yes |
| 7 | **Round 1: 12 blocks, 54 rollouts** (§10.4) | ~2–2.5 h | yes |
| 8 | Blind video scoring into [phase07_eval_log.csv](phase07_eval_log.csv) (§10.6) | ~1 h | yes |

---

# 1 — What changed

| | Phase 0.5 / 0.6 | Phase 0.7 |
|---|---|---|
| Mounting | arms clamped to the table, cameras on separate mounts, positions taped to the table surface | **everything on one acrylic board** — arms, both cameras, tray, fixture |
| Top camera | third-person, front-side, wide; room visible behind the workspace | **bird's-eye, directly above, much higher**; frames the board and little else |
| Cylinder start | standing on one of 10 taped table positions | **in the green tray** |
| Target | top-left hole, taped, defined in the third-person image | **top-left blue hole** of the black fixture, defined on the board |
| Calibration | 2026-09-16 master (+ `shoulder_pan` fix 09-27) | **recalibrated 2026-10-03**, [`calibration/current/`](calibration/current/) |
| Portability | fixed to one table in one room | **set the board down anywhere and record** |

The three changes are deliberately bundled: the board is what makes a repeatable high mount possible, the
high mount is what makes the scene location-independent, and remounting the arms forced the
recalibration. Bundling them means this phase cannot attribute an outcome to any one of them — that is
accepted. The purpose here is a clean portable baseline, not an ablation.

# 2 — Why bird's-eye, from the Phase 0.5 / 0.6 results

Two findings from the predecessors point the same way:

1. **The third-person top view was the aliased one.** It lost to the wrist camera, and even to joint
   state, in every phase but grasp/insert
   ([phase05](phase05_camera_test_results.md)). A view carrying mostly background does not tell apart
   states that need different actions.
2. **Adding it to the wrist camera made reach *worse*.** `top+wrist` reached 6/10 against `top`'s 10/10,
   under-shooting the workspace extremes by 17% ([phase06](phase06_camera_ablation_training.md)) — the
   redundant stream degraded the phase the other camera was carrying.

A top-down view attacks both: it raises the share of task-relevant pixels, and it makes cylinder and hole
position almost affine in image coordinates, which is the geometry the reach phase actually needs. It
also removes the arm's self-occlusion of the table that the front-side view suffered from — though it
introduces the arm occluding the *tray and fixture* from above instead, which §6's preflight has to check.

**What this phase does and does not claim.** It produces the dataset. Whether the bird's-eye view
actually lowers divergence and fixes the fusion degradation is a measurement on this dataset against the
Phase 0.5 numbers — worth doing, but it is a confounded comparison (new frame, new camera, new tray) and
should be read as a sanity check, not as C1c. A clean test would re-mount the third-person camera on the
same board and record both views simultaneously; noted in §9.

# 3 — Rig spec — measure once, write it down

This is the part that makes the rig reproducible after it is knocked over. Fill in before recording.

| Item | Value |
|---|---|
| Acrylic board | ___ × ___ mm, ___ mm thick |
| Follower base position on board | ___ (mark the outline) |
| Top camera height above board | ___ mm |
| Top camera model / `/dev/v4l/by-id` path | ___ |
| Top camera mount | ___ (*Default:* rigid arm bolted to the board, so the view travels with it) |
| Field of view at board level | covers ___ × ___ mm — must include the whole tray, the whole fixture, and the arm's working area |
| Wrist camera model / path | ___ (previously C922) |
| Green tray | ___ × ___ mm, outline marked on the board |
| Black fixture | outline marked on the board; hole pitch ___ mm; target hole marked |
| Cylinder | diameter ___ mm, height ___ mm, hole clearance ___ mm |
| Lighting | ___ (*Default:* the LED panel + diffuser, fixed to the board or a repeatable distance from it) |
| USB | both cameras + both arms on ___ (*Default:* one powered hub, MJPG on both cameras for bandwidth) |
| Cameractrls presets | top ___ , wrist ___ (settings reset on replug — reload every session) |

**Photograph the assembled rig** from the side and from the top camera's own view, and save both into
[`figs/`](figs/). The top-camera still is the reference image for "is the board set up right" at the next
location.

**Repeatability is the whole point.** If the board can be set down somewhere else and the top image
matches the reference still, the rig is portable. If it cannot, the mount is not rigid enough, and no
amount of data will fix that.

# 4 — Calibration

Both arms were recalibrated on **2026-10-03**; the deployed pair is
[`calibration/current/`](calibration/current/) and the previous frames are archived beside it. Full
numbers and the restore procedure: [calibration/README.md](calibration/README.md).

- **The follower's narrowed `shoulder_pan` range is intentional.** Width went 2688 → 2449 counts (−21.0°).
  It covers the working area of the board, which is smaller than the old taped table. Reviewed and kept.
- **Back up before touching anything.** The new pair is already archived; if you re-run
  `lerobot-calibrate` again, copy the live files into a new dated folder under `calibration/archive/`
  *first*.
- **Phase 0.5 / 0.6 assets need their own frame restored** before any replay or rollout. They do not work
  in this one.
- **Sanity check before recording** (§6) is a replay of a Phase 0.7 episode, which means the first few
  throwaway episodes are also the calibration acceptance test.

# 5 — Task card delta

The full card is [toolkit_task_card.md](toolkit_task_card.md). Phase 0.7 changes these entries:

| Entry | Phase 0.7 value |
|---|---|
| Task string | `Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture` |
| Start state | cylinder **in the green tray**, standing upright; arm at home, gripper open |
| Cylinder position | **VARIES** over ___ marked spots **within the tray** (*Default:* 6, two rows of three, each marked on the tray floor) |
| Target | **top-left blue hole** as seen **from the bird's-eye camera** — which is now the board's own top-left, not a camera-dependent direction |
| Success | unchanged: within 30 s, fully seated in that hole, gripper released |
| Stages | unchanged: reached 0.2 · grasped+lifted 0.4 · over the hole 0.6 · inserted 0.8 · seated+released 1.0 |
| Fixed | tray and fixture outlines on the board, lighting, both camera poses, Cameractrls presets, one cylinder, one grasp style |

**Grasping out of a tray is not the same problem as grasping off a table.** The tray walls constrain the
approach and can block a straight-down grasp near an edge. *Default:* keep the straight-down-from-above
style and place the marked spots far enough from the walls that it always works; if it does not, change
the spots, not the style, and write the decision here.

# 6 — Preflight, before the first real episode

1. **Cameras.** Load the Cameractrls presets. In the rerun viewer: the gripper tip is sharp in `wrist`;
   the bird's-eye `top` is not dark, not motion-blurred, and shows the **whole tray and the whole fixture
   with the arm at home**.
2. **Occlusion sweep.** Teleoperate slowly through a full episode and watch `top`. The arm will cross the
   view from above — confirm the **target hole is visible at the moment of insertion** and the
   **cylinder is visible in the tray at the start of every position**. If the arm hides the hole during
   the approach, move the camera or rotate the board, and re-do §3's reference still.
3. **Wrist visibility.** Confirm the cylinder enters the wrist view during the approach, as in Phase 0.5.
4. **Reach test.** Hand-drive the follower to every marked tray spot and to the target hole. All inside
   the narrowed `shoulder_pan` range, with margin.
5. **Five throwaway episodes**, recorded to a scratch `repo_id`. Watch them back, then **replay one**
   (`lerobot-replay`, [01 Step 4](../01_setup_robot.md)) — a clean replay clears the new calibration and
   the mounting. Delete the scratch dataset.

Only then start the real run.

# 7 — Recording

lerobot 0.6.1. Re-check the device paths with `ls /dev/v4l/by-id/` — they change if a camera was swapped.
Keys: **→** ends the episode early, **←** re-records it, **Esc** stops.

```bash
TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0
WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0

lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --dataset.repo_id=HALDijkstraaa/so101_tray_cylinder_portable \
  --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=3 \
  --dataset.fps=30 \
  --dataset.push_to_hub=true \
  --dataset.private=true \
  --display_data=true
```

- **Camera names stay `top` / `wrist`** so training configs and the analysis notebooks carry over
  unchanged — even though `top` now means something different. Note that in the dataset card.
- **New `repo_id`**, so no tooling can silently resume into the Phase 0.5 dataset. 0.6.1 appends the start
  time, giving `so101_tray_cylinder_portable_<timestamp>`; record that full name in §8. To add episodes
  later, pass the full stamped name with `--resume=true`.
- **Timing unchanged** from Phase 0.5 — 30 s episodes, 3 s reset — so episode length stays comparable.

**The throwaway run** of §6 step 5 — 5 episodes, separate `repo_id`, **no push**. Watch these back and
replay one before starting the real 50; delete the dataset afterwards so no tooling can train on it.

```bash
TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0
WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0

lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --dataset.repo_id=HALDijkstraaa/so101_tray_cylinder_portable_test \
  --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=3 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --display_data=true
```

# 8 — Schedule and log

**N = 50 clean (D0) episodes**, *Default:* matching Phase 0.5 so the two sets are the same size when
their divergence numbers are compared. Extend toward the N = 100 per-condition figure in
[08 §A.4](../08_data_quality_research.md) once the rig is proven.

## The datasets, as recorded (2026-10-03)

| | Repo | Episodes | Frames | Use |
|---|---|---|---|---|
| **Raw** | [`so101_tray_cylinder_portable_20261003_211912`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912) | 50 | 35,141 | archive · analysis |
| **Trimmed** | [`..._20261003_211912_trimmed`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed) | 50 | **29,462** (83.8%) | **all training** |

Both private, `v3.0`, 30 fps, `top` + `wrist` at 480×640 AV1. The trimmed copy was built 2026-10-05 by
[colab_phase07_prep_trim.ipynb](../colab_phase07_prep_trim.ipynb) with the `dev > 2.0°` / 5-frame-pad
rule; episodes are 485–743 frames, every one longer than an ACT 100-step chunk. Verified after upload:
all parquet readable, codebase-version tag present, same encoder settings as the source.

**Train on the trimmed copy, not the raw one** — see §8's lead-in note below and
[phase07_data_verification.md §2](phase07_data_verification.md).

## What was asked for, and what actually happened

**Shuffle the positions.** Phase 0.5 recorded position blocks (P1 = ep 0–4 … P10 = ep 45–49), which
confounded position with session time and would have put a whole position in lerobot's default holdout.
For Phase 0.7: visit every marked tray spot once per round, in a **written-down shuffled order per
round**, repeating until 50 episodes are done.

> **Not done.** As recorded, episodes 0–44 are **9 tray-spot blocks of 5 consecutive episodes** (a 3×3
> grid) and **45–49 are random placements** in the tray. So position is confounded with session time
> again, and lerobot's default "last 5" holdout would hold out exactly the 5 off-grid episodes. Training
> therefore passes an explicit reordered `--dataset.episodes` with holdout `[9, 19, 29, 39, 49]`. The
> block structure is recoverable from the data: the grasp pose separates the 9 spots at 27× (verification
> report §5).

**Start moving the instant the episode starts.** A static lead-in becomes an idle attractor the policy
cannot escape; Phase 0.5 measured a median 74 idle frames, and the F3 follow-up found that training on
idle-trimmed data flipped the camera ranking. Trim anyway before training
([`scripts/make_trimmed_dataset.py`](../scripts/make_trimmed_dataset.py)).

> **Not done either**, and this one is the reason the trimmed copy is mandatory rather than optional:
> the measured lead-in is a **median 71 frames (2.38 s)**, essentially unchanged from Phase 0.5's 74,
> with 47/50 episodes idle longer than one 50-step chunk. Two rounds of the same written instruction
> have now produced the same lead-in, so the next round needs a mechanism, not a reminder.

Keep `phase07_log.csv` next to the dataset, one row per episode — it cannot be reconstructed afterwards:

`episode_index, round, tray_spot, success (0/1), stage, dial (D0), location, notes`

`location` is new: the board is portable, so record **where** it was recorded. The first 50 are all at one
location (*Default:* the usual desk) — a second location is a separate dataset and the first real test of
the portability claim.

# 9 — Done when

- The rig is measured and photographed (§3), and the top-camera reference still is in `figs/`.
- A Phase 0.7 episode replays cleanly on the new calibration.
- 50 D0 episodes recorded, pushed, and logged, with no position block and no long lead-ins.
- The dataset card states: new calibration frame, bird's-eye `top`, not compatible with
  `so101_toolkit_cylinder_20260917_165544`.

**Then — revised 2026-10-05.** The original plan was to re-run the divergence analysis and the camera
ablation on this dataset first. **Both are deferred**: they are camera-placement questions, and the
dataset is now wanted for the policy-transfer question (Q2) instead. What runs now:

| Stage | What | Where |
|---|---|---|
| 1 ✅ | **Verify the dataset.** 48 PASS / 7 WARN / 0 BLOCK — clean recording, but the idle lead-in survived, so training needs a trimmed copy | [phase07_data_verification.md](phase07_data_verification.md) · [`scripts/verify_phase07_dataset.py`](../scripts/verify_phase07_dataset.py) |
| 2 ✅ | **Trimmed dataset built and pushed** 2026-10-05 — [`..._trimmed`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed), 50 ep / 29,462 frames (§8) | [colab_phase07_prep_trim.ipynb](../colab_phase07_prep_trim.ipynb) |
| 3 ✅ | **All three trained**, everything but the policy held constant. **ACT** 60k steps / 3.85 h on an L4, eval_loss 0.1792 min; **SmolVLA** 20k / 14.2 h on an L4 — and it **overfits**, eval_loss +61% from its 5k minimum; **π0.5** 10k / 5.7 h on a Colab A100, eval curve not read | [phase07_training.md](phase07_training.md) · [ACT](../colab_phase07_act.ipynb) · [SmolVLA](../colab_phase07_smolvla.ipynb) · [π0.5](../colab_phase07_pi05_Colab.ipynb) |
| 4 ◐ | **Round 1 rollouts: six arms, 54 trials** — two checkpoints per policy, mirrored interleave, 9 grid spots. Written and preflighted (all six load and run on the 5080); **not yet run** | **§10** · [phase07_eval_log.csv](phase07_eval_log.csv) · [`make_phase07_eval_schedule.py`](../scripts/make_phase07_eval_schedule.py) |
| 5 | Round 2: success rates at ±9 points, plus the off-grid spots and seeds | **§11** |

Train/eval split for all three: holdout **`[9, 19, 29, 39, 49]`**, placed in the tail of
`--dataset.episodes` so lerobot's "last N" split takes exactly those. The default split would otherwise
hold out episodes 45–49, which are the only **off-grid** tray placements — see §8 and the verification
report §5.

**Still open, in no particular order:**
1. Re-run the Phase 0.5 divergence analysis on this dataset
   ([phase05_second_camera.ipynb](../phase05_second_camera.ipynb)) and put the numbers beside the old
   ones. Confounded, so read it as a sanity check: does the top view stop being the aliased one?
   (Verification §4b already found one piece of the answer for free: the bird's-eye view is blind to the
   wrist tilt every episode opens with.)
2. Retrain the three-way camera ablation on it. The sharp question is whether `top+wrist` still
   under-reaches — if the degradation was background-driven it should shrink here.
3. Record a second location and compare, which is the only direct test of portability.
4. If the comparison in (1) looks worth doing properly, mount the old third-person camera on the board as
   a third stream and record both views at once. That turns a confounded comparison into a controlled one.

---

# 10 — Round 1 eval: six arms, 54 rollouts

Stage 4 of §9. Written 2026-10-12, after all three training runs finished
([phase07_training.md](phase07_training.md)) and after every checkpoint below was loaded and timed on
this machine (§10.1). **Not yet run.**

**Round 1 is a checkpoint-selection pass, not a success-rate measurement.** Each of the three policies
finished with an unsettled checkpoint question, and all three questions are cheap to answer on the robot
and impossible to answer from the loss curves:

| Policy | The open question | Why the loss cannot answer it |
|---|---|---|
| ACT | Is 60k wasted compute over 40k? | Eval loss is flat from 20k; the 20k–60k spread is checkpoint wobble ([training §2](phase07_training.md)) |
| SmolVLA | Does the +61% eval-loss rise from 5k to 20k show up as worse *behavior*? | [phase06 §2](phase06_camera_ablation_training.md) found held-out loss separated conditions by less than one curve's wobble while the robot showed a decisive dissociation |
| π0.5 | Which checkpoint at all? | Its four `eval_loss` values exist only in wandb run `sxpkx9st`, which this machine has no credentials for ([training §4](phase07_training.md)) |

So round 1 runs **two checkpoints per policy** and lets the robot rank them. Success counts come out of
it, and they are the first real success numbers for this rig — but at 9 trials per arm they carry a ±16
point standard error, so **the rate is round 2's job** (§11). What round 1 settles is which six
checkpoints become three.

> **π0.5's checkpoint is decided here rather than from wandb, deliberately.** Reading the curve would
> cost nothing, but it would only rank the checkpoints by a quantity already known to correlate weakly
> with task success — and for a flow-matching policy a single `eval_loss` sample cannot be told apart
> from its own sampling noise ([`eval_checkpoints_heldout.py`](../scripts/eval_checkpoints_heldout.py)
> exists for exactly that reason). Reading it afterwards is still worth doing, as a check on whether
> loss predicted the rollout ordering; that is in §10.7.

## The six arms

Local paths are what the commands below use. All six are already downloaded (§10.1).

| Arm | Policy | Checkpoint | Samples | Epochs | Held-out loss | Hub repo |
|---|---|---|---|---|---|---|
| `act40k` | ACT | `040000` | 320,000 | **12.0** | in the flat 0.179–0.192 band | [`act_tray_cylinder_s1000_trim`](https://huggingface.co/HALDijkstraaa/act_tray_cylinder_s1000_trim) |
| `act60k` | ACT | `060000` | 480,000 | 18.0 | 0.1792 @50k (min) → 0.1848 @55k | same |
| `sv5k` | SmolVLA | `005000` | 320,000 | **12.0** | **0.1258 — the minimum** | [`smolvla_tray_cylinder_s1000_trim`](https://huggingface.co/HALDijkstraaa/smolvla_tray_cylinder_s1000_trim) |
| `sv20k` | SmolVLA | `020000` | 1,280,000 | 47.9 | rises monotonically; 0.2029 @15k (+61%) | same |
| `pi5k` | π0.5 | `005000` | 80,000 | 3.0 | **not read** | [`pi05_tray_cylinder_s1000_trim_A100`](https://huggingface.co/HALDijkstraaa/pi05_tray_cylinder_s1000_trim_A100) |
| `pi10k` | π0.5 | `010000` | 160,000 | 6.0 | **not read** | same |

**`act40k` and `sv5k` are both exactly 320,000 samples.** 40,000 × 8 and 5,000 × 64 land on the same
number, so round 1 contains an **exactly sample-matched ACT-vs-SmolVLA pair** at 12.0 epochs on
identical data, holdout and seed. That is a far cleaner Q2 comparison than matched *recipes*, which is
all [training §1](phase07_training.md) could promise, and it costs nothing extra — it is a property of
the checkpoints that already exist. Note what it does *not* extend to: π0.5's largest checkpoint is
160,000 samples, half of that, so π0.5 cannot be sample-matched to the other two at all. It stays the
sample-starved arm, and any π0.5 result has to be read with that attached.

**π0.5's pair was chosen by bracketing rather than by loss.** 005000 and 010000 are 3.0 and 6.0 epochs —
a clean 2× step, the same shape as the other two pairs. 002500 (1.5 epochs) was left out as implausibly
early, and 007500 as too close to 010000 to separate. If `pi5k` beats `pi10k`, π0.5 overfits on this
dataset even faster than SmolVLA did and 002500 becomes worth a look in round 2.

## 10.1 — Step 0: one-time setup, already done and measured

Everything in this subsection has been run on this machine; it is recorded so the session does not
rediscover it. Re-run only the parts a reinstall would break.

**The six checkpoints are downloaded** to `outputs/hub/` — 20 GB, `pretrained_model` directories only,
no optimizer state (which is 2.2 GB per checkpoint for π0.5 and is not needed for inference):

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && for spec in "act_tray_cylinder_s1000_trim:040000" "act_tray_cylinder_s1000_trim:060000" "smolvla_tray_cylinder_s1000_trim:005000" "smolvla_tray_cylinder_s1000_trim:020000" "pi05_tray_cylinder_s1000_trim_A100:005000" "pi05_tray_cylinder_s1000_trim_A100:010000"; do R=${spec%%:*}; C=${spec##*:}; hf download HALDijkstraaa/$R --include "checkpoints/$C/pretrained_model/*" --local-dir outputs/hub/$R; done
```

**All six load and run inside the control-loop budget on the 5080.** This was the open hardware risk:
the policies were trained on A100/L4 class cards and π0.5 is 4.14 B parameters, so whether it runs at all
on a 16 GB laptop was unknown. Measured with
[`scripts/bench_rollout_latency.py`](../scripts/bench_rollout_latency.py), which feeds synthetic
observations of the right shape through the real preprocessor and the real policy — no robot needed.
Full output: [`results/phase07_eval/bench_rollout_latency_5080.txt`](../results/phase07_eval/bench_rollout_latency_5080.txt).

| Arm | Params | Peak VRAM | Chunk latency (median / worst) | Headroom at `n_action_steps=25` |
|---|---|---|---|---|
| `act40k` / `act60k` | 51.6 M | **0.27 GiB** | 8–9 ms / 9 ms | **89–94×** |
| `sv5k` / `sv20k` | 450.0 M | **0.91 GiB** | 100–105 ms / 101–106 ms | **7.9–8.3×** |
| `pi5k` / `pi10k` | 4,143.4 M | **8.87 GiB** of 15.5 | 282–288 ms / 316 ms | **2.6×** |

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && python scripts/bench_rollout_latency.py --policy outputs/hub/pi05_tray_cylinder_s1000_trim_A100/checkpoints/010000/pretrained_model --n-action-steps 25 --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb
```

> **A low power reading during a rollout is expected — do not chase it.** `nvidia-smi` sampled during a
> π0.5 run shows something like 38 W of an 80 W cap at 2% utilisation, which looks like the GPU is
> being held back. It is not: that is an **idle sample between chunks**. Measured over a 140 s sustained
> π0.5 loop at 0.25 s resolution, the GPU is at **96–100% utilisation, 78.7 W median (82.1 W peak),
> P0, 51–55 °C** whenever it is actually computing — and it computes for only **35% of the wall clock**,
> because a 277 ms chunk inside an 833 ms budget is a 33% duty cycle. The average therefore *has* to
> look low. See §10.1a before changing any power setting.

Three things follow, and they are what make §10.4's commands as simple as they are:

1. **π0.5 fits, with 6.6 GiB spare.** 8.87 GiB peak of 15.5 GiB available. Its 9.35 GB checkpoint is
   533.6 M F32 parameters plus 3.61 B BF16 ones; loading it under the config's `dtype: bfloat16` lands
   at 8.71 GiB of weights before activations.
2. **Plain sync inference is enough for all three. `--inference.type=rtc` is not needed.** RTC exists
   for policies that cannot compute a chunk inside its execution window, and the slowest arm here uses
   316 ms of an 833 ms budget. Leaving RTC off keeps one fewer thing different between the arms.
3. **`n_action_steps=25` is affordable for every arm**, which is what §10.2 spends it on.

**The Phase 0.6 ACT patch is reverted — done 2026-10-12.** It had been left applied to the local lerobot
checkout since the F2 session, and the first bench run emitted `WARNING: Missing key(s) when loading
model: {'model.camera_id_embed.weight'}` on both ACT arms.

```bash
git -C /home/bj/code/lerobot checkout src/lerobot/policies/act/modeling_act.py
```

**It did not affect the rollout, and that is measured rather than argued.** The patch adds a
`camera_id_embed` that is `nn.init.zeros_`-initialised and added to the camera features; a Phase 0.7
checkpoint carries no such key, so it stays zero and `cam_features + 0` is stock ACT. Checked by
predicting an action from the same fixed input under both versions of the file:

| | action (6 joints, degrees) |
|---|---|
| patched | `[4.041168, -43.549736, 43.588104, 26.216066, 5.247201, 21.356627]` |
| stock | `[4.041168, -43.549736, 43.588104, 26.216066, 5.247201, 21.356627]` |

**Bit-identical, max abs diff 0.0.** So the §10.1 bench numbers taken under the patch stand unchanged,
and no Phase 0.6 result that used a non-F2 checkpoint is in question either.

Reverted anyway, for two reasons. The camera-identity embedding was tested in F2 and
**found no benefit**, so nothing depends on keeping it. And the warning it produced is textually
indistinguishable from the one a *real* state-dict mismatch produces — which lerobot logs rather than
raises, so it is the kind of thing that silently voids a run. A clean ACT load now prints nothing, which
means any warning that does appear from here is worth stopping for. If the outstanding Phase 0.6 F3
blocks 2–3 are ever picked up, `act_both_s1000_camemb` is the one asset that needs
`scripts/patch_act_camera_embed.py` re-applied.

**SmolVLA fetches its VLM backbone at load time.** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` is
pulled from the Hub on every `from_pretrained`, not bundled in the checkpoint. It is in
`~/.cache/huggingface` now, so the session does not need network — but a cleared cache means the
`sv5k`/`sv20k` blocks need it back.

## 10.1a — GPU power: leave it alone, and above all do not change it mid-session

Measured 2026-10-12 under a 140 s sustained π0.5 inference loop, sampled every 0.25 s.

| | Idle, between chunks | **Computing** |
|---|---|---|
| Utilisation | 0–2% | **96–100%** |
| Power | 11–17 W | **78.7 W median, 82.1 W peak** |
| SM clock | 180 MHz | **1,740 MHz median, 1,882 MHz peak** |
| P-state / temp | P8, 40 °C | **P0, 51–55 °C** |

**The card is power-limited, not thermally limited, and not idle-throttled.** `Current Power Limit` and
`Default Power Limit` are both **80 W** against a `Max Power Limit` of **175 W**; `SW Power Cap` reads
Active while `HW Thermal Slowdown` reads Not Active at 51 °C. So the 80 W is an OEM/platform TGP, and
SM clocks sit at 1,740 of a possible 3,090 MHz because of it. Raising it would genuinely make inference
faster.

**Do not.** Three reasons, in order of how much they matter:

1. **It buys nothing here.** π0.5 uses 277 ms of an 833 ms budget — 2.9× headroom. Cutting the chunk to
   ~190 ms changes no policy output (inference is deterministic given the same observation) and changes
   no rollout timing, because the policy is called on a fixed 25-step cadence either way. The eval is
   not latency-bound.
2. **Changing it between blocks would be a real confound** — and one the schedule does *not* protect
   against. §10.3's mirror controls for drift that is continuous in session time; a power-limit change
   at block 7 is a step intervention that lands on specific arms. If an arm's numbers then differ, there
   is no way to separate the checkpoint from the clock.
3. **A 2.5 h session at 175 W risks introducing the drift you were trying to avoid.** At 80 W the card
   settles at 51–55 °C and will hold that all afternoon. At 175 W on a laptop chassis it may thermal-
   throttle partway through, which is exactly a mid-session step change in behaviour.

If you want the higher limit anyway, **set it once before block 1 and never touch it again** — and
record that you did, since it makes these bench numbers non-comparable:

```bash
sudo nvidia-smi -pl 175    # may be refused: on many laptops the TGP is OEM-locked
```

**Where the 80 W cap does cost you is training, not this eval.** Any future local run on the 5080 — the
dial grid's ACT column, say — is a sustained 100%-duty-cycle load, and there the clock difference is the
whole wall clock.

## 10.2 — Every arm re-observes at the same rate: `n_action_steps=25`

All six arms run `--policy.n_action_steps=25`, overriding what each checkpoint stores (ACT 100,
both VLAs 50). This is a deployment-only flag — the policy still predicts its full `chunk_size`; the
flag says how much of the chunk to execute before re-predicting — so it changes no weights and voids
nothing about training.

**Why matched, and why 25 rather than 50.** [phase06 §2](phase06_camera_ablation_training.md) rejected a
rollout setting *specifically* because it would have given the three conditions different effective
control rates, and then had to accept exactly that problem in reverse when the stored 100 throttled the
effect under test down to ~9 observations per 30 s episode. Round 1 compares three policy families whose
stored values differ by 2×, so leaving them alone would confound the cross-family comparison with the
re-observation rate. 25 is the matched value that is also the *tightest* one every arm can afford:
0.83 s of open-loop motion, ~36 observations per 30 s episode, and the measured worst case is 316 ms of
an 833 ms budget (§10.1).

**It is also the value Phase 0.6 F3 confirmed, on trimmed data.** 25 stalled the arm at the home pose on
untrimmed Phase 0.5 data — the idle-attractor failure of
[phase06 §4a](phase06_camera_ablation_training.md) — and worked immediately once the training data was
idle-trimmed. Phase 0.7 trains exclusively on the trimmed copy (§8), so the condition F3 identified is
satisfied by construction.

> If an arm stalls at the home pose anyway, that is a **result, not a setup failure** — do not
> hand-wave at the camera, and do not quietly raise `n_action_steps` for one arm. Record it as a stall
> (§10.5) and keep the flag identical across all six. Changing it mid-session makes the whole comparison
> unreadable.

## 10.3 — Positions, schedule, and what the mirror buys

**Nine marked grid tray spots, `T1`–`T9`** — the 3×3 grid of §8, each of which has 5 training episodes
behind it. Round 1 is entirely in-distribution. The five off-grid placements of episodes 45–49 are
**not** evaluated here; they test a different question (the memorisation hypothesis of
[training §3](phase07_training.md)) and are deferred to round 2, so that there is an in-distribution
baseline to read them against.

> **Assign `T1`–`T9` to the nine physical marks once, write the mapping down here, and do not change
> it.** The labels in the schedule are arbitrary until they are pinned to the board. The recording
> order is recoverable — [the verification report §5](phase07_data_verification.md) separates the nine
> blocks by grasp `shoulder_pan` at 27×, and [`figs/phase07_tray_spots.png`](figs/phase07_tray_spots.png)
> plots them — so the natural choice is `T1`–`T9` in the order episodes 0–44 visited them, which makes
> `T1` the spot episodes 0–4 used and lets any per-spot result be read against the training data.
>
> Mapping (fill in before block 1): `T1` ___ · `T2` ___ · `T3` ___ · `T4` ___ · `T5` ___ · `T6` ___ ·
> `T7` ___ · `T8` ___ · `T9` ___

The schedule is pre-generated into
[phase07_eval_log.csv](phase07_eval_log.csv) — 54 rows, outcome columns blank:

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && python3 scripts/make_phase07_eval_schedule.py --seed 0 > Details/phase07_eval_log.csv
```

`lerobot-rollout` records N episodes of one checkpoint in a row, so a fully interleaved trial order
would mean 54 process restarts and 54 model reloads — at 9.35 GB a time for π0.5. Instead the nine spots
split into two halves and every arm runs once per half, in **mirrored order**:

```
half A = T2 T4 T5 T6 T8  (5 spots)      half B = T1 T3 T7 T9  (4 spots)
  block  1: act40k (A) -> T4  T2  T8  T5  T6      block  7: pi10k  (B) -> T1  T7  T3  T9
  block  2: act60k (A) -> T5  T2  T6  T4  T8      block  8: pi5k   (B) -> T3  T9  T7  T1
  block  3: sv5k   (A) -> T4  T8  T6  T5  T2      block  9: sv20k  (B) -> T9  T3  T7  T1
  block  4: sv20k  (A) -> T4  T6  T8  T2  T5      block 10: sv5k   (B) -> T9  T3  T7  T1
  block  5: pi5k   (A) -> T4  T2  T8  T6  T5      block 11: act60k (B) -> T9  T3  T1  T7
  block  6: pi10k  (A) -> T8  T2  T4  T6  T5      block 12: act40k (B) -> T3  T7  T9  T1
```

**12 blocks, 54 trials, 9 per arm.** Three properties, and the script asserts the third rather than
trusting it:

- **Each arm sees each of the nine spots exactly once**, so tray spot is a within-arm constant and
  cannot confound any arm against any other. This is why 9 trials per arm rather than 10 — a tenth trial
  would have to repeat a spot and break that.
- **Each policy's two checkpoints are adjacent** (blocks 1–2, 3–4, 5–6, and mirrored 7–8, 9–10, 11–12),
  so the primary question — which checkpoint — is settled across a few minutes of session time rather
  than across the whole afternoon.
- **Every arm has the same mean session position**, 6.5 of 12: `act40k` runs blocks 1 and 12, `act60k`
  2 and 11, down to `pi10k` at 6 and 7. So drift that is linear in session time — lighting, servo
  warm-up, operator fatigue — cancels out of *every* comparison instead of landing on whoever ran last.
  This is the 6-arm generalisation of the ABBA interleave [phase06 F2](phase06_camera_ablation_training.md)
  used for 2.

**Budget: ~2–2.5 h.** 54 episodes at up to 60 s plus 10 s resets is ~45–60 min of episode time; 12
model loads add ~10 min (π0.5 dominates); the rest is setup and the between-block cylinder moves.

## 10.4 — Step 1: the twelve commands

Rig setup is [§4a Step 2](phase06_camera_ablation_training.md) unchanged — **follower only** on
`/dev/ttyACM0` (no leader: with no teleop connected the episodic strategy returns the arm to the joint
positions captured at startup between episodes), arm **in the home pose before you start**, Cameractrls
presets loaded, board and fixture on their outlines. Run §10.6's preflight first.

Re-check the device paths with `ls /dev/v4l/by-id/` — they change if a camera was swapped.

> **The assignments must be separated by `&&`, not spaces.** Written as a command *prefix*
> (`ARM=act40k lerobot-rollout … $ARM …`) the shell expands `$ARM` on that same line *before* applying
> the assignment, so the policy path silently becomes a garbage string and `from_pretrained` falls
> through to the Hub with `HFValidationError: Repo id must be in the form 'repo_name' or
> 'namespace/repo_name'`. [phase06 §4a](phase06_camera_ablation_training.md) lost a session to this.

### ACT — blocks 1, 2, 11, 12

No `--rename_map`: ACT trained on `observation.images.top` / `.wrist` directly, which is what the robot
produces. Block 1 (`act40k`, half A, 5 episodes):

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && POLICY=outputs/hub/act_tray_cylinder_s1000_trim/checkpoints/040000/pretrained_model && NAME=rollout_phase07_r1_act40k_a && NEP=5 && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && lerobot-rollout --strategy.type=episodic --policy.path=$POLICY --policy.n_action_steps=25 --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/$NAME --dataset.no_stamp=true --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=$NEP --dataset.episode_time_s=60 --dataset.reset_time_s=10 --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

### SmolVLA — blocks 3, 4, 9, 10

Adds `--rename_map`. **This is mandatory, not cosmetic:** the checkpoint's feature names come from
`smolvla_base` and win over the dataset's, so it expects `camera1`/`camera2` and the run dies on a
feature mismatch without the map. It is the same map training used. Block 3 (`sv5k`, half A):

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && POLICY=outputs/hub/smolvla_tray_cylinder_s1000_trim/checkpoints/005000/pretrained_model && NAME=rollout_phase07_r1_sv5k_a && NEP=5 && RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && lerobot-rollout --strategy.type=episodic --policy.path=$POLICY --policy.n_action_steps=25 --rename_map="$RENAME" --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/$NAME --dataset.no_stamp=true --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=$NEP --dataset.episode_time_s=60 --dataset.reset_time_s=10 --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

### π0.5 — blocks 5, 6, 7, 8

Same shape, different map: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`. Block 5 (`pi5k`,
half A):

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && POLICY=outputs/hub/pi05_tray_cylinder_s1000_trim_A100/checkpoints/005000/pretrained_model && NAME=rollout_phase07_r1_pi5k_a && NEP=5 && RENAME='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && lerobot-rollout --strategy.type=episodic --policy.path=$POLICY --policy.n_action_steps=25 --rename_map="$RENAME" --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/$NAME --dataset.no_stamp=true --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=$NEP --dataset.episode_time_s=60 --dataset.reset_time_s=10 --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
```

### The twelve blocks

Change only `POLICY`, `NAME`, `NEP` and (for the VLAs) `RENAME`. Half A is `NEP=5`, half B is `NEP=4`.

| Block | Arm | `POLICY` | `NAME` | `NEP` |
|---|---|---|---|---|
| 1 | `act40k` | `outputs/hub/act_tray_cylinder_s1000_trim/checkpoints/040000/pretrained_model` | `rollout_phase07_r1_act40k_a` | 5 |
| 2 | `act60k` | `outputs/hub/act_tray_cylinder_s1000_trim/checkpoints/060000/pretrained_model` | `rollout_phase07_r1_act60k_a` | 5 |
| 3 | `sv5k` | `outputs/hub/smolvla_tray_cylinder_s1000_trim/checkpoints/005000/pretrained_model` | `rollout_phase07_r1_sv5k_a` | 5 |
| 4 | `sv20k` | `outputs/hub/smolvla_tray_cylinder_s1000_trim/checkpoints/020000/pretrained_model` | `rollout_phase07_r1_sv20k_a` | 5 |
| 5 | `pi5k` | `outputs/hub/pi05_tray_cylinder_s1000_trim_A100/checkpoints/005000/pretrained_model` | `rollout_phase07_r1_pi5k_a` | 5 |
| 6 | `pi10k` | `outputs/hub/pi05_tray_cylinder_s1000_trim_A100/checkpoints/010000/pretrained_model` | `rollout_phase07_r1_pi10k_a` | 5 |
| 7 | `pi10k` | *(as block 6)* | `rollout_phase07_r1_pi10k_b` | 4 |
| 8 | `pi5k` | *(as block 5)* | `rollout_phase07_r1_pi5k_b` | 4 |
| 9 | `sv20k` | *(as block 4)* | `rollout_phase07_r1_sv20k_b` | 4 |
| 10 | `sv5k` | *(as block 3)* | `rollout_phase07_r1_sv5k_b` | 4 |
| 11 | `act60k` | *(as block 2)* | `rollout_phase07_r1_act60k_b` | 4 |
| 12 | `act40k` | *(as block 1)* | `rollout_phase07_r1_act40k_b` | 4 |

Cylinder order inside each block is the schedule's, listed in §10.3 and in the CSV.

### Why each flag is what it is

Carried from [phase06 §4a](phase06_camera_ablation_training.md), with the Phase 0.7 differences marked.

- **`--policy.n_action_steps=25`** — §10.2. Identical on all six arms. **New in Phase 0.7**: Phase 0.6
  ran the stored 100.
- **`--rename_map` on the VLAs only.** Verified to be wired into the rollout path, not just training:
  `build_rollout_context` passes it to the `rename_observations_processor`
  (`rollout/context.py:473`) and through `rename_stats` for the dataset statistics.
- **Both cameras always connected**, named `top` and `wrist`, for all six arms. The physical scene stays
  identical across arms and both views land on video for review.
- **The third declared camera is fine, and it differs between the two VLAs.** Both checkpoints declare a
  camera the two-camera rig cannot feed — SmolVLA a `camera3`, π0.5 a `right_wrist_0_rgb`. Verified in
  the source and confirmed by actually running both (§10.1): **SmolVLA drops it** (`empty_cameras=0`, so
  `modeling_smolvla.py:374-375` breaks out of the padding loop on the first iteration and 2 images are
  fed), while **π0.5 pads it with −1 and masks it out** (`modeling_pi05.py:1011-1012`). Both match what training did
  exactly, since training also fed only two. Nothing to configure — but it is why the model card's
  auto-generated `lerobot-rollout` command does not work as printed: it carries no rename map.
- **`--dataset.no_stamp=true`** keeps the repo name predictable instead of appending a timestamp.
- **The name must start with `rollout_`.** `build_rollout_context` rejects anything else
  (`rollout/context.py:431`). It is a **runtime** check, so it fires only after the robot and cameras
  have already connected.
- **`episode_time_s=60`**, against a 30 s success window — the window still defines success, but a hard
  cutoff leaves no slack on a loaded CPU. Press **→** when the outcome is clear.
- **`reset_time_s=10`** is the window to move the cylinder to the next scheduled spot.
- **`--display_data=true`** gives the live rerun view; use it to confirm image quality before the first
  episode commits.
- **No `--inference.type=rtc`** — §10.1 measured that sync inference has 2.6× headroom on the slowest
  arm, and leaving RTC off keeps one fewer thing different between arms.
- **`--dataset.push_to_hub=false`.** 54 episodes of two 480×640 streams stay local; episodes land in
  `~/.cache/huggingface/lerobot/HALDijkstraaa/rollout_phase07_r1_<arm>_<half>/`.
- **Stale action queues are handled.** The episodic strategy calls `engine.reset()` between episodes,
  which resets the policy, preprocessor and postprocessor — no leftover chunk carries over.

## 10.5 — Step 2: during the session

- **→** ends the current episode or reset early · **←** discards and re-records · **Esc** stops the block.
- Use **←** only for a *rig* problem — cylinder knocked over during reset, wrong spot placed. **Do not
  re-record because the policy failed.** The failures are the data.
- Between episodes, move the cylinder to the next spot in that block's list.
- **If an arm does not move by ~10 s, record it as a stall and press →.** Do not wave at the camera
  (§10.2).
- Note `duration_s` and anything odd in `notes` as you go; `success` and `fail_stage` are scored from
  video afterwards (§10.6).

## 10.6 — Preflight, and Step 3: scoring

**Preflight — the rig is reported untouched since the 2026-10-03 recording, so this verifies rather than
rebuilds.** Both checks are cheap and both are hard stops:

**1. Replay episode 0 of the portable-rig dataset.** This is the only acceptance test that separates
"the calibration frame is right" from "it looks right": `lerobot-replay` plays a recorded episode's
**actions** straight to the servos — no policy, no camera, no feedback — so if it no longer reaches
what it reached on 2026-10-03, the problem is physical and nothing about the policies is in question.
It is how the Phase 0.6 `shoulder_pan` drift was caught
([phase06](phase06_camera_ablation_training.md) §"The leftward insertion offset").

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && lerobot-replay --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --dataset.repo_id=HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912 --dataset.episode=0
```

Before running it: **arm at the home pose**, cylinder standing in the tray at **the spot episodes 0–4
used** (the first of the nine grid blocks — [`figs/phase07_tray_spots.png`](figs/phase07_tray_spots.png)
shows the layout), fixture on its outline. Then watch the two moments that matter — **the grasp** and
**the insertion** — and compare against the recorded video of the same episode.

- **The raw dataset, not the trimmed one.** Episode 0 is 878 frames (29.3 s) raw against 743 (24.8 s)
  trimmed, and the 135-frame difference is exactly the idle lead-in and tail. The raw episode therefore
  **starts at the home pose**, which is where the arm is; replaying the trimmed copy would begin with
  the arm already mid-reach and snap it there from home. The raw copy is also the archival record of
  what the session actually did.
- **A clean replay clears the arm, the base, the fixture and the calibration frame in one go.** A miss
  means something physical moved, and [`scripts/joint_offset_check.py`](../scripts/joint_offset_check.py)
  plus [phase06 §8](phase06_camera_ablation_training.md) turn the miss into a correction.
- Nothing needs restoring first: the live calibration pair is byte-identical to
  [`calibration/current/`](calibration/current/), verified 2026-10-12.

**2. Compare the top camera against the reference still** in [`figs/`](figs/) with the arm at home.
§3's reference still is the "is the board set up right" check.

If either fails, fix it *before* any rollout: a rig shift and a policy failure are indistinguishable
afterwards, and 54 trials is too many to throw away.

**Scoring — from video, blind to arm.** Per [phase06 §4b](phase06_camera_ablation_training.md), scoring
live favours whichever arm you expect to win. The arm name is in the dataset directory name, so scoring
blind means shuffling the episodes rather than browsing per-arm folders.

Fill in [phase07_eval_log.csv](phase07_eval_log.csv):

| Column | Values |
|---|---|
| `success` | 1 if, **within 30 s**, the cylinder is fully seated in the top-left blue hole and the gripper released — the task card's definition, unchanged (§5) |
| `fail_stage` | `reach` · `grasp` · `transport` · `insert` · `retreat` · `none`, plus **`stall`** if the arm never started (§10.2) |
| `duration_s` | time to the outcome |
| `notes` | and for every grasp attempt, **`shoulder_pan`** — [phase06 §6.5.3](phase06_camera_ablation_training.md) tracks reach by its slope, and the bird's-eye rig is supposed to have fixed exactly that |

**`fail_stage` is the column that carries the result**, not `success`. At 9 trials an arm's success count
has a ±16 point standard error, but the *stage* an arm fails at is a per-trial observation, and a shift
from `reach` to `insert` failures is visible in 9 trials when a rate difference is not. It is also the
direct test of the §2 prediction that the bird's-eye view fixes reach.

## 10.7 — Pre-registration: confirm or edit before the session

[08 §A.2](../08_data_quality_research.md) asks for predicted orderings committed *before* the comparison
is read, and [training §5](phase07_training.md) notes that the loss comparison can no longer be
pre-registered — all three runs are done — but **the rollouts can, because they have not started.**

**Committed 2026-10-12, before any rollout.** Predictions 1–3 and 5 are the operator's call;
4 and 6 are deliberately left open. Reasoning is recorded so the post-hoc read is honest about which
predictions were well-founded and which were guesses.

| # | Prediction | Confidence | Reasoning |
|---|---|---|---|
| 1 | `act40k` ≈ `act60k`, within 2/9 | **high** | Eval loss is flat from 20k and the 20k–60k spread is checkpoint wobble. If it holds, the dial grid drops ACT to ~30k steps and halves its compute across 19 conditions × 3 seeds |
| 2 | `sv5k` > `sv20k` | **medium** | Held-out loss rises monotonically +61%, far stronger than the sub-wobble differences phase06 dismissed. But loss↛success is what phase06 established, so `sv20k` acting better than its loss is live |
| 3 | `pi10k` > `pi5k` | **high** *(raised)* | **The curve was read** (§10.7a): π0.5's eval loss is still falling at the last checkpoint, with no minimum anywhere in range. It is underfit, not overfit — so the later checkpoint should be the better one, and the SmolVLA-style reversal that motivated running both is ruled out |
| 4 | At matched 320k samples, `act40k` vs `sv5k` | **open — no prediction** | 45 episodes is tiny for a 100 M-parameter VLA and ACT's from-scratch chunked bias suits a single-task, low-variance dataset; against that, SmolVLA carries pretrained visual and language priors ACT has none of. Genuinely uncertain, and the cleanest Q2 result round 1 can produce |
| 5 | **π0.5 is the strongest family** | **medium** | Operator's call, on held-out loss: π0.5 sits at ≈0.052 against SmolVLA's best 0.126 and ACT's 0.179, and it is the only one still improving. §10.7a sets out how far that comparison carries — it is sound against SmolVLA and not available against ACT |
| 6 | Dominant `fail_stage` | **open — no prediction** | §2's claim for the bird's-eye view is that it fixes reach, where the third-person top view was aliased and `top+wrist` under-reached 6/10. Whether `insert` overtakes `reach` is what the column is there to find out |

> **Prediction 5 reverses the original draft, which had π0.5 weakest on a sample-count argument** —
> 160k samples against ACT's 480k and SmolVLA's 320k. That argument is not wrong, but it was made
> before the loss curve was read, and an underfit policy that is still the best of the three on
> held-out loss is a stronger signal than its epoch budget is a worry. Both readings are on the record;
> the rollouts settle it.

**What would change the plan.** If every arm scores ≤1/9, the problem is the rig or the task, not the
checkpoints — stop and diagnose before spending round 2. If all six are indistinguishable at 7–9/9, the
task is too easy to separate policies and round 2 needs the off-grid spots to create headroom.

## 10.7a — π0.5's held-out curve, and how far it compares across families

**Read from wandb run [`sxpkx9st`](https://forge.coreweave.com/wandb/bj-huang-university-of-toronto/phase07-policy-transfer/runs/sxpkx9st) on 2026-10-12**, closing the gap
[training §4](phase07_training.md) left open. The curve **falls monotonically across the whole logged
range** — ≈0.0532 at 2,500 down to ≈0.0515 by 7,500 — with **no minimum reached**.

Two things follow, and they point opposite ways:

1. **π0.5 does not overfit on this dataset**, where SmolVLA did badly at the same 45 episodes. At 6.0
   epochs it is still underfit, so `pi10k` should beat `pi5k` (prediction 3) and **more steps would
   probably still help** — the 10,000 the A100 inherited from the L4 notebook was, in hindsight, short
   rather than merely different. That is a Q2-relevant result in its own right: two flow-matching VLAs
   on identical data, holdout and seed, with opposite overfitting behaviour at 48 vs 6 epochs.
2. **The flat eval curve means the checkpoint question was never the risk for π0.5** that it was for
   SmolVLA. Running both π0.5 arms is still the right call — it costs 18 of 54 trials and it is the only
   on-robot evidence either way — but it is now a confirmation rather than a coin flip.

### How far 0.052 vs 0.126 vs 0.179 actually carries

[training §1](phase07_training.md) warns that held-out loss does not compare across the three. That
warning is **right about ACT and too strong about the other two**, and the difference matters because
prediction 5 rests on it. Checked in the source and in the dataset's own statistics:

| | ACT | SmolVLA | π0.5 |
|---|---|---|---|
| Objective | **L1 + KLD on actions** | flow-matching MSE on the velocity field | flow-matching MSE on the velocity field |
| Reduced over | 6 action dims | 6 — truncated before the mean (`modeling_smolvla.py:303`) | 6 — truncated before the mean (`modeling_pi05.py:1081`) |
| Flow timestep sampling | — | `Beta(1.5, 1.0)`, scale 0.999, offset 0.001 | **identical** |
| Action normalisation | `MEAN_STD` | `MEAN_STD` | **`QUANTILES`** |

- **ACT against either VLA: no comparison exists.** An L1 on predicted actions and an MSE on a
  flow-matching velocity residual are different quantities with no conversion between them. Nothing can
  be said about ACT from these numbers, in either direction.
- **SmolVLA against π0.5: comparable, up to one measurable factor.** Same objective, same reduction,
  same timestep distribution, same Gaussian noise — the two were more alike than the blanket warning
  implied. The only difference is the action normaliser, and its size can be computed from the
  dataset's statistics rather than guessed: `QUANTILES` divides by `(q99−q01)/2`, which across the six
  joints is **1.36× the `std`** that `MEAN_STD` divides by, so π0.5's targets are ≈0.74× SmolVLA's.
  Through the flow target `ε − a` that is a **≈1.3× deflation** of π0.5's loss. Correcting for it puts
  π0.5 at an effective ≈0.067 against SmolVLA's best 0.126 — **still roughly half**, so the gap
  survives the correction and prediction 5 is well-founded *against SmolVLA*.
- **The one thing none of it establishes is task success.** [phase06 §2](phase06_camera_ablation_training.md)
  is the standing counterexample: held-out loss separated three conditions by less than one curve's
  wobble while the robot showed a decisive double dissociation. A 2× loss gap is far larger than that —
  but it is still the wrong units for the question, and it says nothing about ACT, which is half of
  prediction 5.

**What would settle it properly, and is cheap.** Evaluate all three on one policy-agnostic metric:
L1 between the predicted action chunk and the ground truth on the held-out episodes, **un-normalised,
in degrees**. That is the same quantity for every policy family, it is directly interpretable, and
every piece needed already exists in
[`eval_checkpoints_heldout.py`](../scripts/eval_checkpoints_heldout.py) — it would take swapping
`policy.forward`'s loss for `predict_action_chunk` plus an inverse-normalisation step. Worth doing
**after** the rollouts, so the loss-vs-success comparison is not read before the robot has spoken.

---

# 11 — Round 2, once round 1 picks the checkpoints

Sketched, not specified — the design depends on round 1's numbers. Recorded so round 1's choices are
made with it in view.

| | Round 1 (§10) | Round 2 |
|---|---|---|
| Question | Which checkpoint per policy | **Success rate** per policy, and off-grid generalisation |
| Arms | 6 (2 checkpoints × 3 policies) | 3 (the winners) × {on-grid, off-grid} |
| Trials | 9 per arm, **54** | 30 per policy on-grid + 12 off-grid, **~126** |
| Resolution | ±16 points — ranks, does not measure | **±9 points** at 50%, per [phase06 §4b](phase06_camera_ablation_training.md) |
| Time | ~2–2.5 h | ~4–5 h, two sessions |

Three things round 2 adds:

1. **Enough trials to quote a rate.** 30 per policy is the [phase06 §4b](phase06_camera_ablation_training.md)
   figure: ±9 points at 50%, where 20 trials gives ±11 and would miss a real 15-point effect.
2. **The off-grid spots**, which test the mechanism [training §3](phase07_training.md) proposed for
   SmolVLA's overfitting: if a policy memorised nine discrete tray positions it should fail specifically
   at unmarked ones. Needs 3 new marks on the tray and a second success rate per policy. Round 1's
   on-grid rate is the baseline that makes the off-grid number mean something.
3. **Seeds.** [phase06 §4b](phase06_camera_ablation_training.md) notes that at 50 episodes seed variance
   is comparable to the effect size, so a single-seed ranking is not a policy-family result. This is the
   expensive item — 3 seeds × 3 policies is 9 training runs — and it is what [08 §A.4](../08_data_quality_research.md)
   already budgets for the dial grid.

**`phase07_log.csv` is still missing** and round 2 does not need it, but the per-episode `success` and
`stage` of the *training* data are not recoverable ([verification §6](phase07_data_verification.md)) —
so if any analysis wants to relate rollout outcomes to the demonstration quality at the same tray spot,
that is gone for these 50 episodes and is a reason to record it for the next dataset.
