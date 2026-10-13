# Phase 0.7 — portable rig: bird's-eye top camera, and a new dataset

Run book for the rebuilt rig, the data collection that follows it (§1–§9), and the **round-1 policy
evaluation** (§10–§13). Everything — both arms, both cameras, the tray and the fixture — is now mounted
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
checkpoint per policy; **§11 has the results**, and **§12 is the agreed round-2 plan**.

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

## The datasets, as recorded

| | Repo | Episodes | Frames | Use |
|---|---|---|---|---|
| **D0 raw** (2026-10-03) | [`so101_tray_cylinder_portable_20261003_211912`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912) | 50 | 35,141 | archive · analysis |
| **D0 trimmed** | [`..._20261003_211912_trimmed`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed) | 50 | **29,462** (83.8%) | round-1 training |
| **D1 raw** (2026-10-13) | [`so101_tray_cylinder_varied_20261013_082911`](https://huggingface.co/datasets/HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911) | 50 | **38,389** | §12.2 — continuously randomised, deliberately varied |

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
| 4a ✅ | **Round 1 run book**: six arms, 54 trials, two checkpoints per policy, mirrored interleave, 9 grid spots | **§10** · [`make_phase07_eval_schedule.py`](../scripts/make_phase07_eval_schedule.py) |
| 4 ✅ | **Round 1 run and scored 2026-10-12** — 54 trials. SmolVLA 6/18, ACT 5/18, π0.5 2/18 (0/18 excluding the two the operator called lucky). Held-out loss ranked the families backwards, and spot difficulty turned out to be policy-specific | **§11** · [phase07_eval_log.csv](phase07_eval_log.csv) |
| 5 ◐ | **Round 2 plan agreed 2026-10-12** — data, not more trials: a continuously-randomised varied dataset, perturbed/recovery episodes, head-only trim, ACT + SmolVLA retrained | **§12** · §13 holds the deferred success-rate design |

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
point standard error, so **the rate is round 2's job** (§13). What round 1 settles is which six
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

> **`T1`–`T9` are the nine spots in recording order** — confirmed 2026-10-12. `T`*k* is the *k*-th
> spot visited during collection, so its five demonstrations are episodes 5(*k*−1)…5(*k*−1)+4 and
> `T1` is the spot episodes 0–4 used. Verified against the operator's dataset screenshots and against
> the grid geometry (§11.6): laid out 3×3 row-major the grasp bearing is a regular raster, +4.1…+5.8°
> along each row and −3.5…−5.6° down each column.
>
> This mapping is what lets rollout outcomes be joined to the demonstrations recorded at the same
> spot, which is the whole of §11.6. **Keep using it; do not relabel.**

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
4 and 6 are deliberately left open. **Scored against the outcome in [§11.2](#112--prediction-scorecard-against-107):
1 and 3 confirmed, 2 and 5 wrong — and both wrong ones were the loss-based ones.** Reasoning is recorded so the post-hoc read is honest about which
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

# 11 — Round 1 results — run and scored 2026-10-12

54 trials, 12 blocks, as §10 specifies. Scored blind from video into
[phase07_eval_log.csv](phase07_eval_log.csv); `episode_index` and `duration_s` recovered from the
recordings by [`analyze_phase07_eval.py`](../scripts/analyze_phase07_eval.py), which also produces
every number and figure below. Summary tables:
[`results/phase07_eval/`](../results/phase07_eval/).

## The four findings to carry forward

1. **Held-out loss does not predict real-robot performance — it ranked the three families exactly
   backwards.** The cause is compounding error: loss is a *per-frame, teacher-forced* quantity scored
   from states the demonstrator reached, while a rollout is scored from states the policy reached.
   §11.3.
2. **A smoothness metric computed from the predicted chunk would close most of that gap, and it can be
   logged to wandb during training with no change to lerobot's training loop.** §11.3a gives the patch.
3. **Keep the tail idle frames.** Trim the head (the [F3](phase06_camera_ablation_training.md) idle
   attractor is real), but the tail is the only place a policy can learn "the task is done — hold
   still", and its absence is measurable: SmolVLA moves 3.6× as much as ACT in the final second. §11.5.
4. **Tray-spot difficulty is a property of the *policy*, not of the position.** The three families
   disagree about which spots are hard (Kendall's W = 0.19; a concordance that low arises from
   sampling noise alone only 3% of the time), and ACT's best spot is SmolVLA's worst. §11.6.

> **Finding 4 is the one with reach beyond this phase.** [08 §A.1](../08_data_quality_research.md)
> records that the 2025 causal-quality line — CUPID and its successors — paid for rigour by making
> quality "**tied to one policy** … the score only holds for that policy," and that 2026's reaction
> was to retreat to model-free proxies (RINSE, PSD scores) on the argument that policy-in-the-loop
> curation does not scale. **Round 1 is a direct empirical instance of the thing the model-free line
> assumes away**: the same 50 demonstrations, with everything held fixed but the policy, produce
> opposite per-position outcomes. A model-free score cannot express that, because the quantity it is
> estimating does not exist independently of the learner. It does not make model-free proxies useless —
> it bounds what they can ever claim, and it says a Q2 answer must be reported per policy family rather
> than pooled.

**Headline: nothing works well yet, and the held-out loss ranking was exactly inverted.** The policy
with the *lowest* held-out loss (π0.5, §10.7a) is the only one that never completed the task on its own;
the policy whose loss said it was 61% overfit (SmolVLA 20k) is the most capable thing in the set.

| | ACT | SmolVLA | π0.5 |
|---|---|---|---|
| Success | 5/18 · **27.8%** | **6/18 · 33.3%** | 2/18 · 11.1% |
| Success, excluding "lucky" | 4/18 · 22.2% | **6/18 · 33.3%** | **0/18 · 0.0%** |
| 95% Wilson | 12.5–50.9% | 16.3–56.3% | 3.1–32.8% |
| Mean progress (of 4 stages) | 2.22 | 2.17 | **0.94** |
| Dominant failure | **grasp** (8) | grasp (6) | **reach (11)** |
| Held-out loss rank | 3rd | 2nd | **1st** |

![success and progress](figs/phase07_eval_success.png)

## 11.1 — Per arm

| Arm | Success | Clean | Progress | Duration | Jerk | Terminal motion | reach / grasp / insert |
|---|---|---|---|---|---|---|---|
| ACT 40k | 2/9 | 2 | 2.33 | 21.5 s | 25 | **4.3** | 0 / 4 / 3 |
| ACT 60k | 3/9 | 2 | 2.11 | 20.3 s | **22** | 7.1 | 1 / 4 / 1 |
| SmolVLA 5k | 3/9 | 3 | 1.78 | 24.7 s | **63** | **28.2** | 2 / 4 / 0 |
| SmolVLA 20k | 3/9 | 3 | **2.56** | 23.4 s | 32 | 12.1 | 1 / 2 / 3 |
| π0.5 5k | 1/9 | **0** | 0.67 | 20.0 s | 32 | 12.7 | 6 / 2 / 0 |
| π0.5 10k | 1/9 | **0** | 1.22 | 23.9 s | 33 | 15.0 | 5 / 1 / 2 |

**Three trials carry a "lucky" note** and they matter disproportionately: π0.5 5k's only success was a
cylinder *kicked* into the hole, π0.5 10k's was another "lucky move", and ACT 60k's was a "lucky grasp".
Excluding them, **π0.5 scores 0/18** — it never performed the task deliberately. They are kept in the
primary count because they meet the task card's definition, and split out everywhere because they are
not evidence of capability.

**Durations are uninformative** (20.0–24.7 s across all six arms, against a 30 s success window and a
60 s recording cap). The operator pressed → once the outcome was clear, so duration mostly measures how
long it took *to become obvious*, not how long the task took. Not a discriminator here.

![failure stages and directional bias](figs/phase07_eval_failure.png)

**No trial failed in transport, across all 54.** Every policy that got the cylinder off the tray got it
over the fixture. That confirms the session impression that transport is the easy stage, and it is the
basis of §11.5's data-collection proposal.

## 11.2 — Prediction scorecard against §10.7

| # | Predicted | Outcome | |
|---|---|---|---|
| 1 | `act40k` ≈ `act60k`, within 2/9 | 2/9 vs 3/9 — within 1, and progress 2.33 vs 2.11 runs the *other* way | ✅ **confirmed** |
| 2 | `sv5k` > `sv20k` | 3/9 vs 3/9 on success, but 20k leads on progress (2.56 vs 1.78) and is half as jerky | ❌ **wrong — reversed** |
| 3 | `pi10k` > `pi5k` | 1/9 each, but progress 1.22 vs 0.67 and 10k is the only one to reach an insert | ✅ **weakly confirmed** |
| 4 | open — `act40k` vs `sv5k` at matched 320k samples | 2/9 vs 3/9, progress 2.33 vs 1.78. Inside noise either way | — **unresolved**, as expected |
| 5 | π0.5 strongest | π0.5 is **last** on every measure, and 0/18 excluding luck | ❌ **wrong, decisively** |
| 6 | open — dominant fail stage | No single stage dominates: π0.5 fails at reach (11/18), ACT and SmolVLA at grasp | — **answered: it is policy-specific** |

**Two of four committed predictions were wrong, and both wrong ones came from held-out loss.** 1 and 3
were loss-based too but happen to be the two where loss and behaviour agreed. Predictions 2 and 5 were
the ones where the loss comparison was load-bearing — and §10.7a's caveat that loss is "the wrong units
for the question" turned out to be the whole story.

## 11.3 — The SmolVLA inversion, and why a per-frame loss cannot see it

This is the most interesting result in round 1. **SmolVLA 5k has the best held-out loss in its family by
61% and behaves worse on the robot than 20k.** The session note called 5k "shaky" and 20k "very smooth",
and that is measurable in the recorded actions rather than a matter of impression:

![smoothness and duration](figs/phase07_eval_smoothness.png)

| | SmolVLA 5k | SmolVLA 20k |
|---|---|---|
| Held-out loss | **0.1258** (best in family) | ~0.2029 (+61%) |
| Success | 3/9 | 3/9 |
| Mean progress | 1.78 | **2.56** |
| **Action jerk** | **63.5** | **32.3** — half |
| Failures | 2 reach, 4 grasp, **0 insert** | 1 reach, 2 grasp, **3 insert** |

5k is **2.0× jerkier** than 20k, and the failure profile shifts: 5k mostly never gets the cylinder, 20k
usually does and then misses the hole. Getting far enough to fail at insertion is strictly more progress
than failing at the grasp.

**Why the loss misses it — the operator's hypothesis, and it is the right one.** Held-out loss is a
*per-frame, teacher-forced* quantity: each prediction is scored against ground truth from a state the
*demonstrator* reached. A rollout is the opposite — the policy is scored from states *it* reached, so a
small per-frame error feeds the next observation and compounds. That is the classic covariate-shift /
compounding-error gap, and it has a specific shape here:

- A **jittery** policy can have excellent per-frame loss. Jitter is high-frequency and roughly
  zero-mean, so it barely moves an L1/MSE average, while on hardware it shakes the gripper off a 
  cylinder that needs millimetre placement. Jerk is a *second-difference* property; the loss is a
  *zeroth-difference* one. They are nearly orthogonal, which is exactly why 5k can win one and lose the
  other.
- **Training longer smooths the policy even as it overfits the frames.** 20k memorises the training
  frames harder (train loss 0.0193, eval 0.2029) but produces a more coherent action sequence. Overfit
  in the per-frame sense is not the same as degraded as a controller.

**What would show it directly, and it is cheap:** the per-stage / per-phase held-out loss both the
session notes and §10.7a asked for. Pool the held-out frames by task stage (reach / grasp / transport /
insert, segmentable by the gripper trace with the detector already in
[`analyze_phase07_eval.py`](../scripts/analyze_phase07_eval.py)) and report loss per stage instead of
one pooled number. The prediction this makes is testable: **5k and 20k should separate at the grasp and
insert stages specifically**, where tolerance is tightest, even though 5k wins pooled. Adding a
smoothness term — the jerk of the *predicted chunk*, which needs no robot — would make the offline
metric track the rollout ordering much better than loss alone.

## 11.3a — Logging chunk smoothness to wandb during training: yes, and it is small

**The training loop needs no change at all.** `MetricsTracker.update_metrics`
(`lerobot/utils/logging_utils.py:163`) *auto-registers an `AverageMeter` for every new scalar key* it
is handed, and `update_policy` passes the policy's whole `output_dict` into it
(`lerobot_train.py:206-207`). So any extra scalar a policy returns alongside its loss is averaged over
the log window and shipped to wandb automatically — the same route `l1_loss` and `losses_after_forward`
already take.

That gives two insertion points, and they answer different questions.

**(a) Train-time, for ACT — one line in the policy.** ACT's `forward` already has the predicted chunk
in hand, so the jerk of its own output is free:

```python
# in ACTPolicy.forward, beside the existing l1_loss / kld_loss entries
jerk = actions_hat.diff(dim=1).diff(dim=1).std(dim=(1, 2)).mean()
loss_dict["chunk_jerk"] = jerk.item()
```

It appears in wandb as `chunk_jerk` with no further work. **This does not transfer to the VLAs as
written:** SmolVLA and π0.5 compute a flow-matching velocity-field loss and never materialise an action
chunk during `forward`, so there is nothing to differentiate. Getting one costs a full denoising loop
(`num_steps=10` extra forward passes), which is far too expensive per training step.

**(b) Eval-time, for all three — the one actually worth doing.** The eval branch
(`lerobot_train.py:653-674`) already pays for a pass over the held-out set at `eval_steps` cadence, and
at that cadence a denoising loop is affordable. Call `predict_action_chunk` on each eval batch, take
the jerk, and log it beside `eval_loss`:

```python
# inside the `if is_eval_step:` loop, after `loss, _ = policy.forward(eval_batch)`
chunk = policy.predict_action_chunk(eval_batch)          # (B, chunk_size, action_dim)
jerk_sum += chunk.diff(dim=1).diff(dim=1).std(dim=(1, 2)).mean().item()
...
# beside the existing eval_loss log
wandb_logger.log_dict({"eval_loss": eval_loss, "eval_chunk_jerk": jerk}, step=step, mode="eval")
```

**Why (b) is the right one.** It measures the jerk of *the chunk the policy would actually execute*,
on held-out observations, for every family on one scale — which is precisely the quantity §11.3 found
tracks rollout behaviour where loss does not. Normalise per joint by the training-set σ (as
[`analyze_phase07_eval.py`](../scripts/analyze_phase07_eval.py) does) so the number is comparable
across policies, and it becomes a second eval curve to read beside the loss.

**Cost.** One denoising loop per eval batch. π0.5's eval pass is ~5 held-out episodes; at the measured
277 ms per chunk this is well under a minute per eval point, against `eval_steps` of 2,500. Negligible.

**The prediction this makes, worth pre-registering before it is run:** retrain SmolVLA with
`eval_chunk_jerk` logged, and the 5k checkpoint should show **~2× the jerk of 20k** — matching the
2.0× measured on the robot (§11.3), and visible *without* a robot. If it does, the metric is a usable
offline proxy for the thing loss misses, and the dial grid can be scored on loss **and** jerk.

> **This has a literature hook.** [08 §A.1](../08_data_quality_research.md) records RINSE (Kulkarni,
> Dhar & Cui, 2026) scoring *demonstrations* by smoothness — spectral arc length plus a
> trajectory-envelope distance — and a PSD-based successor. Round 1 points at the mirror image:
> smoothness of the **policy's output**, not of the training data. The two are complementary, and the
> same spectral-arc-length statistic would work here in place of the plain second difference; jerk is
> the cheaper first cut.

## 11.4 — π0.5: it never learned to reach

**11 of 18 π0.5 trials failed at reach**, against 1 for ACT and 3 for SmolVLA — and the two scored
successes were both annotated lucky, so its deliberate success rate is **0/18**. A policy that cannot
position the gripper over the cylinder has not learned the first thing the task needs, and the operator's
reading — that it frequently goes out of distribution and stalls — matches the failure profile.

Three candidate causes, and they are separable:

1. **Not enough training.** π0.5 saw 160k samples / 6.0 epochs, against ACT's 480k / 18.0 and SmolVLA's
   1.28M / 47.9. Its eval loss was still falling (§10.7a), so it is genuinely underfit. **This is the
   leading explanation and the cheapest to test**: more steps.
2. **Action-expert-only fine-tuning may not be enough.** Only 693 M of 3.3 B parameters trained, with
   `freeze_vision_encoder=true`. If the frozen SigLIP features do not resolve a white cylinder in a green
   tray at this scale, no amount of action-expert training fixes the reach, because the information is
   not in the representation. Testable by unfreezing the vision encoder, or by probing whether the frozen
   features linearly predict cylinder position.
3. **50 episodes is simply too few for a 3.3 B model.** The most likely truth is a combination of this
   and (1).

**What this does not support is "π0.5 is bad at this task".** It is the arm with the least training and
the most parameters, evaluated at 6 epochs. The honest claim is narrower: **at this data and compute
budget, π0.5 is not competitive, and its held-out loss gave no warning of that.**

## 11.5 — Failure stages, and the idle-state observation

**Terminal motion confirms the session note precisely.** The operator observed that ACT parks when the
task is done while SmolVLA keeps moving regardless of the hole's state. Measured as joint motion in the
final second before the episode was stopped:

| | ACT | SmolVLA | π0.5 |
|---|---|---|---|
| Terminal motion | **5.7** | **20.2** — 3.6× ACT | 13.9 |

SmolVLA 5k is the worst at 28.2, ACT 40k the best at 4.3. The operator's explanation for *why* is worth
recording because it is a falsifiable mechanism: ACT appears to gate on the hole's appearance — it moves
when colour shows near the hole and does not when the hole is faked white or black — whereas SmolVLA
moves unconditionally.

**The actionable conclusion is a data-collection change, and it reverses an earlier decision.**
Phase 0.7 trims idle frames from *both* ends of every episode (§8), and the head trim is well justified:
[phase06 F3](phase06_camera_ablation_training.md) showed the idle lead-in creates an attractor the policy
cannot escape at the start. But the **tail** trim removes the only frames that teach "the task is done —
now hold still." Keeping them is how a policy learns a terminal state.

> **For the next dataset: trim the head, keep the tail.** A second or two of deliberate stillness after
> the cylinder is seated and released, recorded on purpose rather than as operator lag. The head and the
> tail are not the same frames and should not share a rule — which is a correction to §8's blanket
> "trim idle" guidance, earned by this round's measurements.

## 11.6 — Per-position: difficulty belongs to the policy, not the spot

**The mapping is known as of 2026-10-12** — `T`*k* is the *k*-th spot in recording order, so its five
demonstrations are episodes 5(*k*−1)…5(*k*−1)+4. Confirmed twice over: the operator's dataset
screenshots (episode 1 → position 1, episode 9 → position 2) and the grid geometry itself. Laid out
3×3 row-major, the grasp bearing moves **+4.1…+5.8°** along every row and **−3.5…−5.6°** down every
column — a regular raster, which is only true if the labelling is right.

Analysis: [`analyze_phase07_positions.py`](../scripts/analyze_phase07_positions.py) ·
[`per_position_joined.csv`](../results/phase07_eval/per_position_joined.csv).

![per-position grid](figs/phase07_position_grid.png)

### T8 is not an easy spot — it is ACT's spot

The session impression that T8 was good was right about T8 and wrong about why. Pooled, T8 is the best
spot (3/6). Split by family, it is the sharpest disagreement in the whole set:

| | ACT | SmolVLA | π0.5 |
|---|---|---|---|
| **T8** | **4.0** — both trials seated | **0.0** — neither trial reached | 2.0 |
| **T5** | **0.5** | **4.0** — both trials seated | 1.5 |

**A clean double dissociation**, in the same sense [phase06 §6.5](phase06_camera_ablation_training.md)
used the phrase: ACT's best spot is SmolVLA's worst, and SmolVLA's best is ACT's worst. Each family's
best spot is different — ACT T8, SmolVLA T5, π0.5 T7 — and so is each family's worst.

![the interaction](figs/phase07_position_interaction.png)

### The interaction survives a noise test, which the raw concordance alone would not

| Statistic | Value | Reading |
|---|---|---|
| Kendall's W, 3 families × 9 spots | **0.192** | 1.0 would be identical rankings |
| Permutation p (null: *no* shared ordering) | 0.823 | cannot reject "no agreement" |
| **Bootstrap p (null: *one* shared ordering, 2 trials/cell)** | **0.030** | **a concordance this low is not just noise** |
| Spearman ρ, ACT vs SmolVLA | **−0.645** (p = 0.066) | negatively correlated, not merely uncorrelated |
| Spearman ρ, ACT vs π0.5 | +0.285 (p = 0.45) | — |
| Spearman ρ, SmolVLA vs π0.5 | −0.228 (p = 0.55) | — |

**The second test is the one that carries the claim, and it is why the first is reported beside it.**
With only 2 trials per (family, spot) cell, sampling noise attenuates W even when every family shares
one ordering — so a low W is, on its own, weak evidence. The bootstrap asks the right question
directly: simulate the world in which position difficulty *is* shared (pooled position effect + a
per-family offset, with residuals resampled from the real data), draw 2 trials per cell, and see how
often W comes out this low. **Answer: 3.0% of the time.**

Supporting it, the ACT↔SmolVLA correlation is **negative**, and noise alone centres ρ at zero, not
below it. Two independent angles, same conclusion.

**Still: 2 trials per cell.** This is a real effect at p ≈ 0.03, not a measured effect size. Round 2
should carry per-position outcomes rather than pooling them.

### Does the training data at a spot explain it? One candidate, flagged not concluded

Spearman ρ across the nine positions, between each training-data property and rollout progress:

| Training metric | pooled | ACT | SmolVLA | π0.5 |
|---|---|---|---|---|
| Divergence of the 5 demos, **reach phase** | +0.05 | **−0.63** (p = .08) | **+0.79** (p = .02) | −0.24 |
| Divergence, whole episode | −0.29 | −0.20 | +0.07 | −0.14 |
| Divergence, carry phase | +0.05 | −0.03 | +0.16 | −0.09 |
| Grasp-pose spread across the 5 demos | +0.14 | −0.21 | +0.36 | +0.11 |
| Mean episode length | −0.50 | −0.64 (p = .07) | +0.32 | −0.56 |
| Mean idle lead-in | −0.50 | −0.54 | −0.03 | +0.05 |

**Nothing is significant pooled** (every p > 0.17) — which is itself consistent with §11.6's main
result: a position property that helps one family and hurts another cannot show up in the pooled
column, because the two cancel.

**The one pattern worth a hypothesis** is `div_reach`, the disagreement among the five demonstrations
during the reach: **ACT −0.63, SmolVLA +0.79 — strong, and opposite in sign.** Where the demonstrations
disagree about how to approach the cylinder, ACT does worse and SmolVLA does better. A plausible
mechanism, and it matches each family's inductive bias: ACT is trained from scratch on 45 episodes and
regresses a chunk, so inconsistent demonstrations are conflicting targets it has no prior to arbitrate;
SmolVLA carries a pretrained visuomotor prior and generates by flow matching, so variation in the
demonstrations is closer to the multimodality that training assumes — and may act as augmentation
rather than noise.

**Stated as one test, it is much stronger — and it is not a workspace effect.** The two opposite
correlations are one claim, so the direct expression of it is a single variable: the **ACT-minus-SmolVLA
progress gap** at each spot, against reach divergence.

| Gap (ACT − SmolVLA progress) vs | ρ | p |
|---|---|---|
| **divergence of the demos, reach phase** | **−0.814** | **0.012** |
| workspace extremity (\|grasp pan\|) | +0.136 | 0.74 |
| mean episode length | −0.542 | 0.14 |
| grasp-pose spread | −0.356 | 0.35 |

That is **one planned test rather than one of forty**, and it survives a Bonferroni correction over the
four candidates above (0.012 × 4 = 0.048). The obvious confound is ruled out directly: `div_reach` is
uncorrelated with workspace extremity (ρ = −0.23, p = 0.56), with grasp-pose spread (p = 0.26) and with
episode length (p = 0.23) — it correlates only with the other divergence measures, as it should. So
"the demonstrations disagree here" is not standing in for "this spot is at the edge of the reachable
area".

**ACT's own training log supplies the missing half of the mechanism.** ACT has exactly one component
for representing multimodal actions — the CVAE latent — and on this dataset
[**its KLD collapsed to 0.0002**](phase07_training.md), because 45 single-operator episodes are almost
unimodal. So the latent went unused, and ACT was in effect a plain L1 chunk regressor. **L1 regresses
toward the conditional median**: where demonstrations genuinely disagree about the approach, the median
path is one no demonstrator took, and it can thread between the modes and miss the cylinder. SmolVLA's
flow matching *samples* the conditional distribution instead of averaging it, so the same spread is
represented rather than collapsed. That is the inductive-bias difference in one line: **the two
families do opposite things with action multimodality, and this dataset's divergence is concentrated
exactly where it hurts one and helps the other.**

**Read this as a hypothesis, not a result.** Three reasons: *n* = 9 positions; the table is 10 metrics
× 4 targets = **40 tests**, where ~2 hits at p < 0.05 are expected by chance alone, and SmolVLA's
p = 0.015 is one hit; and 2 trials per cell. What makes it worth keeping rather than discarding is that
it was not fished for — it is the same ACT↔SmolVLA opposition the W test found independently, now with
a candidate cause attached. **The test is a prediction**: if demonstration divergence during the reach
is really what separates them, then deliberately collecting *more varied* approach demonstrations
should help SmolVLA and hurt ACT. That is a cheap, directional, falsifiable experiment for the next
collection round.

### What this changes about collection

- **Report Q2 per family, never pooled.** A pooled per-position number averages away a real effect.
- **Per-position outcomes in round 2.** Position is not a nuisance variable to average over.
- **The "best" demonstration set may differ by policy.** If the data-quality score the programme is
  chasing is policy-dependent in this way, a single curated subset serving all three families may not
  exist — which is the bound finding 4 puts on the model-free line in [08](../08_data_quality_research.md).

## 11.7 — The leftward miss, and why it is probably not the arm

11 of 41 failures carry a direction: **6 left, 3 top, 2 right**. Phase 0.6 saw a systematic leftward
insertion offset that turned out to be a `shoulder_pan` calibration drift
([phase06](phase06_camera_ablation_training.md)), so a leftward skew deserves a hardware check.

**The split by policy argues against hardware:**

| | off left | off right | off top |
|---|---|---|---|
| ACT | 3 | 1 | 1 |
| π0.5 | 3 | 0 | 1 |
| SmolVLA | **0** | 1 | 1 |

A calibration offset is a property of the robot and would bias all three families equally. SmolVLA
missed left zero times.

**Settled: it is not the arm.** The §10.6 preflight replay was run before the rollouts and passed, so
the calibration frame was good going in — the Phase 0.6 diagnosis does not recur here. The leftward
skew is therefore a **learned** bias in ACT and π0.5, which makes it a data property rather than a
hardware one: both families place the gripper short on the same side, on demonstrations that do not.
Worth checking against the per-position grasp bearings in round 2, since a bias that grows with
|shoulder_pan| would point at the extremes of the workspace rather than at the policy.

## 11.8 — What round 1 settles

- **Checkpoints.** ACT: 40k and 60k are indistinguishable, so **use 40k** and take the 33% compute
  saving (prediction 1, confirmed — this is the one clean win for the dial grid). SmolVLA: **use 20k**,
  not the loss-optimal 5k. π0.5: unsettled, because neither checkpoint works; 10k is ahead on progress.
- **Held-out loss does not rank policies on this task.** It inverted the family ordering completely. The
  §10.7a proposal — un-normalised L1 in degrees — would not have fixed this either, since the failure is
  per-frame-vs-rollout, not units. **Per-stage loss plus a smoothness term is the thing to try** (§11.3).
- **Nothing here is a usable policy.** The best arm is 33% with 9 trials. The gap to a deployable policy
  is a data problem before it is a model problem: 50 episodes, one operator, one grasp style, no failure
  recovery anywhere in the training set.
- **Round 2 cannot just be "more trials of the same".** At these rates, measuring 33% to ±9 points is
  precision about a policy nobody wants to deploy. §12 is revised accordingly.

---

# 12 — Round 2 plan, agreed 2026-10-12

Supersedes the original sketch, which is kept as §13. §11.8's conclusion drives it: the best arm is 33%, so more trials
would buy precision about a policy nobody would deploy. **This round spends its effort on data, not on
measurement.** **§13** stays below as the design to return to once a policy earns it.

| Step | What | Time | Status |
|---|---|---|---|
| 1 ✅ | **Pre-register the divergence prediction** (§12.1) | 0 | committed 2026-10-12, before collection |
| 2 ✅ | **D1: 50 full-task episodes, continuously randomised, deliberately varied** (§12.2) | done 2026-10-13 | 50 ep / 38,389 frames · **P4 PASS** |
| 3 | **D1-perturb: ~30 perturbed / recovery episodes** (§12.3) | ~1.5 h | — |
| 3b | **D2: compositional collection, 130 stage episodes** (§12.4) | **to be timed** | — |
| 4 ◐ | **Processing: head trim + 2 s tail cap, jerk, per-stage loss** (§12.5) | trim ~1 h, local CPU | **trim run 2026-10-13**; metrics pending |
| 5 | **Retrain ACT (local, ~2.2 h) and SmolVLA (Colab, ~21 h)**, both fresh on D1 (§12.6) | ~23 h GPU | notebooks drafted; **local GPU driver broken, see §12.6** |
| 6 | **π0.5 frozen-feature probe**, during training (§12.7) | ~30 min | — |

**Changed since the plan was agreed:** the stage split came back as **§12.4**, run as its own
experiment (D2) with a time-and-performance hypothesis attached, rather than as a replacement for
§12.3's perturbed episodes — the two now test different things and both are recorded. Still dropped: a
third π0.5 training run before §12.7 answers whether more steps can possibly help. **Deferred:** the second location (§12.8), HG-DAgger infrastructure, and the 126-trial
success-rate pass (§13).

## 12.1 — Pre-registration: committed before D1 is recorded

[08 §A.2](../08_data_quality_research.md) asks for predictions committed before the comparison is read.
§11.6 found that demonstration divergence during the reach correlates **−0.63 with ACT** progress and
**+0.79 with SmolVLA** — and as a single planned test, the ACT-minus-SmolVLA gap against reach
divergence gives **ρ = −0.814, p = 0.012**. D1 deliberately raises that divergence, so it is a test of
the mechanism rather than only a bigger dataset.

**The mechanism being tested.** ACT's only component for representing multimodal actions is the CVAE
latent, and its KLD collapsed to 0.0002 on D0 ([training §2](phase07_training.md)) — so ACT behaved as
a plain L1 chunk regressor, and L1 regresses toward the conditional *median*, which where
demonstrations disagree is a path no demonstrator took. SmolVLA's flow matching samples the conditional
distribution instead of averaging it.

**Predict the *gap*, not the levels.** D1 changes several things at once — more data, varied approaches,
tail kept, perturbed episodes — and all of those help both families. Only the **differential** isolates
the mechanism.

| # | Prediction | How it is scored |
|---|---|---|
| **P1** | **The ACT−SmolVLA progress gap moves negative.** Round 1 pooled: ACT 2.22 − SmolVLA 2.17 = **+0.06**. Predict **< −0.3** after D1 | round-2 rollouts, same progress scale |
| **P2** | SmolVLA's absolute progress improves more than ACT's | round-2 vs round-1 per-family progress |
| **P3** | **ACT's KLD does *not* collapse to ~0.0002 this time.** If D1 is genuinely more multimodal the latent has something to encode | free, read off the training log |
| **P4** | Measured `div_reach` on D1 exceeds D0's range (0.24–0.39) | [`analyze_phase07_positions.py`](../scripts/analyze_phase07_positions.py) on D1 |

**If P4 fails, P1–P3 are not tested** — the manipulation did not take, and the round is a data-volume
experiment rather than a mechanism test. Check P4 first, immediately after collection.

**What would falsify the mechanism:** the gap moves *positive* (ACT benefits more from varied data than
SmolVLA), or P4 holds while P1 and P3 both fail. Either would say the §11.6 correlation was a
nine-point coincidence.

### P4 outcome — **PASS**, verified 2026-10-13

D1 recorded as `so101_tray_cylinder_varied_20261013_082911`: 50 episodes, 38,389 frames, median
episode 25.7 s (D0: 23.5 s). Checked with
[`check_d1_divergence.py`](../scripts/check_d1_divergence.py) before anything else was done with the
data, as this section requires.

**The measurement had to be position-controlled or it would have been meaningless.** In D0 the five
demos behind a spot sat at the *same* position, so divergence between them was pure approach
variation. D1 randomises position continuously, so any two episodes differ partly because the cylinder
was elsewhere — a naive whole-dataset divergence would be inflated by that and would "confirm" P4 for
the wrong reason. So divergence is measured within **k = 5 nearest neighbours in grasp-pose space**
(the grasp pose fixes where the cylinder was), and the metric is validated against D0 first:

| | div_reach | neighbourhood position spread |
|---|---|---|
| D0, native 9 blocks of 5 | 0.239–0.389 (mean **0.317**) | 2.78° |
| D0, same kNN metric | 0.231–0.406 (mean **0.322**) | 2.50° |
| **D1, kNN** | 0.344–0.683 (mean **0.478**) | 4.27° |

The kNN mean reproduces the block mean on D0 to **1.4%**, so the two datasets are on one scale.
**D1 − D0 = +0.156, permutation p < 0.0001**, and D1's *minimum* neighbourhood (0.344) sits near D0's
*maximum* (0.406).

**The one real threat was tested, not caveated.** D1's neighbourhoods span 1.71× wider positions, so
some of the rise could be position rather than approach. Restricting both datasets to the pose-spread
range they share (2.57–4.64°, D0 n = 24, D1 n = 31) gives **+0.113, p < 0.0001 — 72% of the
difference survives**. The manipulation genuinely raised approach variation.

**One episode's grasp is mis-detected, and the cause matters for §12.5.** `grasp_seated_index`
assumes the approach-opening falls in the episode's first half and the release-opening in the second —
true for all 50 D0 episodes, and for 49 of 50 D1 episodes. **D1 episode 39 breaks it**: its gripper
trace is perfectly normal (open at f120–160 → hold on the cylinder at f200–320 → release at the
fixture at f360–510 → rest), but the *release* begins at f342, just before the 724-frame episode's
midpoint at f362. Both of the detector's "peaks" then land inside the same release opening, and it
reports the fixture pose as the grasp pose — grasp pan and seated pan come out bit-identical at
+19.38°, with a grasp→seated span of 22 frames against a median of 24% of the episode.

**This is a side effect of the longer tail**, which is otherwise the change we wanted: a 3.2 s hold
plus the retreat pushes the release earlier in relative terms, and the halves assumption was
calibrated when tails were 1.5 s. **It will get worse as tails grow, and §12.5's per-stage held-out
loss depends on exactly this segmentation** — so the detector needs a more robust rule (find the two
openings as contiguous runs above a level derived from the trace, then take the first and the last)
before the per-stage work, not after.

P4 is not affected: dropping episode 39 moves the result from **+0.156 to +0.166** (p < 0.0001;
matched on position spread, +0.113 → +0.122). It also removes an apparent placement outlier that was
never real — with episode 39 excluded, D1's grasp-pan range is **−8.8…+7.3°** against D0's
−11.0…+7.4°, i.e. very slightly *narrower*, which is the expected result when the extra divergence
came from how the operator approached rather than from where the cylinder was put.

**P1–P3 are therefore live.** Measure them after round-2 training and rollouts.

> **A trap when re-running the verifier on a second dataset.** `verify_phase07_dataset.py` writes its
> figures to `Details/figs/phase07_idle.png` and `phase07_tray_spots.png` under **fixed names**, even
> when `--out` points elsewhere — so running it on D1 silently overwrites D0's, which
> [phase07_data_verification.md](phase07_data_verification.md) embeds. D1's copies are kept as
> `phase07_d1_*.png`; restore D0's from git if the verifier is re-run.

### The other two collection changes also took

| | D0 | D1 | |
|---|---|---|---|
| Idle **head** before first motion | 71 frames (2.38 s) | 66 frames (2.20 s) | unchanged, as expected — the head trim handles it |
| Idle **tail** after last motion | 45 frames (1.50 s) | **95 frames (3.17 s)** | **2.1× — the deliberate hold worked** |
| 9 tray-spot blocks separable | **27×** | **1×** | **the block structure is gone** |

![D1 tray spots and idle](figs/phase07_d1_tray_spots.png)

The tray-spot result is the one to read twice. `verify_phase07_dataset.py` reports "9 tray-spot blocks
of 5 are separable" as a WARN because for D0 the *presence* of block structure was the finding; for D1
the *absence* is the success. Within-block spread 2.92° against between-centre 3.90° is **1×
separation** — i.e. no blocks, which is exactly what continuous randomisation was for. The companion
"position is confounded with session time" WARN is hard-coded text from the D0 era and does not apply.

## 12.2 — D1: 50 full-task episodes, continuously randomised

**Three changes from D0, and nothing else.** Everything not listed is deliberately identical, because
D1 must stay mergeable with D0: same cameras and camera *names*, same calibration frame, same task
string, 30 fps, same fixture and tray.

1. **Cylinder position is randomised continuously within the tray** — not the nine marked spots. This
   is the highest-value change in the round. It kills the block confound that has now appeared twice
   (Phase 0.5 and 0.7), removes the nine-point memorisation target, and raises reach divergence as a
   side effect, because different positions force different approaches. **Nothing analytic is lost:**
   the grasp pose recovers cylinder position to ~0.3° (§11.6), so position can be binned or regressed
   after the fact. The nine marked spots stay, for **evaluation only**, where reproducibility matters.
2. **Vary the approach deliberately** — see the box below.
3. **Keep a deliberate tail.** After the cylinder is seated and released, **hold still for ~2 seconds**
   before pressing →. D0's episodes already carry ~1.4 s of tail; the trim is what removed it (§12.5).
   Making it deliberate rather than operator lag is what turns it into a usable "task complete" signal.

> ### Vary the path, not the outcome
>
> This inverts §5's "one grasp style", on purpose, and it is the manipulation §12.1 tests. **Vary:**
> the approach angle to the cylinder, the standoff height before descending, the speed of the
> approach, the path taken from tray to fixture. **Hold fixed:** the final seated pose, the success
> criterion, and the smoothness of each individual path.
>
> **The distinction that matters is modes versus noise.** Several distinct, clean ways of reaching the
> same cylinder broaden the conditional action distribution with *valid* alternatives — that is what a
> flow-matching policy can represent. Jitter, hesitation and unlabelled mid-course corrections are
> noise, and they would hurt both families; they are also what the smoothness literature in
> [08 §A.1](../08_data_quality_research.md) (RINSE, PSD scores) penalises. **Each path smooth; the set
> of paths varied.**

**Timing, from D0's actuals.** D0's 50 episodes ran ~1.5 h, with a median episode of 23.5 s — so
overhead, not episode length, dominates. Budget **~2.5 h**: 20 min preflight, ~1.5–2 h recording,
20 min verify and push. Record the actual wall clock in §8's log table.

`episode_time_s` is raised 30 → **35**. D0's cap was never binding (median 705 frames, max 878 of a
900-frame cap, 0/50 at the cap), but a varied approach plus a 2 s tail eats the margin, and the value
is only a ceiling — episodes still end when → is pressed.

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/so101_tray_cylinder_varied --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=50 --dataset.episode_time_s=35 --dataset.reset_time_s=5 --dataset.fps=30 --dataset.push_to_hub=true --dataset.private=true --display_data=true
```

- **The task string is byte-identical to D0's** — required, both for merging and so the VLAs see one
  instruction.
- **`reset_time_s` 3 → 5.** Placing the cylinder at a *free* position and varying the next approach
  needs more than 3 s, and a rushed reset is how position creeps back into a pattern.
- **Do not re-run `lerobot-calibrate`.** A new frame makes D1 unmergeable with D0, which defeats the
  point. [calibration/README.md](calibration/README.md).
- 0.6.1 appends a timestamp, giving `so101_tray_cylinder_varied_<stamp>` — **record the full stamped
  name in §8** before doing anything else with it.
- Preflight first: §10.6's replay and top-camera check, which also confirms the frame is still D0's.

## 12.3 — D1-perturb: ~30 perturbed and recovery episodes

**This is not the stage split** — that is §12.4, now run as its own experiment. The two were originally
alternatives; they are now complementary, because they cover different things: §12.4 adds clean
single-stage data, while the perturbed episodes below add *in-context recovery*, which no stage clip
contains. Two design hazards drove the original preference, and both still apply to §12.4 — recorded
here because that is where the reasoning lives:

1. **ACT is not language-conditioned.** Verified in the source — it takes images and state only, no
   `task` input. So ACT cannot tell a stage clip from the matching portion of a full episode by
   instruction; only the VLAs could.
2. **Worse, a stage clip injects multimodality exactly where ACT averages.** An insertion clip starts
   from "holding the cylinder, at rest"; in a full episode that same state is passed through *at
   speed*. One observation, two very different action distributions — which is precisely the failure
   §11.6 diagnosed. It would manufacture the problem.

**Perturbed full episodes give stage coverage without the split**, and they supply the
failure-recovery data that was otherwise deferred to R3. Run the whole task, but create the hard case
mid-episode:

| Perturbation | Targets | ~n |
|---|---|---|
| Nudge the cylinder after it is sighted, before the grasp | re-approach, close-range reach | 10 |
| Let the grasp slip or miss, then re-approach and succeed | **grasp recovery** — 14 of 41 round-1 failures | 10 |
| Begin a mis-aligned insert, detect it, lift and re-seat | **insert recovery** — 9 of 41 failures | 10 |

**Record them into D1 with `--resume=true`,** not as a separate repo. They become episodes 50–79 of the
same dataset, which guarantees compatibility and keeps one dataset to manage — and they stay ablatable,
because `--dataset.episodes` selects subsets by index. Training can then compare `0–49` against `0–79`
with nothing else changed.

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && STAMPED=so101_tray_cylinder_varied_CHANGEME && lerobot-record --resume=true --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/$STAMPED --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=30 --dataset.episode_time_s=45 --dataset.reset_time_s=5 --dataset.fps=30 --dataset.push_to_hub=true --dataset.private=true --display_data=true
```

Replace `CHANGEME` with the stamp from §12.2's recorded name. `episode_time_s` is 45 here: a recovery
takes longer than a clean run. **Keep the ~2 s tail.** Log which perturbation each episode used — that
column cannot be reconstructed afterwards, which is the same mistake `phase07_log.csv` already
represents.

## 12.4 — D2: compositional collection, one dataset per stage

**The hypothesis.** Splitting the task into stages and collecting each one separately **(a) takes less
operator time than recording whole episodes, and (b) produces a better policy** — because the budget
goes to the stages that actually fail instead of being spent uniformly on a trajectory whose middle is
already solved. Round 1 is the motivation: 14 of 41 failures were at the grasp and 9 at the insert,
while **no trial in 54 failed in transport**.

| Stage | Episodes | What it covers |
|---|---|---|
| **reach** | 10 | home → gripper positioned above the cylinder in the tray |
| **grasp** | **50** | close on the cylinder and lift clear — round 1's single biggest failure |
| **transport** | 10 | carry from above the tray to above the fixture — already reliable |
| **insert** | **50** | descend, seat in the top-left blue hole, release — round 1's second failure |
| **retrieve** | 10 | return to the home pose with an empty gripper |
| | **130** | |

The 50/10 split is the whole point: five times the data on the two stages that fail, and a token amount
on the three that do not.

**The task string stays the same for every stage.** That is deliberate and it sidesteps the problem that
sank the earlier stage-split proposal (§12.3): ACT takes no language input, so a per-stage instruction
would have been invisible to it. With one instruction, a stage dataset is simply *more data in that part
of the state space*, which every family can use — the policy infers the stage from what it sees, which
is what it has to do at rollout anyway.

### Record all five stages into ONE dataset — this is not optional

**`lerobot-train` cannot train on several datasets at once.** `LeRobotMultiDataset` is declared but
disabled: `train.py:269` raises `NotImplementedError("LeRobotMultiDataset is not currently
implemented.")`, and `datasets/factory.py:124` says the same. So five separate `repo_id`s would need a
merge step before any of them could be used together.

The command below has no `--resume` and no `--dataset.no_stamp`, so **each invocation would create its
own timestamped dataset** — five runs, five datasets, none of them trainable together. The fix is the
pattern §12.3 already uses: let the **first** stage create the dataset, then append the other four with
`--resume=true` and the stamped name. Stages then live at known episode-index ranges in one dataset, and
`--dataset.episodes` selects any combination — which is also what makes the ablation possible.

| Run | Stage | `NEP` | `--resume` | episodes |
|---|---|---|---|---|
| 1 | reach | 10 | — (creates it) | 0–9 |
| 2 | grasp | 50 | `true` | 10–59 |
| 3 | transport | 10 | `true` | 60–69 |
| 4 | insert | 50 | `true` | 70–119 |
| 5 | retrieve | 10 | `true` | 120–129 |

**Write these ranges down as you go.** They are the only record of which episode is which stage — the
same gap `phase07_log.csv` already represents, and here it would make the dataset uninterpretable.

### The command

Run 1 as given, which creates `so101_tray_cylinder_compositional_<stamp>`:

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/so101_tray_cylinder_compositional --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=10 --dataset.episode_time_s=20 --dataset.reset_time_s=5 --dataset.fps=30 --dataset.push_to_hub=false --dataset.private=true --display_data=true --dataset.streaming_encoding=true --dataset.encoder_threads=2
```

Runs 2–5 add `--resume=true` and the stamped name, changing only `NEP`:

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && STAMPED=so101_tray_cylinder_compositional_CHANGEME && NEP=50 && TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0 && WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0 && lerobot-record --resume=true --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" --dataset.repo_id=HALDijkstraaa/$STAMPED --dataset.single_task="Pick up the white cylinder from the green tray and place it in the top left blue hole of the black fixture" --dataset.num_episodes=$NEP --dataset.episode_time_s=20 --dataset.reset_time_s=5 --dataset.fps=30 --dataset.push_to_hub=false --dataset.private=true --display_data=true --dataset.streaming_encoding=true --dataset.encoder_threads=2
```

- **`--dataset.push_to_hub=false`** throughout, then one push at the end with
  [`push_dataset.sh`](../scripts/push_dataset.sh) — which is also what creates the codebase-version tag
  that `hf upload` alone does not (the BLOCK D1 hit).
- **`--dataset.streaming_encoding=true --dataset.encoder_threads=2`** encodes during capture instead of
  writing PNGs first, making `save_episode()` near-instant. That matters more here than anywhere else:
  130 short episodes pay the per-episode cost 130 times, and the per-episode cost is exactly what the
  time hypothesis is about.
- **`episode_time_s=20`** is a generous ceiling for a stage — D1's *whole* episodes ran a median 25.7 s,
  so a single stage should land in single-digit seconds. Press → as soon as the stage is done.
- **Position: randomise continuously**, as §12.2 does, for reach / grasp / insert. Those three are the
  stages where cylinder position matters.
- **Keep the ~2 s tail on `insert` and `retrieve`** — those are the stages that end the task, so they
  carry the "now hold still" signal (§12.5).

### Time the collection — it is half the hypothesis

**Record wall clock per stage**, not just total. The claim is that compositional collection is cheaper,
and the baseline is D1: **50 full episodes in ~1.5–2 h recording**, 38,389 frames.

A rough forward estimate, to be checked against the real numbers: at ~2 s for reach / transport /
retrieve, ~3 s for grasp and ~4 s for insert, 130 stage episodes is **~12,000 frames — under a third of
D1's 38,389**. So the compositional set is expected to be much cheaper *and* much smaller. That cuts
both ways and is why the comparison needs care:

> **Match on operator time, not on episodes or frames.** "130 stage episodes beat 50 full episodes" is
> not a result if the two cost different amounts to collect. The interesting claim — and the one the
> hypothesis actually makes — is **"the same hours of operator time produce a better policy"**. So log
> the hours, and if D2 comes in at half of D1's time, the honest comparison either collects twice as
> much D2 or reports the result as "half the time, equal performance", which would already be a win.

### What to watch in the rollouts

The known risk in a stage split is the **start-state distribution**: a stage clip begins at rest, while
in a full episode that same state is passed through at speed. One observation, two action distributions.

**This makes a specific, falsifiable prediction worth pre-registering now:** if the mismatch matters,
the failure will be **hesitation or a stall at stage boundaries** — the arm arriving at the grasp pose
and pausing before closing, or stopping above the fixture before descending. That is visually distinct
from round 1's failure modes (missing the cylinder, mis-seating), so it can be scored separately in the
round-2 log. If it does not appear, the start-state concern was unfounded and compositional collection
is clean.

**Train D2 against D1 at matched steps**, and keep D1 and D2 as separate training runs rather than
merging them, so the comparison stays interpretable. Merging is a later question, and a good one — a
policy trained on *both* is the obvious third arm if either alone falls short.

## 12.5 — Processing: trim the head, keep a 2 s tail, add two metrics

### The trim, and the one rule that changed

[`make_trimmed_dataset.py`](../scripts/make_trimmed_dataset.py) trimmed **both** ends through round 1.
The head trim stays — [phase06 F3](phase06_camera_ablation_training.md) showed the idle lead-in is an
attractor the policy cannot escape at the start. **The tail is the opposite**: it is the only place a
policy can learn that the task is finished, and §11.5 measured the cost of removing it (SmolVLA moves
3.6× as much as ACT in the final second).

The script now takes **`--tail-cap N`**, keeping up to N frames after the last motion. The default is
unchanged, so D0's trimmed dataset is still reproducible from this script; round 2 passes **60** (2.0 s
at 30 fps).

**Capped rather than kept whole, deliberately.** D1's recorded tails run **47–191 frames (1.6–6.4 s)**.
Keeping all of them would teach a terminal hold that varies 4× in length — which is one more source of
action multimodality at exactly the state where the policy should be doing one unambiguous thing.
Capping gives every episode the *same* ~2 s of stillness, which is what makes it a learnable signal.

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && python scripts/make_trimmed_dataset.py HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911 HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed ~/.cache/huggingface/lerobot/HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed --tail-cap 60
```

**It runs locally and needs no GPU** — measured, since the question came up. `LeRobotDataset` decodes
at **60 frame/s on CPU** here (the verification report's 9 frame/s figure is *random* access, which
re-seeks per item; the trim reads each episode in order). Round 1 did this step on Colab; there was
never a reason to.

**Budget ~1 h, not the ~8 min the decode rate suggests.** Decoding is not the cost — every kept frame
is also re-encoded into the destination videos, and that dominates. The run is CPU-bound and
single-process, so it neither needs nor benefits from the GPU being up. Push afterwards with [`push_dataset.sh`](../scripts/push_dataset.sh), which creates
the codebase-version tag that a plain `hf upload` does not.

### `eval_chunk_jerk` in wandb

§11.3a has the patch and the reasoning: the training loop needs no change, because
`MetricsTracker.update_metrics` auto-registers a meter for every new scalar the policy returns. Use the
eval-branch version — it works for every family and costs one denoising loop per eval batch.

**Pre-registered**: SmolVLA 5k should show **~2× the jerk of 20k**, matching the 2.0× measured on the
robot in §11.3 — *visible with no robot attached*. If it holds, the dial grid can be scored on loss
**and** jerk, and the offline metric finally tracks the thing loss misses.

### Per-stage held-out loss — and a detector fix it needs first

Pool the held-out frames by task stage and report loss per stage instead of one pooled number.
**Pre-registered prediction** from §11.3: SmolVLA 5k and 20k should separate at the **grasp and insert**
stages specifically, where tolerance is tightest, even though 5k wins the pooled number. This is also
what validates the jerk metric — without it there is one unproven proxy and no way to check it.

> **`grasp_seated_index` must be made robust before this, not after.** The segmentation this depends on
> is the detector in [`analyze_phase07_eval.py`](../scripts/analyze_phase07_eval.py), and §12.1 found it
> failing on D1 episode 39: it assumes the approach-opening falls in the episode's first half and the
> release in the second, which **the longer tail breaks**. It is 1/50 on D1 and will get worse as tails
> grow. The fix is to find the two openings as contiguous runs above a level derived from the trace and
> take the first and the last, rather than splitting at the midpoint — and to re-validate it against all
> 100 D0+D1 episodes, confirming it reproduces the existing D0 numbers, before anything depends on it.

## 12.6 — Training: ACT locally, SmolVLA on Colab, both fresh on D1

**Two runs, both fresh, both on D1 alone.** D1 is the same size as D0 (50 episodes), so this is
**size-matched to round 1** and the single variable is the data's character — which is exactly what
§12.1 pre-registered. Notebooks are *derived* from the round-1 ones by
[`make_round2_notebooks.py`](../scripts/make_round2_notebooks.py) rather than rewritten, so the install,
auth, resume and disk-watchdog machinery is identical and only the configuration differs.

| | ACT | SmolVLA |
|---|---|---|
| Notebook | [colab_phase07_r2_act.ipynb](../colab_phase07_r2_act.ipynb) | [colab_phase07_r2_smolvla.ipynb](../colab_phase07_r2_smolvla.ipynb) |
| From | **scratch** | **`lerobot/smolvla_base`** |
| Batch × steps | 8 × **40,000** | 64 × **30,000** |
| Checkpoints | every 10k → 4 | every 5k → 6 |
| Where | **local 5080** | **Colab L4** (or A100) |
| Peak VRAM | **3.7 GiB** of 15.5 | **16.5 GiB** of 24 |
| Est. runtime | **~2.2 h** | **~21 h** |
| Output | `act_tray_cylinder_s1000_d1` | `smolvla_tray_cylinder_s1000_d1` |

### Why fresh, and not continued from the round-1 checkpoint

Warm-starting would carry D0's low-divergence structure into the run and confound the comparison — the
policy would already have absorbed the thing the experiment is trying to vary. It would also make
**P3** unmeasurable: ACT's KLD collapsing or not collapsing is only interpretable from scratch.

The same argument rules out a D0+D1 mixture *for this round*: §11.6 predicts the mixture would help ACT
and hurt SmolVLA relative to D1 alone, so pooling dilutes the manipulation in opposite directions per
family. A 100-episode mixture arm is a good **third** run once the clean test is in — it answers "does
more data win", which is a different question from "does varied data win".

### Why ACT runs locally and SmolVLA cannot

**ACT is faster here than on the L4.** [phase06 §5](phase06_camera_ablation_training.md) measured this
exact machine at **5.4 it/s** at batch 8 with two cameras, 3.7 GiB — against the L4's 4.47 it/s in
round 1. 40,000 steps is **~2.1 h** plus ~10 min of periodic eval. Note that figure is fp32;
`--policy.use_amp=true` is ~1.3× faster but changes the numerics of the reference recipe, so **leave it
off** — round 1 did.

**SmolVLA cannot run here at the round-1 recipe.** Batch 64 peaked at **16.5 GiB** on the L4, and the
5080 has **15.5 GiB** usable. The obvious workaround does not exist: **lerobot has no gradient
accumulation** — `Accelerator()` is constructed without `gradient_accumulation_steps`
(`lerobot_train.py:257-262`) and there is no config field for it — so batch 64 cannot be simulated from
batch 32. Dropping to batch 32 would change the recipe, and with it SmolVLA's absolute performance,
which puts a confound straight into the pre-registered ACT−SmolVLA gap. So SmolVLA goes to Colab at
batch 64, unchanged.

### 30,000 steps gives both arms from one run

Round 1 found SmolVLA's **20k** checkpoint better on the robot than its loss-optimal 5k, so round 2
trains longer. But 20k is also the step count round 1 compared at — so **`save_freq=5000` writes the
20k checkpoint on the way to 30k**, and one run yields both the matched comparison *and* the
train-longer arm. ACT is the mirror: round 1 confirmed 40k ≈ 60k (P1), so 40k is both the matched
checkpoint and the stopping point, and the final 20k of round 1's budget is simply dropped.

**~21 h is past a comfortable Colab session.** Round 1's 14.2 h SmolVLA run survived; the π0.5 run did
not and lost 17 h to a dead session. The notebook's Part C resume rebuilds from the last Hub checkpoint,
so a disconnect costs at most `save_freq` = 5,000 steps ≈ 2.8 h. Two ways to de-risk it: take an **A100**
if the runtime offers one (round 1 measured ~2.5–3.5× over the L4 on π0.5), or **stop at 20k** — that
secures the matched comparison in ~14 h, and Part C resumes to 30k later with no loss.

### The local ACT run, without the notebook

The notebook works locally, but ACT needs none of its Colab scaffolding — this is the whole run:

```bash
cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && HOLDOUT="[9,19,29,39,49]" && EPISODES="[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,9,19,29,39,49]" && lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=true --policy.repo_id=HALDijkstraaa/act_tray_cylinder_s1000_d1 --policy.private=true --save_checkpoint_to_hub=true --dataset.repo_id=HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed --dataset.episodes="$EPISODES" --dataset.eval_split=0.1 --eval_steps=5000 --batch_size=8 --steps=40000 --num_workers=8 --seed=1000 --save_freq=10000 --log_freq=200 --output_dir=outputs/train/act_tray_cylinder_s1000_d1 --job_name=act_tray_cylinder_s1000_d1 --wandb.enable=true --wandb.project=phase07-policy-transfer
```

The episode list is the holdout reordered into the tail, exactly as round 1 did it — see the note in
the notebook's config cell for why the *rationale* changed even though the mechanism did not.

> ### ⚠ The local GPU is currently unusable — fix this before the ACT run
>
> `nvidia-smi` fails with *"couldn't communicate with the NVIDIA driver"*. The cause is a kernel
> upgrade that the NVIDIA kernel-module package did not follow: the machine is running
> **7.0.0-38-generic**, but `linux-modules-nvidia-595-open-generic-hwe-24.04` is still pinned at
> **7.0.0-31**. `/lib/modules/7.0.0-31-generic` has the full set (`nvidia.ko`, `nvidia-drm.ko`,
> `nvidia-modeset.ko`, `nvidia-uvm.ko`, `nvidia-peermem.ko`); `/lib/modules/7.0.0-38-generic` has only
> the in-tree `nvidiafb` and `nvidia-wmi-ec-backlight`, which are not the driver. `dkms` is not
> installed, so nothing rebuilds it automatically.
>
> ```bash
> sudo apt install --reinstall linux-modules-nvidia-595-open-generic-hwe-24.04 && sudo reboot
> ```
>
> The candidate version is already 7.0.0-38.38~24.04.4+1, so this is an upgrade rather than a new
> install. **Booting 7.0.0-31 from GRUB's advanced menu is the zero-risk fallback** — that kernel still
> has working modules. Nothing in §12.5 is blocked either way: the trim is CPU-only.

## 12.7 — The π0.5 gate: can more steps possibly fix the reach?

π0.5 is **0/18 excluding luck**, with 11 of 18 trials failing at reach, and it costs the most compute
and 11.55 GB per checkpoint. §11.4 gives three candidate causes; only one of them is fixed by more
training. **Run this before committing another training run** — ~30 minutes, no training, and it can
be done while ACT and SmolVLA train:

**Probe whether the frozen vision encoder resolves the cylinder at all.** `freeze_vision_encoder=true`
means π0.5's SigLIP features never adapted to this rig. Extract them for held-out frames and fit a
linear regression from features to the cylinder's position (recoverable from the grasp pose, §11.6).

| Outcome | Reading |
|---|---|
| Features predict position well | The representation is fine; the action expert is simply undertrained. **More steps is the right fix.** |
| They do not | **No amount of action-expert training can fix the reach** — the information is not in the representation. Unfreeze the encoder, or drop π0.5 for this phase |

Either way it converts "π0.5 needs more steps" from an assumption into a decision.

## 12.8 — Deferred, and why each is safe to defer

- **Second location.** Still the only direct test of the portability claim this whole phase is built on
  (§9 item 3), and still untested. Deferred because D1 must stay mergeable with D0 and a location
  change adds a variable mid-dataset. **Do not defer it twice** — it belongs in round 3, either as a
  held-out eval site or as its own small dataset.
- **HG-DAgger infrastructure.** §12.3's perturbed episodes deliver the recovery data a round earlier
  without it. Revisit when recovery needs to be *interactive* rather than demonstrated.
- **The 126-trial success-rate pass** (§13). Right again once a policy clears ~50%.
- **Pure stage-split clips.** Only if §12.3's perturbed episodes under-cover the grasp and insert
  stages, and only with the start-state problem solved first.


# 13 — The deferred success-rate design, for when a policy earns it

> **Superseded for now by §12.** 126 trials to measure each policy's success rate to ±9 points is the
> wrong next move at a 33% best arm (§11.8) — it would buy precision about a policy nobody would
> deploy. Kept in full because it becomes the right design again as soon as a policy clears ~50%, and
> because its three components are what round 3 will need.

| | Round 1 (§10) | Round 2 |
|---|---|---|
| Question | Which checkpoint per policy | **Success rate** per policy, and off-grid generalisation |
| Arms | 6 (2 checkpoints × 3 policies) | 3 (the winners) × {on-grid, off-grid} |
| Trials | 9 per arm, **54** | 30 per policy on-grid + 12 off-grid, **~126** |
| Resolution | ±16 points — ranks, does not measure | **±9 points** at 50%, per [phase06 §4b](phase06_camera_ablation_training.md) |
| Time | ~2–2.5 h | ~4–5 h, two sessions |

Three things it adds:

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

**Whenever round 2 runs, write down the `T1`–`T9` mapping first** (§11.6). It is a minute of work and it
is the only thing blocking the per-position analysis.

**`phase07_log.csv` is still missing** and round 2 does not need it, but the per-episode `success` and
`stage` of the *training* data are not recoverable ([verification §6](phase07_data_verification.md)) —
so if any analysis wants to relate rollout outcomes to the demonstration quality at the same tray spot,
that is gone for these 50 episodes and is a reason to record it for the next dataset.
