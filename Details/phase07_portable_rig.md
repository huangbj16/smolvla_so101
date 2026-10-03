# Phase 0.7 — portable rig: bird's-eye top camera, and a new dataset

Run book for the rebuilt rig and the data collection that follows it. Everything — both arms, both
cameras, the tray and the fixture — is now mounted on a single acrylic board, and the top camera has been
raised from a third-person view to a **bird's-eye overview of the board**.

Plan: [08 §A.4](../08_data_quality_research.md) · Task: [toolkit_task_card.md](toolkit_task_card.md) ·
Calibration: [calibration/README.md](calibration/README.md) · Predecessors:
[phase05_camera_test_results.md](phase05_camera_test_results.md),
[phase06_camera_ablation_training.md](phase06_camera_ablation_training.md). Written 2026-10-03.

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

**Shuffle the positions.** Phase 0.5 recorded position blocks (P1 = ep 0–4 … P10 = ep 45–49), which
confounded position with session time and would have put a whole position in lerobot's default holdout.
For Phase 0.7: visit every marked tray spot once per round, in a **written-down shuffled order per
round**, repeating until 50 episodes are done.

**Start moving the instant the episode starts.** A static lead-in becomes an idle attractor the policy
cannot escape; Phase 0.5 measured a median 74 idle frames, and the F3 follow-up found that training on
idle-trimmed data flipped the camera ranking. Trim anyway before training
([`scripts/make_trimmed_dataset.py`](../scripts/make_trimmed_dataset.py)).

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

**Then, in order:**
1. Re-run the Phase 0.5 divergence analysis on this dataset
   ([phase05_second_camera.ipynb](../phase05_second_camera.ipynb)) and put the numbers beside the old
   ones. Confounded, so read it as a sanity check: does the top view stop being the aliased one?
2. Retrain the three-way camera ablation on it. The sharp question is whether `top+wrist` still
   under-reaches — if the degradation was background-driven it should shrink here.
3. Record a second location and compare, which is the only direct test of portability.
4. If the comparison in (1) looks worth doing properly, mount the old third-person camera on the board as
   a third stream and record both views at once. That turns a confounded comparison into a controlled one.
