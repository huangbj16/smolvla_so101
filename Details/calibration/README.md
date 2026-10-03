# Calibration backup — do not lose these

Every file lerobot writes to `~/.cache/huggingface/lerobot/calibration/` is copied here, dated, before it
is replaced. Each folder holds a **complete leader + follower pair**, so any one of them can be restored
on its own.

Live locations: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json` and
`.../teleoperators/so_leader/my_leader.json`.

| folder | frame | what was recorded / trained in it |
|---|---|---|
| [`current/`](current/) | **deployed now** — recalibrated **2026-10-03** | Phase 0.7 onward ([phase07_portable_rig.md](../phase07_portable_rig.md)) |
| [`archive/2026-09-27_phase06_pan_fix/`](archive/2026-09-27_phase06_pan_fix/) | master + `shoulder_pan` +32 counts | nothing recorded; this is the frame the Phase 0.6 policies were *replayed and evaluated* in after the loose-screw fix |
| [`archive/2026-09-16_phase05_06_master/`](archive/2026-09-16_phase05_06_master/) | **the recording-day master** (2026-09-16 20:49) | every Phase 0.5 / 0.6 dataset and every policy trained on them — `so101_toolkit_cylinder_20260917_165544` and all its derivatives |

The leader json is byte-identical in the two archive folders: the leader was not touched between
2026-09-16 and 2026-10-03.

## Why this matters

On Feetech servos `Present_Position = Actual_Position − Homing_Offset`. lerobot stores the homing offset
and the per-joint range here and writes them into the motors' EEPROM on connect. A policy consumes and
emits positions **in the frame its training data was recorded in**, and that dataset's normalization
statistics were computed in the same frame. Change the calibration and every state and action is shifted
by a constant — the old policies are not permanently invalid, but they are wrong by that offset until the
matching frame is restored from the table above.

The calibration step is also **hand-positioned** ("move the arm to the middle of its range and press
ENTER"), and one encoder count is 360/4096 = 0.088°, so a 5° error by hand becomes a permanent 57-count
bias on that joint. Re-running calibration is therefore *not* a safe no-op — which is the whole reason
this folder exists.

## Restoring a frame

Copy the pair back to the live locations, connect the robot, and press ENTER at the
`Press ENTER to use provided calibration file associated with the id my_follower` prompt —
`write_calibration()` then pushes the values into the motors. Verified in `so_follower.py:115` and
`motors/feetech/feetech.py:268`.

```bash
CAL=~/.cache/huggingface/lerobot/calibration
FROM=archive/2026-09-16_phase05_06_master          # or current/ , or the pan_fix folder
cp $FROM/my_follower.json $CAL/robots/so_follower/my_follower.json
cp $FROM/my_leader.json   $CAL/teleoperators/so_leader/my_leader.json
```

Then **replay a recorded episode** from a dataset built in that frame — it is the only acceptance test
that separates "the frame is right" from "it looks right":
[01 Step 4](../../01_setup_robot.md) and
[phase06_camera_ablation_training.md §8](../phase06_camera_ablation_training.md).

---

## 2026-10-03 — full recalibration of both arms, for Phase 0.7

Both arms were recalibrated from scratch when the robot was remounted on the portable acrylic board
([phase07_portable_rig.md](../phase07_portable_rig.md)). This **breaks the Phase 0.5 / 0.6 frame on
purpose**: the arms moved, so the old frame no longer describes the hardware, and Phase 0.7 collects a
new dataset in the new one.

Homing-offset shift from the previously deployed frame (1 count = 0.088°):

| joint | follower Δ | leader Δ |
|---|---|---|
| `shoulder_pan` | **+53 (+4.66°)** | −8 (−0.70°) |
| `shoulder_lift` | −43 (−3.78°) | −17 (−1.49°) |
| `elbow_flex` | +40 (+3.52°) | +27 (+2.37°) |
| `wrist_flex` | +10 (+0.88°) | −41 (−3.60°) |
| `wrist_roll` | −18 (−1.58°) | +5 (+0.44°) |
| `gripper` | +27 (+2.37°) | −4 (−0.35°) |

All of these are inside the few-degrees band hand positioning produces, so the new frame is
self-consistent; it is simply **a different frame**. Recorded range widths also moved: follower
`shoulder_pan` is 2449 counts, down 239 (−21.0°) from the old 2688, and every other joint's width changed
by ≤ 25 counts. The narrower pan range was reviewed and **kept** — it covers the working area of the new
rig, which is smaller than the old taped table.

**Consequences, carried into the phase docs:**
- The Phase 0.5 / 0.6 datasets and the ACT checkpoints trained on them are in the archived master frame.
  To replay or re-evaluate any of them, restore
  `archive/2026-09-16_phase05_06_master/` first (the pan-fix folder for a post-09-27 evaluation).
- No Phase 0.7 data may be mixed with Phase 0.5 / 0.6 data. Beyond the frame change, the top camera's
  viewpoint changed as well, so the two sets share neither an action frame nor an observation space.

## 2026-09-27 — `shoulder_pan` drift correction (previous deployed frame)

The `shoulder_pan` mounting screws worked loose, letting the joint sit a couple of degrees off. Tightened
mechanically; the residual offset was then measured with
[`scripts/joint_offset_check.py`](../../scripts/joint_offset_check.py) against a training-frame landmark:

| joint | delta | applied |
|---|---|---|
| **shoulder_pan** | **+2.37°** measured, **+2.81°** after two replay iterations | **`range_min` 761 → 793, `range_max` 3449 → 3481 (+32 counts)** |
| shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | +0.09 … +0.41° | none — under the ~1.5° hand-positioning noise floor |

`mid` moves 2105 → 2137. The landmark measurement gave +27 counts and replay tuned the last 5:
+10 (still left), then −5 (slightly right). The whole adjustment after the measurement is 0.44°, ~2 mm at
the 270 mm working radius — inside what a hand-held landmark reading can resolve, which is why replay and
not the measurement is the acceptance test. `homing_offset` and the range width (2688 counts) are
untouched, so the joint limits travel with the frame.

**That correction restored the frame the Phase 0.6 policies were trained in — no retraining.** It is
superseded as the deployed frame by the 2026-10-03 recalibration above, but it remains the correct frame
for evaluating those policies.
