# Calibration backup — do not lose these

Copied from `~/.cache/huggingface/lerobot/calibration/` on 2026-09-23. `my_follower.json` is dated
**2026-09-16 20:49**, i.e. before the 2026-09-17 data collection, so it is the exact mapping every
dataset and every trained policy in this project was recorded and trained under.

| file | what it is |
|---|---|
| `my_follower.json` | **the recording-day master** (2026-09-16), the frame every dataset and policy was built in |
| `my_follower.deployed.json` | **currently deployed** — the master with `shoulder_pan` drift corrected, see below |
| `my_leader.json` | leader arm, unchanged |

Live locations: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json` and
`.../teleoperators/so_leader/my_leader.json`.

## 2026-09-27 — `shoulder_pan` drift correction

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

**This restores the frame the policies were trained in — no retraining.**

**Why it matters.** On Feetech servos `Present_Position = Actual_Position - Homing_Offset`. lerobot
stores the homing offset and the per-joint range here and writes them into the motors' EEPROM on
connect. The policies consume and emit positions in *that* frame, and the dataset's normalization
statistics were computed in it. Change the calibration and every state and action is shifted by a
constant — the policies are not permanently invalid, but they are wrong by that offset until it is
restored.

**The calibration step is hand-positioned** ("move the arm to the middle of its range and press
ENTER"), and one encoder count is 360/4096 = 0.088°, so a 5° error by hand becomes a permanent 57-count
bias on that joint. Re-running calibration is therefore *not* a safe no-op.

**Restoring.** This file is the master. Put it back at the path above, connect the robot, and press
ENTER at the "use provided calibration file" prompt — `write_calibration()` writes these values back
into the motors. Verified in `so_follower.py:115` and `motors/feetech/feetech.py:268`.
