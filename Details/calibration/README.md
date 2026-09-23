# Calibration backup — do not lose these

Copied from `~/.cache/huggingface/lerobot/calibration/` on 2026-09-23. `my_follower.json` is dated
**2026-09-16 20:49**, i.e. before the 2026-09-17 data collection, so it is the exact mapping every
dataset and every trained policy in this project was recorded and trained under.

| file | lives at |
|---|---|
| `my_follower.json` | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json` |
| `my_leader.json` | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json` |

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
