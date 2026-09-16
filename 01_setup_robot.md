# 01 — Set up the SO-101 (ports, motors, calibration, camera, teleop)

Goal: a working teleoperation loop — you move the **leader**, the **follower** mirrors, and a
camera streams. Everything downstream (data collection, deployment) depends on this.

> **Ubuntu 24.04 (RTX 5080 laptop), lerobot 0.6.1.** Env install: [Details/ubuntu_env_setup.md](Details/ubuntu_env_setup.md).
> Run `conda activate lerobot` first. Your user must be in `dialout` (check with `groups`).
> Every command has a `--help`; when an arg here differs from 0.6.1, trust `--help` and the
> official guide: https://huggingface.co/docs/lerobot/en/so101

**Your ports (found 2026-09-16):**

| Arm | Port | id |
|---|---|---|
| Follower | `/dev/ttyACM0` | `my_follower` |
| Leader | `/dev/ttyACM1` | `my_leader` |

`ttyACM` numbers follow plug order. **Plug in the follower first, then the leader**, every session. If
they come up swapped, check `ls -l /dev/serial/by-id/` and use those stable paths instead.

## Step 0 — plug in one arm at a time

Plug in **only the follower** first (so the port it grabs is unambiguous), then the leader.

## Step 1 — find the serial ports

```bash
lerobot-find-port
ls -l /dev/ttyACM* /dev/serial/by-id/
```
`lerobot-find-port` asks you to unplug the arm and reports which port disappeared. Done: follower
`/dev/ttyACM0`, leader `/dev/ttyACM1`. If the port vanishes right after plugging in, or the first
command times out, see the `brltty` / ModemManager fixes in [ubuntu_env_setup §8](Details/ubuntu_env_setup.md).

## Step 2 — set up the motors (assign IDs)  [needed once per arm, if not already done]

Each Feetech servo needs a unique ID written to it. The arms were already set up on Windows, so skip
this. Only for a new or replaced motor:
```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1
```

## Step 3 — find your camera paths

```bash
lerobot-find-cameras opencv
ls -l /dev/v4l/by-id/
```
The built-in webcam takes `/dev/video0–3`, so the C920 (top) and C922 (wrist) get higher numbers. Use
the `/dev/v4l/by-id/...-video-index0` paths, which don't change with plug order. Apply the wrist-cam
focus/exposure settings from [ubuntu_env_setup §10](Details/ubuntu_env_setup.md) after each replug.

## Step 4 — calibrate both arms

Calibration records each joint's range so leader and follower agree on angles. Do **both** arms:
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```
- Reuse the **same ids** later so the saved calibration is picked up
  (`~/.cache/huggingface/lerobot/calibration/`).
- Follow the prompts: move each joint through its full range, set the rest/zero pose.

## Step 5 — test teleoperation (no camera yet)

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```
Move the leader — the follower should mirror it. If a joint is inverted or off, recalibrate that arm.

## Step 6 — teleoperation WITH both cameras

```bash
TOP=/dev/v4l/by-id/<C920>-video-index0
WRIST=/dev/v4l/by-id/<C922>-video-index0

lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30} }" \
  --display_data=true
```
- `top` / `wrist` are the camera names (see 06). The **same names must be used at record and train
  time** so the image keys match.
- `--display_data=true` opens a live rerun view; drop it if rerun isn't installed.

## Done when

- Leader → follower mirroring is smooth and correctly oriented.
- Both camera views show up and track the scene.
- You noted: ports, camera paths, and the ids above.

Keep these values handy — every later command reuses them. Next: **[02_read_source_code.md](02_read_source_code.md)**.
