# 01 — Set up the SO-101 (ports, motors, calibration, camera, teleop)

Goal: a working teleoperation loop — you move the **leader**, the **follower** mirrors, and a
camera streams. Everything downstream (data collection, deployment) depends on this.

> Windows note: SO-101 arms connect over USB and appear as **COM ports** (e.g. `COM5`).
> No WSL/usbipd needed on native Windows. Run everything with `conda activate lerobot` first.
> Every command has a `--help`; when an arg here differs from your 0.5.2, trust `--help` and the
> official guide: https://huggingface.co/docs/lerobot/en/so101

## Step 0 — plug in one arm at a time

Plug in **only the follower** first (so the port it grabs is unambiguous), then repeat for the leader.

## Step 1 — find the serial ports

```powershell
lerobot-find-port
```
It lists ports, asks you to unplug the arm, and tells you which COM port disappeared → that's the
arm's port. Do it once with only the follower plugged, note e.g. `COM5`; then only the leader, note
e.g. `COM6`. **Write both down.**

## Step 2 — set up the motors (assign IDs)  [needed once per arm, if not already done]

Each Feetech servo needs a unique ID written to it. If you bought a kit that's already ID'd, or you
already did this, skip. Otherwise:
```powershell
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=COM6
```
Follow the prompts (it walks you through connecting motors one by one). Check `--help` for the exact
flags in 0.5.2.

## Step 3 — find your camera index

```powershell
lerobot-find-cameras
```
It enumerates connected cameras and shows an index/path and resolution for each. Note the index of
your USB webcam (often `0` or `1`), plus a working `width`/`height`/`fps` (e.g. 640×480 @ 30).

## Step 4 — calibrate both arms

Calibration records each joint's range so leader and follower agree on angles. Do **both** arms:
```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=COM6 --teleop.id=my_leader
```
- `--robot.id` / `--teleop.id` name the calibration profile; reuse the **same id** later so your
  saved calibration is picked up (stored under the lerobot calibration dir).
- Follow the on-screen prompts: move each joint through its full range, set the rest/zero pose.

## Step 5 — test teleoperation (no camera yet)

Confirm the mechanical loop before adding vision:
```powershell
lerobot-teleoperate `
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower `
  --teleop.type=so101_leader  --teleop.port=COM6 --teleop.id=my_leader
```
Move the leader arm — the follower should mirror it in real time. If it's mirrored/inverted or a
joint is off, re-run calibration for that arm. (`^` is the PowerShell/cmd line-continuation; you can
also put it all on one line.)

## Step 6 — teleoperation WITH the camera

Add the camera so you see what the policy will see. Camera config is a small dict; verify the exact
syntax with `lerobot-teleoperate --help` (look for `--robot.cameras`):
```powershell
lerobot-teleoperate ^
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower ^
  --teleop.type=so101_leader  --teleop.port=COM6 --teleop.id=my_leader ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" ^
  --display_data=true
```
- `index_or_path: 0` → the webcam index from Step 3.
- `front` is a camera name you choose; remember it — the **same name must be used at record and
  train time** so the model's image key is consistent.
- `--display_data=true` opens a live view (uses rerun); drop it if it's not installed.

## Done when

- Leader → follower mirroring is smooth and correctly oriented.
- The camera view shows up and tracks your scene.
- You noted: follower port, leader port, camera index, and the `--robot.id`/`--teleop.id` you used.

Keep these values handy — every later command reuses them. Next: **[02_read_source_code.md](02_read_source_code.md)**.