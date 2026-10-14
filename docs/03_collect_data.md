# 03 — Collect a demonstration dataset by teleoperation

You'll record episodes of you teleoperating the follower (via the leader) to do a simple task, with
the camera streaming. These `(observation, action-chunk)` windows become SmolVLA's training data —
exactly the "sliding windows of teleop trajectories" from your knowledge notes.

> Prereq: [01_setup_robot.md](01_setup_robot.md) done — you have the follower (`/dev/ttyACM0`) / leader (`/dev/ttyACM1`) ports, the
> camera paths, and calibration ids. `conda activate lerobot` first.
> Ubuntu: blocks marked `powershell` are from the Windows runs. Ports are updated; on Ubuntu swap the `` ` `` / `^` line endings for `\` and use the camera paths from 01.

## The task (v1 — see [00_end_to_end_arc.md](00_end_to_end_arc.md) § ① for the full spec)

**Pick the blue PCB off the white plate and place it into the pocket of the black fixture.** Grip a
**USB connector** (the same one every time → consistent grip pose); PCB **rotation is fixed**, the fixture
is **fixed**; the only thing that varies is the PCB **xy position within ±1 cm** on the marked plane. Keep
the scene, lighting, and camera fixed; the motion is short (~30 s). Consistent scene + one bounded
generalization axis = far less data needed and a real chance of learning on modest hardware.

## Record with `lerobot-record`

```powershell
lerobot-record `
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower `
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader `
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
  --dataset.repo_id=HALDijkstraaa/so101_pick_place_pcb `
  --dataset.single_task="Pick up the blue PCB and place it in the white fixture pocket" `
  --dataset.num_episodes=50 `
  --dataset.episode_time_s=35 `
  --dataset.reset_time_s=2 `
  --dataset.fps=30 `
  --dataset.push_to_hub=true `
  --dataset.private=true `
  --display_data=true
```

update command with two cameras (top-down and wrist):
```powershell
lerobot-record `
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower `
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader `
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" `
  --dataset.repo_id=HALDijkstraaa/so101_pick_place_pcb_v2_test `
  --dataset.single_task="Pick up the blue PCB and place it in the white fixture pocket" `
  --dataset.num_episodes=3 `
  --dataset.episode_time_s=35 `
  --dataset.reset_time_s=2 `
  --dataset.fps=30 `
  --dataset.push_to_hub=true `
  --dataset.private=true `
```

Key args (verify with `lerobot-record --help` for your 0.5.2):
- `--dataset.repo_id` — `you/dataset`. Here `HALDijkstraaa/so101_pick_place_pcb` (your HF handle + a
  PCB-specific name). Saved locally under the lerobot home dir; also uploaded (see `push_to_hub`).
- `--dataset.single_task` — the natural-language instruction. **SmolVLA is language-conditioned**, so
  this text is part of the input — keep it consistent and descriptive (reuse it verbatim at deploy).
- `--dataset.num_episodes` — **50** for this task (longer/more precise than a plain pick-place).
  More/consistent demos > many sloppy ones.
- `--dataset.episode_time_s=35` — max seconds recorded per episode (auto-advances after this). 35 s
  gives comfortable margin over the ~30 s task — you can also end early with the hotkey.
- `--dataset.reset_time_s=2` — pause between episodes to **reset the scene** (move the PCB to the next
  ±1 cm spot, re-home). 2 s is quick — raise it if you need more time to reposition carefully.
- `--dataset.push_to_hub=true` + `--dataset.private=true` — uploads the dataset to the HF Hub as a
  **private** repo. Requires you to be logged in first: `huggingface-cli login` (paste a write token).
  Set `push_to_hub=false` to keep it purely local.
- The **camera name** (`front`) and **camera index** (`1`) must match what you'll use at train/deploy time.

### The recording loop / hotkeys
`lerobot-record` guides you episode by episode: it resets, you teleoperate the task, then it saves.
There are keyboard controls to end an episode early, re-record a bad one, or stop — watch the console
prompt (commonly arrow keys / `esc`). Re-record any episode where you fumbled; **data quality is the
whole game** (your teleop-quality thesis applies directly here — jerky or paused demos teach the
policy to be jerky or to freeze).

## Tips for learnable data on a small budget
- **Consistency:** same start pose, **fixed PCB rotation**, grip the **same USB connector** the same way.
- **One-shot grip:** if the grip fails or tilts the board, **restart the episode** — keep grip demos clean.
- **Coverage:** spread the 50 PCB start positions across the **±1 cm** marked plane (roughly uniform,
  biased toward the edges/corners of the range) so the policy generalizes over that one axis.
- **Smoothness:** move deliberately at a 30 Hz-trackable speed; avoid long pauses (they teach "freeze").
- **Poke-adjust:** the drop is lenient — if it doesn't seat, demonstrate a **consistent** poke to nudge it
  in, so the policy learns a clean corrective motion (not erratic pokes).
- **Length:** keep each episode short; trim idle time at start/end.

## Inspect what you recorded

Visualize episodes (images + action/state traces):
```powershell
lerobot-dataset-viz --repo-id=HALDijkstraaa/so101_pick_place_pcb
```

Replay an episode on the robot (sanity-check the recorded actions actually do the task):
```powershell
lerobot-replay `
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower `
  --dataset.repo_id=HALDijkstraaa/so101_pick_place_pcb `
  --dataset.episode=0
```
If replay reproduces the task, your dataset is clean. If it's off, re-check calibration and re-record.

## Mid-collection quality check — STOP after the first ~5 episodes

Don't record all 50 and *then* discover a systematic problem — a bad camera angle or a calibration drift
multiplies into 50 wasted episodes. Record **~5**, then run the gate below. It's the data analog of the
"smoke-test the pipeline before the real run" instinct.

**1. `lerobot-dataset-viz` — look at the images:**
- [ ] PCB, white plate, black fixture/pocket, **and** the gripper are all in frame every episode (nothing
  cut off).
- [ ] The arm doesn't fully occlude the fixture at the drop moment (a little is fine).
- [ ] No blur / no blown-out glare on the white plate (adjust lighting/angle if so).
- [ ] Framing is *identical* across the 5 (camera didn't move).

**2. `lerobot-dataset-viz` — look at the action/state traces:**
- [ ] Curves are **smooth** — no jitter spikes or step discontinuities (= you moved too fast/jerky).
- [ ] **No long flat segments** (idle/pauses — they teach the freeze failure mode).
- [ ] Episode length is sane (~30 s ≈ ~900 frames at 30 fps); trim if you're dwelling.
- [ ] The **gripper channel** shows a clean *close → hold → open* — not repeated open/close (= grip
  fumbles that should've been a re-record).

**3. `lerobot-replay --dataset.episode=0` on the robot (the open-loop check):**
- [ ] Replaying the *recorded actions* reproduces the task (grabs the connector, places, seats).
- If replay **fails**, the problem is hardware/calibration/recording — **not** anything you'll fix by
  collecting more data. Re-check calibration (`--robot.id=my_follower`) and ports before continuing.
  (This is the same "remove the policy, replay a known-good episode" isolation move from the debugging
  framework — it splits "is it the data or the robot?" in one test.)

**Common problems → fix (then delete the bad episodes and restart the batch):**
| Symptom in viz/replay | Cause | Fix |
|---|---|---|
| PCB/fixture out of frame or occluded | camera pose | reposition camera, re-record |
| glare / white plate blown out | lighting | diffuse the light; angle the plate |
| jittery action traces | moved too fast | slow down to a 30 Hz-trackable speed |
| flat idle stretches | dwelling/pausing | keep moving; trim start/end idle |
| gripper opens/closes repeatedly | grip fumble saved | enforce one-shot grip; re-record on any slip |
| replay misses the task | calibration/port drift | recalibrate; verify `/dev/ttyACM0` / `/dev/ttyACM1`, camera paths |

**Decision gate:** 5 clean + replay works → continue to 50, spot-checking every ~10 episodes. Any
systematic issue → fix the setup first; a clean 20 beats a messy 50.

## Done when
- **50** clean episodes recorded, all with the same camera name (`front`) and instruction.
- `lerobot-dataset-viz` looks sensible; `lerobot-replay` reproduces the task.

Next (if time): **[04_finetune_and_deploy.md](04_finetune_and_deploy.md)**.
