#!/usr/bin/env python
"""Measure how far the arm's joint readings have drifted from the training frame, and say what to
edit in the calibration file to put them back.

Why this exists: `lerobot-replay` plays recorded actions open-loop. If replay misses a target the
policies used to hit, the physical<->encoder mapping has moved, and the fix is to restore the mapping
rather than retrain. See Details/phase06_camera_ablation_training.md.

Usage — park the arm by hand at a physical landmark that also appears in the training data (e.g. the
gripper holding a cylinder in the fixture hole), then:

    python scripts/joint_offset_check.py --ref-episode 25 --ref-frame 600

`--ref-frame` is the frame of that episode where the arm was at the same physical landmark; open the
episode video to pick it. Torque is disabled, so the arm is free to position by hand.
"""

import argparse
import time

import numpy as np

COUNTS_PER_DEG = 4095 / 360  # lerobot DEGREES mode: value = (Present_Position - mid) * 360 / 4095


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--id", default="my_follower")
    p.add_argument("--repo-id", default="HALDijkstraaa/so101_toolkit_cylinder_20260917_165544")
    p.add_argument("--ref-episode", type=int)
    p.add_argument("--ref-frame", type=int)
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args()

    ref = None
    if args.ref_episode is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(args.repo_id, episodes=[args.ref_episode])
        ref = ds[args.ref_frame or 0]["observation.state"].numpy()
        print(f"reference: episode {args.ref_episode} frame {args.ref_frame}")
        print("           " + "  ".join(f"{v:+8.2f}" for v in ref))

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect(calibrate=False)          # never prompts, never writes a new calibration
    robot.bus.disable_torque()               # the arm is now free to move by hand
    names = list(robot.bus.motors)
    print("\ntorque disabled — hold the arm at the landmark. Ctrl-C when the reading is steady.\n")
    print("        " + "".join(f"{n:>14s}" for n in names))

    last = None
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            obs = robot.get_observation()
            cur = np.array([obs[f"{n}.pos"] for n in names])
            last = cur
            line = "now     " + "".join(f"{v:>14.2f}" for v in cur)
            if ref is not None:
                line += "\ndelta   " + "".join(f"{v:>14.2f}" for v in cur - ref)
            print(line, end="\r" if ref is None else "\n")
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()

    if ref is not None and last is not None:
        d = last - ref
        print("\n\n=== correction ===")
        print("delta (deg) = reading now - reading in training, at the same physical pose.")
        print("Add this many counts to BOTH range_min and range_max of that joint in")
        print("~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json,")
        print("then reconnect (press ENTER at the prompt) and re-run lerobot-replay.\n")
        for n, dv in zip(names, d, strict=True):
            print(f"  {n:<16s} delta {dv:+7.2f} deg   ->  range_min/max {dv * COUNTS_PER_DEG:+8.0f} counts")
        print("\nOnly correct a joint whose delta you trust: hand-positioning is worth ~1 deg, so")
        print("anything under ~1.5 deg is noise. BACK UP the json first (Details/calibration/).")


if __name__ == "__main__":
    main()
