#!/usr/bin/env python3
"""Round-3 eval schedule: π0.5 10k alone, nine marked spots, two passes.

One arm, so there is no second arm to counterbalance against and the ABBA-BAAB mirror of round 2
does not apply. What still can be protected, and is asserted below:

  1. Each of the nine spots is run exactly twice.
  2. Each spot appears once in pass 1 and once in pass 2, so no spot is sampled twice from the same
     part of the session.
  3. The halves are reshuffled between passes, so a spot that ran early in pass 1 does not reliably
     run early in pass 2 - session-time drift spreads across spots instead of landing on a few.
  4. Every block is one contiguous `lerobot-rollout` call: 4 process starts, not 18.

What this design CANNOT do is separate π0.5 from the session it ran in. ACT 40k and SmolVLA 20k were
scored on 2026-10-15 in a single session; this runs on another day, so the π0.5-vs-them comparison is
between-session where ACT-vs-SmolVLA was within. Both policies were observed to track top-camera
brightness (§12.9 obs 3), so that is a live confound, not a hypothetical one. Reading a difference of
fewer than ~3 trials in 18 as a policy difference is not supported by this design.

    python3 scripts/make_phase07_r3_pi05_eval_schedule.py --emit csv > docs/details/phase07_r3_pi05_eval_log.csv
    python3 scripts/make_phase07_r3_pi05_eval_schedule.py --emit commands
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

# Chosen by the offline sweep in §12.7a: 10k beats 20k by 30.6% on chunk jerk - ~350x its own
# repeat-to-repeat std - and on pooled held-out loss as well. Not by the wandb minimum at 15k,
# which has no checkpoint and sits inside the eval's own jitter.
ARM, POLICY, CKPT = "pi05_10k_d0d1", "pi05", "010000"
# Pulled off the rented box with rsync, which lands checkpoints flat rather than under
# checkpoints/<step>/pretrained_model/.
MODEL_DIR = "outputs/train/pi05_d0d1_80k/010000"

TASK = ("Pick up the white cylinder from the green tray and place it in the top left blue hole "
        "of the black fixture")
TOP = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0"
WRIST = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0"
# Same as every other arm in rounds 1 and 2. π0.5's chunk is 50 and its worst measured inference is
# 316 ms against the 833 ms this buys - 2.6x headroom (§10.3), so plain sync inference is enough and
# --inference.type=rtc is not needed.
N_ACTION_STEPS = 25


def rename_map_for(model_dir: str) -> dict[str, str]:
    """The camera rename map, read from the checkpoint's OWN saved preprocessor.

    π0.5 was pretrained on `base_0_rgb`/`left_wrist_0_rgb` while the robot produces `top`/`wrist`,
    so without the map `lerobot-rollout` dies on a feature mismatch. Read rather than hardcoded,
    because a map that is merely *wrong* - top and wrist swapped - does not raise. It produces a
    policy quietly driving on the wrong camera for a whole block.
    """
    d = pathlib.Path(model_dir)
    for cfg in (d / "policy_preprocessor.json", d / "pretrained_model" / "policy_preprocessor.json"):
        if cfg.is_file():
            for step in json.loads(cfg.read_text()).get("steps", []):
                if "rename" in step.get("registry_name", ""):
                    return dict(step.get("config", {}).get("rename_map", {}))
            return {}
    raise SystemExit(
        f"cannot read the rename map: no policy_preprocessor.json under {d}.\n"
        "Refusing to emit a rollout command without it - a missing map fails at the robot after "
        "the block is set up, and a wrong one fails silently.")


def build(spots: int, passes: int, seed: int):
    """Blocks of (block_no, half, spot_list, pass_no)."""
    rng = random.Random(seed)
    names = [f"T{i}" for i in range(1, spots + 1)]
    blocks, n = [], 0
    for p in range(1, passes + 1):
        order = names[:]
        rng.shuffle(order)
        cut = (spots + 1) // 2
        for half, chunk in (("A", order[:cut]), ("B", order[cut:])):
            n += 1
            blocks.append((n, half, chunk, p))
    return blocks


def check(blocks, spots: int, passes: int) -> None:
    names = {f"T{i}" for i in range(1, spots + 1)}
    seen: dict[str, list[int]] = {s: [] for s in names}
    for b_no, _half, chunk, p in blocks:
        for s in chunk:
            seen[s].append(p)

    for s, ps in seen.items():
        assert len(ps) == passes, f"{s} ran {len(ps)} times, expected {passes}"
        assert sorted(ps) == list(range(1, passes + 1)), \
            f"{s} ran in passes {sorted(ps)} - every spot must appear once per pass"

    # No spot may sit at the same session position in both passes: that is what reshuffling buys.
    pos: dict[str, list[int]] = {s: [] for s in names}
    order = 0
    for _b, _h, chunk, _p in blocks:
        for s in chunk:
            order += 1
            pos[s].append(order)
    per_pass = spots
    repeats = [tuple(o % per_pass for o in v) for v in pos.values()]
    locked = sum(1 for a, b in repeats if a == b)
    assert locked <= 1, f"{locked} spots hold the same within-pass position in both passes"
    print(f"# checks passed: {spots} spots x {passes} passes = {spots * passes} trials, "
          f"{len(blocks)} blocks, {locked} spot(s) position-locked", file=sys.stderr)


def ds_name(p: int, half: str) -> str:
    return f"rollout_phase07_r3_pi05_10k_p{p}{half.lower()}"


def emit_csv(blocks, out=sys.stdout):
    print("block,arm,policy,checkpoint,pass,half,episode_in_block,tray_spot,episode_index,"
          "duration_s,success,fail_stage,notes", file=out)
    for b_no, half, chunk, p in blocks:
        for i, spot in enumerate(chunk):
            print(f"{b_no},{ARM},{POLICY},{CKPT},{p},{half},{i + 1},{spot},{i},,,,", file=out)


def emit_commands(blocks, out=sys.stdout):
    # The proxy unset is not optional. Clash Verge in TUN mode routes at the network layer and
    # needs no proxy variables, but it still exports all_proxy=socks://127.0.0.1:7897 - and httpx,
    # under huggingface_hub, rejects a bare `socks://` scheme outright ("Unknown scheme for proxy
    # URL"). pi0.5's processor asks for the paligemma tokenizer by hub id on every block start, so
    # that abort lands AFTER the robot is set up and the block is lost.
    pre = ("cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && "
           "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && "
           "unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY")
    rmap = rename_map_for(MODEL_DIR)
    for b_no, half, chunk, p in blocks:
        name = ds_name(p, half)
        print(f"\n### block {b_no} — pass {p} half {half} — {len(chunk)} trials at "
              f"{' '.join(chunk)}", file=out)
        print("```bash", file=out)
        print(f"{pre} && "
              f"POLICY={MODEL_DIR} && "
              f"NAME={name} && NEP={len(chunk)} && "
              f"TOP={TOP} && WRIST={WRIST} && "
              f"RENAME='{json.dumps(rmap)}' && "
              f"lerobot-rollout --strategy.type=episodic --policy.path=$POLICY "
              f'--policy.n_action_steps={N_ACTION_STEPS} --rename_map="$RENAME" '
              f"--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower "
              f'--robot.cameras="{{ top: {{type: opencv, index_or_path: $TOP, width: 640, '
              f"height: 480, fps: 30, fourcc: MJPG}}, wrist: {{type: opencv, "
              f'index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG}} }}" '
              f"--dataset.repo_id=HALDijkstraaa/$NAME --dataset.no_stamp=true "
              f'--dataset.single_task="{TASK}" '
              f"--dataset.num_episodes=$NEP --dataset.episode_time_s=60 "
              f"--dataset.reset_time_s=10 --dataset.fps=30 --dataset.push_to_hub=false "
              f"--display_data=true", file=out)
        print("```", file=out)


def emit_summary(blocks, out=sys.stdout):
    print("| block | pass | half | spots |", file=out)
    print("|---|---|---|---|", file=out)
    for b_no, half, chunk, p in blocks:
        print(f"| {b_no} | {p} | {half} | {' '.join(chunk)} |", file=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--spots", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", choices=("csv", "commands", "summary"), default="csv")
    a = ap.parse_args()

    blocks = build(a.spots, a.passes, a.seed)
    check(blocks, a.spots, a.passes)
    {"csv": emit_csv, "commands": emit_commands, "summary": emit_summary}[a.emit](blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
