#!/usr/bin/env python3
"""Build the Phase 0.7 ROUND-2 rollout schedule: two D1 arms over the nine marked tray spots.

Round 1 ran six arms because the question was *which checkpoint per policy*. Round 2 has one
checkpoint per policy already chosen offline by chunk jerk (section 12.6), so it runs **two arms**
and spends the saved session time on trials instead.

Same mirrored design as round 1, for the same reason. `lerobot-rollout` records N episodes of one
checkpoint in a row, so the nine spots split into halves and each arm runs once per half, with the
arm order reversed between halves:

    pass 1:  half A [act, sv]   half B [sv, act]
    pass 2:  half A [sv, act]   half B [act, sv]      <- and the starting arm alternates per pass

Three properties, asserted rather than assumed:

  * **Each arm sees each of the nine spots exactly `--passes` times**, so tray spot is a within-arm
    constant and cannot confound one arm against the other.
  * **Both arms have the same mean session position**, so drift linear in session time - lighting,
    servo warm-up, operator fatigue - cancels out of the comparison rather than landing on
    whichever arm ran last.
  * **Every block is a contiguous `lerobot-rollout` call**, so the session is `2 * 2 * passes`
    process starts, not one per trial.

Why `--passes` is worth thinking about before running: against round 1's best arm at 3/9, a
round-2 arm needs **8/9 (89%)** to clear p<0.05 on a two-sided Fisher test. At two passes it needs
14/18 (78%), and the smallest detectable ACT-vs-SmolVLA gap within the session drops from 56 points
to 28. One pass ranks; two passes can begin to measure.

    python scripts/make_phase07_r2_eval_schedule.py --passes 2 > docs/details/phase07_r2_eval_log.csv
    python scripts/make_phase07_r2_eval_schedule.py --passes 2 --emit commands
"""
import argparse
import csv
import random
import sys

# (arm, policy, checkpoint, local output dir). Both chosen by offline chunk jerk, not by loss:
# SmolVLA's four D1 checkpoints fall monotonically in jerk to 20k (section 12.6); ACT's are swept
# by the same script. Section 11.3a is why jerk and not the eval-loss curve.
ARMS = [
    ("act40k_d1", "act", "040000", "outputs/train/act_tray_cylinder_s1000_d1"),
    ("sv20k_d1", "smolvla", "020000", "outputs/train/smolvla_tray_cylinder_s1000_d1_resumed"),
]

TASK = ("Pick up the white cylinder from the green tray and place it in the top left blue hole "
        "of the black fixture")
# Cameras are addressed by stable by-id paths: /dev/video* renumbers across reboots and round 1
# lost a block to that.
TOP = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0"
WRIST = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0"
# ACT predicts a chunk of 100 and SmolVLA 50, but both executed 25 steps per inference in round 1.
# Keeping it identical is what makes the round-1 and round-2 jerk numbers comparable.
N_ACTION_STEPS = 25


def build(passes: int, spots: int, seed: int):
    """-> (rows, blocks) where a block is (block_no, arm_tuple, half, [spots])."""
    rng = random.Random(seed)
    order = lambda s: int(s[1:])
    blocks, block_no = [], 0
    for p in range(passes):
        all_spots = [f"T{i}" for i in range(1, spots + 1)]
        # Reshuffle the halves per pass so a spot is not locked to the same session position.
        random.Random(seed + 1000 * p).shuffle(all_spots)
        cut = (spots + 1) // 2          # 5 + 4 for nine
        halves = {"A": sorted(all_spots[:cut], key=order),
                  "B": sorted(all_spots[cut:], key=order)}
        # Alternate which arm leads each pass, then mirror within the pass.
        lead = ARMS if p % 2 == 0 else ARMS[::-1]
        for half, arms in (("A", lead), ("B", lead[::-1])):
            for arm in arms:
                block_no += 1
                blocks.append((block_no, arm, half, halves[half], p + 1))

    rows = []
    for block_no, (arm, policy, ckpt, _dir), half, spot_list, p in blocks:
        for i, spot in enumerate(spot_list, start=1):
            rows.append({"block": block_no, "arm": arm, "policy": policy, "checkpoint": ckpt,
                         "pass": p, "half": half, "episode_in_block": i, "tray_spot": spot,
                         "episode_index": "", "duration_s": "", "success": "",
                         "fail_stage": "", "notes": ""})
    return rows, blocks


def check(rows, blocks, passes, spots):
    """Assert the three properties the design exists to provide."""
    import collections
    errs = []
    n_arms = len(ARMS)
    if len(rows) != n_arms * spots * passes:
        errs.append(f"expected {n_arms * spots * passes} trials, built {len(rows)}")

    for arm, *_ in ARMS:
        seen = collections.Counter(r["tray_spot"] for r in rows if r["arm"] == arm)
        if set(seen) != {f"T{i}" for i in range(1, spots + 1)} or set(seen.values()) != {passes}:
            errs.append(f"{arm} does not see each of {spots} spots exactly {passes}x: {dict(seen)}")

    # Mean session position, measured in blocks. Equal means linear drift cancels.
    pos = collections.defaultdict(list)
    for b_no, (arm, *_), *_ in blocks:
        pos[arm].append(b_no)
    means = {a: sum(v) / len(v) for a, v in pos.items()}
    if len(set(round(m, 6) for m in means.values())) != 1:
        errs.append(f"arms have different mean session positions: {means}")

    if errs:
        raise SystemExit("SCHEDULE IS WRONG:\n  " + "\n  ".join(errs))
    return means


def ds_name(arm: str, p: int, half: str) -> str:
    return f"rollout_phase07_r2_{arm}_p{p}{half.lower()}"


def emit_commands(blocks, out=sys.stdout):
    pre = ("cd /home/bj/Documents/bingjian/robot_learning/smolvla_so101 && "
           "source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot")
    for b_no, (arm, _pol, ckpt, odir), half, spot_list, p in blocks:
        name = ds_name(arm, p, half)
        print(f"\n### block {b_no} — {arm} — pass {p} half {half} — "
              f"{len(spot_list)} trials at {' '.join(spot_list)}", file=out)
        print("```bash", file=out)
        print(f"{pre} && "
              f"POLICY={odir}/checkpoints/{ckpt}/pretrained_model && "
              f"NAME={name} && NEP={len(spot_list)} && "
              f"TOP={TOP} && WRIST={WRIST} && "
              f"lerobot-rollout --strategy.type=episodic --policy.path=$POLICY "
              f"--policy.n_action_steps={N_ACTION_STEPS} "
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=2,
                    help="times each arm sees all nine spots (default 2; see the docstring)")
    ap.add_argument("--spots", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", choices=("csv", "commands", "summary"), default="csv")
    a = ap.parse_args()

    rows, blocks = build(a.passes, a.spots, a.seed)
    means = check(rows, blocks, a.passes, a.spots)

    if a.emit == "commands":
        emit_commands(blocks)
        return 0
    if a.emit == "summary":
        print(f"{len(rows)} trials · {len(blocks)} blocks · {a.passes} pass(es) over "
              f"{a.spots} spots · mean session position {means}")
        for b_no, (arm, *_), half, sl, p in blocks:
            print(f"  block {b_no:2d}  pass {p}  half {half}  {arm:11s}  {' '.join(sl)}")
        return 0

    # lineterminator is explicit: csv.writer defaults to CRLF, which the repo's gitattributes
    # renormalisation had to clean out of an earlier eval log.
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
