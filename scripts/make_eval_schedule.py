#!/usr/bin/env python3
"""Build the Phase 0.6 rollout schedule: which policy at which position, in which order.

`lerobot-rollout` records N episodes of ONE policy in a row, so a fully interleaved trial order would
mean restarting the process for every trial. This splits the positions into two halves instead and runs
every condition once per half:

    half A:  cond1 x 5 positions, cond2 x 5, cond3 x 5     <- first part of the session
    half B:  (conditions rotated) x the other 5 positions  <- second part

Each condition therefore gets the same positions at the same point in the session, so drift over the
session (lighting, fixture nudges, your own reset habits) hits all three roughly equally instead of
landing on whoever ran last.

    python scripts/make_eval_schedule.py --seed 0 > Details/phase06_eval_log.csv
"""
import argparse, csv, random, sys

CONDS = ["top", "wrist", "both"]

p = argparse.ArgumentParser()
p.add_argument("--positions", type=int, default=10)   # P1..P10, the same tape marks used for recording
p.add_argument("--repeats", type=int, default=1)      # 1 -> 30 trials (10 per policy), the behavior pass
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = random.Random(a.seed)
pos = [f"P{i}" for i in range(1, a.positions + 1)]
rng.shuffle(pos)
key = lambda s: int(s[1:])
halves = {"A": sorted(pos[: a.positions // 2], key=key), "B": sorted(pos[a.positions // 2 :], key=key)}

w = csv.writer(sys.stdout)
w.writerow(["block", "condition", "half", "episode_in_block", "position",
            "success", "fail_stage", "notes"])

block = 0
plan = []
for rep in range(a.repeats):
    order = CONDS[:]
    rng.shuffle(order)
    for half in ("A", "B"):
        for cond in order:
            block += 1
            eps = halves[half][:]
            rng.shuffle(eps)                       # position order differs per block
            plan.append((block, cond, half, eps))
            for i, ppos in enumerate(eps, start=1):
                w.writerow([block, cond, half, i, ppos, "", "", ""])
        order = order[1:] + order[:1]              # rotate so the same policy is not always first

print(f"# {block} blocks x {a.positions // 2} episodes = {block * (a.positions // 2)} trials", file=sys.stderr)
print(f"# half A = {' '.join(halves['A'])}   half B = {' '.join(halves['B'])}", file=sys.stderr)
for b, cond, half, eps in plan:
    print(f"#   block {b}: {cond:5s} (half {half})  ->  {'  '.join(eps)}", file=sys.stderr)
print("# fail_stage: reach | grasp | transport | insert | retreat | none", file=sys.stderr)
