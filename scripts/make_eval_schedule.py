#!/usr/bin/env python3
"""Build an interleaved, balanced trial schedule for the Phase 0.6 camera ablation.

Every block visits all three policies once per position, in a shuffled order, so drift over the
session (lighting, fixture nudges, your own reset habits) is spread evenly across conditions
instead of landing on whichever policy ran last.

    python scripts/make_eval_schedule.py --trials-per-position 3 > eval_schedule.csv
"""
import argparse, csv, random, sys

CONDS = ["top", "wrist", "both"]

p = argparse.ArgumentParser()
p.add_argument("--positions", type=int, default=10)          # P1..P10, same tape marks as recording
p.add_argument("--trials-per-position", type=int, default=3)  # 3 -> 30 trials per policy
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = random.Random(a.seed)
w = csv.writer(sys.stdout)
w.writerow(["trial", "block", "condition", "position", "success", "fail_stage", "time_s", "notes"])

n = 0
for block in range(1, a.trials_per_position + 1):
    # one block = every (condition, position) pair exactly once, shuffled
    pairs = [(c, f"P{i}") for c in CONDS for i in range(1, a.positions + 1)]
    rng.shuffle(pairs)
    for cond, pos in pairs:
        n += 1
        w.writerow([n, block, cond, pos, "", "", "", ""])

print(f"# {n} trials = {a.trials_per_position} blocks x {len(CONDS)} conditions x {a.positions} positions",
      file=sys.stderr)
print("# fail_stage: reach | grasp | transport | insert | retreat | none", file=sys.stderr)
