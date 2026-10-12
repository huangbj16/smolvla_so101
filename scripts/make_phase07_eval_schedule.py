#!/usr/bin/env python3
"""Build the Phase 0.7 round-1 rollout schedule: six arms over the nine marked tray spots.

`lerobot-rollout` records N episodes of ONE checkpoint in a row, so a fully interleaved trial
order would mean restarting the process once per trial — 54 restarts, and a 9.35 GB model reload
for every pi0.5 one. This splits the nine spots into two halves and runs every arm once per half,
in **mirrored order**: the arm that goes first in half A goes last in half B.

    half A (5 spots):  act40k act60k sv5k sv20k pi5k pi10k
    half B (4 spots):  pi10k pi5k sv20k sv5k act60k act40k

Two properties come out of the mirror, and they are the whole point of it:

  * **Every arm has the same mean position in the session** (6.5 of 12 blocks), so drift that is
    linear in session time — lighting, servo warm-up, operator fatigue — cancels out of every
    comparison rather than landing on whoever ran last.
  * **Each policy's two checkpoints are adjacent**, so the primary question (which checkpoint) is
    settled over a few minutes of session time rather than across the whole afternoon.

Each arm sees each of the nine spots exactly once, so tray spot is a within-arm constant and
cannot confound any arm against another.

    python scripts/make_phase07_eval_schedule.py --seed 0 > Details/phase07_eval_log.csv
"""
import argparse
import csv
import random
import sys

# (arm, policy, checkpoint) — the six arms of round 1. Order here is the half-A order; half B is
# this list reversed. See Details/phase07_portable_rig.md section 10.
ARMS = [
    ("act40k", "act", "040000"),
    ("act60k", "act", "060000"),
    ("sv5k", "smolvla", "005000"),
    ("sv20k", "smolvla", "020000"),
    ("pi5k", "pi05", "005000"),
    ("pi10k", "pi05", "010000"),
]

p = argparse.ArgumentParser()
p.add_argument("--spots", type=int, default=9, help="marked tray spots, T1..TN (default 9)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = random.Random(a.seed)
spots = [f"T{i}" for i in range(1, a.spots + 1)]
rng.shuffle(spots)
key = lambda s: int(s[1:])
# Half A takes the larger share when the count is odd: 5 + 4 for nine spots.
cut = (a.spots + 1) // 2
halves = {"A": sorted(spots[:cut], key=key), "B": sorted(spots[cut:], key=key)}

# lineterminator is explicit: csv.writer defaults to "\r\n", and a CRLF file here is what the
# repo's .gitattributes renormalization had to clean up out of Details/phase06_eval_log.csv.
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["block", "arm", "policy", "checkpoint", "half", "episode_in_block", "tray_spot",
            "episode_index", "duration_s", "success", "fail_stage", "notes"])

block = 0
plan = []
for half, order in (("A", ARMS), ("B", ARMS[::-1])):
    for arm, policy, ckpt in order:
        block += 1
        eps = halves[half][:]
        rng.shuffle(eps)  # spot order differs per block, so no rhythm is learnable
        plan.append((block, arm, half, eps))
        for i, spot in enumerate(eps, start=1):
            w.writerow([block, arm, policy, ckpt, half, i, spot, "", "", "", "", ""])

trials = sum(len(eps) for _, _, _, eps in plan)
print(f"# {block} blocks, {trials} trials ({trials // len(ARMS)} per arm)", file=sys.stderr)
print(f"# half A = {' '.join(halves['A'])}   half B = {' '.join(halves['B'])}", file=sys.stderr)
for b, arm, half, eps in plan:
    print(f"#   block {b:2d}: {arm:7s} (half {half})  ->  {'  '.join(eps)}", file=sys.stderr)

# The mirror is only worth anything if it actually balances; assert it rather than trust it.
mean_pos = {}
for b, arm, _, _ in plan:
    mean_pos.setdefault(arm, []).append(b)
print("# mean session block per arm (all should be equal):", file=sys.stderr)
for arm, bs in mean_pos.items():
    print(f"#   {arm:7s} blocks {bs} -> {sum(bs) / len(bs):.1f}", file=sys.stderr)
assert len({sum(bs) / len(bs) for bs in mean_pos.values()}) == 1, "the mirror is not balanced"
print("# fail_stage: reach | grasp | transport | insert | retreat | none", file=sys.stderr)
