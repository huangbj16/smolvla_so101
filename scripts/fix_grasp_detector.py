#!/usr/bin/env python3
"""Candidate replacement for `grasp_seated_index`, validated against every D0 and D1 episode.

**The bug.** The current detector locates the two gripper openings as `argmax` of the episode's first
half and `argmax` of its second half. That encodes an assumption about *relative* timing - that the
release-open always falls after the episode's midpoint - and the deliberate 2 s tail introduced in D1
breaks it. A longer tail inflates the episode length while the release happens at the same absolute
moment, so the release migrates toward the first half. The failure condition is exactly:

    len(episode) > 2 * release_start

D1 episode 39 is 724 frames with its release beginning at frame 342, so it needed len < 684 and missed
by 40 frames. Both "peaks" then land inside the same release opening, `hold` is computed over that
opening instead of the carry plateau, and the detector reports the fixture pose as the grasp pose.

**The fix.** Find the openings as contiguous runs above a level derived from the trace, and take the
first and the last. No reference to the midpoint, so episode length cannot affect the result.

This script only *proposes* the replacement: it runs both detectors over all 100 D0+D1 episodes and
prints where they agree. The new one is adopted only if it reproduces D0 exactly, since the
verification report and section 11.6 were computed with the old one.

    python scripts/fix_grasp_detector.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import CACHE, frames, grasp_seated_index as old_detector
from check_d1_divergence import D0, D1


def turning_points(x: np.ndarray, delta: float) -> list[tuple[int, str]]:
    """Significant alternating extrema: the classic zigzag filter.

    Walks the signal tracking a running extremum and emits it once the signal reverses by more than
    `delta`. Noise below `delta` is ignored, so this finds the real opens and closes without any
    level threshold - which is what the gripper trace needs, because neither the hold level nor the
    two opening heights are predictable across episodes.
    """
    pts: list[tuple[int, str]] = []
    mn = mx = float(x[0])
    mni = mxi = 0
    rising: bool | None = None
    for i, raw in enumerate(x):
        v = float(raw)
        if v > mx:
            mx, mxi = v, i
        if v < mn:
            mn, mni = v, i
        if rising is not False and v < mx - delta:
            pts.append((mxi, "max")); mn, mni, rising = v, i, False
        elif rising is not True and v > mn + delta:
            pts.append((mni, "min")); mx, mxi, rising = v, i, True
    return pts


def grasp_seated_index_v2(grip: np.ndarray) -> tuple[int, int]:
    """Grasp and seated frame indices, with no reference to the episode midpoint or to a fixed level.

    A clean episode's gripper trace is: closed at rest -> OPEN to approach -> closed on the cylinder
    (the "hold" plateau, set by its diameter) -> OPEN to release -> closed at rest. The job is to
    find the two openings. Neither a midpoint split nor a global threshold can do it:

      * the midpoint fails when a long idle tail pushes the release before it (D1 ep 39);
      * a global threshold fails in two opposite ways - the release opening can be twice the width
        of the approach, putting the approach peak below the threshold (D1 ep 16), and the hold
        plateau can sit above it, merging the two openings into one run (D0 ep 5, 18, D1 ep 48).

    Turning points avoid both: they are defined by how far the signal REVERSES, which is a property
    of the transitions themselves rather than of any absolute level or position in the episode.
    """
    n = len(grip)
    if n < 10:
        return -1, -1
    span = float(grip.max()) - float(np.percentile(grip, 5))
    if span <= 1e-6:
        return -1, -1
    # Delta ladder, strict first. Across D0 and D1 the smallest real approach-peak -> hold
    # reversal measured is 0.19 of span, so 0.18 is the largest delta that still catches every
    # opening in hand - and a larger delta rejects more noise. The ladder exists for data not yet
    # recorded: if a future episode's reversal is shallower, it relaxes until the structure is
    # found instead of failing. On all 100 D0+D1 episodes the first rung already succeeds.
    maxima: list[int] = []
    for delta_frac in (0.18, 0.15, 0.12, 0.08):
        pts = turning_points(grip, delta_frac * span)
        maxima = [i for i, kind in pts if kind == "max"]
        if len(maxima) >= 2:
            break
    if len(maxima) < 2:
        return -1, -1
    a_peak, r_peak = maxima[0], maxima[-1]        # first and last opening
    if r_peak <= a_peak:
        return -1, -1

    # From here the logic is the original's, unchanged - only the two peaks it works from differ.
    hold = float(grip[a_peak:r_peak].min())
    band_g = hold + 0.15 * max(float(grip[a_peak]) - hold, 1e-6)
    gi = a_peak + int(np.where(grip[a_peak:r_peak] <= band_g)[0][0])
    band_r = hold + 0.20 * max(float(grip[r_peak]) - hold, 1e-6)
    at_hold = np.where(grip[gi:r_peak + 1] <= band_r)[0]
    ri = gi + int(at_hold[-1]) if len(at_hold) else -1
    return gi, ri


def main() -> int:
    for name, repo in (("D0", D0), ("D1", D1)):
        fr = frames(CACHE / repo)
        st = np.stack(fr["observation.state"].to_numpy()).astype(float)
        ep = fr["episode_index"].to_numpy()
        same, diff, failed = 0, [], []
        for e in np.unique(ep):
            s = st[np.where(ep == e)[0]]
            g = s[:, 5]
            o_gi, o_ri = old_detector(g)
            n_gi, n_ri = grasp_seated_index_v2(g)
            if n_gi < 0:
                failed.append(int(e))
            elif (o_gi, o_ri) == (n_gi, n_ri):
                same += 1
            else:
                diff.append((int(e), (o_gi, o_ri), (n_gi, n_ri),
                             float(abs(s[o_gi, 0] - s[o_ri, 0])),
                             float(abs(s[n_gi, 0] - s[n_ri, 0]))))
        print(f"--- {name} ---")
        print(f"  identical to the old detector : {same}/50")
        print(f"  new detector failed outright  : {failed or 'none'}")
        for e, o, nw, o_sep, n_sep in diff:
            print(f"  ep {e:2d} DIFFERS  old grasp/seated {o}  new {nw}")
            print(f"            |grasp pan - seated pan|: old {o_sep:.2f} deg -> new {n_sep:.2f} deg "
                  f"({'new separates them, old did not' if n_sep > o_sep else 'old was better'})")
    print("\nAdoption rule: the replacement is only safe if D0 is 50/50 identical, since the")
    print("verification report and section 11.6 were computed with the old detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
