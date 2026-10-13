#!/usr/bin/env python3
"""P4: did D1's deliberate variation actually raise reach-phase divergence above D0's?

This is the pre-registered gate in [§12.1](../Details/phase07_portable_rig.md): if the manipulation
did not take, P1-P3 are not tested and round 2 is a data-volume experiment rather than a mechanism
test. So it is checked before anything else is done with D1.

**The measurement has to be position-controlled, or it is meaningless.** In D0 the five
demonstrations behind a tray spot were at the *same* position, so divergence between them was pure
approach variation. D1 randomises position continuously, so any two episodes differ partly because
the cylinder was somewhere else — and a naive whole-dataset divergence would be inflated by that and
would "confirm" P4 for the wrong reason.

The fix is to compare like with like: for every episode, take the k episodes whose **grasp pose** is
nearest (grasp pose fixes where the cylinder was, to ~0.3 deg) and measure divergence within that
neighbourhood. For D0 that neighbourhood is essentially the recorded block, which is the control:
running the same kNN metric on D0 should reproduce its block-based numbers. If it does, the two
datasets are on one scale.

Residual position spread inside each neighbourhood is reported alongside, because that is the
quantity that would bias the comparison.

    python scripts/check_d1_divergence.py
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import CACHE, OUT, REPO, episode_table, frames, grasp_seated_index
from analyze_phase07_positions import divergence

D0 = "so101_tray_cylinder_portable_20261003_211912"
D1 = "so101_tray_cylinder_varied_20261013_082911"
K = 5                      # neighbourhood size, matching D0's 5 demos per spot
D0_RANGE = (0.24, 0.39)    # the per-block div_reach range measured on D0


def idle_tail(arm: np.ndarray, thresh: float = 2.0) -> int:
    """Frames after the LAST joint motion — the verification script's definition of the idle tail.

    Deliberately not `len(s) - seated_index`: that would also count the gripper release and the
    retreat to home, which are motion, and would overstate the deliberate hold by about 4x.
    """
    moving = np.abs(np.diff(arm, axis=0)).max(axis=1) > thresh
    return int(len(arm) - 1 - np.where(moving)[0][-1]) if moving.any() else len(arm)


def episode_features(repo: str):
    """Per-episode reach trajectory, grasp pose and length, using the validated grasp detector."""
    ds = CACHE / repo
    fr = frames(ds)
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    sd = st[:, :5].std(axis=0)
    out = {}
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        s = st[idx]
        gi, ri = grasp_seated_index(s[:, 5])
        if gi < 0 or ri < 0:
            continue
        out[int(e)] = {
            "reach": (s[:gi + 1, :5] / sd),      # start -> grasp, normalised
            "grasp_pose": s[gi, :5],
            "grasp_pan": s[gi, 0],
            "tail": idle_tail(s[:, :5]),
            "n": len(s),
        }
    return out


def knn_divergence(feats: dict, k: int = K):
    """For each episode, divergence among its k nearest neighbours in grasp-pose space."""
    eps = sorted(feats)
    P = np.array([feats[e]["grasp_pose"] for e in eps])
    Pn = (P - P.mean(0)) / (P.std(0) + 1e-9)          # scale joints before distances
    rows = []
    for i, e in enumerate(eps):
        d = np.linalg.norm(Pn - Pn[i], axis=1)
        nb = np.argsort(d)[:k]                        # includes self
        grp = [feats[eps[j]]["reach"] for j in nb]
        rows.append({
            "episode": e,
            "div_reach": divergence(grp),
            "pose_spread": float(P[nb].std(axis=0).mean()),   # residual position spread, degrees
            "pan_spread": float(P[nb][:, 0].std()),
        })
    return pd.DataFrame(rows)


def block_divergence(feats: dict):
    """D0's native grouping: nine blocks of five consecutive episodes."""
    rows = []
    for b in range(9):
        grp = [feats[e]["reach"] for e in range(5 * b, 5 * b + 5) if e in feats]
        pose = np.array([feats[e]["grasp_pose"] for e in range(5 * b, 5 * b + 5) if e in feats])
        if len(grp) >= 2:
            rows.append({"block": b, "div_reach": divergence(grp),
                         "pose_spread": float(pose.std(axis=0).mean()),
                         "pan_spread": float(pose[:, 0].std())})
    return pd.DataFrame(rows)


def main() -> int:
    f0, f1 = episode_features(D0), episode_features(D1)
    print(f"episodes with a detected grasp:  D0 {len(f0)}/50   D1 {len(f1)}/50\n")

    b0 = block_divergence(f0)
    k0, k1 = knn_divergence(f0), knn_divergence(f1)

    print("=== control: does the kNN metric reproduce D0's block-based numbers? ===")
    print(f"  D0, native 9 blocks of 5 : div_reach {b0.div_reach.min():.3f}-{b0.div_reach.max():.3f}"
          f"  (mean {b0.div_reach.mean():.3f})   pose spread {b0.pose_spread.mean():.2f} deg")
    print(f"  D0, kNN k={K}             : div_reach {k0.div_reach.min():.3f}-{k0.div_reach.max():.3f}"
          f"  (mean {k0.div_reach.mean():.3f})   pose spread {k0.pose_spread.mean():.2f} deg")
    agree = abs(k0.div_reach.mean() - b0.div_reach.mean()) / b0.div_reach.mean()
    print(f"  -> kNN mean differs from block mean by {agree:.1%}"
          f"  {'(comparable)' if agree < 0.25 else '(NOT comparable - treat D1 numbers with care)'}\n")

    print("=== P4: D1 against D0, same kNN metric ===")
    for name, kk in (("D0", k0), ("D1", k1)):
        print(f"  {name}: div_reach mean {kk.div_reach.mean():.3f}   "
              f"median {kk.div_reach.median():.3f}   range {kk.div_reach.min():.3f}-"
              f"{kk.div_reach.max():.3f}   |   neighbourhood pose spread "
              f"{kk.pose_spread.mean():.2f} deg (pan {kk.pan_spread.mean():.2f})")

    # Permutation test on the difference of means - n = 50 per side, no distributional assumption.
    a, b = k0.div_reach.to_numpy(), k1.div_reach.to_numpy()
    obs = b.mean() - a.mean()
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(0)
    null = np.array([(lambda p: p[len(a):].mean() - p[:len(a)].mean())(rng.permutation(pool))
                     for _ in range(20000)])
    pval = float((np.abs(null) >= abs(obs)).mean())
    print(f"\n  difference D1 - D0 = {obs:+.3f}   permutation p = {pval:.4f}")

    ratio = k1.pose_spread.mean() / k0.pose_spread.mean()
    print(f"  neighbourhood position spread ratio D1/D0 = {ratio:.2f}x "
          f"{'- comparable, so the rise is approach variation' if ratio < 1.5 else '- D1 neighbourhoods span wider positions, so part of the rise is position, not approach'}")

    # The ratio above is the one real threat to P4, so test it rather than caveat it: compare the
    # two datasets only over the pose-spread range they share, where position variation is matched.
    lo = max(k0.pose_spread.min(), k1.pose_spread.min())
    hi = min(k0.pose_spread.max(), k1.pose_spread.max())
    m0 = k0[(k0.pose_spread >= lo) & (k0.pose_spread <= hi)]
    m1 = k1[(k1.pose_spread >= lo) & (k1.pose_spread <= hi)]
    print(f"\n  matched on neighbourhood position spread ({lo:.2f}-{hi:.2f} deg): "
          f"D0 n={len(m0)} mean {m0.div_reach.mean():.3f} | D1 n={len(m1)} mean {m1.div_reach.mean():.3f}")
    if len(m0) >= 5 and len(m1) >= 5:
        a2, b2 = m0.div_reach.to_numpy(), m1.div_reach.to_numpy()
        obs2 = b2.mean() - a2.mean()
        pool2 = np.concatenate([a2, b2])
        null2 = np.array([(lambda p: p[len(a2):].mean() - p[:len(a2)].mean())(rng.permutation(pool2))
                          for _ in range(20000)])
        p2 = float((np.abs(null2) >= abs(obs2)).mean())
        print(f"  matched difference = {obs2:+.3f}   permutation p = {p2:.4f}   "
              f"({obs2 / obs:.0%} of the unmatched difference survives)")
    else:
        p2, obs2 = float("nan"), float("nan")
        print("  too few matched neighbourhoods to test")

    verdict = (k1.div_reach.mean() > D0_RANGE[1] and pval < 0.05)
    print(f"\n  D0 per-block range was {D0_RANGE[0]}-{D0_RANGE[1]}")
    print(f"  P4 {'PASS - the manipulation took; P1-P3 are testable' if verdict else 'FAIL - divergence did not rise; P1-P3 are NOT tested'}")

    OUT.mkdir(parents=True, exist_ok=True)
    k1.to_csv(OUT / "d1_divergence_per_episode.csv", index=False)
    print(f"\nwrote {(OUT / 'd1_divergence_per_episode.csv').relative_to(REPO)}")

    # The idle head/tail is reported authoritatively by verify_phase07_dataset.py (section 11),
    # which owns the threshold; it is not duplicated here with a second definition.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
