#!/usr/bin/env python3
"""Phase 0.7 per-position analysis: is a tray spot easy, or easy *for a given policy*?

The round-1 session flagged T8 as noticeably good. Section 11.6 of the rig doc could rank the
nine spots but not explain them, because the T-label to physical-spot mapping was unrecorded.
It is now known - **T*k* is the k-th spot in recording order, so its five demonstrations are
episodes 5(k-1)..5(k-1)+4** - which lets the rollout outcomes at each spot be joined to the
training data recorded there.

Two questions, and they want different statistics.

**Is position difficulty a property of the position?** If so, the three policy families should
agree on which spots are hard. Measured with Kendall's W over the three families ranking the
nine positions, against a permutation null. Low concordance is evidence of a policy x position
interaction - i.e. "data quality" is not a property of the data alone.

**Does the training data at a spot predict the rollout outcome there?** Per-spot demonstration
consistency (grasp-pose spread, phase-aligned trajectory divergence, episode-length spread)
against per-spot rollout progress, by Spearman rank correlation. With nine positions this can
see a strong effect and nothing subtle; it is hypothesis-generating for the next collection
round, not a result.

    python scripts/analyze_phase07_positions.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import (  # noqa: E402
    CACHE, OUT, FIGS, REPO, TRAIN_DS, POLICIES, POLICY_LABEL, N_STAGES,
    episode_table, frames, grasp_seated_index, load_log, wilson,
)

SPOTS = [f"T{i}" for i in range(1, 10)]
# T_k is the k-th spot in recording order; its demonstrations are episodes 5(k-1)..5(k-1)+4.
# Confirmed two ways: the operator's own screenshots (episode 1 -> position 1, episode 9 ->
# position 2) and the grid geometry - laid out 3x3 row-major the grasp bearing moves +4.1..+5.8
# deg along every row and -3.5..-5.6 deg down every column, which is a regular raster.
SPOT_EPISODES = {f"T{k}": list(range(5 * (k - 1), 5 * (k - 1) + 5)) for k in range(1, 10)}
RESAMPLE = 100


def resample(a: np.ndarray, n: int = RESAMPLE) -> np.ndarray:
    """Linear resample a (T, J) trajectory onto n points of normalised phase."""
    t_old = np.linspace(0, 1, len(a))
    t_new = np.linspace(0, 1, n)
    return np.stack([np.interp(t_new, t_old, a[:, j]) for j in range(a.shape[1])], axis=1)


def divergence(trajs: list[np.ndarray]) -> float:
    """Mean pairwise RMS distance between phase-aligned trajectories, in joint-sigma units.

    This is the Phase 0.5 divergence idea applied within a tray spot: how much do the five
    demonstrations recorded at one position disagree about what to do? Phase-normalising first
    removes pure duration differences, which `len_std` reports separately.
    """
    if len(trajs) < 2:
        return float("nan")
    r = [resample(t) for t in trajs]
    d = [np.sqrt(((a - b) ** 2).mean()) for a, b in itertools.combinations(r, 2)]
    return float(np.mean(d))


def training_positions() -> pd.DataFrame:
    fr = frames(TRAIN_DS)
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    eps = episode_table(TRAIN_DS).set_index("episode_index")
    sd = st[:, :5].std(axis=0)                     # normalise joints by their overall spread

    rows = []
    for spot, episodes in SPOT_EPISODES.items():
        whole, reach, carry, poses, seats, lens, idles = [], [], [], [], [], [], []
        for e in episodes:
            idx = np.where(ep == e)[0]
            if not len(idx):
                continue
            s = st[idx]
            gi, ri = grasp_seated_index(s[:, 5])
            if gi < 0 or ri < 0:
                continue
            arm = s[:, :5] / sd
            whole.append(arm)
            reach.append(arm[:gi + 1])             # start -> grasp
            carry.append(arm[gi:ri + 1])           # grasp -> seated
            poses.append(s[gi, :5])
            seats.append(s[ri, :5])
            lens.append(int(eps.loc[e, "length"]))
            # idle lead-in: frames before any joint moves more than 2 units, the same rule the
            # verification report used.
            mv = np.abs(np.diff(s[:, :5], axis=0)).max(axis=1)
            nz = np.where(mv > 2.0)[0]
            idles.append(int(nz[0]) if len(nz) else len(s))
        poses, seats = np.array(poses), np.array(seats)
        rows.append({
            "tray_spot": spot, "n_demos": len(whole),
            "grasp_pan": float(poses[:, 0].mean()),
            "grasp_pan_std": float(poses[:, 0].std()),
            "grasp_pose_std": float(poses.std(axis=0).mean()),
            "seated_pose_std": float(seats.std(axis=0).mean()),
            "div_whole": divergence(whole),
            "div_reach": divergence(reach),
            "div_carry": divergence(carry),
            "len_mean": float(np.mean(lens)), "len_std": float(np.std(lens)),
            "idle_mean": float(np.mean(idles)),
            **{f"grasp_j{j}": float(poses[:, j].mean()) for j in range(5)},
        })
    return pd.DataFrame(rows)


def spearman(a, b) -> float:
    """Spearman rho without scipy: Pearson on the ranks. Ties get average ranks."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def spearman_p(a, b, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for rho. n = 9 is too small to trust an asymptotic test."""
    rho = abs(spearman(a, b))
    rng = np.random.default_rng(seed)
    b = np.asarray(b, dtype=float)
    null = np.array([abs(spearman(a, rng.permutation(b))) for _ in range(n_perm)])
    return float((null >= rho).mean())


def kendall_w(ranks: np.ndarray, n_perm: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Kendall's coefficient of concordance over (raters, items), with a permutation p-value.

    W = 1 means the raters rank the items identically; W near 0 means they disagree. Here the
    raters are the three policy families and the items are the nine tray spots, so W answers
    "is spot difficulty a property of the spot?" directly. The permutation null shuffles each
    family's ranking independently, which is the right null for "no shared ordering".
    """
    m, n = ranks.shape

    def w_of(r):
        s = r.sum(axis=0)
        return 12 * ((s - s.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))

    w = w_of(ranks)
    rng = np.random.default_rng(seed)
    null = np.array([w_of(np.array([rng.permutation(row) for row in ranks]))
                     for _ in range(n_perm)])
    return float(w), float((null >= w).mean())


def interaction_bootstrap(df, w_obs: float, n_boot: int = 10000, seed: int = 0) -> float:
    """Could 2-trials-per-cell noise alone produce a concordance this low?

    A low Kendall's W is weak evidence for a policy x position interaction on its own: with only
    2 trials per (policy, position) cell, sampling noise attenuates W even when every family
    shares the same position ordering. So the null simulated here is the *interesting* one -
    "position difficulty is shared; families differ only by an overall level" - and the question
    is whether the observed W sits in its low tail.

    Construction: each cell's expected progress is the pooled position effect plus that family's
    offset; two trials are then drawn per cell from the pooled residuals, preserving the real
    noise distribution without assuming a parametric form.
    """
    piv = df.pivot_table(index="policy", columns="tray_spot", values="progress", aggfunc="mean")
    pos_effect = piv.mean(axis=0)                       # shared position difficulty
    pol_offset = piv.mean(axis=1) - piv.to_numpy().mean()
    fitted = np.add.outer(pol_offset.to_numpy(), pos_effect.to_numpy())
    resid = (piv.to_numpy() - fitted).ravel()           # empirical noise pool

    rng = np.random.default_rng(seed)
    n_pol, n_pos = fitted.shape
    out = []
    for _ in range(n_boot):
        sim = fitted + rng.choice(resid, size=fitted.shape, replace=True)
        ranks = np.array([pd.Series(r).rank().to_numpy() for r in sim])
        s_ = ranks.sum(axis=0)
        out.append(12 * ((s_ - s_.mean()) ** 2).sum() / (n_pol ** 2 * (n_pos ** 3 - n_pos)))
    out = np.array(out)
    return float((out <= w_obs).mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_log()
    tr = training_positions()

    # --- rollout outcomes per spot, pooled and per family
    rows = []
    for spot in SPOTS:
        s = df[df.tray_spot == spot]
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        r = {"tray_spot": spot, "n": n, "successes": k, "success_rate": k / n,
             "successes_clean": int(s.success_clean.sum()),
             "ci_lo": lo, "ci_hi": hi, "progress": s.progress.mean()}
        for p in POLICIES:
            sp = s[s.policy == p]
            r[f"progress_{p}"] = sp.progress.mean()
            r[f"success_{p}"] = int(sp.success.sum())
        rows.append(r)
    ro = pd.DataFrame(rows)
    m = ro.merge(tr, on="tray_spot")

    pd.set_option("display.width", 220, "display.max_columns", 60)
    print("=== per position: training demonstrations vs rollout outcome ===")
    print(m[["tray_spot", "n_demos", "grasp_pan", "grasp_pose_std", "div_whole", "div_reach",
             "div_carry", "len_std", "idle_mean", "successes", "progress",
             "progress_act", "progress_smolvla", "progress_pi05"]].round(2).to_string(index=False))

    # --- Q1: do the families agree on which spots are hard?
    prog = m[[f"progress_{p}" for p in POLICIES]].to_numpy().T        # (3 families, 9 spots)
    ranks = np.array([pd.Series(r).rank().to_numpy() for r in prog])
    w, pval = kendall_w(ranks)
    print(f"\n=== policy x position interaction ===")
    print(f"  Kendall's W over 3 families x 9 positions : {w:.3f}  (permutation p = {pval:.3f})")
    print(f"  W = 1 would mean the families rank the positions identically.")
    print("  pairwise Spearman between families' per-position progress:")
    for a, b in itertools.combinations(POLICIES, 2):
        rho = spearman(m[f"progress_{a}"], m[f"progress_{b}"])
        pv = spearman_p(m[f"progress_{a}"].to_numpy(), m[f"progress_{b}"].to_numpy())
        print(f"    {POLICY_LABEL[a]:8s} vs {POLICY_LABEL[b]:8s}: rho = {rho:+.3f}  (p = {pv:.3f})")

    best = {p: m.loc[m[f"progress_{p}"].idxmax(), "tray_spot"] for p in POLICIES}
    worst = {p: m.loc[m[f"progress_{p}"].idxmin(), "tray_spot"] for p in POLICIES}
    p_int = interaction_bootstrap(df, w)
    print(f"  P(W <= {w:.3f} | families share one position ordering, 2 trials/cell) = {p_int:.3f}")
    print(f"  best position per family : { {POLICY_LABEL[k]: v for k, v in best.items()} }")
    print(f"  worst position per family: { {POLICY_LABEL[k]: v for k, v in worst.items()} }")

    # --- Q2: does demonstration quality at a spot predict the outcome there?
    print("\n=== training-data metric vs rollout outcome, across the 9 positions ===")
    print("  (Spearman rho; n = 9, so |rho| > 0.67 is p < 0.05 two-sided)")
    metrics = ["grasp_pan", "grasp_pan_std", "grasp_pose_std", "seated_pose_std",
               "div_whole", "div_reach", "div_carry", "len_mean", "len_std", "idle_mean"]
    corr_rows = []
    for metric in metrics:
        row = {"metric": metric}
        for tgt in ["progress"] + [f"progress_{p}" for p in POLICIES]:
            row[tgt] = spearman(m[metric], m[tgt])
        for tgt in ["progress"] + [f"progress_{p}" for p in POLICIES]:
            row["p_" + tgt.replace("progress", "pooled")] = spearman_p(
                m[metric].to_numpy(), m[tgt].to_numpy())
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    print(corr.round(3).to_string(index=False))

    m.to_csv(OUT / "per_position_joined.csv", index=False)
    corr.to_csv(OUT / "per_position_correlations.csv", index=False)
    print(f"\nwrote per_position_joined.csv and per_position_correlations.csv to "
          f"{OUT.relative_to(REPO)}")

    import make_phase07_position_figs as mf
    mf.render(m, corr, w, pval, p_int, FIGS)
    print(f"wrote position figures to {FIGS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
