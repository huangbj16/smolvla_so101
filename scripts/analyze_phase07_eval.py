#!/usr/bin/env python3
"""Phase 0.7 round-1 eval analysis: fill the log from the recordings, then measure.

Two jobs, in order.

**Fill.** `Details/phase07_eval_log.csv` is scored by hand into `success` / `fail_stage` /
`notes`; `episode_index` and `duration_s` are recoverable from the recorded rollout datasets
and are filled in here rather than typed. Episode n of a block is `episode_index` n-1 in that
block's dataset (lerobot writes them in order, and a re-recorded episode is discarded rather
than appended), and duration is `length / fps`.

**Measure.** Success rate, progress, duration, failure stage, per-position breakdown, and two
things that are only visible in the recorded actions rather than in the scoring:

  * **Smoothness.** The session notes call sv5k "shaky" and sv20k "very smooth". That is a
    claim about the action trajectory, so it is measured: per-joint jerk (the std of the second
    difference of the action signal, in units of that joint's training-data std) and a
    joint-space path-length ratio. Both are policy-output properties; neither is visible in a
    per-frame held-out loss.
  * **Terminal motion.** The notes also report that ACT parks when the task is done while
    SmolVLA keeps moving. Measured as the motion in the last second before the operator ended
    the episode.

    python scripts/analyze_phase07_eval.py            # fill + analyse + figures
    python scripts/analyze_phase07_eval.py --no-fill  # analyse only, leave the CSV alone
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
LOG = REPO / "Details/phase07_eval_log.csv"
FIGS = REPO / "Details/figs"
OUT = REPO / "results/phase07_eval"
CACHE = pathlib.Path.home() / ".cache/huggingface/lerobot/HALDijkstraaa"
TRAIN_DS = CACHE / "so101_tray_cylinder_portable_20261003_211912"

# The task's stages in order. `transport` and `retreat` were in the scoring legend but never
# used: no trial failed between lifting the cylinder and arriving over the hole, and a seated
# cylinder always ended with the arm returning to idle, so there is no retreat failure mode.
# Progress is therefore the number of stages cleared, 0-4, with 4 == success.
STAGES = ["reach", "grasp", "transport", "insert"]
PROGRESS = {s: i for i, s in enumerate(STAGES)}   # failed at stage i -> cleared i stages
N_STAGES = len(STAGES)

POLICIES = ["act", "smolvla", "pi05"]
ARMS = ["act40k", "act60k", "sv5k", "sv20k", "pi5k", "pi10k"]
ARM_POLICY = {"act40k": "act", "act60k": "act", "sv5k": "smolvla",
              "sv20k": "smolvla", "pi5k": "pi05", "pi10k": "pi05"}
ARM_LABEL = {"act40k": "ACT 40k", "act60k": "ACT 60k", "sv5k": "SmolVLA 5k",
             "sv20k": "SmolVLA 20k", "pi5k": "π0.5 5k", "pi10k": "π0.5 10k"}
POLICY_LABEL = {"act": "ACT", "smolvla": "SmolVLA", "pi05": "π0.5"}

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light + dark).
C = {"act": "#2a78d6", "smolvla": "#eb6834", "pi05": "#1baf7a"}
# Blue ordinal ramp, steps 250/350/450/600 - the light-surface ordinal floor is step 250.
STAGE_C = {"reach": "#86b6ef", "grasp": "#5598e7", "transport": "#2a78d6", "insert": "#184f95"}
SUCCESS_C = "#0ca30c"   # status: good
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#86857f"
SURFACE, GRID = "#fcfcfb", "#e4e3df"


# ---------------------------------------------------------------- loading

def block_dataset(arm: str, half: str) -> pathlib.Path:
    return CACHE / f"rollout_phase07_r1_{arm}_{half.lower()}"


def episode_table(ds: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(next((ds / "meta/episodes").rglob("*.parquet")))


def frames(ds: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(next((ds / "data").rglob("*.parquet")))


def fill_log(df: pd.DataFrame) -> pd.DataFrame:
    """Recover episode_index and duration_s from the recorded datasets."""
    for (arm, half), grp in df.groupby(["arm", "half"], sort=False):
        ds = block_dataset(arm, half)
        if not ds.is_dir():
            print(f"  ! missing dataset {ds.name} - leaving {len(grp)} rows unfilled")
            continue
        eps = episode_table(ds).sort_values("episode_index")
        fps = json.load(open(ds / "meta/info.json"))["fps"]
        if len(eps) != len(grp):
            print(f"  ! {ds.name}: {len(eps)} episodes recorded but {len(grp)} rows scored")
        for row_i, (_, row) in enumerate(grp.iterrows()):
            if row_i >= len(eps):
                break
            e = eps.iloc[row_i]
            df.loc[row.name, "episode_index"] = int(e["episode_index"])
            df.loc[row.name, "duration_s"] = round(float(e["length"]) / fps, 1)
    return df


def load_log() -> pd.DataFrame:
    df = pd.read_csv(LOG, dtype={"notes": str, "fail_stage": str, "checkpoint": str})
    df["success"] = df["success"].astype(str).str.strip().str.lower().isin(["y", "1", "yes", "true"])
    df["fail_stage"] = df["fail_stage"].fillna("").astype(str).str.strip()
    df["notes"] = df["notes"].fillna("")
    # A scored success cannot also carry a failure stage. One row does (block 11, "lucky
    # grasp"): the note describes how the grasp went, not a failure, so the stage is cleared
    # and the inconsistency is reported rather than silently kept.
    bad = df[df.success & (df.fail_stage != "")]
    for _, r in bad.iterrows():
        print(f"  ! row {r.name + 2}: success=y with fail_stage={r.fail_stage!r} "
              f"(note {r.notes!r}) - treating as success, clearing the stage")
    df.loc[df.success, "fail_stage"] = ""
    unknown = set(df.loc[~df.success, "fail_stage"]) - set(STAGES)
    if unknown:
        sys.exit(f"unknown fail_stage values: {unknown}")
    df["progress"] = np.where(df.success, N_STAGES, df.fail_stage.map(PROGRESS).fillna(0))
    df["policy_label"] = df.policy.map(POLICY_LABEL)
    df["arm_label"] = df.arm.map(ARM_LABEL)

    # Flags recovered from the free-text notes. These are the operator's own qualifiers and
    # they change what the counts mean, so they are parsed rather than ignored.
    n = df.notes.str.lower()
    df["lucky"] = n.str.contains("lucky")
    df["retry"] = n.str.contains("1st try|first try|retry|failed .* then")
    df["shaky"] = n.str.contains("shaky")
    df["smooth"] = n.str.contains("smooth")
    # A "clean" success excludes the ones the operator called lucky - a cylinder kicked into
    # the hole meets the success definition but is not evidence the policy can do the task.
    df["success_clean"] = df.success & ~df.lucky
    for d in ("left", "right", "top"):
        df[f"off_{d}"] = n.str.contains(f"off to {d}")
    return df


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - the right one at n=9, where normal-approx CIs run off the end."""
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rollout_grasp_pan(ds: pathlib.Path) -> dict[int, float]:
    """shoulder_pan at the grasp, per rollout episode, for episodes that actually grasped.

    Used to recover which physical tray spot each T-label refers to: the cylinder's position
    fixes the bearing the arm must take to pick it up, so a rollout grasp at spot T should land
    on the same shoulder_pan as the five training demonstrations recorded there. Episodes that
    never closed on the cylinder have no two-peak gripper cycle and are skipped.
    """
    fr = frames(ds)
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    out = {}
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        s = st[idx]
        gi, ri = grasp_seated_index(s[:, 5])
        if gi >= 0 and ri >= 0:
            out[int(e)] = float(s[gi, 0])
    return out


def trajectory_metrics(ds: pathlib.Path, train_std: np.ndarray) -> pd.DataFrame:
    """Per-episode smoothness and terminal-motion measures from the recorded actions."""
    fr = frames(ds)
    fps = json.load(open(ds / "meta/info.json"))["fps"]
    rows = []
    for ep, g in fr.groupby("episode_index"):
        a = np.stack(g.sort_values("frame_index")["action"].to_numpy())  # (T, 6) degrees
        if len(a) < 5:
            continue
        # Normalise each joint by its spread in the training data, so joints with a large
        # range do not dominate and the number is comparable across policies.
        an = a / train_std
        d1 = np.diff(an, axis=0)
        d2 = np.diff(an, n=2, axis=0)
        path = np.abs(d1).sum(axis=0)                      # total joint-space travel
        net = np.abs(an[-1] - an[0])                       # net displacement
        tail = an[-int(fps):]                              # last second before the operator stopped
        rows.append({
            "episode_index": int(ep),
            "jerk": float(np.std(d2, axis=0).mean() * fps**2),       # per s^2, normalised units
            "speed": float(np.abs(d1).mean() * fps),                  # per s
            "path_ratio": float(path.sum() / max(net.sum(), 1e-6)),   # 1 = straight, high = wandering
            "terminal_motion": float(np.abs(np.diff(tail, axis=0)).sum(axis=0).mean() * fps),
        })
    return pd.DataFrame(rows)


def grasp_seated_index(grip: np.ndarray) -> tuple[int, int]:
    """Grasp and seated frame indices from the gripper trace.

    Ported verbatim from `verify_phase07_dataset.py:check_structure`, which is the validated
    detector behind the verification report's 27x block separation. A naive "last closing
    transition" rule disagrees with it badly: the gripper creeps tighter while the cylinder is
    pressed into the hole, so a global minimum lands during insertion and reports the fixture
    pose as the grasp pose.
    """
    n = len(grip)
    p1 = int(np.argmax(grip[: n // 2]))                      # approach-open
    p2 = n // 2 + int(np.argmax(grip[n // 2:]))              # release-open
    if p2 <= p1:
        return -1, -1
    hold = float(grip[p1:p2].min())                          # the plateau while carrying
    band_g = hold + 0.15 * max(float(grip[p1]) - hold, 1e-6)
    gi = p1 + int(np.where(grip[p1:p2] <= band_g)[0][0])     # END of the closing transition
    band = hold + 0.20 * max(float(grip[p2]) - hold, 1e-6)
    at_hold = np.where(grip[gi:p2 + 1] <= band)[0]
    ri = gi + int(at_hold[-1]) if len(at_hold) else -1
    return gi, ri


def training_spot_stats() -> pd.DataFrame | None:
    """Per-tray-spot consistency of the five demonstrations behind each grid spot.

    Episodes 0-44 are nine consecutive blocks of five (verification report section 5), so block
    b is episodes 5b..5b+4. For each block: the spread of the grasp pose across its five demos
    (how repeatably the operator placed and grasped the cylinder there), the spread of the
    seated pose, and the spread of their durations. These are the candidate explanations for a
    spot being easier for every policy.
    """
    if not TRAIN_DS.is_dir():
        return None
    fr = frames(TRAIN_DS)
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    eps = episode_table(TRAIN_DS).set_index("episode_index")
    rows = []
    for b in range(9):
        pans, poses, seats, lens = [], [], [], []
        for e in range(5 * b, 5 * b + 5):
            idx = np.where(ep == e)[0]
            if not len(idx):
                continue
            s = st[idx]
            gi, ri = grasp_seated_index(s[:, 5])
            if gi < 0 or ri < 0:
                continue
            pans.append(s[gi, 0])      # shoulder_pan uniquely fixes the bearing to the cylinder
            poses.append(s[gi, :5])    # the full arm pose at the grasp
            seats.append(s[ri, :5])
            lens.append(int(eps.loc[e, "length"]))
        if len(pans) >= 2:
            poses, seats = np.array(poses), np.array(seats)
            rows.append({
                "spot_block": b, "n_demos": len(pans),
                "grasp_pan_mean": float(np.mean(pans)),
                "grasp_pan_std": float(np.std(pans)),
                # Spread of the whole grasp pose, not just the bearing: the quantity a policy
                # has to reproduce. Mean per-joint std over the five arm joints.
                "grasp_pose_std": float(poses.std(axis=0).mean()),
                "seated_pose_std": float(seats.std(axis=0).mean()),
                "len_mean": float(np.mean(lens)), "len_std": float(np.std(lens)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    if not a.no_fill:
        print("filling episode_index / duration_s from the recordings")
        # checkpoint must stay a string: read as int it loses the leading zero and
        # "040000" becomes "40000", which no longer names a checkpoint directory.
        raw = pd.read_csv(LOG, dtype={"checkpoint": str})
        raw = fill_log(raw)
        raw["episode_index"] = raw["episode_index"].astype("Int64")
        raw.to_csv(LOG, index=False, lineterminator="\n")
        print(f"  wrote {LOG.relative_to(REPO)}")

    df = load_log()
    assert len(df) == 54, f"expected 54 trials, found {len(df)}"

    # --- trajectory metrics, joined on (arm, half, episode_index)
    train_std = np.stack(frames(TRAIN_DS)["action"].to_numpy()).std(axis=0) if TRAIN_DS.is_dir() \
        else np.ones(6)
    tm = []
    for (arm, half) in df[["arm", "half"]].drop_duplicates().itertuples(index=False):
        ds = block_dataset(arm, half)
        if ds.is_dir():
            t = trajectory_metrics(ds, train_std)
            t["arm"], t["half"] = arm, half
            tm.append(t)
    tm = pd.concat(tm, ignore_index=True)
    df = df.merge(tm, on=["arm", "half", "episode_index"], how="left")

    # --- per-arm summary
    rows = []
    for arm in ARMS:
        s = df[df.arm == arm]
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        rows.append({
            "arm": arm, "policy": ARM_POLICY[arm], "n": n, "successes": k,
            "success_rate": k / n, "ci_lo": lo, "ci_hi": hi,
            "successes_clean": int(s.success_clean.sum()),
            "success_rate_clean": s.success_clean.mean(), "lucky": int(s.lucky.sum()),
            "mean_progress": s.progress.mean(), "mean_progress_frac": s.progress.mean() / N_STAGES,
            "mean_duration_s": s.duration_s.mean(), "sd_duration_s": s.duration_s.std(),
            "mean_duration_success_s": s[s.success].duration_s.mean() if k else np.nan,
            "jerk": s.jerk.mean(), "path_ratio": s.path_ratio.mean(),
            "terminal_motion": s.terminal_motion.mean(),
            **{f"fail_{st}": int((s.fail_stage == st).sum()) for st in STAGES},
        })
    per_arm = pd.DataFrame(rows)

    rows = []
    for p in POLICIES:
        s = df[df.policy == p]
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        rows.append({"policy": p, "n": n, "successes": k, "success_rate": k / n,
                     "successes_clean": int(s.success_clean.sum()),
                     "success_rate_clean": s.success_clean.mean(),
                     "ci_lo": lo, "ci_hi": hi, "mean_progress": s.progress.mean(),
                     "mean_duration_s": s.duration_s.mean(), "jerk": s.jerk.mean(),
                     "path_ratio": s.path_ratio.mean(),
                     "terminal_motion": s.terminal_motion.mean(),
                     **{f"fail_{st}": int((s.fail_stage == st).sum()) for st in STAGES}})
    per_policy = pd.DataFrame(rows)

    rows = []
    for spot in sorted(df.tray_spot.unique(), key=lambda s: int(s[1:])):
        s = df[df.tray_spot == spot]
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        rows.append({"tray_spot": spot, "n": n, "successes": k, "success_rate": k / n,
                     "ci_lo": lo, "ci_hi": hi, "mean_progress": s.progress.mean(),
                     "half": s.half.iloc[0],
                     **{f"fail_{st}": int((s.fail_stage == st).sum()) for st in STAGES}})
    per_spot = pd.DataFrame(rows)

    # --- print
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== per arm ===")
    print(per_arm[["arm", "successes", "n", "success_rate", "ci_lo", "ci_hi", "mean_progress",
                   "mean_duration_s", "jerk", "path_ratio", "terminal_motion",
                   "fail_reach", "fail_grasp", "fail_insert"]].round(3).to_string(index=False))
    print("\n=== per policy ===")
    print(per_policy.round(3).to_string(index=False))
    print("\n=== per tray spot ===")
    print(per_spot.round(3).to_string(index=False))

    print("\n=== notes: operator qualifiers ===")
    print(f"  'lucky' successes      : {int(df.lucky.sum())}  "
          f"{df.loc[df.lucky, ['arm', 'tray_spot']].to_dict('records')}")
    print(f"  retry-then-succeed     : {int(df.retry.sum())}")
    print(f"  called shaky / smooth  : {int(df.shaky.sum())} / {int(df.smooth.sum())}  "
          f"{df.loc[df.shaky | df.smooth, ['arm', 'notes']].to_dict('records')}")
    d = {k: int(df[f'off_{k}'].sum()) for k in ('left', 'right', 'top')}
    print(f"  directional misses     : {d}  (total {sum(d.values())} of "
          f"{int((~df.success).sum())} failures annotated)")
    byp = df.groupby('policy')[['off_left', 'off_right', 'off_top']].sum()
    print(byp.to_string())

    print("\n=== T8 breakdown (the spot the session notes flagged as easy) ===")
    print(df[df.tray_spot == "T8"][["arm", "success", "lucky", "fail_stage", "notes"]]
          .to_string(index=False))

    ts = training_spot_stats()
    if ts is not None:
        print("\n=== training demonstrations, per spot block (episodes 5b..5b+4) ===")
        print(ts.round(3).to_string(index=False))
        ts.to_csv(OUT / "training_spot_consistency.csv", index=False)

        # --- recover the T-label -> physical spot mapping from the rollout grasps
        gp = {}
        for (arm, half) in df[["arm", "half"]].drop_duplicates().itertuples(index=False):
            ds = block_dataset(arm, half)
            if ds.is_dir():
                for e, pan in rollout_grasp_pan(ds).items():
                    gp[(arm, half, e)] = pan
        df["grasp_pan"] = [gp.get((r.arm, r.half, r.episode_index), np.nan)
                           for r in df.itertuples()]
        # Only trials that actually picked the cylinder up carry a meaningful grasp bearing. A
        # policy that flails still produces a two-peak gripper trace, so the detector fires on
        # failures too and the resulting "grasp pan" is noise - with those included the spread
        # within one T label reaches 32 degrees against a 0.3 degree within-block demo spread,
        # and no mapping is recoverable. progress >= 2 == cleared reach and grasp.
        df.loc[df.progress < 2, "grasp_pan"] = np.nan
        centres = ts.set_index("spot_block")["grasp_pan_mean"]
        rows = []
        for spot in sorted(df.tray_spot.unique(), key=lambda s: int(s[1:])):
            v = df.loc[df.tray_spot == spot, "grasp_pan"].dropna()
            if v.empty:
                rows.append({"tray_spot": spot, "n_grasps": 0})
                continue
            med = float(v.median())
            dist = (centres - med).abs().sort_values()
            rows.append({"tray_spot": spot, "n_grasps": len(v), "grasp_pan_median": med,
                         "grasp_pan_spread": float(v.max() - v.min()),
                         "nearest_block": int(dist.index[0]), "dist_deg": float(dist.iloc[0]),
                         "runner_up_block": int(dist.index[1]),
                         "runner_up_deg": float(dist.iloc[1])})
        mapping = pd.DataFrame(rows)
        print("\n=== recovering T-label -> training spot block, from rollout grasp bearings ===")
        print(mapping.round(3).to_string(index=False))
        ok = mapping.dropna(subset=["nearest_block"]) if "nearest_block" in mapping else mapping
        if len(ok):
            unique = ok.nearest_block.nunique() == len(ok)
            margin = (ok.runner_up_deg - ok.dist_deg).min()
            print(f"  one-to-one: {unique} | smallest margin to the runner-up: {margin:.2f} deg "
                  f"| within-block demo spread is ~0.3 deg")
        mapping.to_csv(OUT / "tray_spot_mapping.csv", index=False)

    per_arm.to_csv(OUT / "summary_per_arm.csv", index=False)
    per_policy.to_csv(OUT / "summary_per_policy.csv", index=False)
    per_spot.to_csv(OUT / "summary_per_spot.csv", index=False)
    df.to_csv(OUT / "trials_with_trajectory_metrics.csv", index=False)
    print(f"\nwrote 4 summary CSVs to {OUT.relative_to(REPO)}")

    if not a.no_figs:
        import make_phase07_eval_figs as mf
        mf.render(df, per_arm, per_policy, per_spot, ts, FIGS)
        print(f"wrote figures to {FIGS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    raise SystemExit(main())
