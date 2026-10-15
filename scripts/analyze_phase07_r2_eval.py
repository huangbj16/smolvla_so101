#!/usr/bin/env python3
"""Phase 0.7 ROUND-2 eval analysis: two D1 arms, 36 rollouts.

Reuses round 1's measurement code (`analyze_phase07_eval.py`) so the two rounds are measured by
the same instrument, and adds what round 2's session notes asked for:

  * **Did the jerk-selected checkpoint execute smoother?** Section 12.6 picked SmolVLA 20k over the
    loss curve's 5k on offline chunk jerk. The robot is the arbiter, and this is the first test.
  * **Stuck/jitter.** The notes report the arm trapped in mid-air "jittering back and forth".
    Measured as windows that are moving but going nowhere, with the threshold calibrated against
    the human demonstrations rather than picked.
  * **Slower, or just pausing more?** D1 was collected at deliberately varied speed. Episode
    duration cannot tell those apart; instantaneous speed and pause fraction can.
  * **Brightness per block**, because the session notes attribute block 6's 3/5 to lighting.

    python scripts/analyze_phase07_r2_eval.py              # fill + analyse
    python scripts/analyze_phase07_r2_eval.py --no-fill    # leave the CSV alone
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_phase07_eval as r1
from analyze_phase07_eval import (CACHE, N_STAGES, PROGRESS, STAGES, episode_table, frames,
                                  grasp_seated_index, trajectory_metrics, wilson)

REPO = pathlib.Path(__file__).resolve().parents[1]
LOG = REPO / "docs/details/phase07_r2_eval_log.csv"
OUT = REPO / "results/phase07_eval"
D1_TRAIN = CACHE / "so101_tray_cylinder_varied_20261013_082911_trimmed"

ARMS = ["act40k_d1", "sv20k_d1"]
ARM_LABEL = {"act40k_d1": "ACT 40k (D1)", "sv20k_d1": "SmolVLA 20k (D1)"}
ARM_POLICY = {"act40k_d1": "act", "sv20k_d1": "smolvla"}
# Round-1 counterparts, for the only comparison that matters: did round 2 move anything?
R1_OF = {"act40k_d1": "act60k", "sv20k_d1": "sv20k"}


def block_dataset(arm: str, p: int, half: str) -> pathlib.Path:
    return CACHE / f"rollout_phase07_r2_{arm}_p{p}{half.lower()}"


# ------------------------------------------------------------------ log

def fill_log(df: pd.DataFrame) -> pd.DataFrame:
    """Recover episode_index and duration_s from the recordings, as round 1 did."""
    for (arm, p, half), grp in df.groupby(["arm", "pass", "half"], sort=False):
        ds = block_dataset(arm, p, half)
        if not ds.is_dir():
            print(f"  ! missing {ds.name} - {len(grp)} rows unfilled")
            continue
        eps = episode_table(ds).sort_values("episode_index")
        fps = json.load(open(ds / "meta/info.json"))["fps"]
        if len(eps) != len(grp):
            print(f"  ! {ds.name}: {len(eps)} recorded vs {len(grp)} scored")
        for i, (_, row) in enumerate(grp.iterrows()):
            if i >= len(eps):
                break
            df.loc[row.name, "episode_index"] = int(eps.iloc[i]["episode_index"])
            df.loc[row.name, "duration_s"] = round(float(eps.iloc[i]["length"]) / fps, 1)
    return df


def load_log() -> pd.DataFrame:
    df = pd.read_csv(LOG, dtype={"notes": str, "fail_stage": str, "checkpoint": str})
    df["success"] = df.success.astype(str).str.strip().str.lower().isin(["y", "1", "yes", "true"])
    df["fail_stage"] = df.fail_stage.fillna("").astype(str).str.strip()
    df["notes"] = df.notes.fillna("")
    bad = df[df.success & (df.fail_stage != "")]
    for _, r in bad.iterrows():
        print(f"  ! row {r.name + 2}: success with fail_stage={r.fail_stage!r} - clearing")
    df.loc[df.success, "fail_stage"] = ""
    unknown = set(df.loc[~df.success, "fail_stage"]) - set(STAGES)
    if unknown:
        sys.exit(f"unknown fail_stage values: {unknown}")
    # Same definition as round 1: stages cleared, 0-4, 4 == success.
    df["progress"] = np.where(df.success, N_STAGES, df.fail_stage.map(PROGRESS).fillna(0))
    df["arm_label"] = df.arm.map(ARM_LABEL)
    n = df.notes.str.lower()
    df["retry"] = n.str.contains("first try|1st try|multi|many times|retry|tries")
    df["stuck_note"] = n.str.contains("stuck|trapped|back and forth|jitter")
    df["jerky_note"] = n.str.contains("jerky|shaky")
    df["hit_note"] = n.str.contains("hit the cap|tip hit|hit cyl")
    return df


# ------------------------------------------------- stuck / jitter, calibrated on the demos

def window_straightness(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (path length, net/path) over a sliding window of w frames.

    net/path is 1 for a straight move and -> 0 for motion that returns where it started. Reported
    together with the path length so a window can be classed as "moving but going nowhere"
    separately from "barely moving at all", which look identical in the ratio alone.
    """
    if len(a) <= w:
        return np.empty(0), np.empty(0)
    step = np.abs(np.diff(a, axis=0)).sum(axis=1)                 # per-frame travel
    cs = np.concatenate([[0.0], np.cumsum(step)])
    path = cs[w:] - cs[:-w]
    net = np.abs(a[w:] - a[:-w]).sum(axis=1)
    return path, np.divide(net, np.maximum(path, 1e-9))


def demo_calibration(train_std: np.ndarray, w: int, w_stuck: int) -> dict:
    """Thresholds taken from the human demonstrations, not chosen.

    Two different things get measured, and the first version of this conflated them: a policy that
    oscillates constantly while still making progress is JITTERY, and one that moves without going
    anywhere is STUCK. Judged on the net/path ratio alone SmolVLA scored 95% "stuck", which is not
    credible for a policy that reached the hole in a third of trials - the ratio was detecting its
    permanent tremor, not the mid-air traps the session notes describe.

      jitter  - short window, low net/PATH ratio: moving, but not in a straight line
      stuck   - long window, low ABSOLUTE net displacement while still moving: going nowhere

    Both floors are the 1st percentile of what the demonstrations do, so each flags behaviour a
    human demonstrator essentially never produces.
    """
    fr = frames(D1_TRAIN)
    paths, straights, steps, nets_s, paths_s = [], [], [], [], []
    for _, g in fr.groupby("episode_index"):
        a = np.stack(g.sort_values("frame_index")["action"].to_numpy()) / train_std
        pth, st = window_straightness(a, w)
        if len(pth):
            paths.append(pth); straights.append(st)
        if len(a) > w_stuck:
            step = np.abs(np.diff(a, axis=0)).sum(axis=1)
            cs = np.concatenate([[0.0], np.cumsum(step)])
            paths_s.append(cs[w_stuck:] - cs[:-w_stuck])
            nets_s.append(np.abs(a[w_stuck:] - a[:-w_stuck]).sum(axis=1))
        steps.append(np.abs(np.diff(a, axis=0)).sum(axis=1))
    paths, straights = np.concatenate(paths), np.concatenate(straights)
    steps = np.concatenate(steps)
    nets_s, paths_s = np.concatenate(nets_s), np.concatenate(paths_s)
    pause_floor = float(np.median(steps) * 0.10)
    # The stuck gate asks only "is it moving at all", NOT "is it moving as much as a typical
    # demo window". The first version used the demo MEDIAN path as the gate and so missed every
    # real trap: the two episodes the session notes call "trapped in mid air" net nothing for
    # 86% and 81% of their 2 s windows, but jitter at ~40% of a demo's amplitude, so a
    # median-path gate threw them out and reported 1%.
    gate_s = paths_s >= w_stuck * pause_floor
    return {"move_floor": float(np.median(paths)),
            "straight_floor": float(np.percentile(straights[paths >= np.median(paths)], 1)),
            "move_floor_stuck": float(w_stuck * pause_floor),
            "net_floor": float(np.percentile(nets_s[gate_s], 1)),
            "pause_floor": pause_floor}


def motion_metrics(ds: pathlib.Path, train_std: np.ndarray, cal: dict, w: int,
                   w_stuck: int) -> pd.DataFrame:
    """Jitter, stuck, and the speed/pause split, per episode."""
    fr = frames(ds)
    fps = json.load(open(ds / "meta/info.json"))["fps"]
    rows = []
    for ep, g in fr.groupby("episode_index"):
        a = np.stack(g.sort_values("frame_index")["action"].to_numpy()) / train_std
        if len(a) < w_stuck + 2:
            continue
        path, straight = window_straightness(a, w)
        jitter = (path >= cal["move_floor"]) & (straight < cal["straight_floor"])

        step = np.abs(np.diff(a, axis=0)).sum(axis=1)
        cs = np.concatenate([[0.0], np.cumsum(step)])
        path_s = cs[w_stuck:] - cs[:-w_stuck]
        net_s = np.abs(a[w_stuck:] - a[:-w_stuck]).sum(axis=1)
        stuck = (path_s >= cal["move_floor_stuck"]) & (net_s < cal["net_floor"])

        moving = step >= cal["pause_floor"]
        rows.append({
            "episode_index": int(ep),
            "jitter_frac": float(jitter.mean()),
            "stuck_frac": float(stuck.mean()),
            "stuck_s": float(stuck.sum() / fps),
            # Longest single stuck run - one 6 s freeze and six 1 s hesitations are different
            # failures and the mean hides which one happened.
            # A run of R consecutive stuck WINDOWS spans R-1+w_stuck frames, not R: the window
            # itself has width. Dividing the run length alone understates every trap by 2 s.
            "stuck_longest_s": float((max(_longest_run(stuck) - 1, 0) + w_stuck) / fps)
                               if stuck.any() else 0.0,
            "pause_frac": float(1.0 - moving.mean()),
            # Speed WHILE MOVING: this is what separates "slower" from "pauses more".
            "speed_moving": float(step[moving].mean() * fps) if moving.any() else 0.0,
        })
    return pd.DataFrame(rows)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def grasp_pose(ds: pathlib.Path, delta: float = 3.0) -> dict[int, np.ndarray]:
    """Arm pose at the FIRST gripper close after the approach-open, per episode.

    Looser than `grasp_seated_index`, which needs the full close->release cycle and so discards
    every trial that grasped but failed to insert - exactly the trials "was the grasp accurate"
    is about.
    """
    from analyze_phase07_eval import turning_points
    fr = frames(ds)
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    out = {}
    for e in np.unique(ep):
        s = st[ep == e]
        pts = turning_points(s[:, 5], delta)
        mx = next((i for i, (_, k) in enumerate(pts) if k == "max"), None)
        if mx is None:
            continue
        nxt = next((pts[j][0] for j in range(mx + 1, len(pts)) if pts[j][1] == "min"), None)
        if nxt is not None:
            out[int(e)] = s[nxt, :5]
    return out


# ------------------------------------------------------------------ brightness

def brightness(ds: pathlib.Path, n: int = 16) -> dict:
    """Mean and p99 top-camera luminance, sampled across the dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    d = LeRobotDataset(f"HALDijkstraaa/{ds.name}", root=ds)
    if len(d) == 0:
        return {"top_mean": np.nan, "top_p99": np.nan, "wrist_mean": np.nan}
    idx = np.linspace(0, len(d) - 1, min(n, len(d))).astype(int)
    tops, wrists = [], []
    for i in idx:
        s = d[int(i)]
        tops.append(s["observation.images.top"].float().mean(0).numpy())
        if "observation.images.wrist" in s:
            wrists.append(float(s["observation.images.wrist"].float().mean()))
    T = np.stack(tops)
    return {"top_mean": float(T.mean() * 255), "top_p99": float(np.percentile(T, 99) * 255),
            "wrist_mean": float(np.mean(wrists) * 255) if wrists else np.nan}


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--window-s", type=float, default=1.0, help="jitter window (seconds)")
    ap.add_argument("--stuck-window-s", type=float, default=2.0,
                    help="stuck window (seconds): long enough that a normal pause is not a trap")
    ap.add_argument("--no-brightness", action="store_true", help="skip video decode")
    a = ap.parse_args()

    df = load_log()
    if not a.no_fill:
        print("filling episode_index / duration_s from the recordings")
        raw = pd.read_csv(LOG, dtype=str)
        raw = fill_log(raw)
        raw.to_csv(LOG, index=False, lineterminator="\n")
        df = load_log()

    tr = frames(D1_TRAIN)
    train_std = np.stack(tr["action"].to_numpy()).std(axis=0)
    fps = json.load(open(D1_TRAIN / "meta/info.json"))["fps"]
    w, w_stuck = int(round(a.window_s * fps)), int(round(a.stuck_window_s * fps))
    print(f"\ncalibrating on {D1_TRAIN.name}: jitter over {w} frames, stuck over {w_stuck}")
    cal = demo_calibration(train_std, w, w_stuck)
    print("  " + " · ".join(f"{k} {v:.3f}" for k, v in cal.items()))
    demo_base = motion_metrics(D1_TRAIN, train_std, cal, w, w_stuck)
    print(f"  human baseline on the D1 demos: stuck {100*demo_base.stuck_frac.mean():.1f}% · "
          f"jitter {100*demo_base.jitter_frac.mean():.1f}% · "
          f"pause {100*demo_base.pause_frac.mean():.1f}% · "
          f"speed(moving) {demo_base.speed_moving.mean():.2f} · jerk n/a (demos, not a policy)")

    # ---- per-episode trajectory + motion metrics, joined onto the log --------------------
    per_ep = []
    for (arm, p, half), grp in df.groupby(["arm", "pass", "half"], sort=False):
        ds = block_dataset(arm, p, half)
        if not ds.is_dir():
            continue
        t = trajectory_metrics(ds, train_std)
        m = motion_metrics(ds, train_std, cal, w, w_stuck)
        j = t.merge(m, on="episode_index", how="outer")
        j["arm"], j["pass"], j["half"] = arm, p, half
        per_ep.append(j)
    traj = pd.concat(per_ep, ignore_index=True)
    df = df.merge(traj, on=["arm", "pass", "half", "episode_index"], how="left",
                  suffixes=("", "_t"))

    # ---- per arm --------------------------------------------------------------------------
    print("\n" + "=" * 86)
    print("  ROUND 2 PER ARM")
    print("=" * 86)
    rows = []
    for arm in ARMS:
        s = df[df.arm == arm]
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        rows.append({
            "arm": arm, "n": n, "successes": k, "rate": k / n, "ci_lo": lo, "ci_hi": hi,
            "mean_progress": s.progress.mean(), "mean_duration_s": s.duration_s.astype(float).mean(),
            "jerk": s.jerk.mean(), "path_ratio": s.path_ratio.mean(),
            "speed_moving": s.speed_moving.mean(), "pause_frac": s.pause_frac.mean(),
            "jitter_frac": s.jitter_frac.mean(),
            "stuck_frac": s.stuck_frac.mean(), "stuck_longest_s": s.stuck_longest_s.max(),
            "n_ep_with_stuck": int((s.stuck_frac > 0).sum()),
            "terminal_motion": s.terminal_motion.mean(),
            **{f"fail_{st}": int((s.fail_stage == st).sum()) for st in STAGES},
        })
    per_arm = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(per_arm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- round 1 vs round 2 --------------------------------------------------------------
    # The round-1 ROLLOUT RECORDINGS no longer exist: not on disk, and section 10.4's commands
    # ran with --dataset.push_to_hub=false, so nothing reached the Hub either. Only the derived
    # summary survives, and its jerk and speed were normalised by D0's action sigma while
    # everything here uses D1's - and those differ by 3.3x on wrist_roll alone (section 12.6).
    # So success, duration and failure stage carry across; the trajectory measures do not, and
    # are reported as unavailable rather than silently compared.
    print("\n" + "=" * 86)
    print("  ROUND 1 -> ROUND 2")
    print("=" * 86)
    r1sum = pd.read_csv(OUT / "summary_per_arm.csv").set_index("arm")
    r1log = r1.load_log()
    print(f"  {'arm':22s} {'success':>9s} {'progress':>9s} {'duration':>9s}   trajectory")
    for arm in ARMS:
        o = R1_OF[arm]
        s1, s2 = r1sum.loc[o], df[df.arm == arm]
        print(f"  {'r1 ' + o:22s} {int(s1.successes)}/{int(s1.n):<7d} "
              f"{s1.mean_progress:9.2f} {s1.mean_duration_s:8.1f}s   "
              f"recordings deleted - not comparable")
        print(f"  {'r2 ' + arm:22s} {int(s2.success.sum())}/{len(s2):<7d} "
              f"{s2.progress.mean():9.2f} {s2.duration_s.astype(float).mean():8.1f}s   "
              f"jerk {s2.jerk.mean():.1f} · jitter {100*s2.jitter_frac.mean():.0f}% · "
              f"stuck {100*s2.stuck_frac.mean():.1f}%")
    print("\n  !! Round-1 rollout recordings were not kept (push_to_hub=false and the local")
    print("     cache is gone), so the cross-round jerk comparison the session notes ask for")
    print("     cannot be made. Keep the round-3 recordings.")

    # ---- brightness per block ---------------------------------------------------------------
    bright = None
    if not a.no_brightness:
        print("\n" + "=" * 86)
        print("  BRIGHTNESS PER BLOCK   (the session notes attribute block 6 to lighting)")
        print("=" * 86)
        brows = []
        for (arm, p, half), grp in df.groupby(["arm", "pass", "half"], sort=False):
            ds = block_dataset(arm, p, half)
            if not ds.is_dir():
                continue
            b = brightness(ds)
            b.update({"block": int(grp.block.iloc[0]), "arm": arm,
                      "successes": int(grp.success.sum()), "n": len(grp)})
            brows.append(b)
        for extra in ["rollout_phase07_r2_act40k_d1_test_brightlighting",
                      "rollout_phase07_r2_sv20k_d1_test_brightlighting"]:
            ds = CACHE / extra
            if ds.is_dir():
                b = brightness(ds)
                b.update({"block": -1, "arm": extra.split("_r2_")[1].replace("_test_brightlighting",
                          "") + " BRIGHT", "successes": -1, "n": -1})
                brows.append(b)
        b = brightness(D1_TRAIN); b.update({"block": 0, "arm": "D1 TRAINING", "successes": -1, "n": -1})
        brows.append(b)
        bright = pd.DataFrame(brows).sort_values("block")
        print(bright.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        reg = bright[bright.block > 0]
        print(f"\n  regular blocks span {reg.top_mean.min():.1f}-{reg.top_mean.max():.1f} "
              f"({100*(reg.top_mean.max()/reg.top_mean.min()-1):.1f}% range) — "
              f"block 6 is {float(reg.loc[reg.block == 6, 'top_mean'].iloc[0]):.1f}")

    # ---- bright-lighting runs: trajectories only, since they are not scored ---------------
    print("\n" + "=" * 86)
    print("  BRIGHT-LIGHTING RUNS - trajectory only (no per-trial scores yet)")
    print("=" * 86)
    brt = []
    for arm, name in [("act40k_d1", "rollout_phase07_r2_act40k_d1_test_brightlighting"),
                      ("sv20k_d1", "rollout_phase07_r2_sv20k_d1_test_brightlighting")]:
        ds = CACHE / name
        if not ds.is_dir():
            continue
        t = trajectory_metrics(ds, train_std)
        m = motion_metrics(ds, train_std, cal, w, w_stuck)
        j = t.merge(m, on="episode_index", how="outer"); j["arm"] = arm; j["cond"] = "bright"
        brt.append(j)
        base = df[df.arm == arm]
        print(f"  {ARM_LABEL[arm]:18s}  n={len(j)}")
        for lbl, col in [("jerk", "jerk"), ("jitter %", "jitter_frac"), ("stuck %", "stuck_frac"),
                         ("speed(moving)", "speed_moving"), ("pause %", "pause_frac")]:
            f = 100 if "%" in lbl else 1
            print(f"      {lbl:14s} normal {base[col].mean()*f:7.2f}   bright {j[col].mean()*f:7.2f}")
    if brt:
        pd.concat(brt, ignore_index=True).to_csv(OUT / "r2_brightlighting_trajectory.csv",
                                                 index=False)

    # ---- grasp decomposed by axis ----------------------------------------------------------
    # The two axes answer different questions and disagree, which is the point: bearing is
    # position-dependent so only REPEATABILITY across the two attempts at a spot is meaningful,
    # while height is position-INdependent (the cylinder sits at a fixed height) so the demos
    # are a valid absolute reference for it.
    print("\n" + "=" * 86)
    print("  GRASP, DECOMPOSED BY AXIS")
    print("=" * 86)
    demo_pose = np.array([v for v in grasp_pose(D1_TRAIN).values()])
    grasp = {"demo_n": len(demo_pose),
             "demo_mean": demo_pose.mean(axis=0).tolist(),
             "demo_sd": demo_pose.std(axis=0).tolist(), "arms": {}}
    rec = {}
    for (arm, p_, half), g in df.groupby(["arm", "pass", "half"], sort=False):
        ds = block_dataset(arm, p_, half)
        if not ds.is_dir():
            continue
        gp = grasp_pose(ds)
        for _, r in g.iterrows():
            if r.fail_stage == "reach":      # never got near the cylinder; not a grasp attempt
                continue
            v = gp.get(int(r.episode_index))
            if v is not None:
                rec.setdefault(arm, []).append((r.tray_spot, v))
    for arm in ARMS:
        items = rec.get(arm, [])
        if not items:
            continue
        pans = {}
        for spot, v in items:
            pans.setdefault(spot, []).append(v[0])
        rep = [abs(v[0] - v[1]) for v in pans.values() if len(v) == 2]
        H = [1, 2, 3]                         # shoulder_lift, elbow_flex, wrist_flex
        hz = float(np.median([np.median([abs((v[i] - demo_pose[:, i].mean())
                                             / demo_pose[:, i].std()) for _, v in items])
                              for i in H]))
        reached = df[(df.arm == arm) & (df.fail_stage != "reach")]
        cleared = int((reached.success | (reached.fail_stage != "grasp")).sum())
        grasp["arms"][arm] = {
            "n_grasps": len(items), "bearing_repeat_median_deg": float(np.median(rep)),
            "bearing_repeat_max_deg": float(np.max(rep)), "height_z": hz,
            "cleared_grasp": cleared, "reached": len(reached)}
        print(f"  {ARM_LABEL[arm]:18s} bearing repeatability (same spot, 2 tries) "
              f"median {np.median(rep):5.2f}°  max {np.max(rep):5.2f}°")
        print(f"  {'':18s} height error vs demos                    {hz:5.2f} sd")
        print(f"  {'':18s} cleared the grasp stage                  {cleared}/{len(reached)} "
              f"= {100*cleared/len(reached):.0f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r2_grasp_axes.json").write_text(json.dumps(grasp, indent=2))
    demo_row = {"arm": "D1_DEMOS", "n": len(demo_base), "successes": -1,
                **{c: float(demo_base[c].mean()) for c in
                   ["jitter_frac", "stuck_frac", "pause_frac", "speed_moving"]}}
    pd.DataFrame([demo_row]).to_csv(OUT / "r2_demo_baseline.csv", index=False)
    df.to_csv(OUT / "r2_per_trial.csv", index=False)
    per_arm.to_csv(OUT / "r2_summary_per_arm.csv", index=False)
    if bright is not None:
        bright.to_csv(OUT / "r2_brightness_per_block.csv", index=False)
    print(f"\nwrote {OUT}/r2_per_trial.csv, r2_summary_per_arm.csv"
          + (", r2_brightness_per_block.csv" if bright is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
