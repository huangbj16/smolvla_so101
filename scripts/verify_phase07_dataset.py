#!/usr/bin/env python
"""Integrity verification for the Phase 0.7 portable-rig dataset.

Implements the **Integrity** row of [08 §A.3](../docs/08_data_quality_research.md) — the checks that
filter rather than rank: frame drops, camera-action sync, frozen frames, exposure drift,
camera-pose drift, joint-limit saturation, leader-follower error. Plus the two Phase 0.7
specifics: the idle head/tail that broke the Phase 0.6 rollouts, and whether the recorded
tray-spot schedule is actually in the data.

Run from the repo root in the `lerobot` conda env:

    python scripts/verify_phase07_dataset.py
    python scripts/verify_phase07_dataset.py --repo-id <other> --out results/other_verification

Every video frame of both cameras is decoded, not sampled: sequential PyAV decode runs at
~560 frame/s here, so the whole 70k-frame pass costs ~2 min. Going through `LeRobotDataset[i]`
instead would be ~9 frame/s because it re-seeks per item, which is why the exhaustive pass reads
the mp4s directly and a short separate check exercises the real training path.

Exit code is 0 when nothing BLOCKs, 1 otherwise. WARNs are reported, not fatal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pandas as pd

REPO_DEFAULT = "HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912"
CAL_DEFAULT = Path(__file__).resolve().parent.parent / "docs" / "details" / "calibration" / "current"
OUT_DEFAULT = Path(__file__).resolve().parent.parent / "results" / "phase07_verification"

# sts3215 has 4096 encoder counts over 360 deg; lerobot uses resolution-1 as the divisor.
COUNTS_PER_REV = 4095.0
DEG_PER_COUNT = 360.0 / COUNTS_PER_REV
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# lerobot's SO-101 default is use_degrees=True: the five body joints are recorded in
# MotorNormMode.DEGREES (degrees from the MIDPOINT of the calibrated sweep, unclamped) while the
# gripper is RANGE_0_100 (percent of the sweep, clamped). Reading a body joint as a percentage is
# what makes a perfectly normal -103.8 look like a limit violation.
BODY = JOINTS[:5]

# Thresholds. Each one is a judgement call, so it is named and justified where it is used.
IDLE_DEV_DEG = 2.0  # same rule as scripts/make_trimmed_dataset.py and the Phase 0.5 notebook
IDLE_PAD = 5
FROZEN_MAE = 0.05  # mean abs luminance diff (0-255) below which two frames are "identical"
FROZEN_RUN = 5  # that many consecutive identical frames is a stuck camera, not a still scene
BLACK_MEAN = 8.0  # frame mean luminance below this is an unusable/black frame
WHITE_MEAN = 247.0
# Gripper events are detected per episode, relative to that episode's own rest and peak levels:
# the absolute opening width varies (ep 30 peaks at 17.6% where most episodes reach 25-30%), so a
# fixed threshold misses episodes rather than finding defects.
GRIP_SEATED_FRAC = 0.20  # how far above the hold level the final rise is considered started

BLOCK, WARN, PASS, INFO = "BLOCK", "WARN", "PASS", "INFO"


class Report:
    """Collects one row per check so the summary table and the JSON stay in sync."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.data: dict = {}

    def add(self, section: str, name: str, status: str, detail: str) -> None:
        self.rows.append((section, name, status, detail))
        print(f"  {status:5s}  {name}: {detail}", flush=True)

    @property
    def blocked(self) -> bool:
        return any(r[2] == BLOCK for r in self.rows)

    def table(self) -> str:
        w = max(len(r[1]) for r in self.rows)
        out, seen = [], None
        for section, name, status, detail in self.rows:
            if section != seen:
                out.append(f"\n{section}")
                seen = section
            out.append(f"  {status:5s}  {name:{w}s}  {detail}")
        return "\n".join(out)


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def load_local(repo_id: str) -> tuple[Path, dict, pd.DataFrame, pd.DataFrame]:
    """Locate the cached snapshot and read the three metadata tables plus the frame data.

    Uses LeRobotDataset only to resolve/populate the cache; everything after that reads the
    files directly so the checks do not depend on lerobot's own loader being correct.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id)
    root = Path(ds.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))]
    ).sort_values("episode_index")
    frames = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.glob("data/chunk-*/file-*.parquet"))]
    ).sort_values("index")
    return root, info, episodes, frames, ds


# --------------------------------------------------------------------------------------------
# 1 - manifest consistency
# --------------------------------------------------------------------------------------------
def check_manifest(rep: Report, repo_id: str, root: Path, info: dict, eps: pd.DataFrame, fr: pd.DataFrame) -> None:
    print("\n[1] manifest and metadata consistency")
    n_ep, n_fr = info["total_episodes"], info["total_frames"]

    rep.add("1 manifest", "info.json vs episodes meta",
            PASS if len(eps) == n_ep else BLOCK,
            f"{len(eps)} episode rows vs total_episodes={n_ep}")
    rep.add("1 manifest", "info.json vs frame parquet",
            PASS if len(fr) == n_fr else BLOCK,
            f"{len(fr)} parquet rows vs total_frames={n_fr}")
    rep.add("1 manifest", "episode lengths sum to total",
            PASS if int(eps["length"].sum()) == n_fr else BLOCK,
            f"sum(length)={int(eps['length'].sum())} vs {n_fr}")

    ep_idx = eps["episode_index"].to_numpy()
    rep.add("1 manifest", "episode_index is 0..n-1",
            PASS if np.array_equal(ep_idx, np.arange(n_ep)) else BLOCK,
            f"{ep_idx.min()}..{ep_idx.max()}, {len(set(ep_idx))} unique")

    # The global `index` column is what LeRobotDataset uses to address frames.
    gi = fr["index"].to_numpy()
    rep.add("1 manifest", "global index contiguous",
            PASS if np.array_equal(gi, np.arange(n_fr)) else BLOCK,
            f"0..{gi.max()}, strictly increasing={bool(np.all(np.diff(gi) == 1))}")

    # dataset_from_index/dataset_to_index must tile [0, n_fr) with no overlap or gap, otherwise
    # episode slicing (and therefore the trim script and the eval split) silently misaligns.
    a = eps["dataset_from_index"].to_numpy()
    b = eps["dataset_to_index"].to_numpy()
    tiled = a[0] == 0 and b[-1] == n_fr and np.array_equal(a[1:], b[:-1])
    rep.add("1 manifest", "episode slices tile the frame table",
            PASS if tiled else BLOCK,
            f"from[0]={a[0]}, to[-1]={b[-1]}, contiguous={bool(np.array_equal(a[1:], b[:-1]))}")

    for key in ("observation.images.top", "observation.images.wrist"):
        f = info["features"].get(key)
        ok = f is not None and f["dtype"] == "video" and f["shape"] == [480, 640, 3]
        rep.add("1 manifest", f"feature {key}", PASS if ok else BLOCK,
                f"{f['dtype']} {f['shape']} codec={f['info']['video.codec']}" if f else "MISSING")

    rep.add("1 manifest", "fps / robot_type / codebase_version", INFO,
            f"fps={info['fps']} robot_type={info['robot_type']} v={info['codebase_version']}")

    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_str = tasks.index[0] if tasks.index.name == "task" else tasks.iloc[0, 0]
    expected = ("Pick up the white cylinder from the green tray and place it in the "
                "top left blue hole of the black fixture")
    rep.add("1 manifest", "task string matches Phase 0.7 §5",
            PASS if str(task_str) == expected else WARN,
            f"{len(tasks)} task(s): {str(task_str)!r}")

    # lerobot resolves a dataset at the git revision named by codebase_version, so the tag must
    # exist on the Hub or training dies minutes in with "must be tagged with a codebase version".
    try:
        from huggingface_hub import HfApi

        tags = [t.name for t in HfApi().list_repo_refs(repo_id, repo_type="dataset").tags]
        ok = info["codebase_version"] in tags
        rep.add("1 manifest", "Hub codebase-version tag", PASS if ok else BLOCK,
                f"tags={tags or 'NONE'}, need {info['codebase_version']}")
    except Exception as e:  # offline is not a data problem
        rep.add("1 manifest", "Hub codebase-version tag", WARN, f"could not check ({type(e).__name__})")

    stats = json.loads((root / "meta" / "stats.json").read_text())
    need = {"mean", "std", "min", "max", "q01", "q99"}
    missing = {k: sorted(need - set(stats.get(k, {}))) for k in ("action", "observation.state")}
    missing = {k: v for k, v in missing.items() if v}
    rep.add("1 manifest", "stats cover mean/std and quantiles",
            PASS if not missing else BLOCK,
            "pi05 needs q01/q99 (QUANTILES norm); " + ("all present" if not missing else str(missing)))
    rep.data["manifest"] = {"episodes": int(n_ep), "frames": int(n_fr), "fps": info["fps"]}


# --------------------------------------------------------------------------------------------
# 2 - timing / frame drops
# --------------------------------------------------------------------------------------------
def check_timing(rep: Report, info: dict, eps: pd.DataFrame, fr: pd.DataFrame) -> None:
    print("\n[2] timing and frame drops")
    fps = info["fps"]
    nominal = 1.0 / fps
    ep = fr["episode_index"].to_numpy()
    ts = fr["timestamp"].to_numpy().astype(float)
    fi = fr["frame_index"].to_numpy()

    bad_fi, bad_t0, worst_dt, bad_len = [], [], 0.0, []
    for i, row in eps.iterrows():
        e = int(row["episode_index"])
        m = ep == e
        f, t = fi[m], ts[m]
        if not (f[0] == 0 and np.all(np.diff(f) == 1)):
            bad_fi.append(e)
        if abs(t[0]) > 1e-6:
            bad_t0.append(e)
        if len(f) != int(row["length"]):
            bad_len.append(e)
        if len(t) > 1:
            worst_dt = max(worst_dt, float(np.abs(np.diff(t) - nominal).max()))

    rep.add("2 timing", "frame_index 0..n-1 per episode",
            PASS if not bad_fi else BLOCK,
            "all 50 contiguous" if not bad_fi else f"broken in {bad_fi}")
    rep.add("2 timing", "timestamp starts at 0 per episode",
            PASS if not bad_t0 else WARN,
            "all 50 start at 0.000" if not bad_t0 else f"nonzero in {bad_t0}")
    rep.add("2 timing", "per-episode length matches meta",
            PASS if not bad_len else BLOCK,
            "all 50 match" if not bad_len else f"mismatch in {bad_len}")
    # lerobot's default tolerance_s is 1e-4; a dt error above it makes delta_timestamps lookups fail.
    rep.add("2 timing", "inter-frame dt uniform",
            PASS if worst_dt <= 1e-4 else BLOCK,
            f"worst |dt - {nominal*1000:.2f}ms| = {worst_dt*1000:.4f} ms (tolerance_s=1e-4 -> 0.1 ms)")

    lens = eps["length"].to_numpy()
    rep.add("2 timing", "episode duration", INFO,
            f"{lens.min()}-{lens.max()} frames ({lens.min()/fps:.1f}-{lens.max()/fps:.1f}s), "
            f"median {int(np.median(lens))} ({np.median(lens)/fps:.1f}s)")
    rep.data["timing"] = {"worst_dt_ms": worst_dt * 1000, "dropped_frames": 0 if not bad_fi else None}


# --------------------------------------------------------------------------------------------
# 3 - video layout vs metadata
# --------------------------------------------------------------------------------------------
def check_video_layout(rep: Report, root: Path, info: dict, eps: pd.DataFrame, cams: list[str]) -> dict:
    print("\n[3] video file layout vs metadata")
    fps = info["fps"]
    layout: dict[str, list[dict]] = {}
    for cam in cams:
        key = f"videos/observation.images.{cam}"
        files: dict[tuple[int, int], list[int]] = {}
        for _, r in eps.iterrows():
            files.setdefault((int(r[f"{key}/chunk_index"]), int(r[f"{key}/file_index"])), []).append(
                int(r["episode_index"]))
        entries = []
        for (ch, fidx), ep_list in sorted(files.items()):
            sub = eps[eps["episode_index"].isin(ep_list)].sort_values("episode_index")
            path = root / key / f"chunk-{ch:03d}" / f"file-{fidx:03d}.mp4"
            # Episodes must be laid out back-to-back inside a file: from_timestamp of the next
            # episode == to_timestamp of the previous one. A gap means a frame went missing
            # between episodes and every later episode in that file reads shifted frames.
            frm = sub[f"{key}/from_timestamp"].to_numpy()
            to = sub[f"{key}/to_timestamp"].to_numpy()
            gap = float(np.abs(frm[1:] - to[:-1]).max()) if len(frm) > 1 else 0.0
            span = float(to.max() - frm.min())
            expect = float(sub["length"].sum()) / fps
            entries.append({"path": path, "episodes": ep_list, "frames": int(sub["length"].sum()),
                            "gap_s": gap, "span_s": span, "expect_s": expect,
                            "exists": path.is_file()})
        layout[cam] = entries

        missing = [e["path"].name for e in entries if not e["exists"]]
        rep.add("3 video layout", f"{cam}: files present",
                PASS if not missing else BLOCK,
                f"{len(entries)} file(s), {sum(e['frames'] for e in entries)} frames"
                + (f"; MISSING {missing}" if missing else ""))
        worst_gap = max(e["gap_s"] for e in entries)
        rep.add("3 video layout", f"{cam}: episodes contiguous in file",
                PASS if worst_gap < 1e-6 else BLOCK,
                f"worst inter-episode timestamp gap {worst_gap*1000:.3f} ms")
        worst_span = max(abs(e["span_s"] - e["expect_s"]) for e in entries)
        rep.add("3 video layout", f"{cam}: timestamp span matches frame count",
                PASS if worst_span < 1e-3 else BLOCK,
                f"worst |span - frames/fps| = {worst_span*1000:.3f} ms")
    return layout


# --------------------------------------------------------------------------------------------
# 4 - exhaustive decode pass
# --------------------------------------------------------------------------------------------
def decode_pass(rep: Report, layout: dict, cams: list[str], n_frames: int) -> dict:
    """Decode every frame of every camera once, keeping cheap per-frame summaries.

    Collected per frame: mean luminance (exposure), std luminance (focus/contrast proxy), and
    mean abs difference from the previous frame (motion energy, which drives both the frozen-frame
    check and the camera-action sync check). Also a 24x32 thumbnail mean per episode for the
    camera-pose drift check.
    """
    print("\n[4] exhaustive video decode (every frame, both cameras)")
    out: dict[str, dict] = {}
    for cam in cams:
        t0 = time.time()
        lum = np.full(n_frames, np.nan, dtype=np.float32)
        sd = np.full(n_frames, np.nan, dtype=np.float32)
        mot = np.full(n_frames, np.nan, dtype=np.float32)
        # Frame-to-frame difference skipping one frame. These videos are encoded with video.g=2,
        # so every second frame is an I-frame and the P-frame between decodes bit-identical to it
        # whenever the scene is static. That makes the naive consecutive difference alternate
        # between "real" and "zero"; comparing at a stride of 2 lands on same-type frames and is
        # what actually detects a stuck camera.
        mot2 = np.full(n_frames, np.nan, dtype=np.float32)
        key = np.zeros(n_frames, dtype=bool)
        thumbs: dict[int, np.ndarray] = {}
        thumb_n: Counter = Counter()
        decoded, errors = 0, []

        for entry in layout[cam]:
            # Frames arrive in order; episode boundaries are known from the frame counts, so the
            # k-th decoded frame of the file belongs to the episode whose cumulative range covers k.
            bounds, acc = [], 0
            for e in entry["episodes"]:
                ln = int(_ep_len(e))
                bounds.append((acc, acc + ln, e))
                acc += ln
            try:
                container = av.open(str(entry["path"]))
            except Exception as exc:
                errors.append(f"{entry['path'].name}: open failed {exc}")
                continue
            prev, prev2 = None, None
            k, bi = 0, 0
            try:
                for frame in container.decode(video=0):
                    g = frame.to_ndarray(format="gray").astype(np.float32)
                    while bi < len(bounds) and k >= bounds[bi][1]:
                        bi += 1
                        prev = prev2 = None  # do not compare across an episode boundary
                    if bi >= len(bounds):
                        k += 1
                        decoded += 1
                        continue
                    lo, hi, e = bounds[bi]
                    gi = int(_ep_from(e)) + (k - lo)
                    lum[gi] = g.mean()
                    sd[gi] = g.std()
                    key[gi] = bool(frame.key_frame)
                    mot[gi] = np.abs(g - prev).mean() if prev is not None else np.nan
                    mot2[gi] = np.abs(g - prev2).mean() if prev2 is not None else np.nan
                    prev2, prev = prev, g
                    # 15x20 block-mean thumbnail, accumulated per episode
                    t = g.reshape(15, 32, 20, 32).mean(axis=(1, 3))
                    thumbs[e] = thumbs.get(e, np.zeros_like(t)) + t
                    thumb_n[e] += 1
                    k += 1
                    decoded += 1
            except Exception as exc:
                errors.append(f"{entry['path'].name}: decode failed at frame {k} ({exc})")
            container.close()
            if k != entry["frames"]:
                errors.append(f"{entry['path'].name}: decoded {k} frames, metadata says {entry['frames']}")

        dt = time.time() - t0
        for e in thumbs:
            thumbs[e] /= max(thumb_n[e], 1)
        rep.add("4 decode", f"{cam}: decoded without error",
                PASS if not errors else BLOCK,
                f"{decoded} frames in {dt:.0f}s ({decoded/max(dt,1e-9):.0f} frame/s)"
                + ("" if not errors else f"; {errors[:3]}"))
        rep.add("4 decode", f"{cam}: frame count matches parquet",
                PASS if decoded == n_frames else BLOCK,
                f"{decoded} decoded vs {n_frames} rows")
        n_nan = int(np.isnan(lum).sum())
        rep.add("4 decode", f"{cam}: every parquet row has a frame",
                PASS if n_nan == 0 else BLOCK,
                f"{n_nan} rows with no decoded frame")
        out[cam] = {"lum": lum, "std": sd, "motion": mot, "motion2": mot2, "key": key,
                    "thumbs": thumbs}
    return out


# --------------------------------------------------------------------------------------------
# 5-7 - image-level integrity
# --------------------------------------------------------------------------------------------
def check_frames(rep: Report, vid: dict, cams: list[str], ep: np.ndarray) -> None:
    print("\n[5] black / frozen frames")
    for cam in cams:
        lum, sd, mot = vid[cam]["lum"], vid[cam]["std"], vid[cam]["motion"]
        dark = int((lum < BLACK_MEAN).sum())
        blown = int((lum > WHITE_MEAN).sum())
        rep.add("5 frames", f"{cam}: no black / blown frames",
                PASS if dark == 0 and blown == 0 else WARN,
                f"{dark} frames below mean {BLACK_MEAN}, {blown} above {WHITE_MEAN}; "
                f"luminance {np.nanmin(lum):.1f}-{np.nanmax(lum):.1f} (mean {np.nanmean(lum):.1f})")
        flat = int((sd < 1.0).sum())
        rep.add("5 frames", f"{cam}: no featureless frames", PASS if flat == 0 else WARN,
                f"{flat} frames with luminance std < 1.0 (min {np.nanmin(sd):.1f})")

        # These videos are AV1 with video.g=2, so exactly every second frame is an I-frame. When
        # the scene is static the P-frame between two I-frames is coded as a skip and decodes
        # BIT-IDENTICAL to the one before it. That shows up as a large number of zero-difference
        # frames which look alarming but are a codec artifact, not dropped camera frames - the
        # giveaway is that they are always isolated singletons on non-keyframes.
        k = vid[cam]["key"]
        ident = (~np.isnan(mot)) & (mot < 1e-3)
        on_key = int((ident & k).sum())
        rep.add("5 frames", f"{cam}: zero-difference frames are the video.g=2 artifact", INFO,
                f"{int(ident.sum())} bit-identical frames ({100*ident.mean():.1f}%), "
                f"{on_key} of them on a keyframe (expect 0); keyframes are "
                f"{100*k.mean():.0f}% of frames")

        # The real stuck-camera test compares frames two apart, which lands on same-type frames
        # and so is immune to the artifact above. A truly frozen camera repeats its I-frames too.
        mot2 = vid[cam]["motion2"]
        frozen_runs, worst_run, run, cur_ep = 0, 0, 0, -1
        for i in range(len(mot2)):
            if ep[i] != cur_ep:
                cur_ep, run = ep[i], 0
                continue
            if not np.isnan(mot2[i]) and mot2[i] < FROZEN_MAE:
                run += 1
                worst_run = max(worst_run, run)
                if run == FROZEN_RUN:
                    frozen_runs += 1
            else:
                run = 0
        rep.add("5 frames", f"{cam}: no frozen-frame runs (stride-2 difference)",
                PASS if frozen_runs == 0 else WARN,
                f"{frozen_runs} runs of >={FROZEN_RUN} frames with stride-2 MAE < {FROZEN_MAE} "
                f"(longest run {worst_run}); min {np.nanmin(mot2):.3f}, "
                f"median {np.nanmedian(mot2):.2f} grey levels")

    print("\n[6] exposure drift across the session")
    for cam in cams:
        lum = vid[cam]["lum"]
        per_ep = np.array([np.nanmean(lum[ep == e]) for e in range(ep.max() + 1)])
        # Spearman on episode index: a monotone trend means the lighting or auto-exposure moved
        # over the session, which is confounded with the tray-spot schedule.
        rho = _spearman(np.arange(len(per_ep)), per_ep)
        first5, last5 = per_ep[:5].mean(), per_ep[-5:].mean()
        status = PASS if abs(rho) < 0.5 else WARN
        rep.add("6 exposure", f"{cam}: no monotone drift", status,
                f"Spearman rho={rho:+.2f} vs episode index; per-episode mean luminance "
                f"{per_ep.min():.1f}-{per_ep.max():.1f} (spread {per_ep.max()-per_ep.min():.1f}), "
                f"first5 {first5:.1f} vs last5 {last5:.1f}")

    print("\n[7] camera-pose drift")
    for cam in cams:
        thumbs = vid[cam]["thumbs"]
        ref = thumbs[0]
        corr, shifts = [], []
        for e in sorted(thumbs):
            t = thumbs[e]
            corr.append(_corr(ref.ravel(), t.ravel()))
            shifts.append(_best_shift(ref, t))
        corr = np.array(corr)
        shifted = [(e, s) for e, s in zip(sorted(thumbs), shifts) if s != (0, 0)]
        # The thumbnail is 32x32-pixel blocks, so a non-zero best shift means the episode-mean
        # image moved by at least one block (>=32 px) relative to episode 0 - a real mount shift.
        rep.add("7 pose drift", f"{cam}: mount stable vs episode 0",
                PASS if not shifted else WARN,
                f"best block-shift non-zero in {len(shifted)} episode(s)"
                + (f": {shifted[:6]}" if shifted else "")
                + f"; episode-mean correlation {corr.min():.3f}-{corr.max():.3f}")


# --------------------------------------------------------------------------------------------
# 8 - camera-action sync
# --------------------------------------------------------------------------------------------
def check_sync(rep: Report, vid: dict, cams: list[str], st: np.ndarray, ep: np.ndarray,
               n_ep: int, fps: int) -> None:
    print("\n[8] camera-action sync")
    # Joint speed and image motion energy are two views of the same event, so if the cameras and
    # the servo reads are written to the same frame index their cross-correlation peaks at lag 0.
    # Measured over the TRANSPORT phase (grasp -> seated), where the arm sweeps horizontally and
    # both views see a large, unambiguous change. Over the whole episode the correlation profile
    # is too flat to resolve a lag (the gain from the best shift is under 0.01).
    speed = np.zeros(len(st))
    for e in range(n_ep):
        m = ep == e
        speed[m] = np.concatenate([[0.0], np.abs(np.diff(st[m], axis=0)).sum(axis=1)])

    grip = st[:, 5]
    windows = {}
    for e in range(n_ep):
        idx = np.where(ep == e)[0]
        g = grip[idx]
        n = len(g)
        p1 = int(np.argmax(g[: n // 2]))
        p2 = n // 2 + int(np.argmax(g[n // 2:]))
        gi = p1 + int(np.argmin(g[p1:p2])) if p2 > p1 else 0
        windows[e] = (gi, p2)

    for cam in cams:
        mot = vid[cam]["motion"]
        lags, peaks = [], []
        for e in range(n_ep):
            idx = np.where(ep == e)[0]
            lo, hi = windows[e]
            if hi - lo < 60:
                continue
            a, b = speed[idx][lo:hi], mot[idx][lo:hi]
            ok = ~np.isnan(b)
            a, b = a[ok], b[ok]
            if len(a) < 60:
                continue
            lag, peak = _best_lag(a, b, max_lag=8)
            lags.append(lag)
            peaks.append(peak)
        lags = np.array(lags)
        med = int(np.median(lags))
        within = float((np.abs(lags) <= 2).mean())
        # A USB webcam grabbed asynchronously is routinely one frame stale, so +-2 frames (67 ms)
        # is the pipeline, not a defect - and it is the same at train and at deployment time. A
        # gross misalignment would show as a large median with most episodes agreeing on it.
        status = PASS if abs(med) <= 2 and within >= 0.8 else WARN
        rep.add("8 sync", f"{cam}: aligned during transport", status,
                f"median lag {med:+d} frame ({med/fps*1000:+.0f} ms), {within*100:.0f}% of "
                f"episodes within +/-2 frames; median peak correlation {np.median(peaks):.2f}")

    # The motion-ONSET edge is a much sharper event than the correlation profile, and the two
    # cameras disagree about it in a way that is informative rather than a defect: see the row
    # below. Reported because it is the number that would otherwise look like a sync fault.
    onset_lag = {}
    for cam in cams:
        mot = vid[cam]["motion"]
        vals = []
        for e in range(n_ep):
            idx = np.where(ep == e)[0]
            s = st[idx]
            dev = np.abs(s - s[0]).max(axis=1)
            if not (dev > IDLE_DEV_DEG).any():
                continue
            j_on = int(np.argmax(dev > IDLE_DEV_DEG))
            m = mot[idx]
            base = m[1:25]  # the static lead-in is a clean sample of the sensor noise floor
            thr = np.nanmedian(base) + 6 * np.nanstd(base)
            above = np.where(m[1:] > thr)[0]
            if len(above) and j_on > 0:
                vals.append(int(above[0]) + 1 - j_on)
        onset_lag[cam] = np.array(vals)
        rep.add("8 sync", f"{cam}: motion-onset edge", INFO,
                f"image motion leaves the noise floor {int(np.median(onset_lag[cam])):+d} frames "
                f"after the joints start moving (median over {len(vals)} episodes, "
                f"range {onset_lag[cam].min():+d}..{onset_lag[cam].max():+d})")

    if "top" in onset_lag and "wrist" in onset_lag:
        # In the first 12 frames after onset the arm moves ~25 deg of wrist_flex and under 1.5 deg
        # of everything else: a wrist tilt. The wrist camera is mounted on that joint so it sees it
        # immediately; from a bird's-eye view the arm's silhouette barely changes. So the top
        # camera's late onset is view geometry, and it is consistent with the transport-phase lag
        # above being ~1 frame for both cameras.
        rep.add("8 sync", "top-vs-wrist onset gap is view geometry, not timing", INFO,
                f"wrist sees onset at {int(np.median(onset_lag['wrist'])):+d} frames but top at "
                f"{int(np.median(onset_lag['top'])):+d}; the first motion of every episode is a "
                "wrist tilt (~25 deg of wrist_flex vs <1.5 deg elsewhere), which is invisible from "
                "directly above. Both cameras agree to ~1 frame once the arm translates")


# --------------------------------------------------------------------------------------------
# 9 - joints vs calibration
# --------------------------------------------------------------------------------------------
def check_joints(rep: Report, cal_dir: Path, st: np.ndarray, ac: np.ndarray, keep: np.ndarray) -> None:
    print("\n[9] joint values against the recorded calibration")
    fol = json.loads((cal_dir / "my_follower.json").read_text())
    lead = json.loads((cal_dir / "my_leader.json").read_text())

    for label, cal in (("follower", fol), ("leader", lead)):
        spans = {j: (cal[j]["range_max"] - cal[j]["range_min"]) * DEG_PER_COUNT for j in JOINTS}
        rep.add("9 joints", f"{label} calibrated sweep", INFO,
                "  ".join(f"{j.split('_')[0][:5]}={spans[j]:.0f}deg" for j in JOINTS))

    # Reading a DEGREES-mode joint as if it were a percentage is the trap: a body joint legitimately
    # spans +/-(sweep/2) degrees, which is +/-104 for this arm, not +/-100.
    half = {j: (fol[j]["range_max"] - fol[j]["range_min"]) / 2 * DEG_PER_COUNT for j in BODY}
    rep.add("9 joints", "body joints are DEGREES-mode, not percent", INFO,
            "legitimate range per joint: "
            + "  ".join(f"{j.split('_')[0][:5]}=+/-{half[j]:.1f}" for j in BODY))

    reach_out, cmd_out = {}, {}
    for i, j in enumerate(JOINTS):
        lo, hi = fol[j]["range_min"], fol[j]["range_max"]
        cs, ca = _to_counts(st[:, i], fol, j), _to_counts(ac[:, i], fol, j)
        # How far outside its own calibrated travel the follower actually went, and how far
        # outside it was *asked* to go (the servo clamps Goal_Position to these EEPROM limits).
        reach_out[j] = np.maximum(lo - cs, 0) + np.maximum(cs - hi, 0)
        cmd_out[j] = np.maximum(lo - ca, 0) + np.maximum(ca - hi, 0)

    worst_reach = max((v.max() for v in reach_out.values()), default=0.0)
    detail = ", ".join(f"{j}:{v.max()*DEG_PER_COUNT:.2f}deg" for j, v in reach_out.items() if v.max() > 0.5)
    rep.add("9 joints", "follower stays inside its calibrated travel",
            PASS if worst_reach * DEG_PER_COUNT < 0.5 else WARN,
            f"worst excursion {worst_reach:.1f} counts = {worst_reach*DEG_PER_COUNT:.2f} deg"
            + (f" ({detail})" if detail else " (none)"))

    clamped = {j: v for j, v in cmd_out.items() if (v > 0.5).any()}
    if clamped:
        for j, v in clamped.items():
            sel = v > 0.5
            rep.add("9 joints", f"commands clamped by the servo: {j}", WARN,
                    f"{int(sel.sum())} frames ({100*sel.mean():.1f}% of all, "
                    f"{100*sel[keep].mean():.1f}% of the trimmed window) commanded past the limit by "
                    f"median {np.median(v[sel])*DEG_PER_COUNT:.2f} deg, worst "
                    f"{v.max()*DEG_PER_COUNT:.2f} deg")
    else:
        rep.add("9 joints", "commands clamped by the servo", PASS, "none")

    # Is the clamping doing any damage? Compare tracking error where clamped vs where not.
    for j, v in clamped.items():
        i = JOINTS.index(j)
        cs, ca = _to_counts(st[:, i], fol, j), _to_counts(ac[:, i], fol, j)
        err = np.abs(cs - ca)
        sel = v > 0.5
        rep.add("9 joints", f"clamping cost on {j}", INFO,
                f"|state-action| median {np.median(err[sel])*DEG_PER_COUNT:.2f} deg where clamped vs "
                f"{np.median(err[~sel])*DEG_PER_COUNT:.2f} deg elsewhere")

    print("\n[10] leader-follower tracking error")
    err = np.abs(ac - st)
    for i, j in enumerate(JOINTS):
        rep.add("10 tracking", f"{j}", INFO,
                f"mean {err[:, i].mean():5.2f}  p99 {np.percentile(err[:, i], 99):6.2f}  "
                f"max {err[:, i].max():6.2f}  (units: {'percent' if j == 'gripper' else 'degrees'})")
    rep.data["tracking_p99"] = {j: float(np.percentile(err[:, i], 99)) for i, j in enumerate(JOINTS)}


# --------------------------------------------------------------------------------------------
# 11 - idle head and tail
# --------------------------------------------------------------------------------------------
def trim_window(st: np.ndarray, ep: np.ndarray, n_ep: int):
    """The `dev > IDLE_DEV_DEG` rule from scripts/make_trimmed_dataset.py, as a mask.

    Returns the keep mask, the frames the trim would cut from each end, and separately the raw
    motion-onset / last-motion indices. Onset is the comparable number to Phase 0.5's "median 74
    idle frames"; the cut is onset minus IDLE_PAD, which is what the trimmed dataset will drop.
    """
    keep = np.zeros(len(st), bool)
    cut_head = np.zeros(n_ep, int)
    cut_tail = np.zeros(n_ep, int)
    onset = np.zeros(n_ep, int)
    offset = np.zeros(n_ep, int)
    for e in range(n_ep):
        idx = np.where(ep == e)[0]
        s = st[idx]
        d0 = np.abs(s - s[0]).max(axis=1)
        d1 = np.abs(s - s[-1]).max(axis=1)
        on = int(np.argmax(d0 > IDLE_DEV_DEG)) if (d0 > IDLE_DEV_DEG).any() else 0
        back = np.where(d1 > IDLE_DEV_DEG)[0]
        off = int(back[-1]) + 1 if len(back) else len(s)
        a = max(0, on - IDLE_PAD)
        b = min(len(s), off + IDLE_PAD)
        keep[idx[a:b]] = True
        cut_head[e], cut_tail[e] = a, len(s) - b
        onset[e], offset[e] = on, len(s) - off
    return keep, cut_head, cut_tail, onset, offset


def check_idle(rep: Report, st: np.ndarray, ep: np.ndarray, n_ep: int, fps: int,
               cut_head: np.ndarray, cut_tail: np.ndarray,
               onset: np.ndarray, offset: np.ndarray) -> None:
    print("\n[11] idle head and tail")
    raw = np.array([(ep == e).sum() for e in range(n_ep)])
    kept = raw - cut_head - cut_tail
    # Phase 0.5 measured a median 74-frame lead-in and Phase 0.6 traced the rollout stall to it;
    # Phase 0.7 §8 asked for "start moving the instant the episode starts", so this is the check
    # of whether that instruction landed.
    rep.add("11 idle", "lead-in before first motion", WARN if np.median(onset) > 25 else PASS,
            f"median {int(np.median(onset))} frames ({np.median(onset)/fps:.2f}s), "
            f"range {onset.min()}-{onset.max()}; >25 frames in {int((onset>25).sum())}/{n_ep} "
            f"episodes, >50 in {int((onset>50).sum())}/{n_ep}  "
            f"[Phase 0.5 for comparison: median 74]")
    rep.add("11 idle", "tail after last motion", WARN if np.median(offset) > 25 else PASS,
            f"median {int(np.median(offset))} frames ({np.median(offset)/fps:.2f}s), "
            f"range {offset.min()}-{offset.max()}")
    # SmolVLA and pi0.5 both have chunk_size=50 fixed by their pretrained action experts, so a
    # lead-in longer than 50 frames means the policy's whole first chunk from home is "stay still"
    # - the self-reinforcing stall from Phase 0.6 §4a, with no larger-chunk escape available.
    rep.add("11 idle", "lead-in vs a 50-step action chunk",
            WARN if (onset > 50).sum() else PASS,
            f"{int((onset>50).sum())}/{n_ep} episodes idle longer than one SmolVLA/pi0.5 chunk "
            f"(50 frames = {50/fps:.2f}s); {int((onset>100).sum())}/{n_ep} longer than an ACT "
            f"100-step chunk")
    rep.add("11 idle", "trimmed dataset size", INFO,
            f"{int(kept.sum())}/{int(raw.sum())} frames kept ({100*kept.sum()/raw.sum():.1f}%); "
            f"episode length {kept.min()}-{kept.max()} frames (all >= one 100-step chunk: "
            f"{bool((kept >= 100).all())})")
    rep.data["idle"] = {"onset_median": int(np.median(onset)), "offset_median": int(np.median(offset)),
                        "onset_gt_50": int((onset > 50).sum()),
                        "kept_frames": int(kept.sum()), "kept_fraction": float(kept.sum() / raw.sum())}


# --------------------------------------------------------------------------------------------
# 12 - task structure: gripper cycle, grasp/release poses, tray-spot schedule
# --------------------------------------------------------------------------------------------
def check_structure(rep: Report, st: np.ndarray, ep: np.ndarray, n_ep: int, out: Path) -> None:
    print("\n[12] task structure (gripper cycle, grasp/release poses, tray-spot schedule)")
    grip = st[:, 5]
    grasp_i, seated_i, bad, reopen, chatter = [], [], [], [], []
    for e in range(n_ep):
        idx = np.where(ep == e)[0]
        g = grip[idx]
        n = len(g)
        # The clean D0 trace has three levels: closed at rest -> open to approach -> close on the
        # cylinder (the hold level is set by its diameter) -> open to release -> closed at rest.
        # The approach-open always lands in the first 40% of the episode and the release-open in
        # the last third, so each half contains exactly one of the two peaks. Peak heights differ
        # between the two (release is usually wider), which is why a single fixed or
        # whole-episode-relative threshold misses episodes instead of finding defects.
        p1 = int(np.argmax(g[: n // 2]))
        p2 = n // 2 + int(np.argmax(g[n // 2:]))
        gi = ri = -1
        if p2 > p1:
            # The grasp is the END OF THE CLOSING TRANSITION - the first frame that reaches the
            # hold plateau after the approach-open - not the global minimum between the two peaks.
            # The gripper creeps a little tighter while the cylinder is pressed into the hole, so
            # the global minimum lands during insertion in 5 of the 50 episodes and reports the
            # fixture pose as the grasp pose.
            hold = float(g[p1:p2].min())
            band_g = hold + 0.15 * max(float(g[p1]) - hold, 1e-6)
            gi = p1 + int(np.where(g[p1:p2] <= band_g)[0][0])
            top = float(g[p2])
            # "Seated" = the last frame still at the hold level before the final opening, i.e. the
            # instant the cylinder is in the hole and the gripper has not yet let go. This is the
            # most repeatable moment in the episode, so it is the fixture-stability probe.
            band = hold + GRIP_SEATED_FRAC * max(top - hold, 1e-6)
            at_hold = np.where(g[gi:p2 + 1] <= band)[0]
            ri = gi + int(at_hold[-1]) if len(at_hold) else -1
        if gi < 0 or ri < 0:
            bad.append(e)
        else:
            # A regrasp (dial D3) means letting go and closing again while carrying: the gripper
            # would climb back toward its open level between grasp and seated. This is the direct
            # test, rather than counting threshold crossings over the whole episode.
            span = max(float(max(g[p1], g[p2])) - hold, 1e-6)  # hold = the plateau level
            mid = g[gi:ri + 1]
            reopen.append(float((mid.max() - hold) / span))
            # Gripper chatter: direction changes larger than 1% of the sweep while carrying.
            dg = np.diff(mid)
            sig = dg[np.abs(dg) > 0.01 * span]
            chatter.append(int(np.sum(np.diff(np.sign(sig)) != 0)) if len(sig) > 1 else 0)
        grasp_i.append(gi)
        seated_i.append(ri)

    rep.add("12 structure", "grasp and seated moment detected", PASS if not bad else WARN,
            f"{n_ep - len(bad)}/{n_ep} episodes" + (f"; failed on {bad}" if bad else ""))
    reopen = np.array(reopen)
    n_regrasp = int((reopen > 0.35).sum())
    rep.add("12 structure", "one grasp per episode (no regrasp)",
            PASS if n_regrasp == 0 else WARN,
            f"{n_regrasp}/{len(reopen)} episodes re-open past 35% of the gripper's opening while "
            f"carrying; worst re-open {reopen.max()*100:.0f}%, median {np.median(reopen)*100:.0f}%")
    chatter = np.array(chatter)
    rep.add("12 structure", "no gripper chatter while carrying",
            PASS if np.median(chatter) <= 2 else WARN,
            f"gripper direction changes between grasp and seated: median {int(np.median(chatter))}, "
            f"max {int(chatter.max())} (episode {int(np.argmax(chatter))})")

    valid = [e for e in range(n_ep) if grasp_i[e] >= 0 and seated_i[e] >= 0]
    gp = np.array([st[np.where(ep == e)[0][grasp_i[e]]] for e in valid])
    rp = np.array([st[np.where(ep == e)[0][seated_i[e]]] for e in valid])

    # The target hole is fixed, so every seated pose should be nearly identical. The arm is
    # redundant in the vertical plane - shoulder_lift, elbow_flex and wrist_flex trade off against
    # each other for the same tip position - so only `shoulder_pan` uniquely fixes the horizontal
    # direction to the hole. Gate on pan; report the rest for information.
    pan_sd = float(rp[:, 0].std())
    rep.add("12 structure", "seated pose repeatable (fixture did not move)",
            PASS if pan_sd < 2.0 else WARN,
            f"shoulder_pan std at the seated moment = {pan_sd:.2f} deg over {len(valid)} episodes "
            f"(range {rp[:, 0].min():.2f}..{rp[:, 0].max():.2f}); pan uniquely sets the direction "
            "to the hole, the other three joints are redundant in the vertical plane")
    rep.add("12 structure", "seated pose, all joints", INFO,
            "per-joint std: "
            + "  ".join(f"{j.split('_')[0][:5]}={rp[:, i].std():.2f}" for i, j in enumerate(JOINTS[:5])))

    # The tray spot varies, so the grasp pose should NOT be repeatable - it should separate into
    # the scheduled groups. Phase 0.7 as collected: 5 episodes at each of 9 grid spots (ep 0-44),
    # then 5 at random spots (ep 45-49).
    rep.add("12 structure", "grasp pose varies with tray spot", INFO,
            "per-joint std at grasp: "
            + "  ".join(f"{j.split('_')[0][:5]}={gp[:, i].std():.2f}" for i, j in enumerate(JOINTS[:5])))

    blocks = [list(range(k, k + 5)) for k in range(0, 45, 5)]
    pos = gp[:, :3]
    within, centres = [], []
    for b in blocks:
        rows = [valid.index(e) for e in b if e in valid]
        if len(rows) < 2:
            continue
        c_ = pos[rows].mean(axis=0)
        centres.append(c_)
        within.append(float(np.linalg.norm(pos[rows] - c_, axis=1).mean()))
    centres = np.array(centres)
    between = [float(np.linalg.norm(centres[i] - centres[j]))
               for i in range(len(centres)) for j in range(i + 1, len(centres))]
    ratio = float(np.mean(between) / np.mean(within)) if within else float("nan")
    # If the 9 declared blocks are real, the spread *between* block centres must dominate the
    # spread *within* a block. A ratio near 1 would mean the blocks are indistinguishable.
    # shoulder_pan alone is the non-redundant coordinate (nothing else sets the bearing to the
    # tray spot), so it gives the undiluted separation; the 3-D figure is blurred by the
    # lift/elbow trade-off, which varies within a spot without the gripper moving.
    pan = gp[:, 0]
    pw, pc = [], []
    for b in blocks:
        rows = [valid.index(e) for e in b if e in valid]
        if len(rows) < 2:
            continue
        c_ = pan[rows].mean()
        pc.append(c_)
        pw.append(float(np.abs(pan[rows] - c_).mean()))
    pbet = [abs(pc[i] - pc[j]) for i in range(len(pc)) for j in range(i + 1, len(pc))]
    pratio = float(np.mean(pbet) / np.mean(pw)) if pw else float("nan")
    rep.add("12 structure", "9 tray-spot blocks of 5 are separable",
            PASS if pratio > 4.0 else WARN,
            f"shoulder_pan: within-block spread {np.mean(pw):.2f} deg vs between-centre "
            f"{np.mean(pbet):.2f} deg -> {pratio:.0f}x separation, closest centres "
            f"{min(pbet):.2f} deg apart. In 3-D (pan/lift/elbow) {ratio:.1f}x, diluted by the "
            f"lift-elbow redundancy")

    rand_rows = [valid.index(e) for e in range(45, 50) if e in valid]
    if rand_rows:
        d = [float(np.linalg.norm(pos[r] - centres, axis=1).min()) for r in rand_rows]
        rep.add("12 structure", "episodes 45-49 are off-grid", INFO,
                "distance from each of the last 5 grasp poses to the nearest grid-block centre: "
                + ", ".join(f"{x:.1f}" for x in d) + " deg")

    rep.add("12 structure", "position is confounded with session time", WARN,
            "episodes 0-44 are 9 blocks of 5 consecutive episodes, so anything that drifted over "
            "the session varies with tray spot (same confound as Phase 0.5; see 08 task card). "
            "Also: lerobot's default 'last 5 episodes' holdout would hold out exactly the 5 "
            "off-grid episodes 45-49")

    np.savez(out / "structure.npz", grasp_pose=gp, seated_pose=rp, valid=np.array(valid),
             grasp_index=np.array(grasp_i), seated_index=np.array(seated_i))
    rep.data["structure"] = {"block_separation_ratio_3d": ratio,
                             "block_separation_ratio_pan": pratio,
                             "seated_pose_std": [float(x) for x in rp[:, :5].std(axis=0)],
                             "seated_pan_std": pan_sd}


# --------------------------------------------------------------------------------------------
# 13 - the real training path
# --------------------------------------------------------------------------------------------
def check_training_path(rep: Report, ds, info: dict) -> None:
    print("\n[13] the path lerobot-train will actually take")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    n = ds.num_frames
    idx = np.linspace(0, n - 1, 24).astype(int)
    bad = []
    for i in idx:
        it = ds[int(i)]
        for k in ("observation.images.top", "observation.images.wrist"):
            v = it[k]
            if v.shape != torch.Size([3, 480, 640]) or torch.isnan(v).any() or v.min() < 0 or v.max() > 1:
                bad.append((int(i), k, tuple(v.shape)))
    rep.add("13 train path", "random-access decode through LeRobotDataset",
            PASS if not bad else BLOCK,
            f"{len(idx)} frames spread over the dataset, both cameras, CHW float32 in [0,1]"
            + ("" if not bad else f"; bad: {bad[:3]}"))

    it = ds[0]
    rep.add("13 train path", "language task string reaches the item",
            PASS if isinstance(it.get("task"), str) and it["task"] else BLOCK,
            f"task={it.get('task')!r} (SmolVLA and pi0.5 condition on this)")

    # ACT asks for a 100-step action chunk; both VLAs ask for 50. delta_timestamps is how lerobot
    # fetches them, and it is the one thing that can fail on short episodes or loose timestamps.
    for name, chunk in (("ACT chunk_size=100", 100), ("SmolVLA / pi0.5 chunk_size=50", 50)):
        dt = {"action": [i / info["fps"] for i in range(chunk)]}
        try:
            probe = LeRobotDataset(ds.repo_id, delta_timestamps=dt, episodes=[0, 25, 49])
            a = probe[0]["action"]
            ok = a.shape[0] == chunk and not bool(torch.isnan(a).any())
            rep.add("13 train path", f"action chunk loads: {name}", PASS if ok else BLOCK,
                    f"action shape {tuple(a.shape)}, no NaN={not bool(torch.isnan(a).any())}")
        except Exception as e:
            rep.add("13 train path", f"action chunk loads: {name}", BLOCK,
                    f"{type(e).__name__}: {str(e)[:160]}")


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def make_figures(out: Path, figs: Path, fps: int) -> list[str]:
    """Two plots: the idle head/tail per episode, and the tray-spot block structure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    written = []

    d = np.load(out / "idle.npz")
    onset, offset, length = d["onset"], d["offset"], d["length"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(onset))
    ax.bar(x, onset, color="#c44", label="idle lead-in")
    ax.bar(x, -offset, color="#48a", label="idle tail")
    ax.axhline(50, ls="--", lw=1, color="k")
    ax.axhline(100, ls=":", lw=1, color="k")
    ax.text(-0.4, 53, "SmolVLA / $\\pi$0.5 chunk (50)", ha="left", fontsize=8)
    ax.text(-0.4, 103, "ACT chunk (100)", ha="left", fontsize=8)
    for b in range(5, 50, 5):
        ax.axvline(b - 0.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("episode  (vertical lines mark the 9 tray-spot blocks of 5; 45-49 are off-grid)")
    ax.set_ylabel("frames")
    ax.set_title(f"Phase 0.7 idle head and tail  (median lead-in {int(np.median(onset))} frames "
                 f"= {np.median(onset)/fps:.2f}s)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    p = figs / "phase07_idle.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)

    s = np.load(out / "structure.npz")
    gp, valid = s["grasp_pose"], s["valid"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    # shoulder_pan vs elbow_flex at the grasp is the cleanest 2-D read-out of where in the tray
    # the cylinder was: pan gives the direction, elbow most of the reach distance.
    for b in range(9):
        rows = [i for i, e in enumerate(valid) if b * 5 <= e < b * 5 + 5]
        ax.scatter(gp[rows, 0], gp[rows, 2], s=42, label=f"spot {b+1} (ep {b*5}-{b*5+4})")
    rows = [i for i, e in enumerate(valid) if e >= 45]
    ax.scatter(gp[rows, 0], gp[rows, 2], s=90, marker="x", c="k", label="ep 45-49 (random)")
    ax.set_xlabel("shoulder_pan at grasp (deg)")
    ax.set_ylabel("elbow_flex at grasp (deg)")
    ax.set_title("Grasp pose separates the 9 tray spots")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    p = figs / "phase07_tray_spots.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p.name)
    return written


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
_EP_LEN: dict[int, int] = {}
_EP_FROM: dict[int, int] = {}


def _ep_len(e: int) -> int:
    return _EP_LEN[e]


def _ep_from(e: int) -> int:
    return _EP_FROM[e]


def _to_counts(vals: np.ndarray, cal: dict, joint: str) -> np.ndarray:
    lo, hi = cal[joint]["range_min"], cal[joint]["range_max"]
    if joint == "gripper":  # RANGE_0_100: percent of the sweep
        return lo + (vals / 100.0) * (hi - lo)
    return (lo + hi) / 2 + vals / DEG_PER_COUNT  # DEGREES: offset from the sweep midpoint


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float))


def _best_shift(ref: np.ndarray, t: np.ndarray, r: int = 2) -> tuple[int, int]:
    best, arg = -2.0, (0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            c = _corr(ref.ravel(), np.roll(np.roll(t, dy, axis=0), dx, axis=1).ravel())
            if c > best:
                best, arg = c, (dy, dx)
    return arg


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 10) -> tuple[int, float]:
    """Lag k maximising corr(a[t], b[t+k]); positive k means the image lags the joints."""
    best, arg = -2.0, 0
    for k in range(-max_lag, max_lag + 1):
        x, y = (a[:len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[:len(b) + k])
        if len(x) < 30:
            continue
        c = _corr(x, y)
        if c > best:
            best, arg = c, k
    return arg, best


# --------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_DEFAULT)
    ap.add_argument("--calibration", type=Path, default=CAL_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--skip-video", action="store_true", help="metadata and joint checks only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== Phase 0.7 dataset verification ===\nrepo : {args.repo_id}")
    root, info, eps, fr, ds = load_local(args.repo_id)
    print(f"local: {root}\ncal  : {args.calibration}")

    for _, r in eps.iterrows():
        _EP_LEN[int(r["episode_index"])] = int(r["length"])
        _EP_FROM[int(r["episode_index"])] = int(r["dataset_from_index"])

    rep = Report()
    cams = [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
    n_ep, n_fr, fps = info["total_episodes"], info["total_frames"], info["fps"]
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ac = np.stack(fr["action"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()

    check_manifest(rep, args.repo_id, root, info, eps, fr)
    check_timing(rep, info, eps, fr)
    layout = check_video_layout(rep, root, info, eps, cams)

    if not args.skip_video:
        vid = decode_pass(rep, layout, cams, n_fr)
        check_frames(rep, vid, cams, ep)
        check_sync(rep, vid, cams, st, ep, n_ep, fps)
        np.savez_compressed(args.out / "per_frame.npz",
                            **{f"{c}_{k}": vid[c][k] for c in cams
                               for k in ("lum", "std", "motion", "motion2", "key")})

    keep, cut_head, cut_tail, onset, offset = trim_window(st, ep, n_ep)
    check_joints(rep, args.calibration, st, ac, keep)
    check_idle(rep, st, ep, n_ep, fps, cut_head, cut_tail, onset, offset)
    check_structure(rep, st, ep, n_ep, args.out)
    check_training_path(rep, ds, info)

    counts = Counter(r[2] for r in rep.rows)
    print("\n" + "=" * 100)
    print(rep.table())
    print("\n" + "=" * 100)
    print(f"  {counts[PASS]} PASS   {counts[WARN]} WARN   {counts[BLOCK]} BLOCK   "
          f"{counts[INFO]} INFO      ({time.time()-t0:.0f}s)")
    verdict = "NOT READY - see BLOCK rows" if rep.blocked else "READY for training"
    print(f"  verdict: {verdict}")
    print("=" * 100)

    (args.out / "verification.json").write_text(json.dumps(
        {"repo_id": args.repo_id, "verdict": verdict, "counts": dict(counts),
         "rows": [{"section": s, "check": n, "status": st_, "detail": d} for s, n, st_, d in rep.rows],
         **rep.data}, indent=2))
    np.savez(args.out / "idle.npz", cut_head=cut_head, cut_tail=cut_tail,
             onset=onset, offset=offset,
             length=np.array([(ep == e).sum() for e in range(n_ep)]))
    print(f"\nwrote {args.out / 'verification.json'}")
    try:
        figs = Path(__file__).resolve().parent.parent / "docs" / "details" / "figs"
        print("wrote figures:", ", ".join(make_figures(args.out, figs, fps)))
    except Exception as e:  # a missing matplotlib must not fail the verification
        print(f"figures skipped ({type(e).__name__}: {e})")
    return 1 if rep.blocked else 0


if __name__ == "__main__":
    sys.exit(main())
