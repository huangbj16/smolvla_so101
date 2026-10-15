#!/usr/bin/env python3
"""Merge the D0 and D1 trimmed datasets into one 100-episode set for the pi0.5 round-2 run.

`lerobot-train` takes a single `--dataset.repo_id`, so training on D0 and D1 together needs them
aggregated first. The two were recorded ten days apart but share a task string, fps, robot type,
camera keys and action names, so `aggregate_datasets` can concatenate them directly.

The script does three things beyond the merge, because an aggregated dataset that is subtly wrong
would not announce itself:

  1. Refuses to merge sources that disagree on anything the policy sees (task, fps, features).
  2. Verifies the episode mapping by LENGTH SIGNATURE rather than assuming source order is
     preserved - the held-out split is specified by episode index, so a silent reordering would
     train on the eval episodes.
  3. Prints the held-out indices in merged coordinates, and the on-disk size for the upload.

    python scripts/merge_d0_d1.py              # merge and verify
    python scripts/merge_d0_d1.py --verify-only   # re-check an existing merge
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

D0 = "HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed"
D1 = "HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed"
MERGED = "HALDijkstraaa/so101_tray_cylinder_d0d1_100ep_trimmed"

# Each source holds out the last episode of blocks 2, 4, 6, 8 plus one off-grid episode. Keeping
# both sources' holdouts means the merged eval set stays balanced across tray spots AND across
# the two recording sessions, which is what makes the eval loss comparable to round 1's.
HOLDOUT_PER_SOURCE = [9, 19, 29, 39, 49]


def episode_lengths(meta) -> list[int]:
    """Per-episode frame counts, however this lerobot version stores them."""
    eps = meta.episodes
    for key in ("length", "episode_length", "num_frames"):
        try:
            if key in eps.column_names:
                return [int(v) for v in eps[key]]
        except Exception:
            pass
    # Fall back to counting rows per episode_index.
    import collections
    counts = collections.Counter()
    for col in ("dataset_from_index", "dataset_to_index"):
        if col not in getattr(eps, "column_names", []):
            break
    else:
        return [int(b) - int(a) for a, b in zip(eps["dataset_from_index"], eps["dataset_to_index"])]
    raise SystemExit(f"cannot read per-episode lengths; columns = {getattr(eps, 'column_names', '?')}")


def describe(meta, label: str) -> dict:
    d = {"episodes": meta.total_episodes, "frames": meta.total_frames, "fps": meta.fps,
         "robot": meta.robot_type, "cameras": tuple(meta.camera_keys),
         "action": tuple(meta.features["action"]["shape"]),
         "tasks": tuple(sorted(meta.tasks.index))}
    print(f"  {label:28s} {d['episodes']:3d} ep · {d['frames']:6d} fr · {d['fps']} fps · "
          f"{len(d['tasks'])} task(s)")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="delete an existing merge first")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.utils.constants import HF_LEROBOT_HOME

    print("sources")
    metas, specs, lengths = [], [], []
    for repo in (D0, D1):
        # Instantiating the dataset (not just the metadata) is what pulls the parquet and videos.
        LeRobotDataset(repo)
        m = LeRobotDatasetMetadata(repo)
        metas.append(m)
        specs.append(describe(m, repo.split("/")[-1]))
        lengths.append(episode_lengths(m))

    # ---- refuse to merge sources that disagree on anything the policy sees -------------------
    a, b = specs
    for k in ("fps", "robot", "cameras", "action", "tasks"):
        if a[k] != b[k]:
            raise SystemExit(
                f"\nREFUSING TO MERGE: the two datasets disagree on {k!r}.\n"
                f"  D0: {a[k]}\n  D1: {b[k]}\n"
                "A policy trained across this boundary would see two different things under one\n"
                "label. Reconcile the sources first.")
    print(f"  -> compatible: same fps, robot, cameras, action spec and task string")
    print(f"     task: {a['tasks'][0]!r}")

    dst = HF_LEROBOT_HOME / MERGED
    if args.force and dst.exists():
        print(f"\n--force: removing {dst}")
        shutil.rmtree(dst)

    if not args.verify_only and not dst.exists():
        from lerobot.datasets.aggregate import aggregate_datasets
        print(f"\nmerging -> {dst}")
        aggregate_datasets(repo_ids=[D0, D1], aggr_repo_id=MERGED, aggr_root=dst)
    elif dst.exists():
        print(f"\nmerge already present at {dst}")
    else:
        raise SystemExit("--verify-only but no merge exists")

    # ---- verify -------------------------------------------------------------------------------
    print("\nverifying")
    mm = LeRobotDatasetMetadata(MERGED, root=dst)
    describe(mm, "merged")

    want_eps = sum(s["episodes"] for s in specs)
    want_fr = sum(s["frames"] for s in specs)
    ok = True
    for label, got, want in (("episodes", mm.total_episodes, want_eps),
                             ("frames", mm.total_frames, want_fr)):
        good = got == want
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} total {label}: {got} (expected {want})")

    # The holdout is specified by INDEX, so a reordering during aggregation would silently move
    # eval episodes into training. Compare the per-episode length sequence, which is effectively a
    # fingerprint: the chance of two different orderings matching on 100 lengths is negligible.
    merged_len = episode_lengths(mm)
    expect_len = lengths[0] + lengths[1]
    same = merged_len == expect_len
    ok &= same
    print(f"  {'OK  ' if same else 'FAIL'} episode order preserved "
          f"(length signature over {len(merged_len)} episodes)")
    if not same:
        bad = [i for i, (g, w) in enumerate(zip(merged_len, expect_len)) if g != w]
        print(f"       first mismatches at merged indices {bad[:10]}")
        print(f"       -> DO NOT use the holdout indices below; the mapping is not D0 then D1.")

    n0 = specs[0]["episodes"]
    holdout = HOLDOUT_PER_SOURCE + [e + n0 for e in HOLDOUT_PER_SOURCE]
    train = [e for e in range(want_eps) if e not in holdout]
    held_fr = sum(merged_len[e] for e in holdout) if same else 0

    print(f"\nheld-out episodes (merged indices): {holdout}")
    print(f"  = D0 {HOLDOUT_PER_SOURCE} and D1 {HOLDOUT_PER_SOURCE} (+{n0})")
    print(f"  {len(train)} train / {len(holdout)} eval episodes"
          f" · {want_fr - held_fr} train / {held_fr} eval frames"
          f" · eval_split = {len(holdout) / want_eps:.2f}")

    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"\non disk: {size / 1e9:.2f} GB at {dst}")
    print(f"  upload this whole directory to the box, then point training at it with")
    print(f"  --dataset.repo_id={MERGED} --dataset.root=<remote path>")

    print("\n" + ("ALL CHECKS PASSED" if ok else "!! SOME CHECKS FAILED - see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
