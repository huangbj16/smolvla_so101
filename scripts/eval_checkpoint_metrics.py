#!/usr/bin/env python
"""The two offline metrics round 1 asked for: chunk jerk, and held-out loss split by task stage.

**Why.** Round 1 found that held-out loss ranked the three policy families backwards
([phase07_portable_rig.md §11.3](../docs/details/phase07_portable_rig.md)). The diagnosis was that
loss is a *per-frame, teacher-forced, zeroth-difference* quantity while what separated SmolVLA's two
checkpoints on the robot was smoothness - a *second-difference* property of the action sequence - and
*where* in the task the error lands. This measures both without a robot:

  * **chunk jerk** - the std of the second difference of the predicted action chunk, per joint, in
    units of that joint's training-data std, scaled to per-s^2. Identical formula to the rollout jerk
    in `analyze_phase07_eval.py:trajectory_metrics`, so the numbers sit on one scale. The pre-registered
    prediction is a *ratio*: SmolVLA 5k should show ~2x the jerk of 20k, matching the 2.0x measured on
    the robot.
  * **per-stage held-out loss** - the same `policy.forward` loss the training curve logs, but averaged
    over the frames of one task stage at a time instead of over all of them. Stages come from the
    gripper trace via `grasp_seated_index`, the detector rewritten on 2026-10-13. Prediction: 5k and
    20k separate at **grasp** and **insert**, where tolerance is tightest, even though 5k wins pooled.

**Post-hoc on saved checkpoints, deliberately.** The obvious alternative was to log these from inside
the training loop's eval branch. For a flow-matching policy `predict_action_chunk` draws noise from the
global RNG, which advances the generator and perturbs every subsequent training batch and dropout mask -
so a run with the metric would not be comparable to one without it, would not reproduce round 1, and
would break `--resume` determinism. The metric would change the thing it measures.

Both numbers are stochastic for the VLAs (flow timestep and noise are sampled per forward), so every
checkpoint is measured `--repeats` times with fixed seeds and reported as mean +/- std. **Differences
smaller than that std are not differences.**

    python scripts/eval_checkpoint_metrics.py \
        --output-dir outputs/hub/smolvla_tray_cylinder_s1000_trim \
        --dataset HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed

Writes `<output-dir>/checkpoint_metrics.json`. Run `--all-round1` to do all six round-1 arms in one go.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import CACHE, frames, grasp_seated_index, turning_points
from eval_checkpoints_heldout import HOLDOUT, find_checkpoints

STAGES = ["reach", "grasp", "transport", "insert"]


def checkpoint_rename_map(path: pathlib.Path) -> dict[str, str]:
    """The rename map the checkpoint was trained with, read from its own saved preprocessor.

    `make_policy` validates the dataset's visual features against the policy config, and the VLA
    checkpoints declare renamed keys - `camera1`/`camera2` for SmolVLA, `base_0_rgb`/
    `left_wrist_0_rgb` for pi0.5 - while the dataset has `top`/`wrist`. Training passed
    `--rename_map` to bridge that; here the same map is recovered from the checkpoint instead of
    hardcoded, so it cannot drift from what the policy actually expects. ACT's is empty.
    """
    cfg = json.loads((path / "policy_preprocessor.json").read_text())
    for step in cfg.get("steps", []):
        if "rename" in step.get("registry_name", ""):
            return dict(step.get("config", {}).get("rename_map", {}))
    return {}

# The six round-1 arms, as section 10 names them.
ROUND1 = [
    ("act40k", "act_tray_cylinder_s1000_trim", 40000),
    ("act60k", "act_tray_cylinder_s1000_trim", 60000),
    ("sv5k", "smolvla_tray_cylinder_s1000_trim", 5000),
    ("sv20k", "smolvla_tray_cylinder_s1000_trim", 20000),
    ("pi5k", "pi05_tray_cylinder_s1000_trim_A100", 5000),
    ("pi10k", "pi05_tray_cylinder_s1000_trim_A100", 10000),
]


# ---------------------------------------------------------------------------------------------
# stage segmentation
# ---------------------------------------------------------------------------------------------
def episode_stages(grip: np.ndarray) -> np.ndarray | None:
    """Per-frame stage index for one episode, from its gripper trace.

    The detector already finds the two gripper openings and the grasp/seated frames between them.
    Four segments fall straight out of those four landmarks, and they are the task card's own stages:

        reach      0        -> approach opening   moving toward the cylinder, gripper opening
        grasp      approach -> grasp              descending and closing on it
        transport  grasp    -> seated             carrying, gripper held at the hold level
        insert     seated   -> end                seating and releasing

    Returns None if the episode has no recognisable structure, so it can be skipped rather than
    silently mislabelled.
    """
    gi, ri = grasp_seated_index(grip)
    if gi < 0 or ri <= gi:
        return None
    span = float(grip.max()) - float(np.percentile(grip, 5))
    maxima: list[int] = []
    for frac in (0.18, 0.15, 0.12, 0.08):
        pts = turning_points(grip, frac * span)
        maxima = [i for i, kind in pts if kind == "max"]
        if len(maxima) >= 2:
            break
    if len(maxima) < 2:
        return None
    a_peak = maxima[0]
    if not (a_peak < gi < ri):
        return None
    lab = np.full(len(grip), STAGES.index("insert"), dtype=np.int64)
    lab[:a_peak] = STAGES.index("reach")
    lab[a_peak:gi] = STAGES.index("grasp")
    lab[gi:ri] = STAGES.index("transport")
    return lab


def stage_labels(dataset_repo: str) -> dict[int, np.ndarray]:
    """Stage labels for every held-out episode, keyed by episode_index."""
    fr = frames(CACHE / dataset_repo.split("/")[-1])
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    out = {}
    for e in HOLDOUT:
        idx = np.where(ep == e)[0]
        if not len(idx):
            continue
        lab = episode_stages(st[idx, 5])
        if lab is None:
            print(f"  ! episode {e}: no stage structure found - excluded from the per-stage split")
            continue
        out[int(e)] = lab
    return out


# ---------------------------------------------------------------------------------------------
# the two measurements
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def chunk_jerk(policy, preprocessor, postprocessor, loader, camera_keys, device, fps,
               train_std: torch.Tensor, seed: int) -> float:
    """Mean jerk of the predicted action chunk over the held-out frames.

    Same formula as the rollout jerk: per joint, the std over time of the second difference of the
    action in units of that joint's training-data std, averaged over joints, scaled by fps^2. The
    chunk is predicted in normalised space, so it is pushed through the postprocessor first - which
    matters because pi0.5 normalises with QUANTILES and the other two with MEAN_STD, and without
    the unnormalisation the three would sit on three different scales.
    """
    torch.manual_seed(seed)
    policy.eval()
    vals = []
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast("cpu", enabled=False))
    with autocast:
        for batch in loader:
            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
            chunk = policy.predict_action_chunk(preprocessor(batch))     # (B, T, A), normalised
            chunk = postprocessor(chunk).to(torch.float32)               # degrees
            if chunk.shape[1] < 3:
                continue
            d2 = torch.diff(chunk / train_std, n=2, dim=1)               # (B, T-2, A)
            vals.append((d2.std(dim=1).mean(dim=1) * fps ** 2).cpu())
    return float(torch.cat(vals).mean()) if vals else float("nan")


@torch.no_grad()
def forward_loss(policy, preprocessor, loader, camera_keys, device, seed: int) -> float:
    """One pass of lerobot's own eval loss over whatever subset `loader` holds."""
    torch.manual_seed(seed)
    policy.eval()
    total, n = 0.0, 0
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast("cpu", enabled=False))
    with autocast:
        for batch in loader:
            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
            loss, _ = policy.forward(preprocessor(batch))
            total += loss.item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------------------------
def build_datasets(dataset_repo: str, policy_cfg, batch_size: int, num_workers: int, labels):
    """The pooled held-out loader plus one loader per stage.

    A stage loader is a Subset of the same dataset, so every frame is preprocessed identically and
    the pooled number is exactly the frame-weighted combination of the stage numbers (up to batch
    boundaries). Separate loaders rather than per-sample losses, because `policy.forward` returns
    one scalar per batch and nothing in lerobot exposes the per-sample vector.
    """
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.utils.collate import lerobot_collate_fn

    meta = LeRobotDatasetMetadata(dataset_repo)
    delta = resolve_delta_timestamps(policy_cfg, meta)
    ds = LeRobotDataset(dataset_repo, episodes=HOLDOUT, delta_timestamps=delta,
                        return_uint8=True, tolerance_s=1e-4, image_transforms=None)
    collate = lerobot_collate_fn if meta.has_language_columns else None

    def mk(subset):
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=collate)

    # Map each row of the held-out dataset to a stage, through its (episode, frame) pair.
    ep_col = ds.hf_dataset["episode_index"]
    fr_col = ds.hf_dataset["frame_index"]
    stage_of = np.full(len(ds), -1, dtype=np.int64)
    for i, (e, f) in enumerate(zip(ep_col, fr_col)):
        lab = labels.get(int(e))
        if lab is not None and int(f) < len(lab):
            stage_of[i] = lab[int(f)]

    loaders = {"pooled": mk(ds)}
    counts = {}
    for si, name in enumerate(STAGES):
        idx = np.where(stage_of == si)[0]
        counts[name] = int(len(idx))
        if len(idx):
            loaders[name] = mk(torch.utils.data.Subset(ds, idx.tolist()))
    return meta, ds, loaders, counts


def measure(output_dir: pathlib.Path, dataset: str, steps, repeats, batch_size, num_workers,
            labels, skip_stages=False):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = find_checkpoints(output_dir)
    if steps:
        ckpts = [(s, p) for s, p in ckpts if s in set(steps)]

    results, meta, loaders, counts, train_std, fps = {}, None, None, None, None, None
    for step, path in ckpts:
        t0 = time.time()
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device.type

        rename = checkpoint_rename_map(path)
        if loaders is None:
            meta, ds, loaders, counts = build_datasets(dataset, cfg, batch_size, num_workers, labels)
            fps = meta.fps
            train_std = torch.as_tensor(np.asarray(meta.stats["action"]["std"], dtype=np.float32),
                                        device=device)
            print(f"  held-out: {ds.num_frames} frames / {ds.num_episodes} episodes · "
                  f"stages {counts} · fps {fps}", flush=True)

        policy = make_policy(cfg, ds_meta=meta, rename_map=rename)
        # No normalizer override: the checkpoint's own saved preprocessor already carries the stats
        # AND the rename map training used (camera1/camera2 for SmolVLA, base_0_rgb/left_wrist_0_rgb
        # for pi0.5). Overriding it with the dataset's raw stats would mismatch those renamed keys.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": device.type}},
            postprocessor_overrides={"device_processor": {"device": device.type}},
        )

        rec = {"jerk": [], **{f"loss_{s}": [] for s in ["pooled"] + STAGES}}
        for i in range(repeats):
            seed = 1000 + i
            rec["jerk"].append(chunk_jerk(policy, preprocessor, postprocessor, loaders["pooled"],
                                          meta.camera_keys, device, fps, train_std, seed))
            for name in ["pooled"] + STAGES:
                if skip_stages and name != "pooled":
                    continue
                if name in loaders:
                    rec[f"loss_{name}"].append(
                        forward_loss(policy, preprocessor, loaders[name], meta.camera_keys,
                                     device, seed))

        summary = {}
        for k, v in rec.items():
            if v:
                m = float(np.mean(v))
                summary[k] = {"mean": m, "std": float(np.std(v)), "runs": v}
        results[step] = summary
        j, lp = summary["jerk"], summary.get("loss_pooled", {"mean": float("nan"), "std": 0.0})
        stages = "  ".join(f"{s}={summary[f'loss_{s}']['mean']:.4f}"
                           for s in STAGES if f"loss_{s}" in summary)
        print(f"  step {step:6d}: jerk {j['mean']:7.2f} +/- {j["std"]:5.2f} | "
              f"loss {lp['mean']:.4f} +/- {lp['std']:.4f} | {stages}   [{time.time() - t0:.0f}s]",
              flush=True)

        del policy, preprocessor, postprocessor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=pathlib.Path, default=None)
    ap.add_argument("--all-round1", action="store_true",
                    help="measure all six round-1 arms under outputs/hub/")
    ap.add_argument("--hub-root", type=pathlib.Path, default=pathlib.Path("outputs/hub"))
    ap.add_argument("--dataset",
                    default="HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--skip-stages", action="store_true", help="jerk and pooled loss only")
    ap.add_argument("--holdout", type=int, nargs="*", default=None,
                    help="held-out episode indices; default is D0's [9,19,29,39,49]. The merged "
                         "D0+D1 set keeps both sources' holdouts: 9 19 29 39 49 59 69 79 89 99")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="combined JSON for --all-round1 (default results/phase07_eval/)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing results file instead of resuming into it")
    args = ap.parse_args()
    if args.holdout:
        # Rebound before anything reads it. Every use in this module resolves the global at call
        # time, so this reaches the loader, the stage labeller and the written JSON alike.
        globals()["HOLDOUT"] = list(args.holdout)

    print(f"dataset : {args.dataset}  (held-out episodes {HOLDOUT})")
    labels = stage_labels(args.dataset)
    print(f"stages  : segmented {len(labels)} of {len(HOLDOUT)} held-out episodes\n")

    if args.all_round1:
        combined, stage_frames = {}, None
        out = args.out or pathlib.Path("results/phase07_eval/checkpoint_metrics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        for arm, repo, step in ROUND1:
            print(f"=== {arm}  ({repo} @ {step}) ===", flush=True)
            res, counts = measure(args.hub_root / repo, args.dataset, [step], args.repeats,
                                  args.batch_size, args.num_workers, labels, args.skip_stages)
            combined[arm] = {"repo": repo, "step": step, **res[step]}
            stage_frames = counts
            # Written after every arm, not once at the end: pi0.5 takes far longer than the other
            # four put together, and a crash or a stop there should not discard what is already
            # measured. The file is complete for the arms it names.
            out.write_text(json.dumps({"dataset": args.dataset, "holdout": HOLDOUT,
                                       "repeats": args.repeats, "stage_frames": stage_frames,
                                       "arms": combined}, indent=2))
            print(f"  -> {out} now holds {len(combined)} arm(s)", flush=True)
        print(f"\nwrote {out}")
        return 0

    if not args.output_dir:
        ap.error("pass --output-dir or --all-round1")

    # One checkpoint per measure() call, writing after each. pi0.5 costs ~an hour an arm, so a
    # single write at the end means a dead battery or a stopped kernel discards the lot - which is
    # exactly what happened on 2026-10-17. Re-running now skips whatever is already in the file,
    # provided the dataset and holdout still match; a changed holdout invalidates the old numbers,
    # so it starts clean rather than mixing two eval sets in one file.
    out = args.out or args.output_dir / "checkpoint_metrics.json"
    want = [s for s, _ in find_checkpoints(args.output_dir)
            if not args.steps or s in set(args.steps)]
    done, counts = {}, None
    if out.is_file() and not args.fresh:
        prev = json.loads(out.read_text())
        if prev.get("holdout") == HOLDOUT and prev.get("dataset") == args.dataset \
                and prev.get("repeats") == args.repeats:
            done = {int(k): v for k, v in prev.get("results", {}).items()}
            counts = prev.get("stage_frames")
            if done:
                print(f"resuming: {sorted(done)} already measured in {out}")
        else:
            print(f"{out} was built with a different dataset/holdout/repeats - starting clean")

    for step in want:
        if step in done:
            continue
        res, c = measure(args.output_dir, args.dataset, [step], args.repeats,
                         args.batch_size, args.num_workers, labels, args.skip_stages)
        done.update(res)
        counts = c or counts
        out.write_text(json.dumps({"dataset": args.dataset, "holdout": HOLDOUT,
                                   "repeats": args.repeats, "stage_frames": counts,
                                   "results": {str(k): done[k] for k in sorted(done)}}, indent=2))
        print(f"  -> {out} now holds {sorted(done)}", flush=True)

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
