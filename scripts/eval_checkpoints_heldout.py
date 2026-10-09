#!/usr/bin/env python
"""Re-measure held-out loss for saved checkpoints, and append the numbers to the wandb run.

Why this exists. `lerobot-train` logs one `eval_loss` per `eval_steps`, and that is a *single sample*
of a stochastic quantity for the flow-matching policies: SmolVLA and pi0.5 draw a random flow timestep
and a random noise vector on every forward, so two evaluations of the same weights on the same data
disagree. Reading an eval curve — which is how the SmolVLA overfitting result in
[phase07_training.md](../Details/phase07_training.md) was found — needs to know how much of its wiggle
is noise. This runs each checkpoint several times with fixed seeds and reports mean +/- std.

The eval loop is copied from `lerobot/scripts/lerobot_train.py` (the `is_eval_step` branch) so the
numbers are directly comparable to the ones already on the curve: same uint8 -> float/255 conversion,
same preprocessor, same `policy.forward` loss, same mean-over-batches reduction, same batch size,
`shuffle=False`, `drop_last=False`.

Usage, in a notebook cell or a shell, on the machine that still has the checkpoints:

    python scripts/eval_checkpoints_heldout.py \
        --output-dir /content/outputs/pi05_tray_cylinder_s1000_trim_L4 \
        --dataset HALDijkstraaa/so101_tray_cylinder_portable_20261003_211912_trimmed \
        --wandb-project phase07-policy-transfer --repeats 3

Results go to `<output-dir>/heldout_eval.json` and, unless `--no-wandb`, to the existing wandb run
under a `recheck/` namespace with its own step axis, so nothing collides with the logged curve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HOLDOUT = [9, 19, 29, 39, 49]


def find_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    d = output_dir / "checkpoints"
    if not d.is_dir():
        raise SystemExit(f"no checkpoints directory under {output_dir}")
    out = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and not p.is_symlink() and p.name.isdigit():
            pm = p / "pretrained_model"
            if (pm / "config.json").exists():
                out.append((int(p.name), pm))
    if not out:
        raise SystemExit(f"no usable checkpoints in {d} (looked for */pretrained_model/config.json)")
    return out


def build_eval_loader(dataset_repo: str, policy_cfg, batch_size: int, num_workers: int):
    """The held-out loader, built the way lerobot's own train script builds it."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.utils.collate import lerobot_collate_fn

    meta = LeRobotDatasetMetadata(dataset_repo)
    delta = resolve_delta_timestamps(policy_cfg, meta)
    # make_train_eval_datasets() takes the LAST ceil(n*eval_split) episodes of --dataset.episodes,
    # and the training runs ordered that list so the tail is exactly HOLDOUT. Selecting those
    # episodes directly reproduces the same eval set without re-deriving the split.
    ds = LeRobotDataset(dataset_repo, episodes=HOLDOUT, delta_timestamps=delta,
                        return_uint8=True, tolerance_s=1e-4, image_transforms=None)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        collate_fn=lerobot_collate_fn if meta.has_language_columns else None,
    )
    return meta, ds, loader


@torch.no_grad()
def eval_once(policy, preprocessor, loader, camera_keys, device, seed: int) -> float:
    """One pass over the held-out set. Mirrors lerobot_train.py's is_eval_step branch exactly."""
    torch.manual_seed(seed)          # the flow-matching timestep and noise are drawn here
    policy.eval()
    total, n = 0.0, 0
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast("cpu", enabled=False))
    with autocast:
        for batch in loader:
            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            total += loss.item()
            n += 1
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="the run's output dir, i.e. the parent of checkpoints/")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="passes per checkpoint; >1 measures the sampling noise (default 3)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="only these checkpoint steps (default: all found)")
    ap.add_argument("--wandb-project", default="phase07-policy-transfer")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None,
                    help="defaults to the output dir's name, which is the job name")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = find_checkpoints(args.output_dir)
    if args.steps:
        ckpts = [(s, p) for s, p in ckpts if s in set(args.steps)]
    print(f"device     : {device}")
    print(f"checkpoints: {[s for s, _ in ckpts]}")
    print(f"dataset    : {args.dataset}  (held-out episodes {HOLDOUT})")
    print(f"repeats    : {args.repeats}\n")

    results, meta, loader, camera_keys = {}, None, None, None
    for step, path in ckpts:
        t0 = time.time()
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device.type

        if loader is None:   # the eval set does not depend on the checkpoint; build it once
            meta, ds, loader = build_eval_loader(args.dataset, cfg, args.batch_size, args.num_workers)
            camera_keys = meta.camera_keys
            print(f"held-out set: {ds.num_frames} frames over {ds.num_episodes} episodes, "
                  f"{len(loader)} batches of {args.batch_size}\n")

        policy = make_policy(cfg, ds_meta=meta)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(path),
            preprocessor_overrides={
                "device_processor": {"device": device.type},
                "normalizer_processor": {
                    "stats": meta.stats,
                    "features": {**policy.config.input_features, **policy.config.output_features},
                    "norm_map": policy.config.normalization_mapping,
                },
            },
        )

        vals = [eval_once(policy, preprocessor, loader, camera_keys, device, seed=1000 + i)
                for i in range(args.repeats)]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        results[step] = {"mean": mean, "std": std, "runs": vals}
        print(f"  step {step:6d}: eval_loss {mean:.4f} +/- {std:.4f}   "
              f"({', '.join(f'{v:.4f}' for v in vals)})   [{time.time() - t0:.0f}s]", flush=True)

        del policy, preprocessor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = args.output_dir / "heldout_eval.json"
    out.write_text(json.dumps({"dataset": args.dataset, "holdout": HOLDOUT,
                               "repeats": args.repeats, "results": results}, indent=2))
    print(f"\nwrote {out}")

    best = min(results, key=lambda s: results[s]["mean"])
    print(f"best checkpoint by held-out loss: step {best} ({results[best]['mean']:.4f})")
    spread = max(r["std"] for r in results.values())
    print(f"largest within-checkpoint std   : {spread:.4f}  <- differences smaller than this are noise")

    if not args.no_wandb:
        import wandb

        name = args.wandb_run_name or args.output_dir.name
        api = wandb.Api()
        path = f"{args.wandb_entity + '/' if args.wandb_entity else ''}{args.wandb_project}"
        match = [r for r in api.runs(path) if r.name == name]
        if not match:
            print(f"\nno wandb run named {name!r} in {path} - skipping the upload "
                  f"(results are still in {out})")
            return 0
        run_id = match[0].id
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         id=run_id, resume="must")
        # A separate namespace with its own x-axis, so these points append cleanly instead of
        # colliding with the already-logged eval/eval_loss series on the training step axis.
        wandb.define_metric("recheck/step")
        wandb.define_metric("recheck/*", step_metric="recheck/step")
        for step in sorted(results):
            wandb.log({"recheck/step": step,
                       "recheck/eval_loss": results[step]["mean"],
                       "recheck/eval_loss_std": results[step]["std"]})
        run.summary["recheck/best_step"] = best
        run.summary["recheck/best_eval_loss"] = results[best]["mean"]
        run.summary["recheck/noise_std"] = spread
        wandb.finish()
        print(f"appended {len(results)} points to wandb run {name} ({run_id}) under recheck/*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
