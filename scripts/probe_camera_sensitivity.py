#!/usr/bin/env python3
"""Which camera is each policy actually using?

The round-2 session notes report that ACT gets *better* when the top camera is over-exposed to the
point of being unreadable, and section 12.9's trajectory measures agree: ACT's stuck fraction falls
from 10.1% to 0.8% and its speed returns to the demonstrations' under bright light. That suggests
the top camera is not helping ACT so much as interfering with it.

This measures the claim directly, offline, with no robot. For each policy, on held-out D1 frames,
one camera at a time is destroyed and the predicted action chunk is compared against the unperturbed
prediction:

    sensitivity = mean |chunk_perturbed - chunk_baseline| / training sigma

A policy that ignores a camera is unmoved when that camera is destroyed.

Two controls that the number is meaningless without:

  * **Sampling floor.** SmolVLA samples its chunk from noise, so two runs on IDENTICAL input already
    differ. Camera sensitivity below that floor is not sensitivity at all. Measured as `seed_repeat`.
  * **Both cameras.** If destroying both barely moves the output, the policy is running open-loop on
    proprioception and the per-camera numbers say nothing about vision.

`top_bright` replicates the actual intervention rather than an abstract one: the top image is scaled
to the mean luminance measured in the bright-lighting rollouts (198/255) and clipped, which is what
the real over-exposure did.

    python scripts/probe_camera_sensitivity.py
    python scripts/probe_camera_sensitivity.py --max-frames 400     # quicker
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
from analyze_phase07_eval import CACHE
from eval_checkpoint_metrics import STAGES, checkpoint_rename_map, stage_labels

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results/phase07_eval/camera_sensitivity.json"
DATASET = "HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed"

ARMS = [
    ("act40k_d1", "outputs/train/act_tray_cylinder_s1000_d1", "040000"),
    ("sv20k_d1", "outputs/train/smolvla_tray_cylinder_s1000_d1_resumed", "020000"),
]
TOP, WRIST = "observation.images.top", "observation.images.wrist"
# Mean top-camera luminance measured in the bright-lighting rollouts, against 138 in the normal
# session and the D1 training data (results/phase07_eval/r2_brightness_per_block.csv).
BRIGHT_TARGET = 198.0 / 255.0


def perturbations(mean_img: dict[str, torch.Tensor]):
    """name -> fn(batch) -> batch, each returning a NEW dict (the loader's tensors are reused)."""
    def blank(cam):
        def f(b):
            o = dict(b)
            o[cam] = mean_img[cam].to(b[cam].device).expand_as(b[cam]).clone()
            return o
        return f

    def bright(cam):
        def f(b):
            o = dict(b)
            x = b[cam]
            # Scale to the measured bright-light mean, then clip - which is what over-exposure
            # does: it raises the level AND destroys detail at the top end.
            scale = BRIGHT_TARGET / max(float(x.mean()), 1e-6)
            o[cam] = (x * scale).clamp(0, 1)
            return o
        return f

    def both(b):
        o = dict(b)
        for cam in (TOP, WRIST):
            o[cam] = mean_img[cam].to(b[cam].device).expand_as(b[cam]).clone()
        return o

    return {"top_blank": blank(TOP), "wrist_blank": blank(WRIST),
            "top_bright": bright(TOP), "both_blank": both}


@torch.no_grad()
def predict(policy, pre, post, batch, camera_keys, seed: int) -> torch.Tensor:
    """One action chunk in degrees, with the sampling noise pinned by `seed`."""
    b = dict(batch)
    for cam in camera_keys:
        if cam in b and b[cam].dtype == torch.uint8:
            b[cam] = b[cam].to(dtype=torch.float32) / 255.0
    torch.manual_seed(seed)          # same noise for baseline and perturbation, or the delta is noise
    return post(policy.predict_action_chunk(pre(b))).to(torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from eval_checkpoints_heldout import HOLDOUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(DATASET)
    labels = stage_labels(DATASET)
    train_std = torch.as_tensor(np.asarray(meta.stats["action"]["std"], dtype=np.float32),
                                device=device)
    print(f"held-out episodes {HOLDOUT} · device {device}")

    results = {"dataset": DATASET, "holdout": HOLDOUT, "bright_target_255": BRIGHT_TARGET * 255,
               "arms": {}}

    for arm, odir, ckpt in ARMS:
        path = REPO / odir / "checkpoints" / ckpt / "pretrained_model"
        if not path.is_dir():
            print(f"  ! {arm}: {path} missing, skipping")
            continue
        t0 = time.time()
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path, cfg.device = path, device.type
        rename = checkpoint_rename_map(path)

        ds = LeRobotDataset(DATASET, episodes=HOLDOUT)
        n = min(args.max_frames, len(ds))
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        sub = torch.utils.data.Subset(ds, idx.tolist())
        loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4)
        # Which stage each sampled frame belongs to, for the per-stage split.
        ep = np.array([ds.hf_dataset[int(i)]["episode_index"] for i in idx])
        fi = np.array([ds.hf_dataset[int(i)]["frame_index"] for i in idx])
        stage_of = np.array([labels[int(e)][int(f)] if int(e) in labels
                             and int(f) < len(labels[int(e)]) else -1 for e, f in zip(ep, fi)])

        policy = make_policy(cfg, ds_meta=meta, rename_map=rename)
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": device.type}},
            postprocessor_overrides={"device_processor": {"device": device.type}})
        policy.eval()

        # Mean image per camera over the probed frames: destroying a camera by replacing it with
        # the dataset mean removes its information while staying closer to the training
        # distribution than a flat grey would.
        acc, cnt = {TOP: 0.0, WRIST: 0.0}, 0
        for b in loader:
            for cam in (TOP, WRIST):
                x = b[cam]
                x = x.to(torch.float32) / 255.0 if x.dtype == torch.uint8 else x.to(torch.float32)
                acc[cam] = acc[cam] + x.sum(dim=0, keepdim=True)
            cnt += b[TOP].shape[0]
        mean_img = {c: (acc[c] / cnt).to(device) for c in (TOP, WRIST)}

        perts = perturbations(mean_img)
        conds = ["seed_repeat", *perts]
        deltas = {c: [] for c in conds}
        stages_seen = []

        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
                    else torch.autocast("cpu", enabled=False))
        with autocast:
            for bi, batch in enumerate(loader):
                base = predict(policy, pre, post, batch, meta.camera_keys, args.seed)
                for cond in conds:
                    if cond == "seed_repeat":
                        # Identical input, DIFFERENT noise: the policy's own sampling spread.
                        other = predict(policy, pre, post, batch, meta.camera_keys, args.seed + 7)
                    else:
                        other = predict(policy, pre, post, perts[cond](batch), meta.camera_keys,
                                        args.seed)
                    d = ((other - base).abs() / train_std).mean(dim=(1, 2))   # per sample
                    deltas[cond].append(d.float().cpu())
                stages_seen.append(stage_of[bi * args.batch_size:
                                            bi * args.batch_size + base.shape[0]])

        st = np.concatenate(stages_seen)
        rec = {}
        for cond in conds:
            v = torch.cat(deltas[cond]).numpy()
            rec[cond] = {"mean": float(v.mean()), "sd": float(v.std()),
                         **{f"stage_{STAGES[s]}": float(v[st == s].mean())
                            for s in range(len(STAGES)) if (st == s).any()}}
        results["arms"][arm] = rec

        floor = rec["seed_repeat"]["mean"]
        print(f"\n  {arm}  ({time.time() - t0:.0f}s, {n} frames)")
        print(f"    {'condition':14s} {'|Δaction| / σ':>14s}   {'× sampling floor':>17s}")
        for cond in conds:
            m = rec[cond]["mean"]
            tag = "  <- sampling floor" if cond == "seed_repeat" else ""
            ratio = "" if cond == "seed_repeat" else f"{m / floor:17.1f}" if floor > 1e-9 else \
                f"{'inf':>17s}"
            print(f"    {cond:14s} {m:14.4f}   {ratio}{tag}")
        del policy, pre, post
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
