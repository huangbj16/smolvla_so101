#!/usr/bin/env python3
"""Are the eval images outside what the policy was trained on?

The round-2 session notes ask whether an image-embedding shift between D1 and the eval sessions
could explain why twice the data bought no success. Pixel statistics already say no - the D1
training data averages 138.4 top-camera luminance against 136.9-138.9 across the eight eval blocks
- but a policy does not see pixels, it sees its encoder's output, and two sets can match on mean
brightness while landing in different places in feature space.

Measured in ACT's OWN ResNet backbone (the vision-dependent policy, per
`probe_camera_sensitivity.py`), two ways, because a centroid shift and an out-of-manifold rate
answer different questions:

  * **Centroid shift**, in units of the training set's own per-dimension spread. 0 = same centre.
    Anything under ~0.2 is small against the variation the training set already contains.
  * **Outside rate**: the fraction of eval frames whose nearest training neighbour is further away
    than the 95th percentile of training-to-training nearest-neighbour distance. A set drawn from
    the training distribution scores ~5% by construction, which is the control.

The bright-lighting rollouts are included as a positive control: if the method cannot see THAT
shift, it cannot be trusted to report the absence of one.

    python scripts/probe_embedding_shift.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import CACHE
from eval_checkpoint_metrics import checkpoint_rename_map

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results/phase07_eval/embedding_shift.json"
TRAIN = "HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed"
ACT_CKPT = "outputs/train/act_tray_cylinder_s1000_d1/checkpoints/040000/pretrained_model"

EVAL_SETS = {
    "r2 rollouts (normal)": ["rollout_phase07_r2_act40k_d1_p1a", "rollout_phase07_r2_act40k_d1_p1b",
                             "rollout_phase07_r2_act40k_d1_p2a", "rollout_phase07_r2_act40k_d1_p2b"],
    "r2 rollouts (bright)": ["rollout_phase07_r2_act40k_d1_test_brightlighting"],
}
TOP = "observation.images.top"


@torch.no_grad()
def features(backbone, imgs: torch.Tensor) -> np.ndarray:
    """Globally pooled backbone features - the representation the transformer encoder sees."""
    f = backbone(imgs)
    f = f[next(reversed(f))] if isinstance(f, dict) else f
    return f.mean(dim=(2, 3)).float().cpu().numpy()


def collect(backbone, repo: str, root: pathlib.Path | None, n: int, device, episodes=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo, root=root, episodes=episodes)
    if len(ds) == 0:
        return np.empty((0, 1))
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    out, buf = [], []
    for i in idx:
        x = ds[int(i)][TOP]
        buf.append(x.to(torch.float32) / 255.0 if x.dtype == torch.uint8 else x.to(torch.float32))
        if len(buf) == 32:
            out.append(features(backbone, torch.stack(buf).to(device))); buf = []
    if buf:
        out.append(features(backbone, torch.stack(buf).to(device)))
    return np.concatenate(out)


def outside_rate(train: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Fraction of `other` beyond the training set's own 95th-pct nearest-neighbour distance."""
    def nn(a, b, exclude_self):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        if exclude_self:
            np.fill_diagonal(d, np.inf)
        return d.min(axis=1)
    thr = float(np.percentile(nn(train, train, True), 95))
    return float((nn(other, train, False) > thr).mean()), thr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from eval_checkpoints_heldout import HOLDOUT
    from lerobot.utils.constants import HF_LEROBOT_HOME

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(TRAIN)
    p = REPO / ACT_CKPT
    cfg = PreTrainedConfig.from_pretrained(p)
    cfg.pretrained_path, cfg.device = p, device.type
    pol = make_policy(cfg, ds_meta=meta, rename_map=checkpoint_rename_map(p)).to(device).eval()
    backbone = pol.model.backbone

    # Two halves of the TRAINING set: the reference, and a control that shows what "no shift"
    # scores. Without it a 7% outside-rate has no meaning.
    train_eps = [e for e in range(50) if e not in HOLDOUT]
    ref = collect(backbone, TRAIN, None, a.n, device, episodes=train_eps)
    held = collect(backbone, TRAIN, None, a.n // 2, device, episodes=HOLDOUT)
    sd = ref.std(axis=0).mean()
    print(f"ACT backbone features: {ref.shape[1]}-d · train n={len(ref)} · per-dim sd {sd:.4f}\n")

    res = {"reference": "D1 training episodes (ACT ResNet backbone, pooled)",
           "n_train": len(ref), "feature_dim": int(ref.shape[1]), "sets": {}}
    print(f"  {'set':26s} {'n':>5s} {'centroid shift':>15s} {'outside rate':>14s}")

    def report(label, X):
        if len(X) == 0:
            print(f"  {label:26s} {'-':>5s}   <empty>"); return
        shift = float(np.linalg.norm(X.mean(0) - ref.mean(0)) / (sd * np.sqrt(ref.shape[1])))
        orate, thr = outside_rate(ref, X)
        res["sets"][label] = {"n": len(X), "centroid_shift_sd": shift, "outside_rate": orate,
                              "nn_threshold": thr}
        print(f"  {label:26s} {len(X):5d} {shift:15.3f} {100*orate:13.0f}%")

    report("D1 held-out (control)", held)
    for label, names in EVAL_SETS.items():
        Xs = [collect(backbone, f"HALDijkstraaa/{n}", HF_LEROBOT_HOME / "HALDijkstraaa" / n,
                      max(a.n // len(names), 40), device)
              for n in names if (HF_LEROBOT_HOME / "HALDijkstraaa" / n).is_dir()]
        report(label, np.concatenate(Xs) if Xs else np.empty((0, ref.shape[1])))

    print("\n  control = frames from the same dataset the reference came from; it scores ~5% by")
    print("  construction. A set that scores near the control is not shifted.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
