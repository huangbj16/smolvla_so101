#!/usr/bin/env python
"""§12.7's gate: do π0.5's frozen SigLIP features resolve where the cylinder is?

**The question.** π0.5 failed 11 of 18 round-1 trials at *reach* and scored 0/18 excluding luck
([phase07_portable_rig.md §11.4](../docs/details/phase07_portable_rig.md)). §11.4 gives three candidate
causes and only one of them - undertraining - is fixed by more steps. This separates that from the
other one, and the answer decides whether ~40 h of A100 time is worth spending (§12.7):

  * π0.5 trains `train_expert_only` with `freeze_vision_encoder=true`, so its SigLIP tower is
    `pi05_base`'s and **never adapted to this rig**. If those frozen features do not carry the
    cylinder's position, no amount of action-expert training can fix the reach, because the
    information the action expert would need is not in its input.
  * If they do carry it, the representation is fine and the action expert is simply undertrained -
    and more steps is the right fix.

**The design.** Extract the pooled SigLIP embedding for frames where the cylinder is sitting in the
tray and unoccluded, and fit a linear map from those features to where the cylinder actually was. The
target is the episode's **grasp pose** - §11.6 established the grasp pose recovers cylinder position to
~0.3°, so it is a faithful stand-in that needs no manual labelling.

Three choices that keep the answer honest:

  * **D1, not D0.** D0 put the cylinder on 9 marked spots, so it offers 9 distinct targets. D1
    randomises position continuously, giving 50. The encoder is frozen and saw neither, so nothing is
    lost by probing on the set with better coverage.
  * **Leave-one-episode-out CV.** Every frame of an episode shares one target, so a random split would
    let the model memorise an episode from its other frames and report a success that is not there.
  * **A raw-pixel control on the same folds and the same ridge grid.** If downsampled pixels predict
    the position and SigLIP does not, the representation is the problem. If neither does, the probe
    itself is suspect and the result means nothing - this is the check that tells those apart.

    python scripts/probe_pi05_vision.py                       # top camera, D1 trimmed
    python scripts/probe_pi05_vision.py --camera wrist
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import CACHE, frames, grasp_seated_index, turning_points

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "outputs/hub/pi05_tray_cylinder_s1000_trim_A100/checkpoints/010000/pretrained_model"
DEFAULT_DS = "HALDijkstraaa/so101_tray_cylinder_varied_20261013_082911_trimmed"
LAMBDAS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


# ---------------------------------------------------------------------------------------------
def episode_targets(repo: str, n_frames: int):
    """Per episode: the frame indices to probe, and the grasp pose that says where the cylinder was.

    Frames come from the first half of the reach phase - the arm is still on its way, so the cylinder
    is in the tray and nothing is in front of it. The approach peak (the gripper opening) ends the
    reach, and sampling before its midpoint keeps the descent out of the set.
    """
    fr = frames(CACHE / repo.split("/")[-1])
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    out = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        grip = st[idx, 5]
        gi, ri = grasp_seated_index(grip)
        if gi < 0 or ri <= gi:
            continue
        span = float(grip.max()) - float(np.percentile(grip, 5))
        maxima: list[int] = []
        for frac in (0.18, 0.15, 0.12, 0.08):
            pts = turning_points(grip, frac * span)
            maxima = [i for i, k in pts if k == "max"]
            if len(maxima) >= 2:
                break
        if len(maxima) < 2 or maxima[0] >= gi:
            continue
        hi = max(4, maxima[0] // 2)                 # first half of the reach
        picks = np.unique(np.linspace(0, hi - 1, n_frames).astype(int))
        out.append({
            "episode": int(e),
            "frames": picks.tolist(),
            "grasp_pan": float(st[idx[gi], 0]),      # shoulder_pan at the grasp == cylinder bearing
            "grasp_lift": float(st[idx[gi], 1]),
            "grasp_elbow": float(st[idx[gi], 2]),
        })
    return out


@torch.no_grad()
def siglip_features(ckpt: pathlib.Path, repo: str, spec, camera: str, device):
    """Pooled SigLIP embedding per selected frame, plus the matching downsampled-pixel control.

    The images go through the policy's own `_preprocess_images`, so they get exactly the resize-with-pad
    and the [0,1] -> [-1,1] rescale SigLIP was trained for. Anything else would probe a different
    encoder than the one π0.5 actually runs.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device.type
    rename = {"observation.images.top": "observation.images.base_0_rgb",
              "observation.images.wrist": "observation.images.left_wrist_0_rgb"}
    ds = LeRobotDataset(repo, return_uint8=True, tolerance_s=1e-4, image_transforms=None)
    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=rename)
    policy.eval()

    pg = policy.model.paligemma_with_expert
    cam_key = rename[f"observation.images.{camera}"]
    # `_preprocess_images` lives on the policy, not on the inner PI05Pytorch, and it returns the
    # PRESENT cameras first followed by -1-padded stand-ins for the declared-but-absent ones. Only
    # one camera is fed here, so the prepared tensor is always images[0] - not the camera's index in
    # config.image_features, which would silently read a padded blank for the wrist.
    print(f"  camera {camera!r} -> {cam_key}, fed alone so it is slot 0 after preprocessing")

    # (episode, frame) -> dataset row, read off the table rather than assuming contiguous episodes.
    # This version of LeRobotDataset exposes no episode_data_index, and the mapping is cheap.
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    fr_col = np.asarray(ds.hf_dataset["frame_index"])
    row_of = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep_col, fr_col))}

    feats, pix, ys, groups = [], [], [], []
    for rec in spec:
        e = rec["episode"]
        for f in rec["frames"]:
            if (e, f) not in row_of:
                continue
            item = ds[row_of[(e, f)]]
            img = item[f"observation.images.{camera}"]
            batch = {cam_key: (img.to(torch.float32) / 255.0).unsqueeze(0).to(device)}
            imgs, _ = policy._preprocess_images(batch)
            emb = pg.embed_image(imgs[0].to(torch.float32)).squeeze(0).float().cpu().numpy()
            # SigLIP hands back the full patch grid, (tokens, channels) - which is what the
            # action expert attends over, so it is what the probe must see. Kept whole here and
            # reduced to feature views below, rather than pooled away at extraction time.
            feats.append(emb)
            # Control: 16x16 greyscale of the same frame, nothing learned.
            small = torch.nn.functional.interpolate(
                img.to(torch.float32).unsqueeze(0) / 255.0, size=(16, 16),
                mode="area").mean(dim=1)
            pix.append(small.flatten().numpy())
            ys.append([rec["grasp_pan"], rec["grasp_lift"], rec["grasp_elbow"]])
            groups.append(e)
        print(f"    ep {e:2d}: {len(rec['frames'])} frames", end="\r", flush=True)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (np.array(feats, dtype=np.float64), np.array(pix, dtype=np.float64),
            np.array(ys, dtype=np.float64), np.array(groups))


# ---------------------------------------------------------------------------------------------
def ridge_loeo(X, y, groups, lam):
    """Leave-one-episode-out ridge. Returns one prediction per episode (frames averaged).

    Closed-form ridge on centred data, so no solver dependency: w = (X'X + lam I)^-1 X'y.
    """
    eps = np.unique(groups)
    pred, true = [], []
    for e in eps:
        te = groups == e
        tr = ~te
        Xt, yt = X[tr], y[tr]
        mx, my = Xt.mean(0), yt.mean()
        Xc = Xt - mx
        A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
        w = np.linalg.solve(A, Xc.T @ (yt - my))
        pred.append(float(((X[te] - mx) @ w + my).mean()))   # one value per held-out episode
        true.append(float(y[te][0]))
    pred, true = np.array(pred), np.array(true)
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot, float(np.sqrt(((true - pred) ** 2).mean())), pred, true


def sweep(X, y, groups, name, target):
    best = None
    print(f"\n  {name}  ({X.shape[1]} features)")
    for lam in LAMBDAS:
        r2, rmse, _, _ = ridge_loeo(X, y, groups, lam)
        flag = ""
        if best is None or r2 > best[1]:
            best, flag = (lam, r2, rmse), "  <-"
        print(f"    lambda {lam:>8.0e}   R2 {r2:7.3f}   RMSE {rmse:6.2f} deg{flag}")
    print(f"    best: R2 {best[1]:.3f}, RMSE {best[2]:.2f} deg on {target}")
    return {"n_features": int(X.shape[1]), "best_lambda": best[0], "r2": best[1], "rmse_deg": best[2]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CKPT)
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--camera", default="top", choices=["top", "wrist"])
    ap.add_argument("--n-frames", type=int, default=8, help="frames sampled per episode")
    ap.add_argument("--refresh", action="store_true",
                    help="re-extract features instead of reusing the .npz cache")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results/phase07_eval/pi05_vision_probe.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint : {args.checkpoint}")
    print(f"dataset    : {args.dataset}\ncamera     : {args.camera}\ndevice     : {device}\n")

    spec = episode_targets(args.dataset, args.n_frames)
    pans = np.array([r["grasp_pan"] for r in spec])
    print(f"episodes   : {len(spec)} usable")
    print(f"target     : grasp shoulder_pan, range {pans.min():+.2f}..{pans.max():+.2f} deg, "
          f"sd {pans.std():.2f} deg")

    cache = args.out.with_suffix(f".{args.camera}.npz")
    if cache.is_file() and not args.refresh:
        z = np.load(cache)
        X, P, Y, g = z["X"], z["P"], z["Y"], z["g"]
        print(f"\nreusing cached features from {cache.name}")
    else:
        X, P, Y, g = siglip_features(args.checkpoint, args.dataset, spec, args.camera, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, P=P, Y=Y, g=g)
        print(f"\ncached features to {cache.name}")
    print(f"extracted  : {X.shape[0]} frames · SigLIP {X.shape[1:]} · pixels {P.shape[1]}")

    # Three views of the same tokens, because they answer different halves of the question. The
    # target is a POSITION, so a view that pools the 256 patches away can be blind to it even when
    # the information is plainly there - and a view that pools the channels away keeps position but
    # throws out what is at each position. Reporting both, plus their concatenation, means a
    # negative result is not an artefact of one pooling choice.
    n_tok, n_ch = X.shape[1], X.shape[2]
    views = {
        f"semantic (mean over {n_tok} patches)": X.mean(axis=1),
        f"spatial (mean over {n_ch} channels)": X.mean(axis=2),
        "both concatenated": np.concatenate([X.mean(axis=1), X.mean(axis=2)], axis=1),
    }

    res = {"checkpoint": str(args.checkpoint), "dataset": args.dataset, "camera": args.camera,
           "n_episodes": len(spec), "n_frames": int(X.shape[0]),
           "siglip_shape": list(X.shape[1:]), "target_sd_deg": float(pans.std())}
    col = 0
    res["grasp_pan"] = {name: sweep(v, Y[:, col], g, f"frozen SigLIP — {name}", "grasp_pan")
                        for name, v in views.items()}
    res["grasp_pan"]["pixels"] = sweep(P, Y[:, col], g, "raw 16x16 pixels (control)", "grasp_pan")

    s = max((res["grasp_pan"][k] for k in views), key=lambda d: d["r2"])
    p = res["grasp_pan"]["pixels"]
    print("\n--- verdict ---")
    print(f"  best SigLIP view : R2 {s['r2']:+.3f}   RMSE {s['rmse_deg']:.2f} deg")
    print(f"  pixel control    : R2 {p['r2']:+.3f}   RMSE {p['rmse_deg']:.2f} deg")
    print(f"  target spread : sd {res['target_sd_deg']:.2f} deg  (R2 <= 0 means no better than "
          f"predicting the mean)")
    if s["r2"] > 0.5:
        print("  -> the representation carries cylinder position. The action expert is undertrained;")
        print("     more steps is the right fix (§12.7).")
    elif p["r2"] <= 0.2:
        print("  -> neither the features nor the CONTROL beat predicting the mean. On its own this")
        print("     cannot separate 'this camera does not see the target at this stage' from 'the")
        print("     probe is mis-built', so it licenses no claim about the encoder. It becomes a")
        print("     usable negative control only against a run of this same probe that DID pass")
        print("     (e.g. --camera top, §12.7); otherwise re-run with more frames or a later stage.")
    else:
        print("  -> pixels carry the position and the frozen features do not. More action-expert")
        print("     training cannot fix the reach (§12.7); unfreeze the encoder or drop π0.5.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
