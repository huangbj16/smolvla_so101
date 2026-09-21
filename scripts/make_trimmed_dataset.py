"""Build an idle-trimmed copy of a LeRobotDataset (Phase 0.6 F3).

Trim rule is the one the Phase 0.5 notebook uses: cut the lead-in before any joint moves more than
THRESH from the start pose, and the tail after the arm has settled at its final pose, keeping PAD
frames of margin on each side.
"""
import sys, glob, shutil, time, numpy as np, pandas as pd, torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC, DST, ROOT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
THRESH, PAD = 2.0, 5
t0 = time.time()

src = LeRobotDataset(SRC)
print(f"source: {src.num_episodes} episodes, {src.num_frames} frames")
if ROOT.exists(): shutil.rmtree(ROOT)

# States come straight from the parquet — reading them through src[i] would decode both videos per frame.
raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(src.root) / "data/chunk-*/*.parquet")))])
raw = raw.sort_values("index")
states = np.stack(raw["observation.state"].to_numpy())
ep_of = raw["episode_index"].to_numpy()

feats = {k: v for k, v in src.features.items()
         if k not in ("index", "episode_index", "frame_index", "timestamp", "task_index")}
dst = LeRobotDataset.create(DST, fps=src.fps, features=feats, root=ROOT,
                            robot_type=src.meta.robot_type, use_videos=True)
img_keys = [k for k in feats if k.startswith("observation.images")]

kept = 0
for ep in range(src.num_episodes):
    lo = int(src.meta.episodes["dataset_from_index"][ep])
    hi = int(src.meta.episodes["dataset_to_index"][ep])
    st = states[lo:hi]
    dev0 = np.abs(st - st[0]).max(axis=1)                     # departure from the start pose
    dev1 = np.abs(st - st[-1]).max(axis=1)                    # departure from the final pose
    a = max(0, (int(np.argmax(dev0 > THRESH)) if (dev0 > THRESH).any() else 0) - PAD)
    back = np.where(dev1 > THRESH)[0]
    b = min(len(st), (int(back[-1]) + 1 if len(back) else len(st)) + PAD)
    for i in range(lo + a, lo + b):                            # only kept frames get decoded
        item = src[i]
        frame = {"task": item["task"]}
        for k in feats:
            v = item[k]
            if k in img_keys:                                  # CHW float [0,1] -> HWC uint8
                v = (v.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            elif isinstance(v, torch.Tensor):
                v = v.numpy()
            frame[k] = v
        dst.add_frame(frame)
    dst.save_episode()
    kept += b - a
    print(f"  ep{ep:02d}: {hi-lo:4d} -> {b-a:4d} frames  (cut {a} lead-in, {len(st)-b} tail)", flush=True)

dt = time.time() - t0
print(f"kept {kept}/{src.num_frames} frames ({100*kept/src.num_frames:.0f}%) in {dt/60:.1f} min "
      f"-> {kept/dt:.0f} frames/s")
