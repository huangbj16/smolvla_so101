"""Build an idle-trimmed copy of a LeRobotDataset (Phase 0.6 F3).

Trim rule is the one the Phase 0.5 notebook uses: cut the lead-in before any joint moves more than
THRESH from the start pose, and the tail after the arm has settled at its final pose, keeping PAD
frames of margin on each side.

    python scripts/make_trimmed_dataset.py <src_repo> <dst_repo> <local_root> [--push] [--tail-cap N]

**--tail-cap changes what happens at the END of the episode, and Phase 0.7 round 2 needs it.** The
head trim is justified by [phase06 F3](../Details/phase06_camera_ablation_training.md): the idle
lead-in is an attractor the policy cannot escape at the start. The tail is the opposite - it is the
only place a policy can learn "the task is finished, hold still", and round 1 measured the cost of
removing it (SmolVLA moved 3.6x as much as ACT in the final second). So D1 onward keeps a tail.

`--tail-cap N` keeps up to N frames after the last motion instead of PAD. The default is PAD, which
reproduces the original both-ends behaviour exactly, so the D0 trimmed dataset is still
reproducible from this script. Round 2 uses `--tail-cap 60` (2.0 s at 30 fps): D1's recorded tails
run 47-191 frames, and capping gives every episode the *same* terminal hold rather than a 4x range,
which is what makes it a learnable signal rather than one more source of variation.
"""
import sys, glob, shutil, time, numpy as np, pandas as pd, torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC, DST, ROOT = sys.argv[1], sys.argv[2], Path(sys.argv[3])   # optional flags: --push, --tail-cap N
THRESH, PAD = 2.0, 5
TAIL_CAP = int(sys.argv[sys.argv.index("--tail-cap") + 1]) if "--tail-cap" in sys.argv else PAD
t0 = time.time()

src = LeRobotDataset(SRC)
print(f"source: {src.num_episodes} episodes, {src.num_frames} frames")
print(f"trim: head to {PAD} frames of margin, tail to {TAIL_CAP} frames"
      f"{'  (default = both ends trimmed, the D0 behaviour)' if TAIL_CAP == PAD else '  (tail kept)'}")
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
    b = min(len(st), (int(back[-1]) + 1 if len(back) else len(st)) + TAIL_CAP)
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

# Close the parquet writers and flush the buffered episode metadata. Without this the files have no
# parquet footer and the dataset is unreadable. `DatasetWriter.__del__` does it at interpreter exit,
# which is why this script worked before the call was added — but push_to_hub() below runs BEFORE
# that, so an unfinalized dataset would upload silently and only fail at training time. In a
# long-lived process (a notebook kernel) nothing saves you at all. Idempotent.
dst.finalize()

dt = time.time() - t0
print(f"kept {kept}/{src.num_frames} frames ({100*kept/src.num_frames:.0f}%) in {dt/60:.1f} min "
      f"-> {kept/dt:.0f} frames/s")

if "--push" in sys.argv:
    # Use the dataset's own push_to_hub: it uploads, writes a dataset card, and — the part a plain
    # `hf upload` misses — tags the revision with the codebase version, without which lerobot refuses
    # to load the dataset at all.
    dst.push_to_hub(private=True)
    print(f"pushed and tagged: https://huggingface.co/datasets/{DST}")
else:
    print(f"not pushed. Either re-run with --push, or use scripts/push_dataset.sh {ROOT} {DST}")
