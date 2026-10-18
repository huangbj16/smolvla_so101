#!/usr/bin/env python3
"""Generate notebooks/phase07_gpufree_pi05_train.ipynb.

Rewrites colab_phase07_pi05_gpufree.ipynb for the gpufree.cn box after the env probe. What the
probe removed from the old notebook, and why:

  - HF auth and `huggingface-cli login`: the box has no route to huggingface.co at all.
  - the model download: /data/models/pi05_base and paligemma-3b-pt-224 are already on the image,
    so the laptop->server upload the plan assumed is ~25 GB of work that does not exist.
  - the paligemma access-request walkthrough: granted, and irrelevant offline anyway.
  - push_to_hub / save_checkpoint_to_hub: nowhere to push.
  - the pip install of lerobot: moved to scripts/gpufree_setup_py312.sh, which has to run in a
    shell before Jupyter can see the env anyway.
  - Colab drive-mount and kernel-restart cells.

What is left is the part that is actually specific to this run.
"""
import json, pathlib

C = []


def md(src):
    C.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": src.strip("\n").splitlines(True)})


# =================================================================================================
md(r"""
# Phase 0.7 — π0.5 on the portable rig, 80,000 steps on a rented 4090

Round 1 stopped π0.5 at 10,000 steps, which is **6.0 epochs** of D0 against SmolVLA's 47.9. This
run matches SmolVLA's sample budget — 80,000 × 16 = **1,280,000 samples** — on D0+D1 together.

### Before this notebook

The box ships **lerobot 0.4.1 on Python 3.11.13**, and 0.4.1's `PI05Config` has neither
`train_expert_only` nor `freeze_vision_encoder`. Without them the card would carry a full 4.14 B
fine-tune (~50 GiB against 23.1 GiB free). lerobot ≥0.5 needs Python ≥3.12, so run this first, in a
**terminal**, not here:

```bash
bash gpufree_setup_py312.sh 2>&1 | tee /root/gpufree-data/setup_py312.log
. /root/gpufree-data/env.sh
python /root/gpufree-data/gpufree_verify_py312.py --load
```

That also installs `rsync` (absent from the gpufree image), registers a **`lerobot 0.6.1 (py3.12)`**
Jupyter kernel, and writes `env.sh` with `HF_HOME`, `HF_HUB_OFFLINE` and — if you set one — the
wandb key. Pick that kernel, or don't: the cells below shell out to the 3.12 interpreter explicitly
and do not care what the kernel is. **Set the wandb key before section 5**, since a run launched
without it logs nowhere but its own log file.

### The three things that differ from the A100 run

| | A100, round 1 | here |
|---|---|---|
| steps / `scheduler_decay_steps` | 10,000 / 10,000 | **80,000 / 80,000** |
| dataset | D0 only, 50 ep | **D0+D1 merged, 100 ep** |
| checkpoints | every 2,500 → Hub | every 10,000 → **local disk only** |

Everything else — `dtype=bfloat16`, `gradient_checkpointing`, the expert-only freeze, batch 16,
`lr 2.5e-5`, warmup 1,000, QUANTILES normalisation, `chunk_size 50` — is round 1's recipe
unchanged, so a difference in the result is attributable to steps and data.
""")

# =================================================================================================
md(r"""
## 1 — Confirm the env, the card and the disk

Fast. If anything here is not `OK`, stop: everything below assumes it.
""")

code(r'''
import json, os, pathlib, shutil, subprocess, sys, threading, time

PY       = "/root/gpufree-data/envs/lr312/bin/python"
DATA     = pathlib.Path("/root/gpufree-data")
HF_HOME  = DATA / "cache/hf"                 # seeded with the paligemma snapshot by the setup script
BASE     = "/data/models/pi05_base"          # already on the image; nothing to download

ENV = {**os.environ, "HF_HOME": str(HF_HOME), "HF_HUB_OFFLINE": "1",
       "PATH": f"{pathlib.Path(PY).parent}:{os.environ['PATH']}",
       "TOKENIZERS_PARALLELISM": "false"}


def run(cmd, **kw):
    """Run in the 3.12 env, streaming output."""
    return subprocess.run(cmd, env=ENV, text=True, **kw)


def pyrun(src):
    return run([PY, "-c", src], capture_output=True).stdout.strip()


print(pyrun(r"""
import lerobot, torch, dataclasses as dc
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
f = lambda c: {x.name for x in dc.fields(c)}
need = ({'train_expert_only','freeze_vision_encoder'} <= f(PI05Config)
        and {'eval_steps','max_eval_samples','rename_map'} <= f(TrainPipelineConfig)
        and 'eval_split' in f(DatasetConfig))
print(f"lerobot      {lerobot.__version__}   capabilities {'OK' if need else 'MISSING'}")
print(f"torch        {torch.__version__}  cuda {torch.version.cuda}  avail {torch.cuda.is_available()}")
free, tot = torch.cuda.mem_get_info()
print(f"VRAM free    {free/2**30:.2f} GiB of {tot/2**30:.2f}   budget 18.18 GiB at batch 16"
      f"  -> {'OK' if free/2**30 > 19 else 'TIGHT'} ({free/2**30-18.18:+.2f} GiB spare)")
"""))

u = shutil.disk_usage(DATA)
print(f"{'data disk':12s} {u.free/1e9:.1f} GB free of {u.total/1e9:.1f}")
print(f"{'base model':12s} {'OK' if pathlib.Path(BASE,'model.safetensors').is_file() else 'MISSING'}  {BASE}")
print(f"{'paligemma':12s} {'OK' if (HF_HOME/'hub/models--google--paligemma-3b-pt-224/refs/main').is_file() else 'NOT SEEDED - run setup stage 5'}")
''')

# =================================================================================================
md(r"""
## 2 — Put the dataset on the box

The only thing that still has to travel. The merged set is **1.37 GB**, 100 episodes, 62,799 frames,
`codebase_version v3.0` — the same version lerobot 0.6.1 reads natively, so there is nothing to
convert.

Merge and verify it **on the laptop** first (it is already done if `scripts/merge_d0_d1.py` has been
run):

```bash
python scripts/merge_d0_d1.py --verify-only
```

Then upload. Either the platform's Jupyter file browser, or — much faster for 1.37 GB — `rsync`
from the laptop (the setup script installs it on the box; the gpufree image has no `rsync`), filling
in the host and port from the instance's SSH panel:

```bash
rsync -avP --no-perms --no-owner --no-group \
  ~/.cache/huggingface/lerobot/HALDijkstraaa/so101_tray_cylinder_d0d1_100ep_trimmed/ \
  -e 'ssh -p <PORT>' root@<HOST>:/root/gpufree-data/datasets/so101_tray_cylinder_d0d1_100ep_trimmed/
```
""")

code(r'''
DATASET   = "HALDijkstraaa/so101_tray_cylinder_d0d1_100ep_trimmed"
DS_ROOT   = DATA / "datasets/so101_tray_cylinder_d0d1_100ep_trimmed"

# The holdout, in merged coordinates: the last episode of blocks 2/4/6/8 plus one off-grid
# episode, taken from BOTH sources so the eval set stays balanced across tray spots and across
# the two recording sessions. Same mechanism as round 1, which is what keeps the eval losses
# comparable. See scripts/merge_d0_d1.py.
HOLDOUT  = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
# eval_split takes the LAST ceil(n * split) episodes of this list, so the holdout has to come last.
EPISODES = [e for e in range(100) if e not in HOLDOUT] + HOLDOUT
EVAL_SPLIT = 0.1
assert len(EPISODES) == 100 and EPISODES[-10:] == HOLDOUT

info = json.loads((DS_ROOT / "meta/info.json").read_text())
size = sum(f.stat().st_size for f in DS_ROOT.rglob("*") if f.is_file())
for k, want in [("codebase_version", "v3.0"), ("total_episodes", 100),
                ("total_frames", 62799), ("fps", 30)]:
    got = info.get(k)
    print(f"  {'OK  ' if got == want else 'FAIL'} {k:18s} {got}  (expected {want})")
print(f"  {'OK  ' if abs(size/1e9 - 1.37) < 0.05 else 'FAIL'} on disk           {size/1e9:.2f} GB  (expected 1.37)")
print(f"\n  cameras {[k for k in info['features'] if k.startswith('observation.images')]}")
print(f"  90 train / 10 eval episodes, holdout = {HOLDOUT}")
''')

# =================================================================================================
md(r"""
## 3 — The recipe

Three flags carry most of the risk, so they are worth naming:

| flag | why |
|---|---|
| `--policy.train_expert_only=true` | 0.693 B of 4.143 B trainable. **Without it the run OOMs** — a full fine-tune needs ~50 GiB. |
| `--policy.gradient_checkpointing=true` | not optional on 24 GB: 18.18 GiB with it, no headroom without. |
| `--policy.scheduler_decay_steps=80000` | `CosineDecayWithWarmupSchedulerConfig` auto-scales its horizon **only downward**. The saved config says 10,000; leaving it there would park the LR at its 2.5e-6 floor for 7/8 of the run. |

`--rename_map` maps the rig's camera keys onto the names π0.5 was pretrained with. It is read from
round 1's own `train_config.json`, so it cannot drift from what worked.
""")

code(r'''
OUT    = DATA / "outputs/pi05_d0d1_80k"
JOB    = "pi05_tray_cylinder_d0d1_80k"
STEPS, BATCH, SEED = 80_000, 16, 1000
SAVE_FREQ, EVAL_STEPS, LOG_FREQ = 10_000, 2_500, 200
NUM_WORKERS = 16          # 128 vCPU on this box; the A100 run used 8 and was 82% update-bound
LR, WARMUP = 2.5e-5, 1000
VIDEO_BACKEND = "torchcodec"

# lerobot tests `step % eval_steps` and `step % save_freq` independently, so a mismatched pair
# leaves some checkpoints with no held-out loss beside them - which is the whole selection signal.
assert SAVE_FREQ % EVAL_STEPS == 0, "every checkpoint must land on an eval step"

RENAME = {"observation.images.top":   "observation.images.base_0_rgb",
          "observation.images.wrist": "observation.images.left_wrist_0_rgb"}

# wandb is reachable from this box (the probe got HTTP 404, i.e. routed). The setup script puts the
# key in /root/gpufree-data/env.sh AND in ~/.netrc via `wandb login`, so check both: a netrc-only
# login would otherwise be silently ignored and the run would log nowhere.
def _wandb_ready() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return "api.wandb.ai" in pathlib.Path.home().joinpath(".netrc").read_text()
    except OSError:
        return False


WANDB = _wandb_ready()
PROJECT = "phase07-policy-transfer"


def train_cmd(job=None, steps=None, batch=None, out=None, save_freq=None, eval_steps=None):
    job = JOB if job is None else job
    steps = STEPS if steps is None else steps
    batch = BATCH if batch is None else batch
    out = OUT if out is None else out
    return [f"{pathlib.Path(PY).parent}/lerobot-train",
        f"--dataset.repo_id={DATASET}", f"--dataset.root={DS_ROOT}",
        f"--dataset.episodes={EPISODES}", f"--dataset.eval_split={EVAL_SPLIT}",
        f"--dataset.video_backend={VIDEO_BACKEND}",
        f"--eval_steps={EVAL_STEPS if eval_steps is None else eval_steps}",
        "--max_eval_samples=0",                     # 0 = the whole held-out split
        f"--rename_map={json.dumps(RENAME)}",

        # --policy.path also satisfies the rename_map precondition: lerobot refuses a rename_map
        # without a pretrained checkpoint (configs/train.py:247). The policy TYPE comes from the
        # checkpoint's own config.json, so passing --policy.type as well would be a conflict.
        f"--policy.path={BASE}",                    # local dir, so no hub lookup
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.push_to_hub=false",
        f"--policy.scheduler_decay_steps={steps}",
        f"--policy.scheduler_warmup_steps={WARMUP}",
        f"--policy.optimizer_lr={LR}",

        f"--batch_size={batch}", f"--steps={steps}", f"--num_workers={NUM_WORKERS}",
        f"--seed={SEED}", f"--log_freq={LOG_FREQ}",
        f"--save_freq={SAVE_FREQ if save_freq is None else save_freq}",
        "--save_checkpoint=true", "--save_checkpoint_to_hub=false",
        f"--output_dir={out}", f"--job_name={job}",
        f"--wandb.enable={'true' if WANDB else 'false'}", f"--wandb.project={PROJECT}",
    ]


print("\n".join(train_cmd()))
print(f"\nwandb: {'on' if WANDB else 'OFF - see stage 7 of gpufree_setup_py312.sh to set a key'}")
''')

# =================================================================================================
md(r"""
## 4 — Smoke test: 300 steps

Measures the two numbers that decide whether to commit: **seconds per step** and **peak VRAM**. It
also writes one real checkpoint, which is the only way to find out that a 9.35 GB save works on this
disk before discovering it 10,000 steps into a multi-day run.

**Measured on this box, 2026-10-17:** `updt_s 2.660 + data_s 0.482 = 3.142 s/step`, **peak 18.52
GiB** (A100: 2.391 s/step, 18.18 GiB), checkpoint written at exactly 9.354 GB. That extrapolates to
**69.8 h ≈ 2.9 days**, with a checkpoint every 8.7 h. If `data_s` falls at steady state the update
cost alone puts the floor at 59 h — section 6 settles which.

The card is 1.9 GiB over the A100's slower-per-step pace and 0.34 GiB over its peak, with ~4.6 GiB
still spare. Nothing here needed a batch step-down.
""")

code(r'''
import re

SMOKE = DATA / "outputs/smoke"
shutil.rmtree(SMOKE, ignore_errors=True)

peak = [0.0]
stop = threading.Event()


def watch_vram():
    while not stop.wait(2.0):
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()[0]
            peak[0] = max(peak[0], float(q) / 1024)
        except Exception:
            pass


t = threading.Thread(target=watch_vram, daemon=True)
t.start()
t0 = time.time()
log = (DATA / "smoke.log").open("w")
proc = subprocess.Popen(train_cmd(job="smoke", steps=300, out=SMOKE,
                                  save_freq=300, eval_steps=150),
                        env=ENV, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, bufsize=1)
tail = []
for line in proc.stdout:
    log.write(line)
    tail.append(line.rstrip())
    if any(k in line for k in ("step:", "eval_loss", "Error", "error", "Traceback", "out of memory")):
        print(line.rstrip()[:160])
proc.wait()
stop.set()
log.close()
wall = time.time() - t0

text = "\n".join(tail)
upd = [float(m) for m in re.findall(r"updt_s:([\d.]+)", text)]
dat = [float(m) for m in re.findall(r"data_s:([\d.]+)", text)]
evl = re.findall(r"step (\d+): eval_loss=([\d.]+)", text)
oom = "out of memory" in text.lower()

print(f"\n{'=' * 70}")
print(f"  exit {proc.returncode}{'   !! OOM' if oom else ''}")
if upd:
    s_step = upd[-1] + (dat[-1] if dat else 0.0)
    print(f"  updt_s {upd[-1]:.3f} + data_s {dat[-1] if dat else 0:.3f} = {s_step:.3f} s/step"
          f"   (A100: 1.949 + 0.442 = 2.391)")
    h = STEPS * s_step / 3600
    print(f"  -> {STEPS:,} steps = {h:.1f} h = {h/24:.1f} days")
    print(f"  -> checkpoints every {SAVE_FREQ:,} steps = every {SAVE_FREQ*s_step/3600:.1f} h")
print(f"  peak VRAM {peak[0]:.2f} GiB   (A100 at batch 16: 18.18)")
print(f"  held-out evals: {evl}")
ck = sorted((SMOKE / "checkpoints").glob("*/pretrained_model/model.safetensors")) if (SMOKE / "checkpoints").is_dir() else []
for c in ck:
    print(f"  checkpoint wrote {c.stat().st_size/1e9:.3f} GB  (expected 9.354)")
print(f"  smoke wall clock {wall/60:.1f} min (includes ~2 min of model load)")
print(f"{'=' * 70}")
''')

code(r'''
# Reclaim the smoke test's ~15 GB before the real run.
shutil.rmtree(SMOKE, ignore_errors=True)
print(f"data disk {shutil.disk_usage(DATA).free/1e9:.1f} GB free")
''')

# =================================================================================================
md(r"""
## 5 — Launch

Two things matter about how this is started:

- **`nohup` + `setsid`**, because a dropped browser tab must not take a 60-hour run with it.
- **the log is the record.** With wandb off it is the *only* record of the held-out loss, so it is
  written to the persistent disk, not to a notebook output cell.

Reconnecting later: skip to section 6, which attaches to the running job without disturbing it.
""")

code(r'''
LOG = DATA / f"{JOB}.log"
assert not (OUT / "checkpoints").exists(), f"{OUT} already has checkpoints - see section 8 to resume"

(DATA / "train_cmd.json").write_text(json.dumps(train_cmd(), indent=1))
# No shell: --dataset.episodes is a 100-element list and --rename_map is JSON, and both would need
# exact quoting through `bash -c`. start_new_session detaches the child, so restarting this kernel
# or losing the browser tab cannot SIGHUP a multi-day run.
logf = LOG.open("w")
proc = subprocess.Popen(train_cmd(), env=ENV, stdout=logf, stderr=subprocess.STDOUT,
                        start_new_session=True)
(DATA / "train.pid").write_text(str(proc.pid))
print(f"pid {proc.pid}   log {LOG}")
print(f"follow it from a terminal with:  tail -f {LOG}")
''')

# =================================================================================================
md(r"""
## 6 — Watch it

Safe to run any time, including after a reconnect. Re-run the cell to refresh.
""")

code(r'''
import re

# lerobot logs `step:{format_big_number(step)}`, and format_big_number(2500) is "2K" - 1,500 and
# 2,500 both print as "2K". So the exact step is NOT recoverable from the metric line. Two things
# that are exact: the count of metric lines (one per log_freq steps) and the eval_loss lines, which
# carry the raw int.
def derived_steps(text):
    return [i * LOG_FREQ for i in range(1, text.count("smpl:") + 1)]


def eval_points(text):
    return [(int(s), float(v)) for s, v in re.findall(r"step (\d+): eval_loss=([\d.eE+-]+)", text)]


def status():
    pid = (DATA / "train.pid").read_text().strip() if (DATA / "train.pid").is_file() else None
    alive = pid and pathlib.Path(f"/proc/{pid}").exists()
    text = LOG.read_text() if LOG.is_file() else ""
    steps = derived_steps(text)
    evl = eval_points(text)
    upd = [float(m) for m in re.findall(r"updt_s:([\d.eE+-]+)", text)]
    dat = [float(m) for m in re.findall(r"data_s:([\d.eE+-]+)", text)]
    done = steps[-1] if steps else 0
    print(f"  process      {'running' if alive else 'NOT running'}  (pid {pid})")
    exact = f"   last exact eval step {evl[-1][0]:,}" if evl else ""
    print(f"  step         ~{done:,} of {STEPS:,}  ({100 * done / STEPS:.1f}%){exact}")
    if upd:
        sps = upd[-1] + (dat[-1] if dat else 0)
        print(f"  pace         {sps:.2f} s/step  -> {(STEPS - done) * sps / 3600:.1f} h remaining")
    print(f"  evals        {evl[-4:] if evl else '-'}")
    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"  gpu          {g}")
    u = shutil.disk_usage(DATA)
    saved = sorted(p.name for p in (OUT / "checkpoints").glob("0*")) if (OUT / "checkpoints").is_dir() else []
    print(f"  disk         {u.free/1e9:.1f} GB free; {len(saved)} checkpoints {saved}")
    print(f"  projected    {len(saved)} kept x 9.354 GB; 8 of them need 74.8 GB + 2.2 for the "
          f"newest optimizer state")
    for line in text.splitlines()[-3:]:
        print(f"  | {line[:150]}")

status()
''')

# =================================================================================================
md(r"""
## 7 — Reap optimizer state, and recover the curves

**Reaping.** Each checkpoint is `pretrained_model/` (9.354 GB) plus `training_state/`
(**2.195 GB**, measured). AdamW keeps its moments in each parameter's own dtype, so the bf16 half of
the expert gets bf16 moments — which is why this is 2.195 GB and not the 5.55 GB that fp32 moments
over 0.693 B trainable would cost. Only the newest needs its optimizer state, since that is what
`--resume` reads; dropping the rest takes eight checkpoints from **92.4 GB to 77.0 GB**.

Run this whenever section 6 shows a new checkpoint. It never touches the newest.
""")

code(r'''
def reap(dry=False):
    cks = sorted((OUT / "checkpoints").glob("0*"))
    freed = 0
    for c in cks[:-1]:                      # never the newest - it is the resume point
        ts = c / "training_state"
        if ts.is_dir():
            n = sum(f.stat().st_size for f in ts.rglob("*") if f.is_file())
            freed += n
            print(f"  {'would drop' if dry else 'dropped'} {ts.relative_to(OUT)}  {n/1e9:.2f} GB")
            if not dry:
                shutil.rmtree(ts)
    print(f"  {'would free' if dry else 'freed'} {freed/1e9:.2f} GB; "
          f"{shutil.disk_usage(DATA).free/1e9:.1f} GB free; kept {len(cks)} checkpoints")

reap(dry=True)     # flip to reap() once the listing looks right
''')

code(r'''
# The held-out loss series, from the log rather than wandb, so checkpoint selection works either
# way. This is the file to bring home alongside the checkpoints.
import csv, re

text = LOG.read_text()
rows = [{"step": s, "eval_loss": v} for s, v in eval_points(text)]
# Train steps are derived from the line ORDER, not parsed from the line (see derived_steps).
losses = [float(m) for m in re.findall(r"\bloss:([\d.eE+-]+)", text)]
train = [{"step": st, "loss": ls} for st, ls in zip(derived_steps(text), losses)]

csvp = DATA / f"{JOB}_curves.csv"
with csvp.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "train_loss", "eval_loss"])
    ev = {r["step"]: r["eval_loss"] for r in rows}
    for r in train:
        w.writerow([r["step"], r["loss"], ev.get(r["step"], "")])
    for s, v in ev.items():                       # eval steps with no train line beside them
        if s not in {r["step"] for r in train}:
            w.writerow([s, "", v])
print(f"{len(train)} train points, {len(rows)} eval points -> {csvp}")
if rows:
    best = min(rows, key=lambda r: r["eval_loss"])
    print(f"lowest held-out loss so far: {best['eval_loss']:.4f} at step {best['step']}")
    for r in rows[-8:]:
        print(f"  {r['step']:>7,}  {r['eval_loss']:.4f}")
''')

# =================================================================================================
md(r"""
## 7b — Recovering a step between two saves

`save_freq=10000` means the held-out minimum at **15,000** has no checkpoint. There is no 15k state
hiding anywhere — a checkpoint *is* the state, and nothing else is retained.

It can be **replayed**, though, not merely approximated. `checkpoints/010000/training_state/` holds
the optimizer moments, the scheduler, the RNG state and the dataloader offset, and lerobot's resume
is sample-exact (`lerobot_train.py:455`), so re-running 10k → 15k retraces the original trajectory
rather than making a new one. It costs those 5,000 steps again: **~3.7 h** at the measured
2.66 s/step. (Exact to within CUDA nondeterminism — this run has `cudnn_deterministic: false`.)

> ### ⚠️ `--steps=15000` would silently hand you a *different* 15k
>
> `CosineDecayWithWarmupSchedulerConfig.build()` rescales the horizon whenever
> `num_training_steps < num_decay_steps` (`schedulers.py:149`). `--steps=15000` compresses the whole
> 80k cosine into 15k, so steps 10,001–15,000 would run at learning rates the original never used.
> Restoring the scheduler state does not save you: `LambdaLR.state_dict()` carries the step counter,
> not the lambda.
>
> Keep `--steps=80000`, make the saves finer with `--save_freq=5000`, and stop once `015000` is on
> disk. `save_freq` changes nothing but when files get written.

Three more things the replay depends on, each of which would spoil it quietly:

| | why |
|---|---|
| `--output_dir=<new>` | the saved config still points at the **live run's** directory |
| `--wandb.enable=false` | the saved config carries the live run's `run_id`; a second writer corrupts its curves |
| `batch_size` / `num_workers` left alone | the dataloader resume offset is computed from them (`lerobot_train.py:466`) |

**Worth 3.7 h?** Probably not on the loss alone. 15k is 0.0650 against 10k's 0.0662 — 1.8% — and it
is a single downward spike between two higher neighbours on a curve that jitters ±0.002 point to
point. Read the repeat-to-repeat std in `checkpoint_metrics.json` first: if it is comparable to
0.0012, that minimum is eval noise and there is nothing there to recover.

### Running it from a terminal instead

The cells above pass `ENV` to `Popen`, so they carry `HF_HOME` and `HF_HUB_OFFLINE`. **A bare shell
does not.** `processor_pi05.py` asks for the paligemma tokenizer by hub id, and without those two
variables the hub client falls back to `HF_ENDPOINT`, which the image presets to `hf-mirror.com` —
unreachable from here. It then retries for five minutes and looks like a hang, not a missing
variable. Check first, in the shell you are about to launch from:

```bash
. /root/gpufree-data/env.sh
env | grep -E 'HF_HOME|HF_HUB_OFFLINE|HF_ENDPOINT'
python -c "from transformers import AutoTokenizer as T; print(len(T.from_pretrained('google/paligemma-3b-pt-224')))"
```

That must print `257153` **instantly**. Any pause at all means the cache is not seeded and it is
reaching for the network — re-run stage 5 of `gpufree_setup_py312.sh`.

Then launch with the environment inline, so sourcing cannot be forgotten:

```bash
cd /root/gpufree-data
setsid env -u HF_ENDPOINT HF_HOME=/root/gpufree-data/cache/hf HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  nohup lerobot-train \
    --config_path=outputs/pi05_d0d1_80k/checkpoints/010000/pretrained_model/train_config.json \
    --resume=true --output_dir=/root/gpufree-data/outputs/pi05_recover_15000 \
    --steps=80000 --save_freq=5000 --wandb.enable=false \
    > recover_15000.log 2>&1 &
echo $! > recover.pid
```

**Finding it again if the pid is stale.** Inside the container `nvidia-smi` shows VRAM in use but
lists no processes — it cannot see across the PID namespace — so a Ctrl-C'd run that leaked its CUDA
context looks like a ghost. Ask `/proc` instead:

```bash
for p in /proc/[0-9]*; do
  ls -l $p/fd 2>/dev/null | grep -q /dev/nvidia && \
    echo "$(basename $p)  $(tr '\0' ' ' < $p/cmdline | cut -c1-90)"
done
```

Kill what that names, then confirm VRAM actually drops before relaunching — a new run needs 18.5 GiB
and will OOM against a process that has not let go.
""")

code(r'''
# Set RECOVER_STEP and run this cell, then the next one.
RECOVER_STEP = 15_000
FROM_STEP    = 10_000
SRC   = OUT / "checkpoints" / f"{FROM_STEP:06d}"
REC   = DATA / f"outputs/pi05_recover_{RECOVER_STEP}"
GRAIN = 5_000          # must divide RECOVER_STEP, and is the ONLY thing changed

assert (SRC / "training_state").is_dir(), (
    f"{SRC}/training_state is missing. Without the optimizer moments, scheduler and RNG state this "
    f"would be a fresh run from {FROM_STEP:,}, not a replay - do not reap it before this runs.")
assert RECOVER_STEP % GRAIN == 0, "RECOVER_STEP must land on a save boundary"
assert shutil.disk_usage(DATA).free / 1e9 > 18, "needs ~11.6 GB for the checkpoint + optimizer state"

rec_cmd = [f"{pathlib.Path(PY).parent}/lerobot-train",
    f"--config_path={SRC}/pretrained_model/train_config.json",
    "--resume=true",
    f"--output_dir={REC}",      # the saved config points at the LIVE run's dir - must override
    f"--steps={STEPS}",         # unchanged, or the cosine is rescaled and the LRs are wrong
    f"--save_freq={GRAIN}",     # the only real change: makes RECOVER_STEP a save point
    "--wandb.enable=false",     # the saved run_id belongs to the live run
]
print("\n".join(rec_cmd))
''')

code(r'''
# The GPU cannot hold this AND the live 80k run at once - 18.5 GiB each against 23.5 - so run it
# after that finishes, or on a second instance. Polls until the checkpoint lands, then stops.
target = REC / "checkpoints" / f"{RECOVER_STEP:06d}" / "pretrained_model" / "model.safetensors"
reclog = (DATA / f"recover_{RECOVER_STEP}.log").open("w")
rp = subprocess.Popen(rec_cmd, env=ENV, stdout=reclog, stderr=subprocess.STDOUT,
                      start_new_session=True)
print(f"pid {rp.pid}  ->  {reclog.name}")

while rp.poll() is None and not target.is_file():
    time.sleep(60)
if target.is_file():
    time.sleep(30)                       # let the write settle before pulling the process
    rp.terminate()
    sz = target.stat().st_size
    print(f"stopped at {RECOVER_STEP:,} · {sz / 1e9:.3f} GB "
          f"({'OK' if sz == 9_354_050_752 else 'UNEXPECTED SIZE'}, expected 9.354)")
else:
    print(f"exited {rp.returncode} before writing {target} - read {reclog.name}")
''')

# =================================================================================================
md(r"""
## 8 — Interruptions, and getting the results home

**Resume** restores optimizer state and the step counter, so it continues rather than restarting.
It needs the newest checkpoint's `training_state/` — which is why section 7 never reaps it:

```python
subprocess.Popen(train_cmd() + ["--resume=true"], env=ENV)   # same output_dir
```

**Bringing it home.** Run from the laptop, one checkpoint at a time — 9.35 GB each:

```bash
rsync -avP -e 'ssh -p <PORT>' \
  root@<HOST>:/root/gpufree-data/outputs/pi05_d0d1_80k/checkpoints/080000/pretrained_model/ \
  ~/Documents/bingjian/robot_learning/smolvla_so101/outputs/train/pi05_d0d1_80k/080000/
```

The small files first — name them, do not pattern-match the tree. `--include='*/'` recurses
everywhere and `--include='*.json'` then matches the conda package cache, which is how a 4 MB pull
became 533 MB:

```bash
DEST=~/Documents/bingjian/robot_learning/smolvla_so101/outputs/train/pi05_d0d1_80k/server
mkdir -p $DEST
# No --include='*/', so rsync never descends: top-level files only.
rsync -avP -e 'ssh -p <PORT>' \
  --include='*.log' --include='*.csv' --include='train_cmd.json' --exclude='*' \
  root@<HOST>:/root/gpufree-data/ $DEST/
# wandb's offline history, named explicitly
rsync -avP -e 'ssh -p <PORT>' \
  root@<HOST>:/root/gpufree-data/outputs/pi05_d0d1_80k/wandb/ $DEST/wandb/
```

`env.sh` is deliberately not in that list — it holds the wandb key.

Take `pi05_d0d1_80k_curves.csv` and the log first — they are kilobytes and they decide which
checkpoints are worth the bandwidth at all.

> **The instance keeps billing while it is stopped if the data disk is kept.** Pull what is needed,
> then decide whether to keep the disk.
""")

# =================================================================================================
nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
dst = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "phase07_gpufree_pi05_train.ipynb"
dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {dst}  ({len(C)} cells)")
