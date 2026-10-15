#!/usr/bin/env python3
"""Generate notebooks/phase07_gpufree_env_probe.ipynb.

A read-only probe of the gpufree.cn box. It installs nothing, downloads nothing and writes
nothing outside its own output, because the box is metered and the point is to decide what to
do BEFORE committing to a ~3-day run.
"""
import json, pathlib

C = []          # cells


def md(src):
    C.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": src.strip("\n").splitlines(True)})


# =================================================================================================
md(r"""
# Phase 0.7 — gpufree.cn environment probe

**Read-only.** Installs nothing, downloads nothing, writes nothing. Takes ~3 minutes.

It exists because the shipped image is **lerobot 0.4.0**, and 0.4.0 cannot run the π0.5 recipe
round 1 used:

| What round 1 used | In lerobot 0.4.0 |
|---|---|
| `--policy.train_expert_only=true` (693 M of 4.14 B trainable) | **absent** — 0.4.0 trains all 4.14 B, a full fine-tune needing ≥70 GB |
| `--policy.freeze_vision_encoder=true` | **absent** |
| `--dataset.eval_split` + `--eval_steps=2500` (held-out loss) | **absent** — its `eval_freq` only drives gym rollouts, skipped when there is no env |

So either lerobot gets upgraded to 0.6.1 on the box, or 0.4.0 gets patched. This notebook
measures which is possible, rather than guessing.

### How to run

1. Start the instance, open Jupyter Lab, upload this notebook.
2. **Run all cells in order** (`Run > Run All Cells`).
3. Copy the single block the **last cell** prints and paste it back into the chat.

Nothing here depends on the dataset or on Hugging Face being reachable — failures are the
*result*, not an error. Every check is wrapped, so one failure never stops the rest.
""")

# ---------------------------------------------------------------------------------------------
md("## 0 — Setup\n\nDefines the reporting helpers. Everything after this appends to `REPORT`, which the last cell prints as one block.")

code(r'''
import json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

REPORT = []


def sec(title):
    bar = "=" * 78
    for line in ("", bar, f"  {title}", bar):
        print(line); REPORT.append(line)


def out(*parts):
    s = " ".join(str(p) for p in parts)
    print(s); REPORT.append(s)


def sh(cmd, timeout=30):
    """Run a shell command, never raise. Returns (returncode, combined output)."""
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"<timed out after {timeout}s>"
    except Exception as e:
        return -1, f"<{type(e).__name__}: {e}>"


def guard(fn):
    """Run a probe section; a failure is reported as data, not raised."""
    try:
        fn()
    except Exception as e:
        out(f"  !! this section failed: {type(e).__name__}: {e}")


print("ready - run the cells below in order")
''')

# ---------------------------------------------------------------------------------------------
md("## 1 — Host\n\nCPU count and RAM matter for `--num_workers`; round 1 used 8 workers.")

code(r'''
assert "REPORT" in globals(), "Run the setup cell (section 0) first."

def _host():
    sec("1 - HOST")
    out(f"  platform     {platform.platform()}")
    out(f"  python       {sys.version.split()[0]}  at {sys.executable}")
    out(f"  cpu count    {os.cpu_count()}")
    _, model = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")
    out(f"  cpu model   {model}")
    try:
        mem = {l.split(":")[0]: l.split(":")[1].strip()
               for l in Path("/proc/meminfo").read_text().splitlines()[:3]}
        for k, v in mem.items():
            out(f"  {k:12s} {v}")
    except Exception as e:
        out(f"  /proc/meminfo unreadable: {e}")
    _, who = sh("whoami"); out(f"  user         {who}")
    out(f"  cwd          {Path.cwd()}")

guard(_host)
''')

# ---------------------------------------------------------------------------------------------
md("## 2 — GPU\n\n`mem_get_info` is the number that matters: **free** VRAM, not nominal. The expert-only peak measured on an A100 was 18.18 GiB with gradient checkpointing on.")

code(r'''
def _gpu():
    sec("2 - GPU")
    code_, smi = sh("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,"
                    "compute_cap --format=csv,noheader")
    out(f"  nvidia-smi   {smi if code_ == 0 else '<unavailable>'}")
    _, full = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    out(f"  other procs  {full or '<none>'}")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out(f"  torch sees   {torch.cuda.get_device_name(0)}")
            out(f"  VRAM free    {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
            out(f"  capability   {torch.cuda.get_device_capability(0)}")
        else:
            out("  !! torch.cuda.is_available() is False")
    except Exception as e:
        out(f"  torch probe failed: {type(e).__name__}: {e}")

guard(_gpu)
''')

# ---------------------------------------------------------------------------------------------
md("""
## 3 — Disks

Decides where ~80 GB of checkpoints land. The image ships a **30 GB system disk** (holding `/root`
and therefore `~/.cache`) plus a **data disk**, so the Hugging Face cache must be redirected or it
quietly fills the system disk.

Writability is tested with a 1 KB temp file that is deleted immediately — a mount can be present
and read-only.
""")

code(r'''
def _disks():
    sec("3 - DISKS")
    _, df = sh("df -h -x tmpfs -x devtmpfs -x overlay 2>/dev/null")
    out(df); out("")
    cands = ["/root/gpufree-data", "/root/autodl-tmp", "/root/data", "/data", "/workspace",
             "/mnt/data", "/root", "/tmp"]
    out(f"  {'path':24s} {'total':>9s} {'free':>9s}  writable")
    for p in cands:
        # Every step guarded separately: one unreadable path must not cost the whole section.
        try:
            exists = Path(p).is_dir()
        except Exception as e:
            out(f"  {p:24s} {'-':>9s} {'-':>9s}  <{type(e).__name__}>"); continue
        if not exists:
            out(f"  {p:24s} {'-':>9s} {'-':>9s}  <does not exist>"); continue
        try:
            u = shutil.disk_usage(p)
            tot, free = f"{u.total / 1e9:.0f} GB", f"{u.free / 1e9:.0f} GB"
        except Exception:
            tot = free = "?"
        try:
            t = Path(p) / ".probe_write_test"
            t.write_bytes(b"x" * 1024); t.unlink(); w = "yes"
        except Exception as e:
            w = f"NO ({type(e).__name__})"
        out(f"  {p:24s} {tot:>9s} {free:>9s}  {w}")
    out("")
    for v in ("HF_HOME", "HF_LEROBOT_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE"):
        out(f"  env {v:20s} {os.environ.get(v, '<unset>')}")

guard(_disks)
''')

# ---------------------------------------------------------------------------------------------
md("## 4 — Installed libraries\n\nThe image's torch build constrains whether lerobot 0.6.1 can be installed: 0.6.1 requires `torch>=2.7,<2.12`.")

code(r'''
def _libs():
    sec("4 - LIBRARIES")
    try:
        import torch
        out(f"  torch            {torch.__version__}")
        out(f"  built for CUDA   {torch.version.cuda}   | cudnn {torch.backends.cudnn.version()}")
        out(f"  cuda available   {torch.cuda.is_available()}")
    except Exception as e:
        out(f"  torch import failed: {type(e).__name__}: {e}")
    for mod in ("torchvision", "torchaudio", "transformers", "tokenizers", "accelerate",
                "safetensors", "datasets", "huggingface_hub", "numpy", "wandb", "torchcodec",
                "av", "draccus"):
        try:
            m = __import__(mod)
            out(f"  {mod:16s} {getattr(m, '__version__', '<no __version__>')}")
        except Exception as e:
            out(f"  {mod:16s} <not importable: {type(e).__name__}>")
    out("")
    _, nv = sh("nvcc --version | tail -2"); out(f"  nvcc: {nv or '<absent>'}")

guard(_libs)
''')

# ---------------------------------------------------------------------------------------------
md("""
## 5 — lerobot capability probe

**The decisive section.** It reads the *installed* dataclasses rather than trusting the version
number, because the image may have been patched. Four questions:

1. Does `PI05Config` have `train_expert_only` / `freeze_vision_encoder`? Without them π0.5 is a
   full 4.14 B fine-tune and will not fit 24 GB or 48 GB.
2. Does `TrainPipelineConfig` have `eval_steps`, and `DatasetConfig` have `eval_split`? That is
   the held-out loss every 2,500 steps.
3. Is lerobot an **editable** install? If so, patching it in place is straightforward.
4. Is the dataset `CODEBASE_VERSION` `v3.0`, matching the recorded datasets?
""")

code(r'''
def _lerobot():
    sec("5 - LEROBOT CAPABILITY")
    import dataclasses
    try:
        import lerobot
    except Exception as e:
        out(f"  !! import lerobot failed: {type(e).__name__}: {e}"); return
    root = Path(lerobot.__file__).resolve().parent
    out(f"  lerobot version  {getattr(lerobot, '__version__', '<unknown>')}")
    out(f"  installed at     {root}")
    editable = "site-packages" not in str(root) and "dist-packages" not in str(root)
    out(f"  editable install {editable}   (True = patching in place is easy)")
    _, origin = sh(f"{sys.executable} -m pip show lerobot 2>/dev/null | grep -iE 'location|editable'")
    out(f"  pip show         {origin or '<none>'}")

    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    except Exception:
        try:
            from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
        except Exception as e:
            CODEBASE_VERSION = f"<unreadable: {type(e).__name__}>"
    out(f"  dataset CODEBASE_VERSION  {CODEBASE_VERSION}   (recorded datasets are v3.0)")

    def fields_of(label, cls, wanted):
        try:
            have = {f.name for f in dataclasses.fields(cls)}
        except Exception as e:
            out(f"  {label}: <{type(e).__name__}>"); return
        out(f"\n  {label}  ({len(have)} fields)")
        for w in wanted:
            out(f"    {'YES' if w in have else '-- ':4s} {w}")
        return have

    try:
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        fields_of("PI05Config", PI05Config, [
            "train_expert_only", "freeze_vision_encoder", "dtype", "gradient_checkpointing",
            "compile_model", "tokenizer_max_length", "scheduler_decay_steps",
            "scheduler_warmup_steps", "optimizer_lr", "normalization_mapping", "chunk_size",
            "empty_cameras", "use_peft"])
        d = PI05Config()
        out(f"    default normalization_mapping = "
            f"{ {k: getattr(v, 'value', v) for k, v in d.normalization_mapping.items()} }")
        out(f"    default dtype={d.dtype}  chunk_size={d.chunk_size}  "
            f"tokenizer_max_length={d.tokenizer_max_length}")
        out(f"    default scheduler: warmup={d.scheduler_warmup_steps} "
            f"decay_steps={d.scheduler_decay_steps} lr={d.optimizer_lr} "
            f"decay_lr={d.scheduler_decay_lr}")
    except Exception as e:
        out(f"  PI05Config probe failed: {type(e).__name__}: {e}")

    try:
        from lerobot.configs.train import TrainPipelineConfig
        fields_of("TrainPipelineConfig", TrainPipelineConfig, [
            "eval_steps", "eval_freq", "env_eval_freq", "save_freq", "steps", "batch_size",
            "num_workers", "log_freq", "seed", "resume", "output_dir", "rename_map",
            "max_eval_samples"])
    except Exception as e:
        out(f"  TrainPipelineConfig probe failed: {type(e).__name__}: {e}")

    try:
        from lerobot.configs.default import DatasetConfig
        fields_of("DatasetConfig", DatasetConfig, [
            "eval_split", "episodes", "repo_id", "root", "revision", "video_backend",
            "streaming", "use_imagenet_stats"])
    except Exception as e:
        out(f"  DatasetConfig probe failed: {type(e).__name__}: {e}")

    out("")
    for mod, what in [("lerobot.datasets.aggregate", "aggregate_datasets (dataset merge)"),
                      ("lerobot.datasets.factory", "make_train_eval_datasets (held-out split)")]:
        try:
            m = __import__(mod, fromlist=["x"])
            name = what.split()[0]
            out(f"  {'YES' if hasattr(m, name) else '-- '}  {mod}.{what}")
        except Exception as e:
            out(f"  --   {mod} <{type(e).__name__}>")

    _, w = sh("which lerobot-train lerobot-record lerobot-eval 2>/dev/null")
    out(f"\n  CLI on PATH:\n{w or '    <none on PATH>'}")
    try:  # PATH in a notebook kernel is not a login shell's, so also ask the package metadata
        from importlib.metadata import entry_points
        eps = [e.name for e in entry_points(group="console_scripts") if "lerobot" in e.name]
        out(f"  console_scripts declared: {sorted(eps) or '<none>'}")
    except Exception as e:
        out(f"  console_scripts probe failed: {type(e).__name__}")

guard(_lerobot)
''')

# ---------------------------------------------------------------------------------------------
md("""
## 6 — Network

The box is stated not to reach Hugging Face. This measures *what* it can reach, because the answer
decides three separate things: whether lerobot 0.6.1 can be installed (PyPI), whether the dataset
can be pulled directly instead of uploaded by hand (HF), and whether training metrics can stream
(wandb).

Each request has a short timeout, so a blocked host costs seconds, not minutes. A failure here is
an expected outcome, not a problem with the notebook.
""")

code(r'''
def _net():
    sec("6 - NETWORK")
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
              "NO_PROXY", "no_proxy"):
        if os.environ.get(v):
            out(f"  env {v:12s} {os.environ[v]}")
    out("")
    import urllib.request
    targets = [
        ("PyPI            ", "https://pypi.org/simple/"),
        ("PyPI (tsinghua) ", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
        ("PyPI (aliyun)   ", "https://mirrors.aliyun.com/pypi/simple/"),
        ("HuggingFace     ", "https://huggingface.co"),
        ("HF api          ", "https://huggingface.co/api/models/lerobot/pi05_base"),
        ("HF mirror hf-m  ", "https://hf-mirror.com"),
        ("wandb api       ", "https://api.wandb.ai"),
        ("github          ", "https://github.com"),
    ]
    import urllib.error
    for label, url in targets:
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "probe"})
            with urllib.request.urlopen(req, timeout=8) as r:
                out(f"  {label} OK    HTTP {r.status}  ({time.time() - t0:.1f}s)")
        except urllib.error.HTTPError as e:
            # An HTTP status - even 401/403/404 - means the host answered, which is what is
            # being measured. Only a transport failure means "cannot reach".
            out(f"  {label} OK    HTTP {e.code} (reachable)  ({time.time() - t0:.1f}s)")
        except Exception as e:
            out(f"  {label} FAIL  {type(e).__name__}: {str(e)[:70]}  ({time.time() - t0:.1f}s)")
    out("")
    _, cfg = sh(f"{sys.executable} -m pip config list 2>/dev/null")
    out(f"  pip config:\n{cfg or '    <empty - using default index>'}")
    out(f"  env PIP_INDEX_URL  {os.environ.get('PIP_INDEX_URL', '<unset>')}")

guard(_net)
''')

# ---------------------------------------------------------------------------------------------
md("""
## 7 — Could lerobot 0.6.1 be installed?

`pip install --dry-run` resolves the dependency graph and **installs nothing**. This is the single
most useful answer in the notebook: if 0.6.1 resolves against the image's torch, the run keeps exact
parity with round 1 and with the ACT and SmolVLA arms, and no patching is needed.

Watch for whether it proposes to **replace torch** — that is the risk, since the image's CUDA build
is known-good and a resolver-chosen wheel may not be.

Skipped automatically if no package index was reachable in section 6.
""")

code(r'''
def _upgrade():
    sec("7 - CAN LEROBOT 0.6.1 BE INSTALLED? (dry run, installs nothing)")
    import urllib.request
    index = None
    for url in ("https://pypi.org/simple/", "https://pypi.tuna.tsinghua.edu.cn/simple/",
                "https://mirrors.aliyun.com/pypi/simple/"):
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "probe"}),
                                   timeout=8)
            index = url; break
        except Exception:
            continue
    if not index:
        out("  no package index reachable - skipping. lerobot must be PATCHED rather than upgraded.")
        return
    out(f"  using index  {index}")
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--no-input",
           "--index-url", index, "lerobot[pi,dataset,training]==0.6.1"]
    out(f"  $ {' '.join(cmd)}\n")
    rc, txt = sh(cmd, timeout=420)
    lines = [l for l in txt.splitlines()
             if l.startswith(("Would install", "Requirement already", "ERROR", "error:"))
             or "torch" in l.lower() or "Collecting lerobot" in l]
    out("\n".join(f"  {l}" for l in (lines[-40:] or txt.splitlines()[-30:])))
    out(f"\n  exit code {rc}   ({'RESOLVES' if rc == 0 else 'DOES NOT RESOLVE'})")

guard(_upgrade)
''')

# ---------------------------------------------------------------------------------------------
md("""
## 8 — Local models and the offline tokenizer

The image ships `pi05_base` and `paligemma-3b-pt-224`, so the base model probably does **not** need
uploading — only the dataset does.

The catch is that `processor_pi05.py` hardcodes `tokenizer_name="google/paligemma-3b-pt-224"` and
resolves it through the Hub. On a box with no Hugging Face access that fails ~10 minutes into a run,
after the model is already on the GPU. This section finds the local copy and tests whether the
tokenizer loads **offline**, both from the hub id and from the directory.
""")

code(r'''
def _models():
    sec("8 - LOCAL MODELS + OFFLINE TOKENIZER")
    roots = ["/root/data/models", "/data/models", "/root/models", "/root/gpufree-data/models",
             "/workspace/models", "/root/data", "/data"]
    seen = []
    for r in roots:
        try:
            p = Path(r)
            if not p.is_dir():
                continue
            children = sorted(p.iterdir())[:25]
        except Exception as e:
            out(f"  {r:28s} <{type(e).__name__}>"); continue
        for child in children:
            try:
                if child.is_dir() and any(k in child.name.lower()
                                          for k in ("pi0", "pi05", "paligemma", "gemma", "smolvla")):
                    seen.append(child)
            except Exception:
                continue
    if not seen:
        out("  no pi05/paligemma directory found under: " + ", ".join(roots))
    for d in seen:
        rc, sz = sh(f"du -sh '{d}' 2>/dev/null | cut -f1")
        out(f"\n  {d}   ({sz or '?'})")
        try:
            for f in sorted(d.iterdir())[:14]:
                out(f"      {f.stat().st_size:>13,d}  {f.name}" if f.is_file()
                    else f"      {'<dir>':>13s}  {f.name}/")
        except Exception as e:
            out(f"      <cannot list: {type(e).__name__}>")

    out("")
    _, found = sh("find / -maxdepth 6 -type d \\( -name 'pi05_base' -o -name 'paligemma-3b-pt-224' \\) "
                  "-not -path '*/proc/*' 2>/dev/null | head -10", timeout=90)
    out(f"  filesystem search:\n{found or '    <nothing found>'}")

    out("\n  --- offline tokenizer test (HF_HUB_OFFLINE=1, subprocess so a hang cannot kill the kernel) ---")
    snippet = (
        "import os, sys\n"
        "os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'\n"
        "from transformers import AutoTokenizer\n"
        "src = sys.argv[1]\n"
        "t = AutoTokenizer.from_pretrained(src)\n"
        "print('OK vocab', t.vocab_size)\n")
    sp = Path("/tmp/_tok_probe.py"); sp.write_text(snippet)
    cands = ["google/paligemma-3b-pt-224"] + [str(d) for d in seen if "paligemma" in d.name.lower()]
    for src in cands:
        rc, txt = sh([sys.executable, sp.as_posix(), src], timeout=120)
        last = (txt.strip().splitlines() or ["<no output>"])[-1][:150]
        out(f"  {'OK  ' if rc == 0 else 'FAIL'}  {src}\n          {last}")
    sp.unlink(missing_ok=True)

guard(_models)
''')

# ---------------------------------------------------------------------------------------------
md("## 9 — Paste this back\n\nOne block, everything above. Copy it into the chat.")

code(r'''
print("\n\n" + "#" * 78)
print("#  COPY EVERYTHING BELOW THIS LINE")
print("#" * 78 + "\n")
print("\n".join(REPORT))
print("\n" + "#" * 78)
print("#  END - that is the whole report")
print("#" * 78)
''')

# =================================================================================================
nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

dst = pathlib.Path("/home/bj/Documents/bingjian/robot_learning/smolvla_so101/notebooks/"
                   "phase07_gpufree_env_probe.ipynb")
dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {dst}  ({len(C)} cells)")
