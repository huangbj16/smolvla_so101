#!/usr/bin/env python3
"""Verify the Python 3.12 + lerobot 0.6.1 env built by gpufree_setup_py312.sh.

Read-only. Prints a report to paste back. Expected values come from the A100 run's saved
checkpoint (outputs/hub/pi05_tray_cylinder_s1000_trim_A100), so a match means this box will
reproduce that recipe rather than something near it.

  python gpufree_verify_py312.py            # everything except the 14 GB model load
  python gpufree_verify_py312.py --load      # also load pi05_base and check the freeze + dtypes
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import pathlib
import sys
import traceback

# Expected, measured from the A100 checkpoint's safetensors header.
EXP_TOTAL_B = 4.1434
EXP_F32_B = 0.5336
EXP_BF16_B = 3.6099
EXP_TRAINABLE_B = 0.6934  # gemma_expert + the action/time projections

BAR = "=" * 78


def head(n: str) -> None:
    print(f"\n{BAR}\n  {n}\n{BAR}")


def out(s: str) -> None:
    print(s, flush=True)


def ok(cond: bool) -> str:
    return "OK  " if cond else "FAIL"


def section_env() -> None:
    head("1 - interpreter and libraries")
    out(f"  python       {sys.version.split()[0]}  at {sys.executable}")
    for name in ("lerobot", "torch", "torchvision", "torchcodec", "av", "transformers",
                 "tokenizers", "accelerate", "safetensors", "datasets", "huggingface_hub",
                 "numpy", "wandb", "draccus", "scipy", "einops"):
        try:
            out(f"  {name:16s} {md.version(name)}")
        except md.PackageNotFoundError:
            out(f"  {name:16s} <absent>")
    for var in ("HF_HOME", "HF_HUB_OFFLINE", "HF_LEROBOT_HOME"):
        out(f"  env {var:18s} {os.environ.get(var, '<unset>')}")


def section_cuda() -> None:
    head("2 - GPU")
    import torch
    out(f"  torch        {torch.__version__}  built for CUDA {torch.version.cuda}")
    out(f"  available    {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        out("  >>> nothing below will run. The CUDA build does not match driver 575.57.08.")
        return
    free, total = torch.cuda.mem_get_info()
    out(f"  device       {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    out(f"  VRAM free    {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    out(f"  bf16 matmul  OK  (sum {(x @ x).float().sum().item():.1f})")
    out(f"  budget       18.18 GiB measured on the A100 at batch 16 with gradient checkpointing "
        f"-> {ok(free / 2**30 > 19.0)} fits with "
        f"{free / 2**30 - 18.18:+.2f} GiB spare")


def section_capability() -> None:
    head("3 - the capabilities 0.4.1 was missing")
    import lerobot
    out(f"  lerobot version  {lerobot.__version__}")
    out(f"  installed at     {pathlib.Path(lerobot.__file__).parent}")

    from lerobot.configs.train import TrainPipelineConfig
    checks: list[tuple[str, bool]] = []

    def fields_of(cls) -> set[str]:
        import dataclasses
        return {f.name for f in dataclasses.fields(cls)}

    try:
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        pf = fields_of(PI05Config)
        d = PI05Config()
        for f in ("train_expert_only", "freeze_vision_encoder", "dtype",
                  "gradient_checkpointing", "scheduler_decay_steps", "use_peft"):
            checks.append((f"PI05Config.{f}", f in pf))
        out(f"\n  PI05Config ({len(pf)} fields)")
        for f in ("train_expert_only", "freeze_vision_encoder", "dtype", "gradient_checkpointing",
                  "chunk_size", "num_inference_steps", "tokenizer_max_length", "max_state_dim",
                  "paligemma_variant", "action_expert_variant", "optimizer_lr",
                  "scheduler_warmup_steps", "scheduler_decay_steps", "normalization_mapping"):
            out(f"    {ok(f in pf)} {f:26s} default={getattr(d, f, '<n/a>')}")
    except Exception as e:
        out(f"  PI05Config FAILED to import: {e}")

    tf = fields_of(TrainPipelineConfig)
    out(f"\n  TrainPipelineConfig ({len(tf)} fields)")
    for f in ("steps", "batch_size", "save_freq", "eval_steps", "max_eval_samples",
              "env_eval_freq", "rename_map", "num_workers", "log_freq", "seed", "resume",
              "output_dir", "save_checkpoint"):
        got = f in tf
        out(f"    {ok(got)} {f}")
        if f in ("eval_steps", "max_eval_samples", "rename_map"):
            checks.append((f"TrainPipelineConfig.{f}", got))

    try:
        from lerobot.configs.default import DatasetConfig as DC
        df = fields_of(DC)
        out(f"\n  DatasetConfig ({len(df)} fields)")
        for f in ("repo_id", "root", "episodes", "eval_split", "video_backend",
                  "use_imagenet_stats", "streaming"):
            got = f in df
            out(f"    {ok(got)} {f}")
            if f == "eval_split":
                checks.append(("DatasetConfig.eval_split", got))
    except Exception as e:
        out(f"  DatasetConfig FAILED: {e}")

    out("")
    for mod, attr in (("lerobot.datasets.factory", "make_train_eval_datasets"),
                      ("lerobot.datasets.aggregate", "aggregate_datasets"),
                      ("lerobot.policies.factory", "make_policy")):
        try:
            got = hasattr(importlib.import_module(mod), attr)
        except Exception:
            got = False
        out(f"  {ok(got)} {mod}.{attr}")
        checks.append((f"{mod}.{attr}", got))

    bad = [n for n, good in checks if not good]
    out(f"\n  >>> {len(checks) - len(bad)}/{len(checks)} required capabilities present")
    if bad:
        out("  >>> MISSING: " + ", ".join(bad))


def section_video() -> None:
    head("4 - video decoding (the dataset is mp4-backed)")
    for name in ("torchcodec", "av"):
        try:
            m = importlib.import_module(name)
            out(f"  {ok(True)} import {name} {getattr(m, '__version__', '?')}")
        except Exception as e:
            out(f"  {ok(False)} import {name}: {type(e).__name__}: {str(e)[:90]}")
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        out("  OK   torchcodec.decoders.VideoDecoder imports "
            "(ffmpeg libs resolve) -> --dataset.video_backend=torchcodec")
    except Exception as e:
        out(f"  FAIL torchcodec decoder: {type(e).__name__}: {str(e)[:90]}")
        out("       -> use --dataset.video_backend=pyav instead (av ships its own ffmpeg)")


def section_tokenizer() -> None:
    head("5 - the offline tokenizer (hardcoded hub id in processor_pi05.py)")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    targets = ["google/paligemma-3b-pt-224"]
    targets += [c for c in ("/data/models/paligemma-3b-pt-224",
                            "/root/data/models/paligemma-3b-pt-224")
                if pathlib.Path(c, "tokenizer.json").is_file()]
    for target in targets:
        try:
            tk = AutoTokenizer.from_pretrained(target)
            out(f"  OK   {target}  vocab {len(tk)}")
        except Exception as e:
            out(f"  FAIL {target}\n         {type(e).__name__}: {str(e)[:160]}")
    out("\n  The hub id is what matters: processor_pi05.py asks for it by name, so a FAIL on the")
    out("  first line means training dies at step 0 even though the files are on disk.")


def section_model(base: str) -> None:
    head("6 - load pi05_base and check the freeze (the decisive test)")
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    # Exactly the features the A100 run resolved to, so the comparison below is apples to apples.
    cfg = PI05Config(
        dtype="bfloat16",
        gradient_checkpointing=True,
        freeze_vision_encoder=True,
        train_expert_only=True,
        chunk_size=50,
        n_action_steps=50,
        input_features={
            "observation.images.base_0_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.left_wrist_0_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.right_wrist_0_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.state": PolicyFeature(FeatureType.STATE, (32,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    out(f"  loading {base} ... (14.5 GB off disk, a few minutes)")
    policy = PI05Policy.from_pretrained(base, config=cfg)

    tot = sum(p.numel() for p in policy.parameters())
    trn = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    hist: dict[str, list[int]] = {}
    for p in policy.parameters():
        e = hist.setdefault(str(p.dtype).replace("torch.", ""), [0, 0])
        e[0] += 1
        e[1] += p.numel()

    out(f"\n  total params      {tot / 1e9:.4f} B   expected {EXP_TOTAL_B:.4f} B   "
        f"{ok(abs(tot / 1e9 - EXP_TOTAL_B) < 0.01)}")
    out(f"  trainable params  {trn / 1e9:.4f} B   expected {EXP_TRAINABLE_B:.4f} B   "
        f"{ok(abs(trn / 1e9 - EXP_TRAINABLE_B) < 0.01)}")
    out(f"  frozen            {(tot - trn) / 1e9:.4f} B  ({100 * (tot - trn) / tot:.1f}%)")
    out("\n  dtype histogram (A100 checkpoint: float32 558 tensors / 0.5336 B, "
        "bfloat16 255 / 3.6099 B)")
    for d, (n, numel) in sorted(hist.items()):
        exp = EXP_F32_B if d == "float32" else EXP_BF16_B if d == "bfloat16" else None
        tag = "" if exp is None else f"   expected {exp:.4f} B  {ok(abs(numel / 1e9 - exp) < 0.01)}"
        out(f"    {d:10s} {n:5d} tensors  {numel / 1e9:7.4f} B{tag}")

    out("\n  trainable top-level modules:")
    seen: dict[str, int] = {}
    for name, p in policy.named_parameters():
        if p.requires_grad:
            key = ".".join(name.split(".")[:3])
            seen[key] = seen.get(key, 0) + p.numel()
    for k, v in sorted(seen.items(), key=lambda kv: -kv[1]):
        out(f"    {k:55s} {v / 1e9:.4f} B")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        policy.to("cuda")
        w = torch.cuda.memory_allocated() / 2**30
        free, _ = torch.cuda.mem_get_info()
        out(f"\n  weights on card   {w:.2f} GiB   (measured on the laptop: 8.72)")
        out(f"  free after        {free / 2**30:.2f} GiB   "
            f"-> gradients 2.58 + AdamW 5.17 + activations ~1.42 = {ok(free / 2**30 > 9.2)} "
            f"{free / 2**30 - 9.17:+.2f} GiB spare at batch 16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="also load pi05_base (14.5 GB, slow)")
    ap.add_argument("--base", default="/data/models/pi05_base")
    a = ap.parse_args()

    print("#" * 78)
    print("#  COPY EVERYTHING BELOW THIS LINE")
    print("#" * 78)
    for fn in (section_env, section_cuda, section_capability, section_video, section_tokenizer):
        try:
            fn()
        except Exception:
            head(f"{fn.__name__} RAISED")
            traceback.print_exc(limit=4, file=sys.stdout)
    if a.load:
        try:
            section_model(a.base)
        except Exception:
            head("6 RAISED")
            traceback.print_exc(limit=6, file=sys.stdout)
    else:
        head("6 - skipped (pass --load to run the decisive freeze/dtype check)")
    print("\n" + "#" * 78)
    print("#  END")
    print("#" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
