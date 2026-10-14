#!/usr/bin/env python
"""Measure, on this machine, how long one action chunk takes and how much VRAM it needs.

Why this exists. A rollout at 30 fps with `n_action_steps=N` calls the policy every N/30 seconds.
If one chunk takes longer than that to compute, the control loop cannot keep up: the
`SyncInferenceEngine` runs inference inline (`rollout/inference/sync.py`), so a slow chunk is a
visible stall in the arm's motion, not a dropped frame. Whether that happens depends on the card,
and the Phase 0.7 policies were trained on A100/L4 class hardware — nothing in
[phase07_training.md](../docs/details/phase07_training.md) says what they cost on the 5080 laptop that
has to run the rollouts.

This answers it before any robot time is spent, with no robot attached: synthetic observations of
the right shape are fed through the real preprocessor and the real policy, and the per-chunk
latency and peak VRAM are reported against the control-loop budget.

    python scripts/bench_rollout_latency.py \
        --policy outputs/hub/pi05_tray_cylinder_s1000_trim_A100/checkpoints/010000/pretrained_model \
        --n-action-steps 50 --fps 30

Reads the camera keys out of the checkpoint's own config, so it needs no --rename_map: it builds
the batch under the names the policy was trained on, which is exactly what the rename map produces
at rollout time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

TASK = (
    "Pick up the white cylinder from the green tray and place it in the top left blue hole "
    "of the black fixture"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, type=Path, help="a pretrained_model directory")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override the stored value, as the rollout command does")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="which policy camera keys the robot will actually feed "
                         "(default: every visual input the checkpoint declares)")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    cfg_raw = json.loads((args.policy / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = args.policy
    cfg.device = device.type
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps

    visual = [k for k, v in cfg_raw["input_features"].items() if v["type"] == "VISUAL"]
    fed = args.cameras if args.cameras is not None else visual
    state_shape = tuple(cfg_raw["input_features"]["observation.state"]["shape"])

    print(f"policy          : {cfg_raw['type']}  ({args.policy})")
    print(f"device          : {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else ''})")
    print(f"chunk_size      : {cfg.chunk_size}")
    print(f"n_action_steps  : {cfg.n_action_steps}"
          f"{'  (overridden)' if args.n_action_steps is not None else '  (as stored)'}")
    print(f"declared cameras: {visual}")
    print(f"fed cameras     : {list(fed)}")
    print(f"state shape     : {state_shape}")

    # Stats are only needed so the normalizer has something to divide by; the latency and the
    # memory do not depend on their values, only on the tensor shapes.
    stats = {k: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)} for k in visual}
    stats |= {
        "observation.state": {"mean": torch.zeros(state_shape), "std": torch.ones(state_shape),
                              "q01": -torch.ones(state_shape), "q99": torch.ones(state_shape)},
        "action": {"mean": torch.zeros(cfg_raw["output_features"]["action"]["shape"]),
                   "std": torch.ones(cfg_raw["output_features"]["action"]["shape"]),
                   "q01": -torch.ones(cfg_raw["output_features"]["action"]["shape"]),
                   "q99": torch.ones(cfg_raw["output_features"]["action"]["shape"])},
    }

    # `make_policy` wants dataset metadata or an env config to derive features from. A finished
    # checkpoint already carries its own, so it is loaded directly instead.
    policy = get_policy_class(cfg.type).from_pretrained(args.policy, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.policy),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
        },
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"parameters      : {n_params:,}")
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"VRAM after load : {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()

    def fresh_obs() -> dict:
        # uint8 HWC numpy is exactly what an opencv camera hands the engine. The conversion to
        # batched CHW float is then done by lerobot's own `prepare_observation_for_inference`,
        # called below, rather than reimplemented here — so the measured path is the real one.
        obs = {k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in fed}
        obs["observation.state"] = np.zeros(state_shape[0], dtype=np.float32)
        return obs

    rng = np.random.default_rng(0)

    # One `select_action` in N refills the chunk; the rest pop from the queue. Timing the refill
    # is the number that matters, so the policy is reset before each measured call.
    lat = []
    with torch.inference_mode():
        for i in range(args.warmup + args.repeats):
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            obs = prepare_observation_for_inference(fresh_obs(), device, TASK, "so101_follower")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            action = postprocessor(policy.select_action(preprocessor(obs)))
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                lat.append(dt)

    lat.sort()
    budget = cfg.n_action_steps / args.fps
    median, worst = lat[len(lat) // 2], lat[-1]
    print(f"\naction shape    : {tuple(action.shape)}")
    if device.type == "cuda":
        print(f"peak VRAM       : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"chunk latency   : median {median * 1e3:.0f} ms, worst {worst * 1e3:.0f} ms "
          f"over {args.repeats} calls")
    print(f"control budget  : {budget * 1e3:.0f} ms  "
          f"(n_action_steps {cfg.n_action_steps} / {args.fps:g} fps)")

    headroom = budget / worst if worst else float("inf")
    if headroom >= 2:
        verdict = f"OK - {headroom:.1f}x headroom. Sync inference is comfortable."
    elif headroom >= 1:
        verdict = (f"TIGHT - only {headroom:.1f}x headroom. Sync inference will keep up on average "
                   f"but a slow chunk shows as a stutter. Consider --inference.type=rtc.")
    else:
        verdict = (f"TOO SLOW - the chunk costs {1 / headroom:.1f}x its budget. Sync inference "
                   f"will stall the arm every refill. Use --inference.type=rtc, or raise "
                   f"n_action_steps.")
    print(f"verdict         : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
