# SmolVLA on SO-101 — hands-on guide

Goal: get a real vision-language-action policy running on your SO-101, and connect
the production code back to what you built in W2–W4 (attention, cross-attention,
flow-matching action expert, chunking).

## Your setup (confirmed)

| | |
|---|---|
| OS | Windows 11 (native — SO-101 motors appear as **COM ports**, no WSL passthrough) |
| Env | conda env **`lerobot`** (lerobot **0.5.2**, editable install) |
| lerobot source | `D:\SOARM101-Testing\lerobot\src\lerobot\` |
| Robot | SO-101 **leader + follower** pair |
| Camera | 1 USB webcam |
| GPU | RTX 3060 **6 GB** |
| Model | SmolVLA (~450M) — ships with lerobot at `lerobot.policies.smolvla` |

Activate the env in every terminal first:
```powershell
conda activate lerobot
```

## Reality check on the 6 GB GPU (read before you plan time)

- **Inference / deploy SmolVLA:** fine on 6 GB (450M in bf16 ≈ ~1 GB weights + activations).
- **Fine-tuning SmolVLA:** **tight.** Community configs assume 16–24 GB. On 6 GB you must:
  freeze the VLM backbone, tiny batch size (1–2) + gradient accumulation, bf16/AMP, short
  chunk. Expect to fight OOM; a small proof-of-concept fine-tune is realistic, a full one is not.
  Alternative: fine-tune in the cloud (Colab/A100) and copy the checkpoint back to deploy locally.
- **Honest note:** the *pretrained* `lerobot/smolvla_base` is a base model — running it zero-shot
  on your SO-101 will move the arm but **won't do your task** until you fine-tune on your own data.
  So "deploy" first means "prove the inference pipeline works," not "it does the task."

## Roadmap

**Start with [00_end_to_end_arc.md](00_end_to_end_arc.md)** — it frames this whole folder as one
design→build→debug→train→eval loop and ties each stage to the onsite rounds (Chelsea/Michael/Adrian).

0. **[00_end_to_end_arc.md](00_end_to_end_arc.md)** — the spine: the arc + which interview round each
   stage feeds. Read first.
1. ~~**[01_setup_robot.md](01_setup_robot.md)**~~ — ✅ **DONE** (teleop loop working). Reference only.
2. **[02_read_source_code.md](02_read_source_code.md)** — guided tour of the SmolVLA source,
   mapping every piece back to your W2/W3/W4 code. Includes a Python snippet to load the model and
   watch the shapes you now understand flow through it.
3. **[03_collect_data.md](03_collect_data.md)** — teleoperate to record a demonstration dataset,
   visualize and replay it. (Treat it as a *coverage experiment*, per 00/05.)
4. **[04_finetune_and_deploy.md](04_finetune_and_deploy.md)** — smoke-test the fine-tune pipeline
   locally (6 GB-friendly), then do the real fine-tune on ColabPro, and run the policy on the robot.
5. **[05_eval_and_design.md](05_eval_and_design.md)** — the eval protocol (matched conditions, staged
   metrics, CIs) + a design-decisions log. **Your Chelsea design-round centerpiece.**
6. **[06_expand_data_multi_env.md](06_expand_data_multi_env.md)** — multi-env + negatives collection
   plan, and the **field results log** (v1 → v3 → v3b): the empty-scene bug, the camera-geometry
   confound, and the dual-camera rig that fixes the ~1 cm grasp error.
7. **[07_measure_data_quality.md](07_measure_data_quality.md)** — turning *Data Quality in Imitation
   Learning* into numbers on your own parquet files. Companion: `measure_data_quality.ipynb`.
8. **[08_data_quality_research.md](08_data_quality_research.md)** — **Project A research
   plan**: how to quantitatively measure dataset quality. Field overview, metrics, sim + real experiments
   with injected defects across ACT/SmolVLA/π0.5, the second-camera test and the haptic-teleop study, on
   an 8-week timeline.
9. **[09_home_deployment_and_hardware.md](09_home_deployment_and_hardware.md)** — **Project B** (one
   household task at home), lessons carried forward, gaps, and the full hardware/compute bill of
   materials (GPU memory, π0.5 deployment memory, pricing, SO-101 repeatability).

## Authoritative references (versions drift — check these when a command differs)

- LeRobot SO-101 guide: https://huggingface.co/docs/lerobot/en/so101
- SmolVLA docs: https://huggingface.co/docs/lerobot/en/smolvla
- SmolVLA blog (architecture): https://huggingface.co/blog/smolvla
- Always: `lerobot-<command> --help` shows the exact args for *your* installed 0.5.2.

## The CLI you'll use (confirmed present in your install)

`lerobot-find-port` · `lerobot-find-cameras` · `lerobot-setup-motors` · `lerobot-calibrate` ·
`lerobot-teleoperate` · `lerobot-record` · `lerobot-replay` · `lerobot-dataset-viz` ·
`lerobot-train` · `lerobot-eval` · `lerobot-info`
