# SmolVLA on SO-101

Hands-on robot learning on a pair of SO-101 arms: set up the robot, collect teleop demonstrations,
fine-tune vision-language-action policies (SmolVLA, ACT, π0.5), evaluate them on the real arm, and
research what makes a robot dataset good. Along the way, the production code is mapped back to what
was built in W2–W4 (attention, cross-attention, flow-matching action expert, chunking).

**Start with [docs/00_end_to_end_arc.md](docs/00_end_to_end_arc.md).** Current work is in
[docs/08_data_quality_research.md](docs/08_data_quality_research.md) (Project A).

## Layout

```
smolvla_so101/
├── README.md
├── docs/          plan docs 00–09, read in order
│   └── details/   what the plan docs link to: experiment write-ups, eval logs, figures, calibration
├── notebooks/     Colab training notebooks and local analysis notebooks
├── scripts/       data, training and eval tools; run from the repo root
└── results/       small committed outputs: verification JSON, eval summaries, photos
```

Not in git (see [.gitignore](.gitignore)): `outputs/` (local training runs and checkpoints), the
notebooks' `.npz` caches, datasets and videos. Datasets and trained models live on the Hugging Face
Hub under `HALDijkstraaa/`.

## Plan docs — `docs/`

| # | Doc | What it covers |
|---|---|---|
| 00 | [End-to-end arc](docs/00_end_to_end_arc.md) | The spine: one design → build → debug → train → eval loop, and which interview round each stage feeds. **Read first.** |
| 01 | [Set up the robot](docs/01_setup_robot.md) | Ports, motors, calibration, cameras, teleop. ✅ Done; reference only. |
| 02 | [Read the source code](docs/02_read_source_code.md) | Guided tour of the SmolVLA source, mapped back to the W2–W4 code. |
| 03 | [Collect data](docs/03_collect_data.md) | Record, visualize and replay a teleop dataset, treated as a coverage experiment. |
| 04 | [Fine-tune and deploy](docs/04_finetune_and_deploy.md) | Local smoke test, the real fine-tune on Colab, then run the policy on the arm. |
| 05 | [Eval and design](docs/05_eval_and_design.md) | Eval protocol (matched conditions, staged metrics, CIs) and the design-decisions log. |
| 06 | [Expand data, multi-env](docs/06_expand_data_multi_env.md) | Multi-env + negatives collection and the v1 → v3b field log: camera-geometry confound, dual-camera fix. |
| 07 | [Measure data quality](docs/07_measure_data_quality.md) | *Data Quality in Imitation Learning* turned into numbers on your own parquet files. |
| 08 | [Data-quality research](docs/08_data_quality_research.md) | **Project A:** which data defects hurt which policies (ACT / SmolVLA / π0.5), and whether cheap metrics catch them. |
| 09 | [Home deployment and hardware](docs/09_home_deployment_and_hardware.md) | **Project B:** one household task at home, lessons carried forward, hardware and compute costs. |

Docs 03, 04 and 06 were written on the old Windows laptop (RTX 3060, 6 GB): their PowerShell commands,
`D:\` paths and 6 GB workarounds are historical. Translate them to the setup below.

### Details — `docs/details/`

Plan docs stay short and link here for tables, commands and caveats.

| File | What it holds |
|---|---|
| [ubuntu_env_setup.md](docs/details/ubuntu_env_setup.md) | Fresh Ubuntu machine → conda env → `lerobot-calibrate` |
| [calibration/](docs/details/calibration/README.md) | Arm calibration files (`current/` + dated archives) and how to restore them |
| [toolkit_task_card.md](docs/details/toolkit_task_card.md) | The real-robot task: cylinder into the toolkit fixture |
| [robomimic_dataset.md](docs/details/robomimic_dataset.md) | robomimic as a sim test dataset for Project A |
| [phase05_camera_test_results.md](docs/details/phase05_camera_test_results.md) | Phase 0.5: does a second (wrist) camera help? |
| [phase06_camera_ablation_training.md](docs/details/phase06_camera_ablation_training.md) | Phase 0.6: ACT trained on top / wrist / both cameras |
| [phase07_portable_rig.md](docs/details/phase07_portable_rig.md) | Phase 0.7: portable rig, new dataset, round-1 eval, round-2 plan |
| [phase07_data_verification.md](docs/details/phase07_data_verification.md) | Phase 0.7: is the new dataset fit to train on? |
| [phase07_training.md](docs/details/phase07_training.md) | Phase 0.7: ACT, SmolVLA and π0.5 trained on the same data |
| `phase0*_eval_log.csv` | Per-rollout eval logs, scored by hand |
| `figs/` | Figures used by the docs above |

## Notebooks — `notebooks/`

| Notebook | Runs on | Purpose |
|---|---|---|
| [colab_finetune_smolvla.ipynb](notebooks/colab_finetune_smolvla.ipynb), [`_updated`](notebooks/colab_finetune_smolvla_updated.ipynb) | Colab | 04: first SmolVLA fine-tune (PCB pick-and-place). `_updated` is the run that produced the model. |
| [measure_data_quality.ipynb](notebooks/measure_data_quality.ipynb) | local | 07: data-quality metrics on the PCB dataset |
| [phase05_second_camera.ipynb](notebooks/phase05_second_camera.ipynb) | local | Phase 0.5: second-camera test |
| [colab_phase06_f2_f3.ipynb](notebooks/colab_phase06_f2_f3.ipynb) | Colab | Phase 0.6 follow-ups: F2 (camera embedding), F3 (idle-trimmed data) |
| [colab_phase07_prep_trim.ipynb](notebooks/colab_phase07_prep_trim.ipynb) | Colab | Phase 0.7 Stage 2: build the idle-trimmed dataset |
| `colab_phase07_*_gpufree.ipynb`: [ACT](notebooks/colab_phase07_act_gpufree.ipynb) · [SmolVLA](notebooks/colab_phase07_smolvla_gpufree.ipynb) · [π0.5](notebooks/colab_phase07_pi05_gpufree.ipynb) | rented GPU | Phase 0.7 Stage 3: fine-tune each policy on the same data |
| [colab_phase07_pi05_Colab.ipynb](notebooks/colab_phase07_pi05_Colab.ipynb) | Colab | Phase 0.7: π0.5 on Colab; also the template for round 2 |
| [colab_phase07_r2_smolvla.ipynb](notebooks/colab_phase07_r2_smolvla.ipynb) | Colab | Phase 0.7 round 2: continue SmolVLA on D1. Generated by `scripts/make_r2_notebook.py`; edit the script, not the notebook. |

The local notebooks write their `.npz` caches next to themselves, so open them with `notebooks/` as
the working directory (VS Code and Jupyter do this by default).

## Scripts — `scripts/`

Run from the repo root (`python scripts/<name>.py`). Each file's docstring says what it does and how to
call it.

- **Data:** `verify_phase07_dataset.py`, `make_trimmed_dataset.py`, `push_dataset.sh`,
  `fix_grasp_detector.py`, `check_d1_divergence.py`
- **Training:** `train_camera_ablation.sh`, `resume_training.sh`, `push_models.sh`,
  `patch_act_camera_embed.py`, `make_r2_notebook.py`, `eval_checkpoints_heldout.py`
- **Robot eval:** `make_eval_schedule.py` (0.6), `make_phase07_eval_schedule.py`,
  `joint_offset_check.py`, `bench_rollout_latency.py`, `analyze_phase07_eval.py`,
  `analyze_phase07_positions.py` (the `make_phase07_*_figs.py` files are imported by the two analyzers)

## Setup

| | |
|---|---|
| Machine | RTX 5080 Laptop (16 GB), Ubuntu 24.04, since 2026-09-15 |
| Env | conda env `lerobot`, lerobot **0.6.1** (editable install). Steps: [ubuntu_env_setup.md](docs/details/ubuntu_env_setup.md) |
| Robot | SO-101 leader + follower. Plug in the follower first: follower `/dev/ttyACM0`, leader `/dev/ttyACM1` |
| Cameras | Top (bird's-eye) + wrist |
| Cloud | Colab (L4 / A100) and rented GPUs for the longer training runs |

Activate the env in every terminal first:

```bash
conda activate lerobot
```

## References

Versions drift, so check these when a command differs:

- LeRobot SO-101 guide: https://huggingface.co/docs/lerobot/en/so101
- SmolVLA docs: https://huggingface.co/docs/lerobot/en/smolvla
- SmolVLA blog (architecture): https://huggingface.co/blog/smolvla
- `lerobot-<command> --help` shows the exact arguments for the installed version.

CLI used here: `lerobot-find-port` · `lerobot-find-cameras` · `lerobot-setup-motors` ·
`lerobot-calibrate` · `lerobot-teleoperate` · `lerobot-record` · `lerobot-replay` ·
`lerobot-dataset-viz` · `lerobot-train` · `lerobot-eval` · `lerobot-info`
