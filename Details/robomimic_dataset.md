# robomimic as a test dataset for Project A

Details behind the "Summary of findings" paragraph in [08 §A.2](../08_data_quality_research.md). Checked 2026-09-14.

| | Details |
|---|---|
| **Code** | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic): MIT license, v0.5.0 (June 2025), robosuite v1.5. Includes BC-RNN, BC-Transformer, Diffusion Policy and IQL, plus training and closed-loop eval scripts |
| **Data** | [amandlek/robomimic](https://huggingface.co/datasets/amandlek/robomimic) on Hugging Face: v1.5 files, MIT, ~6.6 GB total. Tasks: Lift, Can, Square, Transport, Tool Hang |
| **Quality labels** | **MH** (Lift, Can, Square, Transport): 300 successful demos, 50 per operator. Operator groups are HDF5 filter keys: `better`, `okay`, `worse`, and pairs like `worse_okay`. Train on one with `config.train.hdf5_filter_key` · **Can-Paired:** 100 start states × {success, deliberate failure} · **MG:** mixed-quality rollouts from RL checkpoints |
| **Files** | `demo_v15.hdf5` (MuJoCo states + actions) and `low_dim_v15.hdf5` (end-effector pose, gripper, joints, object state). Can MH = 80 MB + 113 MB. No image files: regenerate them by replaying the states with `dataset_states_to_obs.py --camera_names agentview robot0_eye_in_hand` |
| **Closed-loop eval** | robosuite (MuJoCo) runs on CPU or any GPU, with no RTX/Isaac requirement |

**How the paper defines quality:**
- **Operator groups** by experience.
- **Trajectory length** is the only number attached to quality (Table 4).
  - Can: 143 / 181 / 304 steps for better / okay / worse.
  - The spread also grows as skill drops.
- **Labels are per operator, not per episode.**

**How it fits the plan:**
1. **Offline metric check (week 2, ~3 h on top of the metric code).**
   - Run integrity, Tier 1 and Tier 2 metrics on the MH `low_dim` files.
   - Report AUC for better vs. worse, and success vs. failure on Can-Paired.
   - Always compare against trajectory length alone; a metric that doesn't beat length adds nothing.
2. **Q1 on natural variation (optional, weeks 3–5, compute runs in background).**
   - Train robomimic's own BC-RNN or Diffusion Policy on top-N / bottom-N / random-N subsets of the 300 MH
     demos, picked by each metric.
   - Roll out in robosuite.
   - This complements the injected-defect dials with variation nobody scripted.
   - RINSE, DemInf and CUPID report results on the same data, so you have reference points.
3. **Backup if LeIsaac slips in week 2.** Q0 still needs your dials, but Q1 can proceed on robomimic.

**Caveats:**
- **Per-operator labels:** a "better" operator can still record a bad episode.
- **Length confound:** slow ≠ bad, the same reason the N1 control exists.
- **Different robot and action space:** a Franka with 7-D delta end-effector actions, not SO-101 joint
  targets. Compute smoothness on end-effector trajectories in both so the numbers compare.
- **Not LeRobot-native:** there's no env integration, and I didn't find a LeRobot-format copy on the Hub.
  ACT/SmolVLA/π0.5 would need a converter plus an eval wrapper, so keep Q2 on LeIsaac.
- **Updated MuJoCo bindings:** the v1.5 files use them, and the docs warn results may not exactly match
  the 2021 paper.

**Sources:** [robomimic GitHub](https://github.com/ARISE-Initiative/robomimic) · [datasets on Hugging Face](https://huggingface.co/datasets/amandlek/robomimic) · [v0.1 dataset docs](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) · [dataset format and filter keys](https://robomimic.github.io/docs/datasets/overview.html) · [generate_paper_configs.py (filter key names)](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/generate_paper_configs.py)
