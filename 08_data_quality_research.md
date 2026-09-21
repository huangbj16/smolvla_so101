# 08 — Project A: what makes a robot dataset good?

Docs 00–07 took you from "teleop works" to a 160-episode multi-env dataset, three trained checkpoints, a
confirmed camera confound and a data-quality notebook. This doc is the research plan that follows.

The rest now lives in [09](09_home_deployment_and_hardware.md):
- Project B (home deployment).
- The four lessons carried forward from 06/07.
- Gaps.
- The full hardware/compute analysis.

# PART A — What makes a robot dataset good? (8-week plan)

> **Revision 4 (2026-09-14).**
> - Follows your 8-week timeline; operator state is in Future todo (§A.7).
> - Adds the second-camera test, the haptic study, and a field overview in §A.2.
> - Project B and the full hardware notes moved to [09](09_home_deployment_and_hardware.md).

## A.1 — Questions

| ID | Question | When |
|---|---|---|
| **C1** Second camera | Does adding the gripper camera raise visual state diversity and lower action divergence? | Week 1 — [done, passed](Details/phase05_camera_test_results.md) |
| **C1b** Camera ablation | Does the wrist camera actually raise ACT success rate, and does divergence predict it? | Week 2 — [done, partly](Details/phase06_camera_ablation_training.md) |
| **Q0** Defects | Which data defects (jerky, hesitant, fumbled, inconsistent) hurt closed-loop success, and by how much? | Sim wk 3–5 · real wk 6–7 |
| **Q1** Ranking | Does training on the top-N episodes by metric M beat random-N and bottom-N? | Sim wk 3–5 · real wk 6–7 |
| **Q2** Policy transfer | Do Q0/Q1 hold across ACT, SmolVLA and π0.5? | Sim wk 3–5 · real wk 6–7 |
| **H1** Haptics | On a separate chip-grasp task: does vibrotactile feedback from gripper tactile sensors make grip force more consistent across episodes and users, and policies better? | Week 8 |

**Framing:** not a new metric. The question is which defects matter to which policies, and whether cheap
metrics can find them before training.

## A.2 — Literature

### How the field evolved (2021 → 2026)

**Before 2023: quality meant "who collected it".** The problem was visible before it had a vocabulary.
Mandlekar et al.'s robomimic study (CoRL 2021) trained the same algorithms on demonstrations from
operators rated "worse", "okay" and "better". Proficiency changed policy performance a lot, and
mixed-quality human data was harder to learn from than proficient-only data. But "quality" was only a
label on the operator. Gandhi, Karamcheti, Liao & Sadigh (CoRL 2022) went one step further. People pick
different, equally valid strategies for the same task, and mixing *incompatible* strategies hurts the
policy. So they scored each new demonstration by its compatibility with a base policy, and coached
operators toward compatible behavior.

**2023: quality became a property of the data.** Belkhale, Cui & Sadigh (NeurIPS 2023) turned those
observations into properties you can reason about without naming the operator:
- **Action divergence:** how inconsistent actions are at similar states. It's a formal version of
  Gandhi's incompatibility.
- **Transition diversity:** noise in what happens after a given state and action.
- **State diversity** is not automatically good.

This set the agenda but left two gaps. The metrics were analysis tools on controlled datasets rather than
a scalable way to score episodes. And "good" was argued from distribution shift, not measured against how
the policy actually performs on the robot (closed-loop success).

**2024: the view zoomed out to whole datasets and diversity.** Large multi-robot datasets like Open
X-Embodiment shifted attention from the episode to the dataset. Hejna et al.'s Re-Mix (2024) asked how
much weight each *domain* should get when pretraining a generalist policy. It learned the weights with
distributionally robust optimization instead of choosing them by hand. Lin et al. (2024) measured data
scaling laws. Diversity of environments and objects dominates, and beyond a threshold, more demonstrations
per environment add little. That made Belkhale's diversity point concrete enough to plan data collection
around. Meanwhile, human-robot interaction researchers approached quality from the demonstrator's side.
Sakr, Van der Loos, Kulić & Croft ("Consistency Matters", arXiv 2024, later ACM Transactions on
Human-Robot Interaction) showed that how *consistent* a person is across repetitions predicts task
success. They measured this as variance in path length, jerk, curvature and joint effort. It is
essentially a kinematic, per-person version of action divergence.

**2025: from proxies to causal quality.** The main shift was to stop guessing which property matters and
measure each demonstration's actual effect on the policy.
- **Learned estimates without rollouts.** Hejna et al.'s DemInf (RSS 2025) replaced hand-picked
  properties with a learned estimate of the mutual information between states and actions.
  Demonstrations whose actions are unpredictable from their states (high action divergence) contribute
  little, and get filtered out. Zhang et al.'s SCIZOR (2025) scored below the episode level. A
  self-supervised progress predictor removes transitions that don't advance the task, and duplicate
  state-action segments are removed too.
- **Rollout-based quality.** Chen, Lessing, Liu & Finn's Demo-SCORE (2025) trains a policy, rolls it out,
  and learns a classifier that separates successful from failed rollouts. It then keeps the
  demonstrations that classifier favors, so quality means "reproducible by this robot".
- **Causal quality.** Agia et al.'s CUPID (CoRL 2025) makes this rigorous. It adapts influence functions
  (Koh & Liang, 2017) to estimate how much each training demonstration raises or lowers the policy's
  *expected closed-loop return*. That closes Belkhale's gap: the score is tied directly to downstream
  outcomes.
- **Multi-task selection.** Dass et al.'s DataMIL (2025, ICLR 2026) uses datamodels (Ilyas et al., 2022)
  to choose which prior data, e.g. from Open X-Embodiment, helps a target task. It optimizes a cheaper
  stand-in objective instead of running rollouts.

The price was that quality became **tied to one policy and expensive**. You need a trained policy, often
rollouts, and the score only holds for that policy.

**2026: cheap again, and moving into data collection.** There were two reactions.
- **Cheaper causal scoring.** Lee et al.'s QoQ (ICRA 2026) keeps influence functions but measures
  influence on the loss over a few held-out validation demonstrations, so no rollouts are needed.
- **Back to model-free signals.**
  - **Smoothness.** Kulkarni, Dhar & Cui's RINSE (2026) scores demonstrations by smoothness (spectral arc
    length plus a trajectory-envelope distance) from joint data alone. Its argument is that curation
    needing a policy in the loop doesn't scale. Cui co-authored Belkhale's paper, so this line comes
    straight out of that group.
  - **A metric that runs in seconds.** Sojib, Arthanat & Begum (2026) made smoothness scoring even cheaper
    with a power-spectral-density score. They validated it by fine-tuning π0.5 on demonstrations from
    older adults, which moves the question from expert teleoperators to lay users.
- **Scoring during collection.** Narayanan et al. (2026) compute cheap telemetry scores after every
  episode and feed them back to the operator: smoothness, gripper chatter, stalling, joint-limit
  saturation. The same fix-it-at-the-source idea appears in haptics: Cuan, Okamura & Khansari (IEEE
  Transactions on Haptics, 2024) found that feedback during teleoperation raised both data throughput and
  policy success.

**The arc in one line:** a label on the operator (2021–22) → properties of the data (2023) → dataset
composition and diversity (2024) → causal effect on a specific policy (2025) → cheap proxies checked
against causal scores, and pushed into data collection (2026). The scoring unit moved the other way:
dataset or domain → episode → transition.

**Categories of metrics:**
1. **Property-based, no model needed:** diversity, action divergence and consistency, smoothness,
   telemetry. Cheap and policy-independent, but they are proxies.
2. **Learned estimators without rollouts:** mutual information (DemInf), progress prediction (SCIZOR),
   validation-loss influence (QoQ). Medium cost.
3. **Causal, tied to a policy:** rollout influence (CUPID), datamodels (DataMIL), rollout classifiers
   (Demo-SCORE). Expensive and policy-specific, but closest to ground truth.
4. **Dataset or mixture level:** Re-Mix, scaling laws.
5. **Interventions during collection:** compatibility coaching, operator feedback, haptics.

**What's still open, and where this project sits:**
- **One policy family per study.** Most papers validate on one kind of policy, so no one has shown
  whether cheap proxies and causal scores agree across ACT, SmolVLA and π0.5. That's Q2.
- **Defect types are inferred, not tested.** Which defects actually hurt has not been tested by
  deliberately injecting them. That's Q0.
- **Haptics lacks consistency measures.** The haptics evidence reports throughput and policy success,
  not the grip-force consistency measures H1 targets.

**Links:**
- 2021–2023: [robomimic](https://robomimic.github.io/study/) · [Gandhi et al.](https://arxiv.org/abs/2210.08073) · [Belkhale et al.](https://arxiv.org/abs/2306.02437)
- 2024: [Re-Mix](https://arxiv.org/abs/2408.14037) · [Data Scaling Laws](https://arxiv.org/abs/2410.18647) · [Consistency Matters](https://arxiv.org/abs/2412.14309) · [Cuan et al.](https://arxiv.org/abs/2211.03020)
- 2025: [DemInf](https://arxiv.org/abs/2502.08623) · [Demo-SCORE](https://arxiv.org/abs/2503.03707) · [CUPID](https://arxiv.org/abs/2506.19121) · [DataMIL](https://arxiv.org/abs/2505.09603) · [SCIZOR](https://arxiv.org/abs/2505.22626)
- 2026: [QoQ](https://arxiv.org/abs/2603.09056) · [RINSE](https://arxiv.org/abs/2604.23000) · [PSD metric](https://arxiv.org/abs/2605.01544) · [Narayanan et al.](https://arxiv.org/abs/2605.26349) · [FlexiTac](https://arxiv.org/abs/2604.28156)

### Reading list

| Week | Paper | Takeaway | Status |
|---|---|---|---|
| 1 | Belkhale et al., NeurIPS 2023 | Action divergence and transition diversity. Diversity isn't always good | ✅ read |
| 1 | RINSE + PSD metric (read together) | Cheap smoothness proxies, the first metrics you'll implement | |
| 1 | CUPID, CoRL 2025 | The causal reference point. Read the problem statement and influence derivation closely; skim experiments | |
| 2 | Demo-SCORE | The causal method you can actually run in sim | |
| 2 | Consistency Matters | Consistency metrics for H1 | |
| 2 | Cuan, Okamura & Khansari, IEEE ToH 2024 | H1's closest prior work | |
| Optional | DemInf | The bridge between cheap and causal scoring | |

**Skim if time allows:**
- QoQ, DataMIL, SCIZOR.
- Data Scaling Laws.
- robomimic MH (operator-quality labels).
- Narayanan et al. (telemetry feedback).
- FlexiTac.

**Pre-register before the week-3 training runs** (dated git commit):
- The predicted success drop per defect × policy.
- The predicted metric ordering.
- The predicted haptic effect.

### Summary of findings

robomimic is open source (MIT code and data on Hugging Face), and its 300-demo multi-operator sets come
with worse/okay/better labels built in. Plan: in week 2, score those demos with our metrics and check
whether any metric separates better from worse operators more reliably than trajectory length alone.
Optionally, test Q1 on this naturally varying data with robomimic's own policies in simulation. Details
and caveats: [Details/robomimic_dataset.md](Details/robomimic_dataset.md).

## A.3 — Metrics (frozen end of week 2)

| Group | Metrics | Cost |
|---|---|---|
| **Integrity** (filter, don't rank) | Frame drops, camera–action sync, frozen frames, exposure drift, camera-pose drift, joint-limit saturation, leader–follower error | Free |
| **Tier 1**: proprioception | SAL, TED, PSD, LDLJ, jerk / path length, static fraction, duration, gripper chatter | Free |
| **Tier 2**: one GPU pass | Visual & robot-state diversity (DINOv2), action divergence (visual vs. state), initial-state spread, cross-camera consistency, near-duplicates | Minutes |
| **Tier 3**: needs a trained policy | Demo-SCORE (sim only, optional) | Sim rollouts |
| **Tactile** (H1) | Peak grip force, force CV (within and across participants), overshoot, time-to-stable-grasp, slip events, chip break rate | Free |
| **Baselines** | Random-N, all data, blind rater score (~150 shuffled clips, ~2.5 h of their time) | — |

## A.4 — Designs

### Real task: toolkit assembly (task card)

Pick up the white cylinder from the table and place it upright in the top-left hole of the fixed black
fixture. Success means it is fully seated and released within 30 s. Only the cylinder's position varies,
over 10 taped positions. Full card (success stages, fixed factors, episode protocol, log format):
[Details/toolkit_task_card.md](Details/toolkit_task_card.md).

### Phase 0.5 — Second-camera test (week 1, 4 h)

**Question.** Does adding the wrist (gripper) camera to the top camera make the observation tell apart
states that need different actions? Extends 07 §3.

**Data (recorded 2026-09-17).** 50 clean (D0) episodes, one operator, 10 cylinder positions × 5
recorded in position blocks (`ep 0-4 = P1 … 45-49 = P10`, so position is confounded with session time), both
cameras, 25 s on average (628–880 frames).
- Hub: [HALDijkstraaa/so101_toolkit_cylinder_20260917_165544](https://huggingface.co/datasets/HALDijkstraaa/so101_toolkit_cylinder_20260917_165544)
- Local copy used for analysis: `~/.cache/huggingface/lerobot/HALDijkstraaa/so101_toolkit_cylinder_20260917_165544`
- `..._163836` (5 episodes) and `..._164823` (3 episodes) in the same folder are test runs; don't use them.

**Hypotheses.** All episodes are clean demos from one operator. So when similar-looking frames have
different actions, the cause is mostly what the camera can't see, not inconsistent teleoperation.
1. **H-a — Wrist beats top where precision matters.** In the grasp and insertion phases, wrist-only action
   divergence is lower than top-only. *Why:* the next move depends on the gripper's offset from the
   cylinder or hole, a few mm. That fills the wrist image but is a few pixels in the top image, and the arm
   often hides it from above.
2. **H-b — Top beats wrist during reach and transport.** In the first part of the episode, wrist-only
   divergence is higher than top-only. *Why:* before the cylinder is in the wrist view, wrist frames show
   similar table surface for all 10 positions, so they can't tell which way to go. The top view shows
   where the cylinder is.
3. **H-c — The two views complement each other (the pass test).** Top + wrist has lower divergence than
   either view alone, over the whole episode, at every k. The drop is larger than for the control (top +
   wrist embeddings shuffled between frames). *Why:* following H-a and H-b, each view resolves the
   ambiguity the other can't. The shuffled control has the same size and value range but no matching
   information, so it can't help.
4. **H-d — Aliasing exists and sits in the fine phases.** Some frame pairs from different episodes are close
   in top space but far in wrist space, with different actions. They cluster in grasp and insertion.
   *Why:* same reasoning as H-a. Showing these pairs side by side is the direct evidence.
5. **H-e — Joint state alone is ambiguous at the start.** Robot-state divergence is highest early in the
   episode, where top-camera divergence is low. *Why:* every episode starts from the same home pose, but
   the cylinder is in one of 10 places. The joints can't know where to go; the top camera can. Later, the
   pose itself mostly sets the action, so state divergence falls. State is a reference, not a competitor.
6. **H-f — Wrist frames are more varied.** Visual diversity is higher for wrist than top. *Why:* the wrist
   camera moves with the arm, so its whole image changes; the top view is mostly a static scene. Treat
   this as descriptive only: raw cosine distances aren't on the same scale across two cameras.

**Analysis (notebook [phase05_second_camera.ipynb](phase05_second_camera.ipynb)).** Paired: the same
frames are embedded as top, wrist, top + wrist and controls, with DINOv2, L2-normalized, then concatenated.
Divergence is reported at k = 5, 10, 20, by episode phase, with bootstrap intervals over episodes.
Changes from the 07 notebook:
- **Trim idle frames:** static frames at the start and end of each episode are cut before sampling.
- **Neighbors from other episodes only:** neighbors from the same episode are nearly identical frames a few
  hundredths of a second apart, and would make every space look consistent.
- **Action = where the arm goes next** (commanded position 10 frames ahead minus the current position),
  not the absolute command. The absolute command is almost the current pose, so any space that
  encodes the pose would look consistent. Absolute actions are still reported, to compare with 07.
- **Control = shuffled wrist, not a copy of top.** With normalized embeddings, top + a copy of top has
  exactly the same neighbors as top, so it would always "pass". The copy is kept only as a sanity check.

**Pass:** at every k, with episode-bootstrap intervals above zero, top + wrist divergence is (1) lower
than top-only and (2) lower than top + shuffled wrist. A test run showed shuffled dims *raise* divergence,
so subtracting the control's change from the gain would make the test too easy.

**Result (2026-09-19): C1 passes — the wrist camera helps — but the surprises are bigger than the result.**
The top camera, not the wrist camera, is the aliased view (it loses to wrist and even to joint state in
every phase but grasp/insert), and top+wrist beats wrist alone by only 0.023. Per-phase numbers are
confounded by action scale, so H-a/H-b stay provisional. Follow-ups (notebook §9–13) then showed the
phase picture reverses once divergence is normalized *within* each phase (top wins reach/transport, wrist
wins grasp/insert), that a short frame history is the single biggest gain, and that in the fine phases no
observation beats chance — which sets a data-collection guideline: spend episodes and discipline on grasp
and insert. Full numbers, verdicts and reasoning:
[Details/phase05_camera_test_results.md](Details/phase05_camera_test_results.md).

**Recording (lerobot 0.6.1)**, as run. First load the camera presets and check the images
([01](01_setup_robot.md) Step 6). Keys: **→** ends the episode early, **←** re-records it, **Esc** stops.

```bash
TOP=/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A8C83F4F-video-index0
WRIST=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3ADD8F-video-index0

lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --dataset.repo_id=HALDijkstraaa/so101_toolkit_cylinder \
  --dataset.single_task="Pick up the white cylinder and place it in the hole of the black fixture" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=3 \
  --dataset.fps=30 \
  --dataset.push_to_hub=true \
  --dataset.private=true \
  --display_data=true
```

Changes from the 03 command:
- **Ports and cameras:** Ubuntu ports, `/dev/v4l/by-id` paths, names `top`/`wrist`.
- **MJPG:** `fourcc: MJPG` so two cameras fit on USB bandwidth.
- **One run of 50 episodes.** Without `--dataset.no_stamp`, 0.6.1 appends the start time to `repo_id`, so
  the dataset is `so101_toolkit_cylinder_20260917_165544`. To add episodes later, pass that full name with
  `--resume=true`.
- **Timing:** 30 s episodes to match the success window; 3 s reset.

### Phase 0.6 — Camera ablation: train and evaluate (week 2)

**Question.** Phase 0.5 measured whether the second camera *should* help. This phase measures whether it
*does*: train ACT three times — `top`, `wrist`, `top+wrist` — on the same 50 episodes and compare success
on the robot. It is also the first test of whether action divergence predicts policy performance.

Masking a camera at inference on a single jointly-trained policy does **not** work: in lerobot's ACT each
camera adds ~300 tokens to one encoder sequence, so dropping one is an out-of-distribution input rather
than an ablation. Measured on the 5080 laptop at batch 8: 72 ms/step single-camera, 134 ms/step with both,
so all three 60k-step runs take **~6.3 h sequentially** in fp32 and no cloud GPU is needed. Training is
GPU-bound, so larger batches and parallel runs both buy nothing (three concurrent runs finish 8% *later*).

Pass 1 is one seed and 30 rollouts, aimed at the **emergent behavior** of a top-only vs wrist-only
policy rather than at success rates; the statistical comparison is a later pass.

**Done 2026-09-21** ([wandb](https://wandb.ai/bj-huang-university-of-toronto/phase06-camera-ablation?nw=nwuserbjhuang)).
Held-out loss separated the three by less than one curve's checkpoint wobble, and success rates (3/10,
2/10, 2/10) were noise — but *where* they failed was decisive. `top` reached the cylinder 10/10 and never
failed at reach; `wrist` reached 3/10 and failed at reach 7 times (Fisher p = 0.003), and the three
positions it did reach were exactly the three nearest its fixed default trajectory (p = 0.008). H-a and
H-b confirmed as a double dissociation. H-c was **not** supported: `top+wrist` reached only 6/10,
under-shooting the workspace extremes by 17%, so the redundant stream degraded the phase the other camera
was carrying. Divergence predicts what information is *available*, not whether a network can fuse it
without harm. Note the run book also
fixes the train/eval split: episodes were recorded in position blocks, so lerobot's default "last 5"
holdout would have removed position P10 from training entirely.

Full run book — held-fixed table, step-by-step training scripts, eval protocol, pre-registered
predictions: [Details/phase06_camera_ablation_training.md](Details/phase06_camera_ablation_training.md).
Scripts: [`scripts/train_camera_ablation.sh`](scripts/train_camera_ablation.sh),
[`scripts/make_eval_schedule.py`](scripts/make_eval_schedule.py).

### Dials: sim and real data with known defects

| Dial | Instruction | Should be flagged by | Sim / real episodes |
|---|---|---|---|
| **D0 Clean** | Best effort, one strategy, one-shot grasp | — | 150 / 100 |
| **D1 Jerky** | Small corrective oscillations; still succeed | SAL, LDLJ, PSD | 50 / 50\* |
| **D2 Hesitant** | 2–3 pauses of 1–3 s | Static fraction, duration | 50 / 50\* |
| **D3 Fumbled** | Miss, regrasp, succeed | Gripper chatter | 50 / — \* |
| **D4 Inconsistent** | Switch between 2–3 approach strategies | Action divergence | 50 / — \* |
| **N1 Slow-clean** (control) | Like D0 at half speed | Should *not* be flagged | 50 / 50 |

\*Real track uses the **2 dials that hurt most in sim** (D1/D2 shown as placeholders).
**Sim ≈ 400 episodes on `so101_pick_orange`. Real ≈ 250 on toolkit assembly.**

**Rules:**
- **Fixed N = 100** episodes per training condition.
- **Dial blocks mixed** within every session.
- **Dials verified** with an independent measure (e.g. velocity sign reversals for D1), never with the
  metric under test.
- **Labels hidden** from metric code.
- **Circularity guard.** Injected defects answer Q0. Q1 is judged on the mixed pool. N1 catches metrics
  that punish harmless slowness.

### Training + eval grid

| Track | Datasets | ACT | SmolVLA | π0.5 | Eval |
|---|---|---|---|---|---|
| **Sim** | Q0: clean + 5 dials × {25, 50}% = 11 · Q1: top/bottom × 3 metrics + 2 random = 8 | 19 × 3 seeds = 57 | 8 (clean, 5 dials @50%, top-N, random-N) | 4 (clean, worst dial, top-N, random-N) | 100 rollouts / checkpoint, fixed seeds |
| **Real** | Clean, worst dial @50%, top-N, random-N | 4 | 4 | 2 (clean, worst dial) | 40 interleaved rollouts / condition, video-scored blind by rater |
| **Haptic** (chip task) | Feedback ON vs. OFF, pooled across participants, same N | 2 | — | — | 30 rollouts / condition; grip force + break rate logged |

- **Compute:** ~180–270 GPU-h.
  - ACT on 4090s.
  - SmolVLA and π0.5 on A100s.
  - Sim eval on 4090s (Isaac Sim can't run on A100/H100).
- **Resolution:** differences you can reliably detect.
  - Sim ACT: ~11–15 pts.
  - Sim SmolVLA/π0.5: ~20 pts.
  - Real: ~30 pts, so hardware confirms *direction*, not size.
- **Held constant:** hyperparameters (frozen after one clean run), seeds and initial states, and
  normalization stats. Normalization stays fixed by selecting subsets from one pool with
  `--dataset.episodes`.
- **π0.5 memory:** real deployment fits on the 5080. If Isaac Sim + π0.5 won't share 16 GB, serve π0.5
  from a cloud 4090/A100; the sim waits for actions, so results don't change.

### H1 — Haptic teleop study (chip-grasp task; hardware wk 4–6, study wk 8)

- **Task:** a separate chip-grasp task, used only for tactile testing.
  - Pick up a chip and place it without breaking or dropping it.
  - Use uniform chips (e.g. stacked-can style) so object variation stays small.
- **Hardware:**
  - FlexiTac V2 pads (32×12 taxels, ~$2.5/pad, 100 Hz serial readout) on the follower gripper fingers.
  - Vibration motors on the operator's wrist/hand, driven by an MCU.
  - Intensity ∝ total grip pressure, plus a pulse at the target force.
  - Measure sensor-to-vibration latency.
- **Tactile recorded in both conditions**; only the feedback toggles.
  - Log it as a timestamped side stream aligned to episodes.
  - Policies stay vision + proprio, so H1 tests *data quality*, not a new input.
- **Participants:** you + ~5 others, ~1–1.5 h each.
  - Each does 5 practice + ~20 recorded episodes per condition.
  - Block order counterbalanced across participants: half OFF-ON-ON-OFF, half ON-OFF-OFF-ON.
- **Measures:**
  - **Primary:** grip-force CV, within each participant (episode to episode) and across participants
    (spread of per-person mean peak force).
  - **Secondary:** chip break rate, drops, overshoot, time-to-stable-grasp, success, duration, action
    divergence.
- **Analysis:** mixed-effects model (feedback fixed, participant random). Report it with and without your
  own sessions, since you're far more practiced.
- **Training:** pool ON episodes vs. OFF episodes across participants (same N, ~100 each) → ACT → 30 real
  rollouts each. Log the policy's own grip force and break rate.
- **Gate:** hardware bench-tested and participants booked by end of week 6; otherwise week 8 shrinks to
  hardware + a pilot.

## A.5 — Timeline (8 weeks, ~240 h)

`(+)` = something your draft didn't include that the plan depends on.

| Week | Tasks (hours) | Output / gate |
|---|---|---|
| **1** · ~34 h | Infrastructure: Ubuntu, LeRobot, envs, lighting control (8) · define toolkit-assembly task (2) · Phase 0.5 camera test (4) · read Belkhale, RINSE, PSD, CUPID (16) + summary (4) | Teleop works on Ubuntu · task card · C1 result · paper notes |
| **2** · ~30 h | Read Demo-SCORE, Consistency Matters, Cuan et al.; finalize metrics (12) · (+) implement integrity + Tier 1 metrics, port Tier 2 from notebook (6) · Isaac Sim + LeIsaac, teleop `so101_pick_orange` (4) · eval + logging harness; **Phase 0.6 camera ablation**: 3 ACT runs on the 5080 (~11 h unattended) + 90 interleaved rollouts (8) | **Gate:** harness runs unattended; clean ACT at 50–80% success (else adjust N or task randomization) · C1b result |
| **3** · ~30 h | Sim collection, ~400 episodes (20) · score, bin into manifests, pre-register (4) · cloud training recipes (6) · ACT grid, 57 runs in background (1–2 days on parallel 4090s) | Manifests + pre-registration committed |
| **4** · ~25 h | ACT sim eval + findings (8) · launch 8 SmolVLA + 4 π0.5 runs (4) · haptic hardware: order FPC/PCB + motors, print mounts (8) · (+) chip-task card + recruit participants (3) · buffer (2) | ACT Q0/Q1 results |
| **5** · ~30 h | SmolVLA + π0.5 sim eval (4) · review sim results; pick the 2 worst dials + best metric for real (8) · (+) lock real rig + collection protocol (8) · start real collection (10) | Sim report (Q0–Q2) · real conditions frozen |
| **6** · ~30 h | Real collection → ~250 episodes (20) · score + bin (3) · launch real grid: ACT 4, SmolVLA 4, π0.5 2 (2) · (+) haptic bench test (5) | Real checkpoints · **haptic hardware gate** |
| **7** · ~30 h | Real eval: 10 conditions × 40 rollouts, interleaved, blind video scoring (24) · analysis (6) | Real results: does the direction match sim? |
| **8** · ~27–30 h | Haptic sessions: ~5 participants × ~1.5 h (8) · tactile analysis (4) · ACT ON vs. OFF on cloud · real eval 2 × 30 (5) · Part A write-up (10) | H1 result · final report |

**Risks (the plan is ~240 h; my earlier estimate for similar scope was 300–470 h):**
1. **Week 2 sim setup is the most optimistic slot.** Isaac Sim on Blackwell plus the LeIsaac ↔ LeRobot
   policy bridge is unverified. If the harness isn't running by end of week 2:
   - Collect anyway in week 3.
   - Run the sim grid with ACT only.
2. **Metric code wasn't budgeted.** Added 6 h in week 2; Claude can draft the Tier 1 functions.
3. **Sim eval throughput is unknown until week 2.** 69 checkpoints × 100 rollouts may need several
   4090s in parallel.
4. **Haptic hardware is the tightest dependency.** It must work, and participants must be booked, by end
   of week 6.

**Cut order:**
1. π0.5 in sim.
2. SmolVLA on real.
3. 25% dose levels.
4. Q1 down to 2 metrics.

**Never cut:** fixed N, matched eval, the N1 control, pre-registration.

## A.6 — Deliverables

1. **`robot-data-quality` repo:** scoring API, grid launcher, LeIsaac eval harness.
2. **Dial-labeled SO-101 datasets** (sim + real) on HF.
3. **Figures:**
   - Camera-view effect (C1).
   - Defect × policy dose-response (Q0).
   - Ranking / transfer matrix (Q1–Q2).
   - Haptic consistency + policy effect (H1).
4. **Short report:** pre-registration next to outcomes, negatives included.

## A.7 — Future todo

- **Operator state (old Q3):** detect fatigue / low effort from the order of episodes within a session.
  Uses baseline drift, time-on-task slopes, change points, reset time and self-reports. Keep recording
  timestamps now so this stays possible.
- **Other data sources:** UMI handheld gripper, egocentric human video.
- **Public-dataset scale-up:** descriptive scoring of community SO-100/101 datasets and a DROID subset.
- **Tier 3 at scale:** CUPID, QoQ, DataMIL.

## A.8 — Decisions

**Settled:**
- **Operator:** you (data collection).
- **Real task:** toolkit assembly: white cylinder into the fixture's top-left hole ([task card](Details/toolkit_task_card.md)).
- **π0.5:** sim + real on the 5080.
- **Cloud:** 4090 + A100.
- **Rater:** available.
- **H1 participants:** you + others (~5 suggested).
- **H1 task:** a separate chip-grasp task, used only for tactile testing.

**Open:**
1. **Participant count**, and whether your own sessions count in the pooled training data.
2. **Human-subjects review.** If H1 results will be published, check whether your institution requires
   review (IRB) before week 8. Reviews can take weeks, so ask during week 1–2.

---

# Part D — Resources for Part A

The full bill of materials is in [09](09_home_deployment_and_hardware.md) Part D: GPU memory math, π0.5
memory sources, GPU pricing, laptop vs. desktop, and SO-101 repeatability.

## D.1 — Compute: what runs where

| Model | Params | Inference memory | Training memory | Train on | Run / eval on |
|---|---|---|---|---|---|
| ACT | 80 M | 0.16 GB weights | ~5 GB @ bs 8 | Cloud 4090 (grid) · 5080 (pilots) | 5080 |
| SmolVLA | 450 M | ~3–4 GB served | 10–16 GB @ bs 8 | Cloud A100 | 5080 |
| π0.5 | 3.3 B | 6.6 GB weights · ~8–12 GB served (PyTorch bf16) | >22.5 GB LoRA · >70 GB full | Cloud A100 (LoRA) | 5080 (real) · 5080 or cloud policy server (sim) |

- **Isaac Sim needs RTX GPUs.** Run sim eval on cloud 4090s; A100/H100 are unsupported.
- **5080 laptop setup:** Ubuntu 22.04 dual-boot, NVIDIA driver ≥ 570.144 (open kernel modules),
  CUDA 12.8+.
- **π0.5 on the 5080:** serve it with LeRobot PyTorch bf16 (+ RTC).
  - If you use openpi's JAX stack, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - Log `torch.cuda.max_memory_allocated()` on first run; no source publishes a measured figure.
- **Q2 caveat:** if π0.5 gets LoRA while ACT/SmolVLA get full fine-tunes, adaptation method is confounded
  with model class. State it in the write-up.

## D.2 — Hardware and software checklist

| Item | For | Status | Est. cost |
|---|---|---|---|
| SO-101 leader + follower · top C920 + gripper C922 | Everything | ✅ have | — |
| RTX 5080 laptop · cloud 4090 + A100 | Everything | ✅ have | — |
| Rigid camera/arm mounts + fixed LED panel with diffuser | Week 1 rig + lighting control | 🔴 week 1 | $80–200 |
| Spare servo set | A dead servo mid-plan stalls a week | 🔴 | $60–150 |
| Toolkit-assembly parts · uniform chips | Real task · H1 task | 🔴 | small |
| **FlexiTac V2 pads** (32×12 taxels) + readout board ([site](https://flexitac.github.io/), [hardware repo](https://github.com/FlexiTac/FlexiTac_Hardware_Repo)) | H1 grip force | 🔴 order week 4 | ~$2.5/pad + est. $30–80 board |
| Vibration motors (LRA/ERM) ×2–4 + haptic driver (e.g. DRV2605L) + MCU (ESP32/Arduino) | H1 feedback | 🔴 week 4 | est. $20–50 |
| Printed finger mounts, wrist strap · 3D printer | H1, fixtures | Noisebridge ✅ | filament |
| Backup (external SSD + cloud) | Datasets | 🔴 | $80–150 |
| LeRobot · Isaac Sim 5.x + Isaac Lab + [LeIsaac](https://github.com/LightwheelAI/leisaac) · HF Hub · W&B or TensorBoard · DuckDB + Parquet | Everything | 🔴 week 1–2 | free |

---

## Immediately next (week 1)

1. **Infrastructure** (8 h): Ubuntu dual-boot, LeRobot, envs, lighting control. Step-by-step:
   [Details/ubuntu_env_setup.md](Details/ubuntu_env_setup.md).
2. **Toolkit-assembly task card** (2 h). Done: [Details/toolkit_task_card.md](Details/toolkit_task_card.md).
3. **Phase 0.5 camera test** (4 h, §A.4). Teleop with both cameras: [01](01_setup_robot.md) Step 6 (follower `/dev/ttyACM0`, leader `/dev/ttyACM1`).
4. **Read RINSE + PSD, then CUPID** (§A.2), plus a summary.
5. **Phase 0.6 camera ablation** (week 2, §A.4). Train three ACT policies overnight, then evaluate:
   [Details/phase06_camera_ablation_training.md](Details/phase06_camera_ablation_training.md).
