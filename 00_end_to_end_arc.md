# 00 — The End-to-End Arc (read this first)

This folder is no longer just "deploy SmolVLA." It's a **full design → build → debug → train → eval loop**
on a real robot, structured so the hands-on *doubles as interview material* for all three focus rounds:

- **Chelsea (design):** every stage below forces a design decision + an eval choice — that's your
  system-design and eval-methodology story.
- **Michael (debugging):** every stage has a characteristic failure mode — that's your diagnostic story.
- **Adrian (coding) / Karol (vision):** you've read the production `VLAFlowMatching` and can map it to the
  code you wrote from scratch.

> Status: **`01_setup_robot.md` is DONE** — teleop loop works. So you start at the *data* stage. The
> pipeline below is a loop, not a line: eval feeds the next data collection (the flywheel).

```
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              ▼
   ①  SPEC ──►  ②  DATA ──►  ③  TRAIN ──►  ④  DEPLOY ──►  ⑤  EVAL ──►  (targeted data)
  (task, gen.   (03_collect   (04_finetune  (04 deploy    (05_eval_and_design)
   axes,        _data)        _and_deploy)   section)      → find failures, collect there
   success)                                                 → retrain
```

---

## ① SPEC — pin the problem before you touch the robot

The rule: decide and **write down** the task, the generalization axes, the success criterion, and the
constraints *before* collecting data (this is literally your answer to Chelsea's "how would you design
this"). Below is the **locked v1 spec** for this project.

### v1 SPEC — LOCKED (2026-07-21)

![v1 task setup: SO-101 gripper (left); the blue PCB with USB connectors sitting on a white plate (right);
the black 3D-printed fixture with a white-marked pocket = the drop target (top).](results/pick_and_place_pcb.jpg)

**Task:** pick a blue PCB (fixed orientation) off a white plate and place it into the white concave pocket
of a **fixed** black 3D-printed fixture; poke to seat it if the first drop doesn't land.

**The three sub-skills (where the difficulty lives):**
1. **Precise grip of a thin part** — a small tilt slips/tilts the PCB; the grip xy must be accurate.
2. **Stable hold in transit** — consistent gripper close ("clutch") so the board doesn't shift mid-air.
3. **Low-tolerance insertion + corrective poke** — seat it in a tight pocket; nudge if needed.

**Generalization axes (v1 varies exactly ONE):**

| Factor | v1 decision |
|---|---|
| PCB **xy position** | **VARY** — ±1 cm on a clearly marked 2D plane (the single generalization axis) |
| PCB **rotation** | FIXED (keeps the grip pose consistent) |
| Fixture (drop target) pos/rot | FIXED |
| PCB instance / shape / color | FIXED — one blue PCB |
| Lighting / scene / camera | FIXED |

**Success criterion (pre-registered):**
- **Binary:** within **30 s**, the PCB is seated flat in the pocket and the gripper has released.
- **Staged (for more signal per rollout):** reached (0.2) / grasped+lifted (0.4) / over the pocket (0.6) /
  dropped in (0.8) / seated flat + released, including a successful poke-adjust (1.0).

**Data plan:** 50 demos to start. Quality bar: smooth & slow (trackable at 30 Hz); **one-shot grip**
(restart the episode if the grip fails — keep grip demos clean); drop is lenient and **poke-adjust is
demonstrated consistently** (so the policy learns a clean corrective motion). Cover the ±1 cm range
roughly uniformly and bias toward its edges/corners. Be ready to add 30–50 demos targeted at whatever
positions fail at eval (the flywheel, ⑤).

**Constraints:** 30 Hz control loop; SmolVLA inference fits the 6 GB GPU (~near real-time); first rollouts
at reduced speed with a hand on the e-stop.

### Mechanical items — status

1. **Thin-PCB acquisition — RESOLVED.** The PCB has a **USB connector on each side** that stands proud of
   the board, so the parallel gripper pinches a connector rather than trying to get under a 1.6 mm edge.
   Keep it identical every demo: grip the **same** connector, same approach → this *is* your fixed grip
   pose. (Watch: pinch the connector body squarely; an off-center grab is what tilts/slips the board.)
2. **Chamfer the pocket — recommended.** The SO-101's repeatability is a few mm; a **lead-in bevel/funnel**
   around the pocket lets a near-miss slide in ("drop-in with chamfer," not press-fit) and raises success
   more than any training change. If the printed fixture's pocket is a straight wall, consider a v2 reprint
   with a chamfer; the poke-adjust behavior covers the gap meanwhile.
3. **Contrast — good, no action.** Blue/silver PCB on a white plate (pick) and a **white-marked pocket on
   the black fixture** (place) both read cleanly. Target contrast doesn't even matter much in v1 (fixed
   fixture → the policy memorizes the drop location), but it sets you up well for v2.

**Design talking point:** "I scoped generalization to a single bounded axis — PCB xy within ±1 cm — fixed
rotation so the grip pose stays consistent, and fixed the fixture. I moved the hard tolerance problem into
the **fixture design** (a chamfered pocket) rather than demanding sub-mm precision from a $150 arm."

## ② DATA — the real bottleneck → `03_collect_data.md`

Build the dataset as a deliberate experiment, not a pile of demos:
- **Coverage over the axes you chose** (vary cube position across episodes); hold the rest fixed.
- **Quality control** (your teleop-quality thesis): smooth, deliberate motions; trim idle frames (pauses
  teach the policy to freeze); consistent language instruction; re-record fumbles.
- **20–50 clean episodes** to start. Then `lerobot-dataset-viz` + `lerobot-replay` to sanity-check.
- Note the **normalization stats** get computed from this data — they must match at inference (a top
  deploy-time bug).

**Debugging failure mode (Michael):** garbage-in — inconsistent instruction string, wandering camera,
jerky demos. **Design talking point (Chelsea):** "data *coverage* is the independent variable; I biased
collection toward the hard positions the policy would later fail on."

## ③ TRAIN — pipeline first, convergence second → `04_finetune_and_deploy.md`

Two-phase, matching your compute answer:
1. **Local smoke test (RTX 3060 6 GB):** freeze the VLM backbone, `batch_size=1`, bf16, ~50 steps. Goal
   is **"does the pipeline run end-to-end without OOM/shape errors,"** *not* convergence. This is where
   you'll hit and fix the real integration bugs cheaply.
2. **Real fine-tune on ColabPro:** once the pipeline is proven, run the actual training (more steps,
   larger effective batch via grad-accum), then copy the checkpoint back to deploy locally.

**Monitor:** FM loss trends down but **won't hit zero** (irreducible velocity-variance floor — you know
this). Loss is a checkpoint-selection heuristic; the real metric is closed-loop eval (④/⑤).

**Debugging failure modes (Michael):** OOM (→ freeze more / batch 1 / bf16), loss NaN (→ LR/clip), loss
flat (→ overfit one batch, check trainable params — see `bug_hunt_1_training.py`).

## ④ DEPLOY — run YOUR policy on the arm → `04_finetune_and_deploy.md` (deploy section)

`lerobot-record` with `--policy.path` and **no `--teleop`** — the policy drives the follower.
- **Same camera name + instruction + normalization** as training, or the input keys mismatch → garbage.
- Hand on the e-stop, reduced speed for the first rollouts.

**Debugging failure modes (Michael's wheelhouse):** trained-fine-but-bad-on-robot — normalization
mismatch, input-pipeline drift, chunk-boundary jerk (→ real-time chunking), latency stalling the control
loop, covariate-shift drift. This *is* Scenario 2 in `../onsite_prep/C_debugging_michael.md`.

## ⑤ EVAL + FLYWHEEL — measure honestly, then feed the next round → `05_eval_and_design.md`

Run the eval protocol (matched initial conditions, staged/graded metrics, CI, seen-vs-unseen, blind
scoring), review failure videos, and **collect targeted data where it fails** → retrain. The loop is the
product.

**This is your Chelsea centerpiece.** See `05_eval_and_design.md` for the full protocol and the design
decisions to be able to defend.

---

## The one story this whole folder buys you
> "I scoped a task and its generalization axis, collected coverage-controlled teleop data, smoke-tested
> the fine-tune pipeline locally then trained on cloud, deployed on the real SO-101, and evaluated it with
> matched initial conditions and staged metrics — then used the failures to decide what data to collect
> next. I hit and fixed real bugs at every layer: OOM, normalization mismatch, chunk-boundary jerk."

That single paragraph answers a design question, a debugging question, and a "why PI" question at once.
