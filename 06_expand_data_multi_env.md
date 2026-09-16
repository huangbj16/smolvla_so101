# 06 — Expand the dataset for env + position generalization (and add negatives)

Your v1 policy (50 home demos) worked **in the collection scene** but failed two ways when moved:
1. **New space → task fails.** Classic **scene/visual covariate shift** — the vision features were
   entangled with the one home background/lighting.
2. **Empty scene → it *still* runs the lift-clench motion.** The smoking gun: the policy learned an
   **unconditional trajectory replay**, not a visually-gated skill. In *every* v1 demo the object was
   present, in ~the same place, same background, and the arm always acted → object-presence carried
   zero information, so gradient descent had no reason to condition on it. It memorized the motion.

This doc fixes both by (a) adding **scene/lighting diversity**, (b) forcing **visual conditioning**
via wide position variation, and (c) adding **negatives** so "no object → don't act" becomes a concept.

> Prereqs: [01_setup_robot.md](01_setup_robot.md), [03_collect_data.md](03_collect_data.md) done.
> `conda activate lerobot`. Reuse your ports/camera index/ids. Eval protocol lives in
> [05_eval_and_design.md](05_eval_and_design.md).
> Ubuntu: blocks marked `powershell` are from the Windows runs. Ports are updated; on Ubuntu swap the `` ` `` / `^` line endings for `\` and use the camera paths from 01. `--dataset.root` paths below are Windows paths too.

---

## Scope decision (deliberate): only claim **position + environment**

We **keep PCB rotation fixed and fixture position fixed**, same PCB instance. Rationale: SmolVLA won't
reliably learn rotation-invariance from ~50 samples/condition, and spending the data budget on axes we
won't *evaluate* is waste (the doc-05 talking point). So the **generalization claim is exactly two axes:
PCB xy position and environment (scene+lighting)** — and we will measure precisely those.

Everything held constant across the *whole* dataset (rotation, fixture, instance, camera name/fps,
instruction string) is intentional and bounded — just be honest that the policy is **not** expected to
generalize over them.

---

## The collection plan

| Component | Amount | Detail |
|---|---|---|
| **Positive spots** | **8 total** = home (already have) + **7 new** | Different rooms/surfaces + **different lighting** each |
| Demos per spot | **30** | Split **6 PCB positions × 5 demos** |
| PCB position range | **WIDE** | Spread across the full reachable workspace — *not* ±1 cm. This is what forces the policy to *look* to find the PCB. |
| PCB rotation / fixture | **fixed** | Bounded scope (see above) |
| **Negatives** | **50 total** = **10 per scene** | Same instruction string, **no PCB**; mix empty-table **and distractor** episodes; idle action |
| **Held-out test env** | **1 (a 9th spot)** | **Never recorded.** Closed-loop eval only (see below) |

Net new data to collect: **7 × 30 = 210 positives + 50 negatives = 260 episodes**, appended to the
existing 50. Final training set ≈ **310 episodes across 8 environments**.

**Why 8×30 and not 5×50:** demos-per-condition *saturates* fast; **diversity is the lever** (data-scaling-laws
result). The same budget spread over more scenes generalizes better than more depth in fewer scenes.
If a spot is cheap to set up, prefer *another spot* over more demos in an existing one.

### Position coverage (the anti-memorization rule)
For each spot, use **6 marked PCB positions** spread widely (corners + center + mid-edges of the
reachable plane), **5 demos each**. Bias toward the extremes of reach. If the positions are tightly
clustered, the pick stays memorizable and you haven't fixed the root cause.

### Negatives (the anti-"empty-scene-acting" rule)
- **Same `--dataset.single_task` string** as positives (`"Pick up the blue PCB..."`). This is critical:
  it teaches *same instruction + no object visible → idle*. If you relabel them "do nothing," they
  never fire at deploy (you always send the pick instruction) — useless.
- **No PCB present.** Two sub-types, ~half each:
  - **Empty**: just table + background (+ the fixture is fine).
  - **Distractor**: a *non-PCB* object in view (a pen, a cap) that must **not** be grabbed.
- **Action = idle**: start at the normal home pose, hold still / tiny natural jitter, end. Keep them
  **short** (`--dataset.episode_time_s≈8–10`). Don't flood with long static clips — a huge "do nothing"
  mass can teach the policy to freeze even when the PCB *is* present. 50/310 ≈ 16% is a sane ratio.

---

## Storage: **append** to the existing dataset (pause/continue friendly)

Add to the **same** `repo_id` with `--resume=true`. Your home 50 becomes "spot #1"; you only collect
the 7 new spots + negatives. One `repo_id` = no merge step at train time and natural pause/continue
across sessions.

**Your actual dataset (verified locally):**
- `repo_id` = `HALDijkstraaa/so101_pick_place_pcb_20260721_183543` (it's **timestamped** — the plain
  `so101_pick_place_pcb` does not exist locally; the `_175109`/`_181041` folders were early gate runs).
- local `root` = `C:/Users/bingjian/.cache/huggingface/lerobot/HALDijkstraaa/so101_pick_place_pcb_20260721_183543`
  (default is `~/.cache/huggingface/lerobot/<repo_id>`; override via `HF_LEROBOT_HOME`).

**`--resume=true` requires an explicit `--dataset.root`** in this lerobot version — without it, `resume()`
tries to write into the read-only Hub snapshot cache and aborts with
`resume() requires an explicit 'root' directory`. Point `root` at the local dataset dir above.

**Before you start — protect v1:** note the current HF commit hash (or duplicate the dir to
`..._v1_frozen`) so you can always recover the pristine 50-only dataset for reproducibility.

**Invariants that must be identical every session** (a drift here fragments the dataset):
- `--dataset.repo_id` **and** `--dataset.root` — the timestamped values above, every time
- camera name `front`, index, `width/height/fps`
- `--dataset.single_task` string — **verbatim**, positives *and* negatives
- `--dataset.fps`

### Record a positive batch (one spot = one session, resume on)
```powershell
lerobot-record `
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower `
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader `
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
  --dataset.repo_id=HALDijkstraaa/so101_pick_place_pcb_20260721_183543 `
  --dataset.root="C:/Users/bingjian/.cache/huggingface/lerobot/HALDijkstraaa/so101_pick_place_pcb_20260721_183543" `
  --dataset.single_task="Pick up the blue PCB and place it in the white fixture pocket" `
  --dataset.num_episodes=30 `
  --dataset.episode_time_s=35 `
  --dataset.reset_time_s=3 `
  --dataset.fps=30 `
  --resume=true `
  --dataset.push_to_hub=true `
  --dataset.private=true `
  --display_data=true
```

### Record a negatives batch (shorter episodes, no PCB)
```powershell
lerobot-record `
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower `
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader `
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
  --dataset.repo_id=HALDijkstraaa/so101_pick_place_pcb_20260721_183543 `
  --dataset.root="C:/Users/bingjian/.cache/huggingface/lerobot/HALDijkstraaa/so101_pick_place_pcb_20260721_183543" `
  --dataset.single_task="Pick up the blue PCB and place it in the white fixture pocket" `
  --dataset.num_episodes=10 `
  --dataset.episode_time_s=10 `
  --dataset.reset_time_s=3 `
  --dataset.fps=30 `
  --resume=true `
  --dataset.push_to_hub=true `
  --dataset.private=true `
  --display_data=true
```

> **Two gotchas, both hit and fixed:**
> 1. `--resume=true` is a **top-level** flag (`lerobot-record --help`: `[--resume str]`), **not**
>    `--dataset.resume` — the latter errors with `unrecognized arguments`.
> 2. `--resume=true` **requires `--dataset.root`** (a local dir). Without it: `ValueError: resume()
>    requires an explicit 'root' directory` (it won't write into the read-only Hub snapshot cache).
>
> It appends `num_episodes` **more** episodes to the dataset at `root`. **Dry-run first:** set
> `num_episodes=1` and `push_to_hub=false`, then confirm the count went 50 → 51 via
> `lerobot-dataset-viz --repo-id=<id> --root=<path>` before committing a full 30-episode batch. Then
> flip `push_to_hub=true`.

### Session rhythm (pause / continue)
1. Set up spot *k* (move robot + fixture, arrange lighting, re-mount camera to the **same** framing).
2. **5-episode gate** (below) → if clean, record the remaining 25.
3. Stop. Next session: new spot, `--resume=true` again. Repeat until 7 spots + negatives done.

---

## Per-spot quality gate (don't multiply a mistake by 30)

Reuse the [03_collect_data.md](03_collect_data.md) gate at **every new spot** (setup changes each time):
- [ ] PCB + plate + fixture + gripper all in frame; camera framing consistent *within* the spot.
- [ ] No blur / no blown-out glare (new lighting = new glare risk — this is the whole point, but it
      must still be *legible*).
- [ ] Action/state traces smooth; **no long flat idle** in positives.
- [ ] Gripper channel = clean close→hold→open.
- [ ] `lerobot-replay --dataset.episode=<newest>` reproduces the task (isolates robot/calib from data).

For **negatives**: viz should show the arm holding home pose, **no** grab motion, PCB absent.

**Camera note:** you *are* deliberately varying lighting/scene, but keep the **camera-to-workspace
geometry reproducible per spot** (same mount height/angle). Uncontrolled camera-pose drift is a
separate hidden axis that will muddy your env-generalization result.

---

## Eval: how it actually works (answers your two questions)

### Closed-loop, on-robot — **the** metric (no data collection)
Bring the robot to the **held-out 9th env** it never saw, run the policy live, and **score task
success** with the staged rubric. You do **not** collect teleop data to do this — the value of a
held-out env is that it's unseen; recording there would only give a weak offline number and tempt you
to train on it.

Use the doc-05 protocol verbatim:
- **Matched, interleaved** runs when comparing checkpoints (A,B,A,B) on the **same marked PCB positions**.
- **Staged success credit** (reached / grasped+lifted / over-pocket / dropped / seated) — more signal
  per rollout than binary.
- **Wilson CI** on N≈15–20 — only large gaps are real at this N.

**Three closed-loop tests that directly check the v1 failures:**
| Test | What it proves |
|---|---|
| Held-out env, PCB present, varied positions | env + position generalization (the claim) |
| Held-out env, **empty scene** | negatives worked → policy now **idles** instead of lift-clench |
| Held-out env, **distractor only** | policy conditions on *object identity*, not just "something's there" |

### Offline / open-loop action error — a **weak proxy**, checkpoint-picking only
Yes, this is the "compare predicted action vs. real action given the image sequence" idea (action MSE /
your FM val loss on held-out episodes). Useful to **rank checkpoints**, but it is **not** the verdict:
- Low action-MSE ≠ task success.
- It feeds **ground-truth** observations each step, so it **never sees compounding error** — the exact
  thing that breaks BC policies in the real world.

So: use offline error to pick which checkpoints to roll out; let the **robot decide**. (If val loss and
closed-loop disagree, trust the robot — doc 05.)

---

## Done when
- [ ] v1 dataset commit backed up / duplicated before appending.
- [ ] 7 new spots × 30 (6 positions × 5) appended; camera/instruction/fps identical throughout.
- [ ] 50 negatives (empty + distractor, same instruction, idle) appended.
- [ ] `lerobot-dataset-viz` shows all 8 envs + negatives; `lerobot-replay` on a new episode works.
- [ ] Held-out 9th env chosen (kept **out** of training) for closed-loop eval.

Then retrain ([04_finetune_and_deploy.md](04_finetune_and_deploy.md)) and run the closed-loop tests
above. If the empty-scene test still triggers a grab, the negatives are too few/too long — rebalance
before adding more positives.

---

## Results log — v3 (160 eps: 140 positive + 20 negative, 3 new envs)

**What I actually ran** (not the full 8-env plan yet — 3 new envs so far):
- Dataset: **160 episodes** = **140 positive** (home + 3 new envs, ~30 each) + **20 negative** (10/env × 2 envs).
- Train: **SmolVLA fine-tune, A100 (Colab), 20k steps, ~2 h.** Training **stable**, nothing anomalous.
- Tracking: [wandb v3](https://wandb.ai/bj-huang-university-of-toronto/smolvla_so101_pcb_v3) ·
  checkpoints: [HF smolvla_so101_pcb_finetuned_v3](https://huggingface.co/HALDijkstraaa/smolvla_so101_pcb_finetuned_v3).

### First-scene test — Noisebridge Hackertorium (an env that IS in the training set)

**✅ What works (the negatives paid off — this is the headline win):**
- **Empty scene → the arm stays put.** The v1 "lift–clench on an empty table" bug is **gone**. Only minor
  jitter near the start pose. **The negatives did exactly their job:** "same instruction + no object → idle."
- **Correction behavior emerged from <10 correction episodes.** When it dropped the PCB slightly off, it
  re-approached and nudged it in. Learning a corrective policy from so few demos is a genuinely good sign.
- **Returns to home** after placing, most of the time. ✓

**⚠️ What doesn't / open concerns:**
- **Reaches the *same* spot regardless of PCB position.** This is the big regression: **v2 (home-only)
  generalized over position better than this v3.** That's backwards from what more data should do.
- **Startup wait when PCB present:** the arm hesitates before its first move. Likely because in the teleop
  demos the *initial tip motion happened before the tip was visible* → no visual anchor for "start reaching."
- **Over-correction:** sometimes it keeps nudging even after the PCB is already seated — it can't *see* the
  seated result at this camera angle, so it's guessing.
- **Fixture placement is memorized, not visual** (fixture is fixed → the carry-to-pocket leg doesn't need
  vision; expected, and by design).

### Prime suspect: a **camera-geometry regression**, not a generalization limit
At this env the **camera was mounted too far forward** — it **can't see the PCB (or the tip) during the
grasp.** That single defect plausibly explains the three worst symptoms at once: no visual anchor for the
reach (→ same-spot reaching + startup wait) and no view of the result (→ over-correction). It violates the
[03](03_collect_data.md)/[06 quality-gate](#per-spot-quality-gate-dont-multiply-a-mistake-by-30) rule that
**the PCB, tip, and pocket must be in frame** — and worse, it violates it *inconsistently across envs*, so
camera pose is now a **hidden confound** contaminating the position-generalization result.

> **Honest read:** I can't attribute the position-generalization failure to data/model yet — the bad camera
> angle confounds it. Fix the geometry and re-test before concluding anything about scaling or coverage.

### Next steps (prioritized)
1. **Fix camera geometry first (highest leverage).** Re-mount so the **PCB + gripper tip + pocket are visible
   through the whole grasp**, per env. Make "tip visible during first reach" a **hard gate** (it wasn't).
   Re-collect the offending env(s) with the corrected angle.
2. **Ablation — drop the bad-angle env, retrain, re-eval position gen** (my hunch). If position generalization
   *returns*, the bad camera data was poisoning the reach; if not, the cause is elsewhere. Clean isolation.
3. **Run the real [doc-05](05_eval_and_design.md) closed-loop protocol** on a **good-angle held-out env** —
   matched marked positions, staged rubric, Wilson CI. Right now the v3 assessment is anecdotal (one env,
   in-distribution). I don't have an honest position-generalization *number* yet.
4. **Close the negatives coverage gap** (see knowledge.md §12): the negatives only taught "PCB seated **+ at
   home** → idle," not "PCB seated **+ at any pose** → return home." Add demos that **start from varied
   non-home poses with the PCB already seated and return to home** so the general rule is actually covered.
5. **Only then** finish the remaining envs to hit the 8×30 plan — no point scaling data on top of a camera
   confound.

### ▶ Committed next actions (this session)
Two experiments to separate "bad camera angle" from "genuine generalization failure":

**A. Re-test v3 at a good-angle env (no retrain, cheap first).**
- Pick an env whose training data had a **clean camera angle** (PCB + tip + pocket visible through the grasp) —
  i.e. home, or another good env — and run the current v3 checkpoint there.
- Use varied PCB positions; watch specifically whether the **reach now tracks the PCB** (vs. the same-spot
  reaching seen at Noisebridge). Score with the [doc-05](05_eval_and_design.md) staged rubric.
- **Read:** if position tracking *returns* at a good angle, the Noisebridge failure was a **camera confound**,
  not a model/data limit → the fix is observability, not more data. If it *still* reaches the same spot even
  with a good angle → the problem is real (data/model), go to B and beyond.

**B. Ablation — retrain without the bad-camera-angle env, re-eval.**
- Rebuild the training set **excluding the Noisebridge (bad-angle) episodes**; keep everything else identical
  (same steps, LR, seed) so it's a clean one-variable ablation.
- Re-eval on the **same good-angle positions as A** (matched conditions).
- **Read:** if position generalization **improves** with the bad data removed → the bad-angle episodes were
  *poisoning* the reach (mislabeled visual grounding), and the lesson is "one env with broken observability
  can degrade the whole policy." If **no change** → the bad data was inert, and the gap is elsewhere (coverage
  of off-home / non-tracking states — step 4).
- Keep the **full-data v3 checkpoint** as the baseline; this is a paired A-vs-ablation comparison, so log both
  on the same marked positions.

> Order: do **A first** (no training, immediate signal). Only run **B** if A is ambiguous or you want to
> confirm the confound quantitatively. Record both numbers in this log when done.

---

## Results log — v3b (test A done: good-angle, in-distribution env — camera confound CONFIRMED)

Ran the current v3 checkpoint at the **upstairs electronics room** (in-distribution, **good side camera angle**).

**✅ Confirmed / works:**
- **Position tracking works.** Reach goes to *different table areas* as the PCB moves → **visual conditioning
  is real**; the Noisebridge same-spot reaching was the **bad camera geometry, not a model/data limit.** This
  is the clean confirmation of test-A's hypothesis.
- **Retry behavior** (sometimes): if a pickup fails and the EE is *still in the grasp area* at chunk end, it
  re-attempts. If it has already moved toward the fixture, it does **not** go back — **because no demo ever
  showed a fixture→retry return.** (Coverage = behavior, again.)
- **Negatives generalize:** distractor objects + a moving hand in view → **no motion.**

**❌ Fails / open:**
- **Grasp precision ~1 cm off → ~0% pickup success.** **Reframe:** position *tracking* passed; grasp
  *precision* failed. This is an **observability/precision** problem, **not** a coverage problem — a single
  **side** camera has **depth ambiguity along its optical axis**, and the chunk executes semi-open-loop (no
  fine correction once committed). **More position data won't fix it.**
- **"Confused" on task restart.** Finish task → place PCB back on table → the arm does spurious lift→home
  cycles until the old frames clear. Mechanism is **stale internal state**, not "context length exceeded":
  the **action-chunk queue** (and/or short obs history `n_obs_steps`) still holds a prediction from the
  *completed* scene. Secondary cause: the "just-completed → object-reappears" transition is **OOD** (every
  training episode starts fresh). See fixes below.
- **No task-progress belief (correlation, not causation).** The flat policy can't represent "I have/haven't
  grasped," so it can't gate "verify grasp → then go to fixture." (Prior work: **ECoT**, hierarchical
  **RT-H / π0.5**, **τ0-VLA** — see knowledge.md §12.)

### Corrected diagnosis → what actually fixes each failure
| Symptom | Wrong attribution | Real cause | Fix |
|---|---|---|---|
| Reaches right *area*, misses grasp by ~1 cm | "position gen too weak" | monocular **side-cam depth ambiguity** + open-loop chunk | **wrist / eye-in-hand camera** for final approach; shorter replan / closed-loop; recheck calibration |
| Spurious lift→home on restart | "context window overflow" | stale **action queue** + obs history; OOD hand-off state | **`policy.reset()` on every restart** (flush queue+buffer); check `n_obs_steps`; (opt.) cover the transition in data |
| No back-and-forth retry near fixture | — (correct read) | no fixture→retry demo | add retry/verify **transition demos** |
| Doesn't "know" it grasped | — (correct read) | flat BC has no progress belief | ECoT / hierarchical (research tier) |

### Revised next steps (supersedes the generic list where they conflict)
1. **Camera geometry is the #1 lever, and it fixes the *grasp*, not just the reach.** Add a **wrist camera**
   (eye-in-hand) for the last-cm approach + keep a **fixed side cam** for scene context. Mount both **rigidly
   and reproducibly**. This is the "how humans coordinate hand+eye" instinct made concrete.
2. **Markers = eval tool, not a smaller training set.** Use black-marker positions to measure **seen vs.
   unseen** (doc-05 style). **Do NOT** collapse *training* to a few marked spots — that re-introduces
   discrete-position **memorization**. Train on **wide/continuous** coverage (dense grid + jitter); reserve
   **interstitial off-grid** positions as held-out unseen.
3. **Consolidate into a clean v4:** fixed dual-camera geometry, **drop the bad-angle (Noisebridge) episodes**,
   re-collect + retrain. Then run the real doc-05 closed-loop eval (matched marked positions, staged rubric,
   Wilson CI) to get an **honest** position-generalization *and* grasp-success number.
4. **Deployment hygiene now (free):** wire `policy.reset()` into your restart flow so the stale-state confusion
   stops without any retraining.

---

## Hardware progress — dual-camera rig built (implements v3b next-step #1)

Acting on the v3b diagnosis (**grasp miss = monocular side-cam depth ambiguity**, not a coverage
problem), I built the **eye-in-hand + fixed-scene** two-camera setup. This is the concrete fix for the
~1 cm grasp error and directly changes the observation space for the upcoming **v4** collection.

**The two cameras and *why* each is configured this way:**

| Camera | Role | Model | Exposure | Focus | Rationale |
|---|---|---|---|---|---|
| **Wrist (eye-in-hand)** | last-cm grasp approach | **Logitech C922** | **1/120 s** shutter, **600 ISO** | **manual, 44%** (fixed near) | Tilted so the **gripper tip sits at FOV center** — resolves the side-cam depth ambiguity for the final approach. |
| **Top-down (fixed scene)** | scene / xy position context | **Logitech C920** | **auto** | **0%** (far/∞) | Overhead view gives clean **PCB xy** with no depth ambiguity for the reach; carries scene+lighting context. |

**Why the manual settings on the wrist cam matter (not incidental):**
- **Manual focus @ 44% (near):** the tip and PCB are at a **short, roughly constant working distance**.
  Autofocus would *hunt* as the scene changes frame-to-frame, injecting blur + a moving focal plane —
  a hidden nuisance variable. A **fixed near-focus** keeps the grasp region sharp and consistent across
  every episode (reproducibility, same discipline as fixed camera pose).
- **1/120 s shutter (fast):** the wrist cam **moves with the arm**, so it's the one that suffers **motion
  blur** during the reach. A fast shutter freezes the tip/PCB at the moment fine correction matters most.
- **600 ISO:** compensates the light lost to the fast shutter so the near scene stays properly exposed.
- The **top-down C920 stays on auto** because it's **static** (no motion blur) and views a **wider, farther**
  scene — its job is scene/position context, where auto-exposure robustness across lighting > sharpness of
  the near field, and **focus 0% (far)** matches its longer working distance.

**Geometry intent:** wrist cam **tilted to center the gripper tip** = the tip (and PCB during the final
approach) is always in frame — the exact observability the v3/Noisebridge run lacked. Top-down keeps the
scene/xy anchor. Together they cover **reach (top-down xy) + grasp (wrist depth)**, the two legs that a
single side cam couldn't serve at once.

> **Carries into v4 — update the record invariants:** collection now uses **two** cameras, so
> `--robot.cameras` must list **both** (e.g. `{ top: {...C920...}, wrist: {...C922...} }`) and the camera
> **names, indices, resolution, fps, exposure/focus settings** join the **"invariants that must be identical
> every session"** list above. v4 is a **fresh geometry**, so it does
> **not** resume onto the single-cam v1–v3 dataset — start a new `repo_id` and re-collect (the v3b plan to
> **drop the bad-angle episodes** and consolidate a clean v4 already required a re-collect). Lock the wrist
> manual focus/exposure **before** the first episode and don't touch them mid-dataset — a mid-run change to
> a manual setting fragments the data the same way a camera-pose drift would.

**Still TODO before v4 collection:** rigid/reproducible mount for both cams; confirm the wrist tilt keeps
the tip centered through the *whole* grasp arc (not just at home); re-run the [per-spot quality
gate](#per-spot-quality-gate-dont-multiply-a-mistake-by-30) with both feeds; wire `policy.reset()` into the
restart flow (the free v3b deployment-hygiene fix, independent of the cameras).

---

## Interview talking point (feeds the Chelsea design round)
> "v1 failed two ways off-site: it ignored the object (ran the motion on an empty table) and broke in
> new lighting. I diagnosed the first as a *coverage/spurious-correlation* bug — object presence was
> constant in my data, so the policy had no reason to condition on it — and fixed it with **negatives
> carrying the same instruction** plus **wide position variation** that makes the action impossible to
> produce without looking. The second is **scene covariate shift**, so I spent the budget on **scene
> diversity over demo depth** (8 envs × 30, not 5 × 50 — diversity is the lever), and I reserved an
> **unseen env for closed-loop eval** so the generalization number is honest. I deliberately did *not*
> claim rotation/fixture generalization — I don't evaluate it, so I don't pay for it."