# 09 — Project B (home deployment) + hardware, compute and background notes

Split out of [08](08_data_quality_research.md) on 2026-09-14, so 08 holds only the Project A
research plan. This doc keeps everything else:
- **Part 0:** four lessons carried forward from 06/07.
- **Part B:** one household task at home, for your mom.
- **Part C:** gaps worth naming.
- **Part D:** the full bill of materials: GPU memory math, π0.5 deployment memory sources, GPU pricing,
  laptop vs. desktop, SO-101 repeatability.
- **Part E:** how the two projects feed each other.

They are deliberately different in kind:

| | **Project A — Data Quality** | **Project B — Home Deployment** |
|---|---|---|
| Type | **Research.** Open question; negative results still count | **Engineering.** Known-solvable; the difficulty is integration + reliability |
| Success = | A defensible *number*: "metric X predicts closed-loop success at ρ=0.7, metric Y doesn't" | A *working thing*: mom presses a button, the arm does the chore, 9 times out of 10 |
| Generalization | is the whole subject | **explicitly not required** — one scene, one task, forever |
| Main cost | **evaluation** (robot rollouts), not training | **your time on site survey + fixture design**, not ML |
| Risk | you measure something that doesn't transfer | you build something that works once and then rots |

They share hardware, the LeRobot stack, and the eval protocol from [05](05_eval_and_design.md) — and B
doubles as a second, messier testbed for A's metrics. Run them in parallel: A is compute-bound (long
runs you start and walk away from), B is hands-bound (measuring, printing, mounting).

---

# Part 0 — What you already know, carried forward

Four hard-won facts from 06/07 that constrain everything below. Don't re-derive them.

1. **BC learns correlations; coverage is the only lever.** If `(state, action)` isn't in the data, the
   policy has no idea — even when the right action is obvious to a human (knowledge.md §12).
2. **Observability confounds everything.** The v3 "position generalization regression" was a camera that
   couldn't see the PCB during grasp. **Any data-quality result measured across episodes with drifting
   camera geometry is invalid.** Lock the rig before you measure anything.
3. **Tracking ≠ precision.** Reaching the right *area* and grasping within *mm* are different failures
   with different fixes (coverage vs. observability). Your metrics must be able to tell them apart.
4. **Offline loss is a checkpoint-picker, not a verdict.** Closed-loop success is the metric. Which is
   exactly why Project A is hard and worth doing: everyone wants an offline number that predicts the
   closed-loop one.

---

# PART B — Deploy a real household task, at home, for your mom

**The reframe that makes this tractable:** this is not a robotics research problem. It's a **product
reliability problem** wearing a robotics costume. You explicitly don't need generalization. You need one
task, in one place, under one lighting setup, working **on the 50th try as well as the 5th**, and safe
enough to sit near a person who never signed a consent form.

The failure mode isn't "the policy can't learn it." It's "it worked in August, and by October the camera
had been nudged 2 cm and nobody could fix it."

## B.1 — Choosing the task (do this before buying anything)

**Hard gates — a candidate failing any of these is out, however appealing:**
- **Payload ≤ ~300 g** at the SO-101's *measured* capability at full extension, not the datasheet number.
- **No sharp, hot, wet, electrical, or heavy items.** No knives, no kettles, no glassware near a person.
- **A failure is boring.** Worst case = an object on the floor. Not a burn, a flood, or a cut.
- **Fixed workspace** — a dedicated table/tray that doesn't move and isn't in a walkway.
- **She can reach the e-stop faster than the arm can reach her.**

**Soft scoring (1–5 each):** genuine daily value · repeatability of object pose · tolerance (does a 1 cm
error still succeed?) · visual legibility (contrast, glare, occlusion) · need for force feedback · cycle
time · frequency per day.

| Candidate | Value | Tolerance | Risk | Verdict |
|---|---|---|---|---|
| Sort pills into a weekly organizer | very high | **terrible** (tiny, medical) | **medical — hard fail** | ✗ Wrong first task, wrong consequences |
| **Fetch a specific item from a fixed tray** (glasses, remote, phone) | high | good | low | ✓✓ **Strong candidate** — static poses, big tolerance, real daily value |
| Bin small trash from a fixed tray | medium | good | low | ✓ Easy — but is it actually useful to her? |
| Load/unload a dish rack | high | poor (wet, specular, heavy, variable) | medium | ✗ v3 material, not v1 |
| Water a plant (fixed pot, small can) | medium | medium | **water near electronics** | ~ only with a tiny can |
| Press a fixed appliance button / open a microwave door | high | good | medium | ✓ literally the task the PSD paper studied with older adults |
| **Sort mail/cards into two labeled slots** | medium | **excellent** | very low | ✓✓ **Best pure-v1 task** — near-identical to your PCB task, so the whole pipeline transfers day one |
| Pass a snack/cup from a dispenser to a fixed spot | high | medium | spill | ✓ with a lidded container |

**Recommendation:** pick the task by **asking your mom what annoys her**, then filter through the gates —
don't pick it from this table. But bias toward whatever is closest to
`pick rigid object from marked spot → place in fixed receptacle`, because you've already debugged that
exact motion end-to-end. **Fetch-item-from-a-fixed-tray** is the sweet spot: real value, huge tolerance,
trivially safe, and it degrades gracefully (it drops the thing; she picks it up).

> **Design talking point:** "I chose the task by consequence-of-failure first and capability second. The
> first household robot task should be one where the worst outcome is mild inconvenience."

## B.2 — Site survey (the unglamorous part that decides the outcome)

Do this **with a tape measure and a notebook**, before any data collection, and **write the numbers back
into this section** — they're the spec for everything downstream.

- [ ] **Reach envelope.** Measure the SO-101's working radius at your actual payload; tape the reachable
      footprint onto the table. **Every object pose must sit inside it, with margin, at the required
      gripper orientation** — reach is orientation-dependent, and this kills more home deployments than
      anything else.
- [ ] **Table height vs. her seated/standing height.** The arm should work *below* chest height so
      nothing can swing toward her face.
- [ ] **Mounting.** The base **must be bolted or clamped down**. An arm that can drag its own base is an
      arm whose calibration is a lie; vibration from a wobbly table silently corrupts every demo.
- [ ] **Lighting, measured across a day.** Photograph the scene at 8am / noon / 6pm / 10pm. **Window
      light is the enemy** — it changes the scene more between morning and evening than your entire 8-env
      dataset did. Decide now: (a) block daylight and install a **fixed LED lamp** so lighting is a
      constant, or (b) collect across all four conditions and pay the data cost. **(a) is correct for
      v1.** Note glare and specular hotspots on the work surface.
- [ ] **Camera geometry, fixed forever.** Top-down + wrist (you have both). The top-down mount must be
      rigid and **outside the arm's swept volume** — draw the sweep before drilling.
- [ ] **Power + network.** Dedicated outlet; cable routing that can't be tripped over or snagged by the
      arm; a power strip with a **physically obvious master switch** she can hit.
- [ ] **The compute box.** Where it lives, how loud it is, whether it survives a power cut, and whether
      it auto-starts the stack on boot (it must).
- [ ] **Clearance.** Mark a no-go zone; nothing she needs daily should be inside it.
- [ ] **Cat / dog / grandchild factor.** Be honest about who else is in that room.

**Photograph the finished setup from four angles and save them in `results/`.** That photo set *is* your
reproducibility record — when it stops working in three months, you diff against it.

## B.3 — Task decomposition and the data plan

Apply your own §12 insight directly: **enumerate the states, then cover them.**

1. **Decompose** into sub-skills: approach → pre-grasp align → grasp → lift → transport → position →
   release → retreat → home.
2. **At every seam, enumerate the off-nominal states:** grasp slipped, object dropped mid-transport,
   placed crooked, started from a non-home pose, object already placed, object absent, object moved by a
   human mid-episode, arm bumped.
3. **Collect demos for those states.** This is the structured-DAgger idea from §12 — choose states by
   task analysis instead of waiting for rollouts to surface them. **Budget ~30–40% of the data for
   off-nominal and recovery states.** It feels wasteful, and it is the single thing separating "the demo
   worked once" from "it works on a Tuesday."
4. **Negatives from varied poses.** Your v3 lesson applied: "task done → return home" must be shown
   **from many starting poses**, not only from home.
5. **Continuous coverage, not marked spots.** Train on a dense, jittered continuous range; reserve marked
   interstitial positions purely for eval (06 §v3b step 2).

**Size estimate:** 150–300 episodes for one fixed-scene task with clean dual-cam geometry. Fewer than the
multi-env project because you have **one environment** — the 06 diversity argument cuts the other way
here: with no generalization claim, depth in one condition is exactly right.

## B.4 — Model choice for B (be boring on purpose)

| Model | Fit for a fixed household task | Call |
|---|---|---|
| **ACT** | Single task, fixed scene, no language needed. Small, fast, trains locally, low-latency inference on modest hardware | **Start here.** Ship the whole loop with ACT first |
| **Diffusion Policy** | Strong on multimodal fine manipulation, still single-task | Fallback if ACT's precision falls short |
| **SmolVLA** | Worth it once you want *language* ("bring my glasses" vs. "bring the remote") or several task variants in one policy | v2 — when you add a second item |
| **π0.5** | Generalization you explicitly don't need, ~3B params, 24 GB+ to fine-tune, heavier inference | Skip for B; use it in **Project A**, where model class *is* the question |

> **Design talking point:** "I used the smallest model that could do the job, because for a fixed
> household task the bottleneck is reliability and latency, not capability. The VLA is for the research
> project; the chore gets a 100 MB policy."

## B.5 — Reliability engineering (where B is actually won)

The gap between "trained a good policy" and "mom uses it" lives entirely here. None of it is ML.

- **`policy.reset()` at every task boundary** — your free v3b fix. Non-negotiable in the run loop.
- **Homing + calibration check on startup.** Refuse to run if joints are outside expected bounds.
- **Watchdogs:** per-episode timeout (stop and return home), max-velocity cap, torque/current cap with a
  soft-stop, and a "no object detected → don't move" gate (your negatives, plus a cheap detector as
  belt-and-braces).
- **A physical one-button start** — an arcade button on USB/GPIO, not a terminal command. If she needs a
  keyboard, the project has failed.
- **A physical e-stop** that cuts servo power, within her reach, clearly marked. Test it monthly.
- **Autostart on boot** plus service supervision — the box *will* lose power; assume it.
- **Log every run** (video + actions + success flag): remote debugging, and free flywheel data.
- **Weekly self-check:** run the task once from a known state and compare against a stored reference —
  catches calibration drift and camera nudges before she does.
- **Camera-drift alarm.** Store a reference frame at deploy time, compare on each startup, warn if the
  scene shifted. **This one check would have caught your entire v3 Noisebridge confound.**

## B.6 — Safety (she did not consent to be a test subject)

The SO-101 is a low-power hobby arm, not a certified cobot, and nothing here makes it one. Use
[ISO 10218 / ISO TS 15066](https://www.evsint.com/collaborative-robot-safety-standards-2026-iso-10218-2025-ts-15066/)
(power-and-force limiting, speed-and-separation monitoring) and **ISO 13482** (personal care robots) as a
*mindset*, not a claim:

- **Software speed cap**, well below the arm's max, always. Fast robots near people are a choice.
- **Blunt everything.** No sharp gripper edges, no pinch points at the base. Print rounded, compliant
  fingertips.
- **Workspace separation.** Her normal hand path and the arm's swept volume must not overlap.
- **Never unattended in v1.** You are present for every run until you have *counted* dozens of
  consecutive clean cycles; then supervised-by-her. "Alone in the house with it" is a much later
  milestone, if ever.
- **Teach her the stop first** — before she ever sees it do the task.
- **Write down what happens when power is cut mid-motion.** If the arm goes limp and drops its payload,
  that must be safe by design — another argument for light, soft, boring objects.

## B.7 — Acceptance criteria ("done" must be a number)

Pre-register these, like any other eval:

- [ ] **≥90% success over 30 consecutive runs**, staged rubric, across the **full day's lighting range**
      she'll actually use it in.
- [ ] **Zero unsafe events** (no contact with a person, no object thrown beyond the workspace, no
      uncommanded motion) across those 30.
- [ ] **Mom can start and stop it herself**, unassisted, after one demonstration.
- [ ] **Cold-start test:** power-cycle everything; it comes back and runs with no laptop and no terminal.
- [ ] **One-week soak:** still passes the above a week later **without you touching it.** This is the
      criterion that separates a deployment from a demo.
- [ ] **She says it's useful.** If she's politely humoring you, you picked the wrong task — go back to
      B.1. That's not a failure, it's the finding.

## B.8 — Milestones

| # | Milestone | Rough effort |
|---|---|---|
| B1 | Task chosen *with her*; site survey numbers + photos recorded | 1 weekend |
| B2 | Mount, lighting, fixtures/jigs printed, cameras locked | 1–2 weeks |
| B3 | Teleop works in the home setup; per-spot quality gate passes | a few days |
| B4 | 150–300 episodes incl. off-nominal / recovery / negatives | 1–2 weeks of sessions |
| B5 | ACT trained; closed-loop eval with the staged rubric | 1 week |
| B6 | Reliability harness: button, e-stop, watchdogs, autostart, logging, drift check | 1–2 weeks |
| B7 | 30-run acceptance test + one-week soak | 2 weeks (mostly waiting) |
| B8 | Handover — she uses it, you watch and log | ongoing |

**Realistic total: ~2–3 months of evenings and weekends**, of which the ML is maybe 15%. Plan for that
honestly rather than being surprised by it.

---

# Part C — Gaps worth naming explicitly

Things not yet in your docs that will bite otherwise:

1. **Evaluation is the budget, not training.** Every plan above is priced in *rollouts*. Build the
   logging/scoring harness first; it's the compounding investment.
2. **Dataset versioning.** You already lived this (`_175109`, `_181041`, "protect v1"). Adopt a real
   convention: immutable `repo_id` per camera geometry, a `metadata.json` per session (camera poses,
   lighting, operator, exposure/focus, env id), and **never mutate a dataset you've trained on**.
3. **Operator as a hidden variable.** If anyone else ever teleoperates, that's a confound — log it. (Also
   a free bonus factor for Project A: can a metric detect *who* collected an episode?)
4. **Calibration drift over weeks.** Log a calibration fingerprint per session, or a metric that's really
   detecting drift will masquerade as a data-quality result.
5. **Thermal / duty cycle.** Hobby servos sag when hot, so a session's last 10 episodes may differ
   systematically from its first 10 — exactly what a quality metric *should* catch, and exactly what
   silently ruins an experiment if you don't log it.
6. **A second arm doubles your data rate** and lets you run A/B evals in parallel — the cheapest
   throughput upgrade available to you.
7. **Blind scoring, no cherry-picking.** Pre-register, score from video without knowing the condition,
   report every rollout including the embarrassing ones.
8. **Publishing.** Both projects are worth writing up; a repo plus a clear report is worth more than
   either raw result, and it's the artifact other people can check.
9. **Reproducibility from someone else's machine.** Pin versions, commit configs, record the exact
   `lerobot` commit — your Windows-path patch ([PR #2940](https://github.com/huggingface/lerobot/pull/2940))
   is a reminder that your stack carries local modifications.
10. **Sim is a legitimate accelerant, not a compromise** — for *metric selection*. Just never let a sim
    number be the final claim about real data.

---

# Part D — Bill of Materials

**✅ have** · **🔴 buy now** (blocks the plan) · **🟡 buy later** (unblocks a phase) · **⚪ optional/skip**.
Prices are approximate street/list figures as of **September 2026**, sourced where noted — verify before
ordering.

> **Note (2026-09-14):** written before you settled on the **RTX 5080 laptop + cloud 4090/A100**. The
> purchase advice in D.1 is kept for reference. The current Project A setup is in
> [08](08_data_quality_research.md) Part D.

## D.0 — GPU memory per model (computed + cross-checked against published figures)

**How the numbers are derived.** Training memory = **optimizer/weight states** (exactly computable) +
**activations** (batch-dependent, measured). With AdamW mixed precision the state cost is
`4 B (fp32 master) + 4 B (fp32 grad) + 8 B (Adam m,v) = 16 B/param`; a **frozen** weight costs only
`2 B/param` (bf16, no grad, no optimizer slot); inference is `2 B/param` plus a small activation buffer.
Applying that:

| Model | Params | Inference (bf16) | **Full fine-tune** states | Frozen-backbone states | Reported real-world total |
|---|---|---|---|---|---|
| **ACT** | 80 M | 0.16 GB | **1.3 GB** | n/a | ~5 GB @ bs 8 · ~21 GB @ bs 64, 2 cams |
| **Diffusion Policy** | ~250 M | 0.5 GB | **4.0 GB** | n/a | ~8 GB @ bs 8 · ~24 GB @ bs 64 |
| **SmolVLA** | 450 M | 0.9 GB | **7.2 GB** | 2.3 GB (`train_expert_only`) | **10–16 GB @ bs 8**; bs 64 recipe assumes A100 80 GB |
| **π0 / π0.5** | 3.3 B (2.3 B VLM + 0.3 B action expert) | 6.6 GB | **52.8 GB** | 11.0 GB (`train_expert_only`) · 7.1 GB (LoRA) | **openpi: >8 GB infer · >22.5 GB LoRA · >70 GB full** |

**The computation validates the published numbers**: openpi's ">70 GB" for a full π0.5 fine-tune is my
52.8 GB of states + ~17 GB of activations; their ">22.5 GB" LoRA figure is 7.1 GB of states + ~15 GB of
activations. So the activation term is **~15–17 GB regardless of mode** — which is why LoRA saves far
less than the 7× parameter-state reduction suggests.

### What that means for a 32 GB card

| Workload | 32 GB (RTX 5090) | 24 GB (5090 laptop / 4090 / 3090) | 16 GB | 6 GB (your 3060) |
|---|---|---|---|---|
| ACT train | ✅ bs 64 | ✅ bs 48 | ✅ bs 32 | ✅ bs 4–8 |
| Diffusion Policy train | ✅ bs 48 | ✅ bs 32 | ✅ bs 16 | ⚠️ bs 2–4 |
| SmolVLA full fine-tune | ✅ bs 24–32 | ✅ bs 16 | ⚠️ bs 8 | ❌ (frozen VLM, bs 1–2 only) |
| **π0.5 LoRA** | ✅ (tight, bs 4–8) | ⚠️ barely — 22.5 GB floor | ❌ | ❌ |
| **π0.5 full fine-tune** | ❌ | ❌ | ❌ | ❌ — needs A100 80 GB / H100 or multi-GPU FSDP |
| π0.5 **inference** (real-robot + sim eval) | ✅ | ✅ | ✅ | ❌ — 6.6 GB of weights alone |

> **⚠️ Confound alert for Project A's Q2.** If ACT and SmolVLA get **full** fine-tunes but π0.5 only gets
> **LoRA**, the "metric × policy class" comparison is confounded by *adaptation method*, not just model
> class. Pick one and state it: **LoRA everything** (cheap, consistent, my recommendation) or full-FT
> everything (cloud-only, expensive). Don't mix without saying so.

### Deployment note: chunking decouples inference from control rate
A 50-action chunk at 30 Hz is **1.67 s of motion**, so even π0.5 has a ~1,667 ms inference budget per
chunk. Measured π0.5 latencies for reference:

| Setup | π0.5 latency | % of a 50-step chunk budget |
|---|---|---|
| RTX 4090, bf16, 5 denoise steps (vanilla) | **76 ms** | 4.6% |
| RTX 4090, same + Real-Time Chunking | 97 ms | 5.8% |
| RTX 5090 desktop, optimized engine (FlashRT) | 12–20 ms (**50–80 Hz**) | ~1% |
| Jetson AGX Thor, optimized engine | ~43 ms (23 Hz) | 2.6% |
| **RTX 5080 laptop** (est. 0.3–0.5× desktop 5090) | **~130–150 ms** vanilla | **~9%** |

> Note the 4090 and 5090 rows come from *different software stacks* (openpi/RTC paper vs. an optimized
> inference engine), so they aren't a clean apples-to-apples GPU comparison — but both are far inside the
> deadline, which is the point.

**Latency is not the deployment constraint; VRAM capacity is.** And real-robot eval **cannot be rented**
(the arm is on your desk, over USB), so your local VRAM floor is set by the largest model you want to run
*on hardware*: **π0.5 ⇒ 12–16 GB local, minimum.**

**Corollary for laptops:** inference is a **~5–10% duty cycle burst**, not a sustained load, so the
thermal-throttling penalty that cripples laptops for *training* is irrelevant for *deployment*. A laptop
is a perfectly good policy-serving box; it's a poor trainer.

### D.0.1 — What sources actually say about π0.5 *deployment* memory (checked 2026-09-13)

| Source | Type | What it says |
|---|---|---|
| [pi.website — π0.5 blog](https://www.pi.website/blog/pi05) (Apr 2025) and [openpi blog](https://www.pi.website/blog/openpi) (Feb 2025) | **Official PI website** | **No hardware or memory figures at all.** Only the 300 M action-expert size |
| [openpi README](https://github.com/Physical-Intelligence/openpi) | **Official PI repo** | Inference **"> 8 GB"**, example **RTX 4090**; single-GPU estimate; Ubuntu 22.04 only; JAX **preallocates 75%** of GPU memory by default |
| [openpi DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md) | Official PI repo | Run the server on "a powerful GPU (~NVIDIA 4090)"; "0.5 – 1 sec latency per chunk is normal" (remote, incl. network) |
| [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05) | Trainer (Hugging Face) | **No inference figure.** Training recipe "sized for a single 80 GB GPU" (bs 64, bf16, grad checkpointing); `train_expert_only` = "reduced memory" |
| [NVIDIA Jetson AI Lab — π0.5 on Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) | Trainer (NVIDIA) | Weights **"~6 GB+"**; BF16 ~132 ms → TensorRT FP8 ~53 ms → FP8+NVFP4 ~49 ms. Thor only; no Orin guidance |
| [FlashRT](https://github.com/flashrt-project/FlashRT) | Community inference engine | 5090 FP8 **17.6 ms** · **5060 Ti FP8 41.4 ms** · L40 26.6 ms · Thor FP8 44 ms / NVFP4 23–32 ms · AGX Orin INT8 124 ms / BF16 216 ms. **No VRAM figures** |
| [abdul004/pi05_so101_checkpoint](https://huggingface.co/abdul004/pi05_so101_checkpoint) | Community, **SO-101** | Trained on A100 80 GB; served on **RTX 4090 24 GB — "borderline for memory, occasional OOM during model loading"**; 270 ms end-to-end with JPEG, 600 ms without (remote) |
| [openpi issue #599](https://github.com/Physical-Intelligence/openpi/issues/599) | Community (**π0**, not π0.5) | ~**37 GB** used on a 48 GB GPU via `serve_policy.py`; unanswered |
| [HF community guide (Tonic)](https://huggingface.co/blog/Tonic/training-and-inference-with-pi05), [EmbodiFlow](https://io-ai.tech/platform/en/guides/Pipeline/LeRobot/Pi0/), Intel Open Edge, SVRC | Trainers | **No π0.5 memory figures** (SVRC's "24 GB+" is for OpenVLA) |

**Reading the conflict.** The official ">8 GB" matches the arithmetic (3.3 B × 2 B bf16 = 6.6 GB + buffers)
and NVIDIA's "~6 GB+" weights. The scary community numbers are mostly the **stock JAX stack**: 37 GB on a
48 GB card is ≈ **77%**, i.e. JAX's 75% preallocation — a reservation, not a requirement. The 4090
load-time OOMs are real but plausibly a **checkpoint-restore peak** (e.g. transient fp32 or duplicate
param trees) rather than steady state — *my hypothesis, not confirmed by any source.*

**Practical floor depends on the stack, not just the model:**

| Stack | Practical VRAM floor | Basis |
|---|---|---|
| openpi **JAX**, stock | **16–24 GB**; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` | official >8 GB, but 24 GB load-OOM reports + 75% prealloc |
| LeRobot / openpi **PyTorch**, bf16 | **~8–12 GB** steady state; **16 GB comfortable** | 6.6 GB weights (computed) + NVIDIA's ~6 GB+ |
| **TensorRT / FlashRT** FP8 · NVFP4 · INT8 | **< 8 GB plausible** (FP8 ≈ 3.3 GB weights, computed) | runs on AGX Orin, 5060 Ti. FP8 needs Ada/Blackwell — **not** Ampere (3090/3060) |

**Nobody publishes a measured steady-state VRAM number for π0.5 inference.** Measure it yourself on first
deploy (`torch.cuda.max_memory_allocated()` for PyTorch; disable JAX preallocation first) and record it
here — that number is more useful than any row above.

## D.1 — Compute (the one real decision)

> **⚠️ Price correction (2026-09-13).** An earlier draft of this doc quoted the RTX 5090 at its **$1,999
> launch MSRP**. That number is obsolete and was misleading here: the 50-series market is broken
> (GDDR7 + CoWoS shortages expected to persist into 2027). **Actual observed prices:**
>
> | Card | **Observed price (user-recorded, 2026-09-13)** | ≈ USD @ 7.15 | US median for comparison |
> |---|---|---|---|
> | **RTX 3090 24 GB (used)** | **¥8,000** | ~$1,120 | — |
> | **RTX 5090 32 GB** | **¥46,000** | ~$6,430 | $4,700 (Aug) → $5,690 (Sep 2026) |
>
> **That's 5.75× the price for ~2.5× the throughput and +8 GB — roughly half the performance per yuan,
> and the extra 8 GB unlocks no new capability** (24 GB already clears π0.5 LoRA's 22.5 GB floor; neither
> card can do a full π0.5 fine-tune, which needs >70 GB). **The 5090 is no longer the recommendation at
> these prices.**

| Item | Why | Status | Cost |
|---|---|---|---|
| **RTX 3060 6 GB** (current) | Fine for ACT/SmolVLA **inference** and all tier-1/tier-2 metrics. Too small for serious fine-tuning, and **cannot run π0.5 at all** (6.6 GB of weights alone) | ✅ have | — |
| **RTX 3090 24 GB (used)** | **The buy — *if* you have or build a desktop host.** 24 GB clears π0.5 LoRA *and* all local/real-robot inference; Ampere, 936 GB/s, no FP8/FP4, ~2.5× slower than a 5090. ⚠️ **The ¥8,000 is the card only**: your 3060 is 6 GB, i.e. the *laptop* variant (desktop 3060s ship 8/12 GB), so a 3090 needs a tower — CPU, board, 32 GB RAM, **850–1000 W PSU**, case — roughly **¥3,000–5,000 more** (rough estimate). See §D.1.2 | 🔴 | **¥8,000 card · ~¥11,000–13,000 all-in** |
| **A second RTX 3090** | 2× ¥8,000 = ¥16,000 — still **35% of one 5090**, and gives **48 GB total + two parallel experiments**. For a *grid* study (Project A), parallelism beats per-job speed. 3090s also still support NVLink | 🟡 **seriously consider** | ¥8,000 |
| RTX 5090 32 GB | Only justified if the price collapses or your time is worth more than ¥38,000 of difference | ⚪ **skip at current pricing** | ¥46,000 |
| **GPU laptop (5090 laptop 24 GB / 5080 laptop 16 GB)** | ~40–50% of desktop 5090 throughput, throttles under sustained load, costs more than a desktop. **Fine as a policy-serving box, poor as a trainer** — and your existing laptop already serves policies | ⚪ **skip** | ~$3,500–4,200 |
| China-market **modded 96 GB RTX 4090/5090** (Alibaba, ~$3,888) | The *only* consumer-priced route to a **full π0.5 fine-tune** (>70 GB). Grey market: modified VBIOS, driver-compatibility risk, no warranty, uncertain resale | ⚪ eyes open | ~¥28,000 |
| **Cloud GPU credits** | π0.5 fine-tunes, the sim grid, anything over 24 GB. RTX 4090 from **~$0.34/hr** (RunPod), A100 **~$0.68–2.21/hr**, H100 **~$1.49–3.99/hr** | 🔴 | **$200–500** for the whole plan |
| Colab Pro+ | already in use; convenient, not cost-optimal for long runs | ✅/🟡 | ~$50/mo |
| **Deployment box for B** — Jetson Orin Nano Super ($249) or a mini-PC | ACT inference doesn't need a 3060, and a dedicated always-on box is what makes B a *deployment* rather than a demo | 🟡 (phase B6) | $249–600 |
| Jetson AGX Orin 64 GB ($1,999) / AGX Thor ($3,499) | for running a ~3B VLA on-robot | ⚪ skip — neither project needs it | — |
| 2 TB NVMe SSD | Datasets are smaller than you'd think (161 eps ≈ ~1 h of 640×480 video per camera). 2 TB covers everything including checkpoints | 🟡 | $100–150 |
| Backup (external SSD + cloud) | Losing a 300-episode dataset costs weeks | 🔴 | $80–150 |

**Revised decision rule (at ¥46,000 for a 5090):** the arithmetic now points hard at **local 3090 + cloud
for sweeps**. One 5090's price converts to roughly **19,000 h** of a rented RTX 4090 ($0.34/hr), **4,300 h**
of A100, or **3,200 h** of H100 — against a whole-plan need of maybe **300–600 GPU-hours**. The 5090's
price tag buys **10–30× more compute than this plan consumes** if spent on cloud instead.

> **Buy the ¥8,000 3090 for the inner loop** (dev, debugging, *all* real-robot eval — which can't be
> rented) **and rent cloud for the sim training grid (08 §A.4).** If you want to spend more, a **second ¥8,000 3090** buys
> parallelism, which a grid study values more than single-job speed. Don't buy the 5090, and don't buy
> a laptop and a desktop.

### D.1.1 — Desktop vs. laptop silicon (the names lie)

A "laptop 5090" is **not** a desktop 5090. It's closer to a desktop 5080 die, underclocked:

| | CUDA cores | Boost | VRAM | Bus | Bandwidth | Power |
|---|---|---|---|---|---|---|
| **RTX 5090 desktop** | 21,760 | 2.41 GHz | **32 GB** | 512-bit | **1,792 GB/s** | 575 W |
| RTX 5080 desktop | 10,752 | ~2.6 GHz | 16 GB | 256-bit | 960 GB/s | 360 W |
| RTX 5070 Ti desktop | 8,960 | ~2.5 GHz | 16 GB | 256-bit | 896 GB/s | 300 W |
| **RTX 5090 laptop** | **10,496** | 1.52 GHz | 24 GB | 256-bit | 896 GB/s | 95–150 W (+25 W boost) |
| RTX 5080 laptop | 7,680 | — | 16 GB | 256-bit | ~896 GB/s | 80–150 W |
| RTX 5070 Ti laptop | 5,888 | — | **12 GB** | 192-bit | ~672 GB/s | 60–115 W |

**Measured, not extrapolated** (StorageReview, UL Procyon, same benchmark both platforms):

| Workload | 5090 laptop | 5090 desktop | Ratio |
|---|---|---|---|
| Llama-3 text gen | 107.9 tok/s | 214.3 tok/s | **0.50×** |
| Mistral text gen | 125.6 tok/s | 255.9 tok/s | **0.49×** |
| Phi text gen | 163.1 tok/s | 314.4 tok/s | **0.52×** |
| SDXL FP16 image gen | 13.39 s/img | 5.22 s/img | **0.39×** |

**So: the laptop 5090 is ≈ 40–50% of the desktop 5090 on real AI work**, has 24 GB instead of 32 GB, and
costs *more*. Two further laptop-specific penalties that don't show up in spec sheets:
- **TGP lottery.** The same "RTX 5090 laptop" ships at 95 W or 150 W depending on chassis — a gap worth
  up to ~40% of performance. The SKU name tells you nothing; check the TGP of the specific laptop.
- **Sustained-load throttling.** Gaming benchmarks are bursty; **training is a 4-hour sustained load**,
  which is the worst case for laptop thermals. Expect to land at the low end of the range, not the high.

**Estimated wall-clock for the reference job** (SmolVLA, 20k steps ≈ 4 h on an A100 per the HF docs),
scaled by spec ratios and the measured desktop:laptop gap — *estimates, not benchmarks of this workload*:

| GPU | VRAM | Est. 20k-step SmolVLA fine-tune | π0.5 LoRA possible? |
|---|---|---|---|
| A100 80 GB (reference) | 80 GB | ~4 h | ✅ (full FT too) |
| **RTX 5090 desktop** | 32 GB | **~3–4 h** | ✅ |
| RTX 5090 laptop | 24 GB | ~7–8 h | ⚠️ barely (22.5 GB floor) |
| RTX 5070 Ti desktop | 16 GB | ~7–8 h | ❌ |
| RTX 5070 Ti laptop | 12 GB | ~12–14 h | ❌ |

**Capacity is binary; speed is linear.** A 16 GB card cannot run π0.5 LoRA *at any speed*. A slower 32 GB
card just makes you wait. For this plan, **prioritize VRAM over throughput.**

### D.1.2 — RTX 5080 laptop vs. 3090 desktop build (the real choice)

| | **RTX 5080 laptop** (e.g. Legion Y9000P / Pro 7i, ROG Strix SCAR) | **3090 desktop build** + keep the 3060 laptop |
|---|---|---|
| Cost | **~¥18,000–22,000** (Y9000P 5080 launch/subsidy pricing — check JD, 2026 shortages may push it up) | **~¥11,000–13,000** all-in |
| VRAM | 16 GB | 24 GB |
| π0.5 LoRA **training** | ❌ (22.5 GB floor) → cloud | ✅ tight, bs 2–4 |
| SmolVLA full fine-tune | ⚠️ bs 8 / grad accum | ✅ bs 16 |
| π0.5 **deployment** | ✅ bf16 PyTorch (~8 GB) | ✅ |
| FP8 / NVFP4 TensorRT speedups | ✅ Blackwell | ❌ Ampere |
| Runs π0.5 **away from your desk** (held-out envs, Noisebridge) | ✅ | ❌ 3060 laptop can't hold it; stream from desktop over network instead |
| Sustained training | throttles; 175 W max | full 350 W, no throttling |
| Per-step training speed | roughly comparable to a 3090 (±25%, *estimate*) — VRAM and throttling are the real differences | |

**What the ¥7,000–9,000 price gap buys in cloud:** a π0.5 fine-tune on a rented A100 80 GB is **~3–4 h ≈
$6–8 per run** (reported for an SO-101 π0.5 checkpoint), so the gap ≈ **~150 π0.5 fine-tunes**. The laptop's
premium is *not* justified by compute — only by **portability** and **replacing an aging 3060 laptop**.

**Decision rule — buy one, not both:**
- **3060 laptop still healthy →** build the **3090 desktop**; keep the laptop as the robot client; stream
  π0.5 from the desktop (openpi/LeRobot remote inference) when evaluating away from the desk.
- **Laptop due for replacement anyway, or most eval happens off-site →** buy the **5080 laptop** *instead*,
  and send all π0.5 training and the sim grid to cloud.

**If buying the 5080 laptop, spec it like this:**
1. **175 W TGP (150 W + 25 W Dynamic Boost), verified for that exact chassis.** Thick models (Legion Pro 7i /
   Y9000P, Strix SCAR) sustain it; thin ones don't. VRAM is 16 GB on every 5080 laptop — no choice there.
2. **System RAM 32 GB minimum, 64 GB preferred** — dataloaders, DINOv2 metric passes, checkpoint loading.
3. **2 TB SSD** or a free second M.2 slot (π0.5 checkpoints are several GB each).
4. **Enough independent USB ports** for leader + follower + 2 cameras; put the two cameras on **different
   controllers/ports**, not one hub (UVC bandwidth allocation failures).
5. **Hardware MUX / dGPU-only mode** — makes Linux NVIDIA drivers far less painful.
6. **Dual-boot Ubuntu 22.04** alongside your working Windows setup: openpi officially supports Ubuntu 22.04
   only, and Blackwell needs **driver ≥ 570.144 (open kernel modules) + CUDA 12.8+**. Keep Windows-native
   LeRobot for SO-101 control (proven, COM ports); avoid WSL2 for robot I/O.

**Software recipe on the 5080 laptop:**

| Model | Train locally? | Deploy |
|---|---|---|
| ACT | ✅ bs ~32 | LeRobot PyTorch |
| Diffusion Policy | ✅ bs ~16 | LeRobot PyTorch |
| SmolVLA | ⚠️ bs 8 + grad accum — or cloud | LeRobot bf16 (~3–4 GB) |
| π0.5 | ❌ → cloud A100 (~$6–8/run) | **LeRobot PyTorch bf16 + RTC** first; TensorRT FP8/NVFP4 later. Avoid stock openpi JAX, or set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Record `torch.cuda.max_memory_allocated()` on first run (§D.0.1) |

## D.2 — Robots and motion hardware

| Item | Why | Status | Est. cost |
|---|---|---|---|
| **SO-101 leader + follower** | Project A's testbed; Project B's deployed arm | ✅ have | — |
| **2nd SO-101 follower (+ leader)** | Parallel data collection and A/B eval — and it lets **B stay permanently deployed at mom's** while A continues at your bench. That's the real reason to buy it | 🟡 **strongly consider** | ~$150–500/arm depending on kit and printed parts |
| Spare servo set + control board | Servos die mid-project; a dead servo at week 6 with a 3-week lead time is a stalled project | 🔴 | $60–150 |
| **Rigid mounting** — clamps, aluminum extrusion / steel plate, table clamps | Base rigidity is a *data quality* requirement, not a nicety | 🔴 | $50–120 |
| **Physical e-stop** (latching, cuts servo power) | Safety gate for B | 🔴 | $20–40 |
| **Arcade start button** (USB/GPIO) | The "mom can use it" requirement | 🟡 | $10–25 |
| Bigger arm — **Seeed reBot B601** (767 mm reach, 1.5 kg payload, 0.2 mm repeatability): **$1,499 assembled / $1,197 kit** | Only if the chosen household task exceeds the SO-101's payload/reach/precision. **Decide after the site survey, not before** | ⚪/🟡 | $1,197–1,499 |

### D.2.1 — Repeatability: SO-101 vs. reBot B601

| | **SO-101** (Feetech STS3215 servos) | **reBot B601-DM** (Damiao DM-J4310-2EC / 4340P) | **reBot B601-RS** (Robstride) |
|---|---|---|---|
| Official repeatability | **None published** (TheRobotStudio has no spec) | ±0.2 mm (product page; "<0.2 mm" in GitHub README) | ±0.1 mm (product page); GitHub table says <0.2 mm |
| Test method published? | — | No | No |
| Third-party figure | ±2–4 mm (SVRC comparison, no method given). Ignore the "±0.5 mm" SEO pages | — | — |
| Encoder resolution | 12-bit, 4,096/rev = 0.088°/count (≈0.4 mm per count at 0.46 m reach) | 14-bit, 16,384/rev = 0.022°/count, **output-side** ("2EC" = dual encoder) (≈0.17 mm per count at 0.77 m) | — |
| Measured joint slop | Single STS3215: **0.62° unloaded / 1.30° loaded** backlash (≈1.1 / 2.3 mm at a 100 mm lever) | Not published | Not published |

**Estimated SO-101 tip repeatability** (a Monte-Carlo model using the URDF link lengths and the measured backlash above; ISO 9283 RP = mean + 3σ distance from the centroid):

| Condition | Mid-table pose (tip ~0.26 m) | Fully extended (~0.46 m) |
|---|---|---|
| Same approach direction, same load (encoder + 1–2 count deadband) | **~±0.9 mm** | ~±1.3 mm |
| Mixed approach directions, light load | **~±3 mm** | ~±5 mm |
| Mixed directions, loaded backlash (pessimistic: gravity actually seats lift/elbow on one side) | ~±6.5 mm | ~±10 mm |

Treat the reBot's ±0.1–0.2 mm as a best case (slow motion, same direction, no load); it is about one encoder count at full reach. Even so, it is roughly an order of magnitude better than the SO-101.

**When this matters and when it doesn't:**
- **Closed-loop visuomotor policies absorb most of it.** The cameras close the loop, and demo-to-demo variance plus policy error are usually centimeter-scale. For a ≥1 cm-tolerance household task, SO-101 repeatability is not the bottleneck.
- **It matters for:**
  - Tasks with tolerance under ~5 mm (PCB into a slot, plugs, lids).
  - Open-loop replay of recorded episodes: the replay error floor is the arm's repeatability.
  - Cross-session consistency: recalibration shifts joint zeros, so joint states recorded on different days may not line up.
- **Project A:** measure the hardware noise floor once. A SAL/TED/jerk difference between episodes smaller than what the arm produces repeating the *same* command is not a data-quality signal.

**Measure your own arm (~30 min):**
1. Pick three poses: near, mid and far.
2. Command each pose 30× and alternate the approach direction.
3. Read the tip with a $15 dial indicator, or an AprilTag on the gripper seen by the top-down C920 (roughly sub-mm with calibration). A pencil tip on graph paper also works.
4. Report RP = mean + 3σ of the distance from the centroid.
5. Re-run after a recalibration to get the cross-session drift.

## D.3 — Sensing

| Item | Why | Status | Est. cost |
|---|---|---|---|
| **Logitech C922 (wrist) + C920 (top-down)** | Your v3b fix, already built and tuned (manual focus 44%, 1/120 s, 600 ISO) | ✅ have | — |
| **3rd camera (side/context)** | A third view enables the cross-camera consistency check and makes the **observability metric** in 08 §A.3 much stronger | 🟡 | $40–80 |
| **Rigid camera mounts** (articulating arms, printed brackets, thread-locked) | Camera-pose drift is your #1 historical confound. Spend real money here | 🔴 | $40–100 |
| **Fixed LED panel + diffuser** | Turns lighting from a nuisance variable into a constant for B and a *controlled* variable for A | 🔴 | $40–100 |
| Calibration targets (ChArUco/checkerboard on a rigid board) | Camera-pose fingerprinting, drift detection, UMI calibration | 🟡 | $10–30 |
| Depth camera (RealSense D405/D435) | Attacks depth ambiguity head-on — but the wrist cam already addresses it, and it's another modality to integrate | ⚪ defer | $250–350 |
| Force/tactile sensing | Would give a real "did I grasp it" signal, but that's a project of its own | ⚪ skip v1 | — |

## D.4 — Haptic teleop hardware

Moved to [08](08_data_quality_research.md) Part D (Project A, H1).

## D.5 — Fabrication & workshop

| Item | Why | Status | Est. cost |
|---|---|---|---|
| **3D printer access** | Fixtures, chamfered pockets, camera brackets, UMI gripper, rounded fingertips. You will print *constantly* — jigs are how you convert a precision problem into a geometry problem (your own 00 talking point) | **Noisebridge** ✅ | — |
| **Own printer** (Bambu A1 mini ≈ $200–250 · A1 ≈ $350–400 · P1S ≈ $600–700) | Iteration speed: a 2-hour turnaround at home beats a trip to the space. Worth it *only* if you're printing weekly | 🟡 | $200–700 |
| Filament (PLA + PETG + TPU) | TPU for compliant fingertips — a soft fingertip buys more grasp tolerance than most training changes | 🟡 | $60–120 |
| **Calipers, tape measure, small square, level** | The site survey and every fixture | 🔴 | $30–60 |
| Bench tools: hex drivers, thread-locker, zip ties, cable management, clamps | Rigidity and repeatability | 🔴 | $50–100 |
| Marking supplies: matte tape, non-reflective mat, printed position grids | Eval positions, workspace boundaries, glare control | 🔴 | $20–40 |
| Label maker | Cable/port/session labeling — trivial-sounding, saves hours | ⚪ | $25 |

## D.6 — Software & services

| Item | Status | Cost |
|---|---|---|
| LeRobot (+ your local Windows patches), PyTorch, DINOv2, openpi | ✅ | free |
| Hugging Face Hub (private datasets + checkpoints); Pro for more storage | ✅ / 🟡 | free–$9/mo |
| Weights & Biases | ✅ | free (personal) |
| **Isaac Sim 5.x + Isaac Lab + [LeIsaac](https://github.com/LightwheelAI/leisaac)** (sim track; LIBERO via `lerobot-eval` as fallback) | 🔴 week 2 | free |
| DuckDB + Parquet (metrics store) | 🔴 week 2 | free |
| Git + DVC (or plain HF revisions) for dataset versioning | 🟡 | free |
| Cloud storage backup | 🔴 | ~$5–10/mo |

## D.7 — Budget tiers

| Tier | What you get | Spend |
|---|---|---|
| **Minimum** (cloud-only, no new compute) | Mounts, lighting, e-stop, tools, spares, backup, cloud credits. Both projects still run — but **π0.5 can never be evaluated on the real arm**, because the 3060's 6 GB can't hold it | **~$500–800** |
| **Recommended** | Minimum **+ ¥8,000 used 3090 (24 GB) + 2nd SO-101 + deployment box for B** | **~$2,000–3,000** |
| **Maximal** | Recommended + a **second 3090** (parallel experiments) + own 3D printer + depth camera + a larger arm (reBot) if the site survey demands it | **~$4,000–5,000** |

Note what *dropped out* versus the first draft: the ¥46,000 5090 and the $3,500–4,200 GPU laptop. At
current pricing neither earns its place — together they'd consume more than the entire "maximal" budget
while unlocking nothing a ¥8,000 3090 plus cloud credits doesn't already cover.

**If you buy only three things:** (1) **rigid camera/arm mounts and a fixed lamp** — they protect every
measurement you will ever take; (2) **the ¥8,000 3090**, which is the difference between "π0.5 is in the
study" and "π0.5 is a citation"; (3) **a second SO-101**, so B can stay deployed at mom's while A keeps
running at your bench.

---

# Part E — How the two projects feed each other

- B's home setup is **environment #9** — a real, uncontrolled, lived-in scene, and perfect A test data.
- A's metrics are B's **quality gate**: score each collection session before training and catch the bad
  session the same day, instead of after a 20k-step run.
- B's reliability harness (logging, drift check, one-button runs) *is* A's eval harness.
- B's failure clusters are A's ground-truth labels for "bad episodes" — A's scarcest resource.
- And the combined story is a strong one: *"I built the measurement, then used it to ship something a
  person actually uses."*

---

## Immediately next

1. **B1 — ask your mom** what actually annoys her, then do the site survey with a tape measure. Write the
   numbers into §B.2 of this file.
