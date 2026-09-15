# 05 — Eval Protocol + Design Decisions (your Chelsea centerpiece)

The hands-on isn't "done" when the arm moves — it's done when you can state, honestly and with error
bars, *how well* it works and *why you built it this way*. This doc makes the SO-101 build concrete
enough to defend in the design round. Pairs with `../onsite_prep/B_design_chelsea.md`.

---

## Part 1 — A concrete eval protocol for "pick the blue PCB, seat it in the fixture"

Don't eyeball "it kinda works." Run this:

**Fixed initial-condition set.** Pick **N = 20** PCB start positions on the marked ±1 cm plane and mark
them (a printed grid / logged photos) so they're reproducible. Split: **~14 "seen"** positions inside the
trained ±1 cm range + **~6 "unseen"** positions at/just beyond the boundary (≈±1.5 cm) → you get an
in-distribution *and* an extrapolation number from one session. (Fixture is fixed, so the only variable
is PCB xy.)

**Run matched + interleaved.** When comparing two checkpoints (or fine-tuned vs base), run **both on the
same 20 positions**, interleaved (A,B,A,B) to cancel session drift. Same PCB, same lighting, same
instruction string.

**Pre-registered, staged success criterion** (graded beats binary — more signal per rollout):
| Stage | Credit |
|---|---|
| reached the PCB | 0.20 |
| grasped + lifted (stable, no tilt/slip) | 0.40 |
| moved over the pocket | 0.60 |
| dropped into the pocket | 0.80 |
| seated flat + released within 30 s (incl. successful poke-adjust) | 1.00 |
Log the stage reached for every rollout, plus **#pokes/interventions** and time. (Grip slip and
failed-seating are your two expected failure modes — track which one dominates.)

**Blind scoring.** Decide success by the pre-registered rule; ideally have someone else (or a re-watched
video) score without knowing which checkpoint it was. Log video + actions + obs for every rollout.

**Report like a scientist** (not "70%"):
> "Seen positions: 10/14 full success (Wilson 95% CI [48%, 87%]); unseen (±1.5 cm): 1/6; mean staged
> score 0.71; 4 pokes total; same 20 marked positions for both policies."

**Wilson CI cheat (small-N binomial, better than ±1.96·SE near 0/1):**
```
p̂ = k/n ; z=1.96
center = (p̂ + z²/2n) / (1 + z²/n)
half   = z/(1+z²/n) · sqrt( p̂(1-p̂)/n + z²/4n² )
CI = center ± half
```
For 9/12 → ≈ [0.49, 0.91]. The point you want to *say out loud*: **with N≈10–20, only large gaps are
real; I wouldn't claim a 10-point improvement without matched conditions or more trials.**

**Closed-loop is the metric.** The FM validation loss from training is only for picking checkpoints to
*then* roll out. If val loss disagrees with closed-loop success, trust the robot.

---

## Part 2 — Design-decisions log (fill this as you build; it's your system-design answer)

For each decision: **choice → why → what you'd ablate if you had more robot time.** This turns "I ran a
tutorial" into "I designed a system."

| Decision | Your choice (v1) | Why | Ablation you'd run |
|---|---|---|---|
| Pretrained vs scratch | fine-tune `smolvla_base` | inherit VLM + cross-embodiment priors; far less data | scratch (expect much worse) |
| Freeze backbone? | freeze VLM, train action expert | fits 6 GB; less overfit on tiny data | unfreeze on cloud, compare |
| Action space | whatever the SO-101 config uses (joint targets) | matches teleop recording | EE-delta (may generalize over position better) |
| Chunk size | config default | temporal coherence vs reactivity | sweep {1, default, 2×} — expect chunking smoother, fewer freezes |
| Data coverage | vary PCB **xy ±1 cm**, fix rotation/fixture/instance | that's the only axis I claim to generalize | add PCB rotation or fixture-position variation, re-eval |
| Data amount | 50 clean episodes | consistent > many-sloppy; longer/precise task | learning curve: 20/35/50 eps vs success |
| Denoise steps @ deploy | ~10 Euler | rectified-flow paths near-straight → few steps suffice | sweep 5/10/20; try Heun for fewer steps |
| Normalization | per-dim stats from *this* dataset | isotropic FM noise assumes comparable scales | (bug guard, not really an ablation) |

**Design talking point:** "I only collected position diversity because position was my sole
generalization claim — spending the data budget on lighting I wasn't going to evaluate would've been
waste. If eval showed a lighting sensitivity, *then* I'd add that axis."

---

## Part 3 — The flywheel (how eval decides the next data)

1. Cluster the failures from the eval videos (e.g. "grip slips when the PCB sits at the far corner of the
   ±1 cm range" / "misses the pocket to one side" / "seats only after 2+ pokes").
2. That's a **covariate-shift diagnosis**: the deployment distribution has states your data underrepresents.
3. **Collect targeted demos in exactly those states** (DAgger-flavored), retrain, re-eval on the *same*
   marked positions → paired comparison shows whether the gap closed.
4. Repeat. Each turn of the loop is a small, honest experiment.

**Design talking point:** "Eval isn't the end of the pipeline, it's the input to the next data
collection. I'd let the failure clusters — not my guesses — allocate the next data budget."

---

## Part 4 — Map every stage to an interview answer

| You did / debugged | Round it feeds | The line |
|---|---|---|
| scoped task + generalization axis | Chelsea (design) | "coverage on the axis I claim, nothing else" |
| matched-condition, staged, CI'd eval | Chelsea (eval methodology) | "N≈20 only resolves big gaps; I pair conditions" |
| closed-loop > val loss | Chelsea | "val loss picks checkpoints; the robot decides" |
| OOM / loss-flat / NaN while training | Michael (debugging) | "overfit one batch; check trainable params" |
| normalization / chunk-jerk / latency on robot | Michael (robot debugging) | "log raw action; open-loop replay; RTC for the seam" |
| read `VLAFlowMatching`, built it from scratch | Adrian / Karol | "SmolVLA is the fm_loss + sampler I wrote, with a VLM prefix" |
