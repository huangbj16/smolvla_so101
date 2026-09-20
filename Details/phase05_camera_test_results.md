# Phase 0.5 results — does the wrist camera help? (C1)

Run 2026-09-19 with [phase05_second_camera.ipynb](../phase05_second_camera.ipynb) on
`HALDijkstraaa/so101_toolkit_cylinder_20260917_165544` (50 clean episodes, 4000 sampled frames after
trimming idle frames). Hypotheses and method: [08 §A.4](../08_data_quality_research.md).

Action divergence = spread of the next motion among a frame's k nearest neighbors **from other episodes**,
divided by the spread over all frames. Lower = the observation tells apart states that need different actions.

## Headline numbers (motion target)

| Space | k=5 | k=10 | k=20 |
|---|---|---|---|
| joint state | 0.457 | 0.520 | 0.569 |
| top | 0.544 | 0.641 | 0.711 |
| wrist | 0.464 | 0.522 | 0.576 |
| **top+wrist** | **0.440** | **0.499** | **0.553** |
| top + shuffled wrist (control) | 0.750 | 0.834 | 0.884 |

By phase (k = 10):

| Space | reach | grasp | transport | insert | retreat |
|---|---|---|---|---|---|
| state | 0.580 | 0.531 | 0.465 | 0.485 | 0.499 |
| top | 0.742 | 0.477 | 0.605 | 0.484 | 0.620 |
| wrist | 0.562 | 0.471 | 0.497 | 0.482 | 0.517 |
| top+wrist | 0.553 | 0.469 | 0.440 | 0.494 | 0.494 |

Aliasing rate (fraction of a frame's 10 top-camera neighbors that are far in wrist space *and* have a
different next motion): 0.6% overall, at most 1.0% in any phase. Visual diversity: wrist 0.468, top 0.186.

## Verdicts

| | Hypothesis | Verdict |
|---|---|---|
| **H-c** | top+wrist beats top, beyond the control | **Confirmed.** Passes at every k, both conditions, bootstrap intervals well clear of 0. Gain over top: 0.142 at k=10, CI [0.127, 0.156] |
| **H-f** | wrist frames more varied | **Confirmed** (descriptive): 0.468 vs 0.186 |
| **H-a** | wrist beats top in grasp/insert | **Not supported.** The direction is right but the gaps are 0.006 and 0.002 — noise. The wrist camera's real advantage is in reach, transport and retreat (0.10–0.18) |
| **H-b** | top beats wrist in reach | **Refuted, and backwards.** In reach, wrist 0.562 vs top 0.742 |
| **H-d** | aliasing concentrated in grasp/insert | **Not supported quantitatively.** If top and wrist distances were unrelated, the rate would be ~25% by construction (two median splits). At 0.6% the top camera's neighbors are almost always genuinely similar in wrist space too. The example pairs are real and instructive, but rare |
| **H-e** | joint state most ambiguous at the start | **Refuted.** State is the *strongest* single space overall (0.520 vs top 0.641 at k=10) and beats the top camera in reach (0.580 vs 0.742) |

**C1 answer: yes, the second camera helps** — but not for the reason the hypotheses assumed. The wrist camera
is the strong view and the top camera is the weak one, almost everywhere in the task.

## Why the results came out this way

- **The top camera is the aliased view, not the wrist camera.** Its diversity is 0.186 against the wrist's
  0.468: from above, the arm and cylinder are a small part of a mostly static scene, so a global DINOv2
  embedding is dominated by the unchanging table and fixture. Frames from different moments end up close
  together. The wrist image changes completely as the arm moves, so it encodes arm pose *and* the local
  geometry, which is most of what sets the next motion.
- **Why top+wrist adds so little over wrist alone (0.023 at k=10).** Concatenating two unit-length
  embeddings makes the distance the *average* of the two cameras' distances, so the weak view gets equal
  weight and dilutes the strong one. On top of that, the wrist view already carries much of what the top
  view knows (the arm's pose implies where the cylinder is, once the operator has aimed at it). A weighted
  combination would show how much of the loss is the equal weighting.
- **There is a floor that no camera can cross.** These are human demos: from the same observation the
  operator can legitimately move at different speeds or along slightly different paths. Around 0.44–0.50
  may be mostly that irreducible spread, not missing information. Nothing in this run estimates the floor,
  so "0.499 is good" is currently unjudgeable.
- **The phase comparison is confounded by action scale** (your observation, and I think it's right).
  Divergence is normalized by the spread of motions over the *whole dataset*. In grasp and insert the arm
  barely moves, so the local spread is small for every space, and every curve drops. That is why the top
  camera looks as good as the wrist there. Phase conclusions (H-a, H-b) should be treated as provisional
  until the metric is normalized within each phase.
- **Aliasing is rare but the examples are informative.** Motion blur makes frames from different phases look
  alike, and a grasped cylinder is a few pixels from above. That suggests the embedding, not the camera,
  limits what the metric can see: a single frame at 640×480 through a small global embedding.

## Caveats

- One operator, one session, clean demos only: this measures observability, not operator inconsistency.
- No per-episode position log, so nothing here separates the 10 cylinder positions.
- DINOv2-small global embeddings; no temporal context; equal weighting in every concatenation.

---

# Follow-up run (2026-09-20, notebook Sections 9–13)

## 9 — Divergence normalized within each phase

Normalizing by the motion spread *inside* each phase, and restricting neighbors to the same phase, removes
the scale artifact that made Section 5 unreadable. Random baseline ≈ 0.90–0.94.

| Space (same-phase neighbors, k=10) | reach | grasp | transport | insert | retreat |
|---|---|---|---|---|---|
| top | **0.514** | 0.897 | **0.436** | 0.937 | 0.504 |
| wrist | 0.538 | **0.868** | 0.529 | **0.898** | 0.503 |
| top+wrist | 0.523 | 0.868 | 0.442 | 0.907 | 0.495 |
| state | 0.510 | 0.839 | 0.399 | 0.896 | 0.517 |
| random | 0.926 | 0.901 | 0.917 | 0.924 | 0.944 |

- **Each camera wins where it should.** Top is better in reach (−0.024) and transport (−0.093); wrist is
  better in grasp (−0.029) and insert (−0.039). This is **H-a and H-b, both supported** once the metric is
  phase-normalized — the opposite of the Section 5 reading, which was dominated by how much the arm moves.
  Section 9c plots this zoomed in.
- **In grasp and insert, every space is near the random baseline.** No observation — camera, both cameras,
  or joint state — predicts the next motion there.

### What "near random" means, and the guideline that follows

It means: given everything the robot can observe, the demonstrations at that instant move in **different
directions across episodes**. Three causes, not separable with this data:
1. genuine multi-modality — several acceptable ways to close the last millimetres;
2. small-amplitude corrections and jitter, which dominate when the overall motion is small;
3. the observation genuinely lacking the mm-level detail (the original H-a idea).

**Guideline for future data collection (derived from this):**
- **The fine phases need the most demonstrations and the most discipline.** Reach and transport are already
  near-deterministic given an observation; extra episodes there add little. Grasp and insert are where a
  policy has to guess.
- **Standardize the fine phases**: one approach direction, one closing speed, no exploratory wiggling. The
  task card's "one grasp style" rule should be enforced hardest in the last second before contact and before
  release.
- **When adding episodes, add them for the fine phases** — more positions, deliberate slow approach — rather
  than more full episodes of everything.
- **Track per-phase divergence as a quality metric** for each new batch. It is the only number here that
  changed when the protocol changed.
- Open check before over-trusting this: recompute with a longer action horizon (e.g. H = 30 instead of 10).
  If divergence falls a lot, cause 2 (jitter) dominates; if it stays, causes 1 and 3 do.

### Why joint state is the strongest single space

State is lowest in almost every phase (0.399 in transport vs 0.436 for top). Partly real, partly a measuring
artifact:

- **Real:** the task is stereotyped and there is one operator. The arm's pose says where the task is and what
  comes next; the future command is the same signal, one third of a second later. The cameras have to infer
  what the joints state directly.
- **Artifact of dimensionality:** state is 6-d Euclidean, the embeddings are 384-d cosine. In 6-d the 10
  nearest neighbors are genuinely almost the same pose; in 384-d distances concentrate and "nearest" is much
  farther away in relative terms. Spaces of very different dimension are therefore not strictly comparable —
  a caveat for any table that ranks them against each other.
- The formula is identical for all spaces (same k, same targets, same normalization), so it is not a
  different metric, only a different geometry. A fair check would reduce the image embeddings to ~6-d (PCA)
  before the comparison. Not done yet.

## 10 — Cameras on top of joint state, and weighting

| Space | k=5 | k=10 | k=20 |
|---|---|---|---|
| state | 0.468 | 0.538 | 0.587 |
| state+top | 0.444 | 0.511 | 0.567 |
| state+wrist | 0.427 | 0.480 | 0.522 |
| state+top+wrist | 0.423 | 0.475 | 0.524 |

- Cameras **do** add over proprioception: state+wrist is 11% below state at k=10, state+top+wrist 12%. Small
  in absolute terms because state is genuinely strong (above), not because the cameras are useless.
- **Weight sweep is flat**: 0.499 at equal weighting, 0.498 at the best weight (0.6 on top). So equal
  weighting was *not* why top+wrist barely beat wrist — the two views are largely redundant.
- **Decision: keep the equal 50/50 concatenation** in all further analyses.

## 11 — Temporal context (the big win)

| Space | k=10 (all neighbors) |
|---|---|
| top | 0.641 |
| top hist (decayed) | 0.508 |
| wrist | 0.522 |
| wrist hist (decayed) | 0.497 |
| top+wrist | 0.499 |
| **top+wrist hist (decayed)** | **0.479** |
| top +Δ / wrist +Δ | 0.562 / 0.507 |

- **History helps most where the single frame is weakest.** The top camera gains 0.133, nearly closing the
  gap to the wrist camera: most of its apparent weakness was not seeing *which way the task was going*.
- **Real history beats a difference vector.** `+Δ` (current frame plus direction of change) is clearly worse
  than keeping the lagged frames in their own slots. Use lagged embeddings, not deltas.
- **Which number predicts policy training?** It depends on the architecture's observation window. ACT and
  SmolVLA condition on a single timestep (plus state), and π0-style models likewise; Diffusion Policy
  typically uses 2 observation steps. So: judge data for a single-frame policy with the single-frame numbers,
  and use the history numbers to describe what is *achievable* with an architecture that looks back. When
  the target architecture is known, set `LAGS` to match its observation window.
- **Why the same-phase bars are higher than the all-neighbor bars** (they are not comparable):
  1. different denominators — all-neighbor bars are divided by the spread over the whole dataset,
     same-phase bars by the spread inside the phase, which is smaller in the fine phases;
  2. a smaller candidate pool (insert has only ~280 frames) means the nearest neighbor is farther away;
  3. it is a strictly harder question: telling apart states *within* a stage, with the easy
     between-stage separation removed.
  Compare within a colour, never across.

## 12 — Representation probe: no effect

`dinov2-base` (0.639 vs 0.641 for top) and a cropped top view (0.647) change nothing, and diversity barely
moves. The top camera's weakness is **not** encoder capacity or background clutter — it is the missing
temporal context (Section 11).

## Revised verdicts

| | Verdict after follow-ups |
|---|---|
| **H-a** wrist wins in grasp/insert | **Supported** with phase-normalized, same-phase neighbors (Section 9) |
| **H-b** top wins in reach | **Supported** in the same analysis, and strongly in transport |
| **H-c** top+wrist beats top | Still confirmed; the gain over wrist alone stays small because the views are redundant, not because of weighting |
| **H-e** state ambiguous | Still refuted — state is the strongest single space, with a dimensionality caveat |
| **H-d** aliasing | **Refuted as stated.** With temporal embeddings the aliased share drops to 0.1–0.6% and same-phase matching rises: most single-frame "aliasing" was phase confusion, not two different states that look alike |

## 13 — Aliasing with temporal embeddings

| Scan (neighbors from any phase, k=10) | same-phase neighbors | aliased | reach | grasp | transport | insert | retreat |
|---|---|---|---|---|---|---|---|
| top → wrist, single frame | 0.701 | 0.007 | 0.005 | 0.000 | 0.011 | 0.000 | 0.008 |
| top → wrist, history | **0.792** | 0.001 | 0.001 | 0.000 | 0.002 | 0.000 | 0.001 |
| wrist → top, single frame | 0.850 | 0.009 | 0.005 | 0.019 | 0.014 | 0.005 | 0.005 |
| wrist → top, history | **0.885** | 0.006 | 0.004 | 0.013 | 0.009 | 0.005 | 0.004 |

- **History makes the matches sane.** Same-phase neighbor share rises (top 0.701 → 0.792, wrist 0.850 →
  0.885) and the aliased share falls (top 0.007 → 0.001). So most of what Section 6 counted as "aliasing"
  was the single frame confusing one *stage* of the task with another, not two genuinely different states.
- **The top camera still can't separate reach from transport**, because the arm occludes the cylinder from
  that viewpoint — visible in the Section 13b pairs, where "A transport" and "B reach" look nearly identical
  from the top and differ obviously at the wrist (cylinder held between the fingers vs. empty gripper above
  the table). This is the clearest picture of what the second camera buys.
- **The wrist camera is close to a phase detector**, which is unsurprising: the phases were *defined* from the
  gripper signal, and the wrist view shows the gripper and whatever is between the fingers. Its same-phase
  rate is the highest even on single frames (0.850).

### Artifact worth knowing: the background is not constant

In the wrist → top pairs, the strongest "difference in the top camera" is often **a dog walking through the
background**, plus a hand at the frame edge and changing light. Consequences:

- **The visual-diversity numbers for the top camera partly measure the room**, not the task. Treat Section 7
  (H-f) as even softer than stated.
- **For training it is probably mild augmentation rather than harm**: the distractors are uncorrelated with
  the cylinder position and with the phase, so a policy has no incentive to key on them, and they add the
  kind of nuisance variation domain randomization aims for. It would only hurt if a distractor correlated
  with the task (e.g. someone always reaching in at the same moment).
- **For metrics it is noise that must be controlled**: it inflates top-camera distances and can push genuinely
  similar frames apart. If a future batch is used for careful measurement, either keep the background clear
  or crop the top view before embedding (Section 12 shows cropping costs nothing on divergence).

## Open questions

- Longer action horizon (H = 30) to separate jitter from genuine multi-modality in the fine phases.
- PCA-matched dimensionality before ranking state against the image spaces.
- Per-position breakdown — still blocked on a per-episode position log.
