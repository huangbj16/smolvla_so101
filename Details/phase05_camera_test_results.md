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

## Follow-ups considered

See [08 §A.4](../08_data_quality_research.md) for the list and what was chosen.
