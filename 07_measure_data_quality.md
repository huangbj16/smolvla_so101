# 07 — Measuring data quality on your PCB dataset (state diversity, action divergence, transition diversity)

This turns the theory from *Data Quality in Imitation Learning* (Belkhale et al.) into numbers you
run on your own LeRobot parquet files. The companion notebook is
[measure_data_quality.ipynb](measure_data_quality.ipynb).

> **Dataset:** `HALDijkstraaa/so101_pick_place_pcb_20260721_183543` (v3.0, **161 episodes /
> 111,776 frames**, verified consistent across info/data/meta). 6-DoF SO-101, one `front` camera.

---

## The one idea everything rests on

Embed every frame into a **state vector**, then all three metrics are simple statistics on those
vectors + the recorded actions. We build **two** state representations on purpose, because comparing
them is the experiment:

| Representation | How | Distance | What it captures |
|---|---|---|---|
| **Visual state** | image → frozen **DINOv2** embedding | cosine | *what the policy can see* (occlusion lives here) |
| **Robot state** | the 6-DoF **joint vector** (normalized) | Euclidean | *the true physical configuration* (occlusion-blind) |
| **Action** | recorded 6-DoF joint target, **per-dim normalized** | Euclidean | what the demonstrator did |

**The decisive design choice:** measure **action divergence in the *visual* space** (what the policy
conditions on), and use the **robot-state** version as a *control*. Your occlusion problem only exists
in the image — so it must show up in visual-space divergence and *vanish* in robot-state divergence.
That contrast is a fingerprint of observation-space information loss, not physical ambiguity.

---

## Section 1 — Load & preprocess (run once, then cached)

- Read all 7 data parquets → `action`, `observation.state`, `episode_index`, `frame_index`.
- **Subsample** ~40 frames/episode (uniform stride) → ~6.4k frames — enough for stable statistics,
  cheap enough to embed on the GPU in a couple of minutes.
- For each sampled frame: decode its image (LeRobot + PyAV), run **DINOv2-small** → 384-d embedding;
  keep the 6-DoF joint state; keep the action.
- **Normalize actions per-dim** by dataset std (so no single joint dominates the variance — this is the
  paper's whole normalization point).
- Save everything to a single `.npz` cache so Sections 2–4 never touch the parquet/video again.

**Group labels for comparison:** you tag episode-index ranges as `good_angle` vs `upfront_camera`
(edit the dict at the top). Every downstream plot can compare any set of groups.

## Section 2 — State diversity ("how much ground does the data cover?")

- **Visual diversity (DINOScore):** mean pairwise **cosine distance** of DINOv2 embeddings. ~0 = all
  frames look alike; →1 = very diverse.
- **Robot-state diversity:** spread of the normalized joint vectors (mean pairwise distance +
  effective volume via PCA).
- **Plots:** per-group bar charts + a 2-D PCA scatter (visual and robot-state) colored by group, so you
  can *see* whether the upfront-camera frames collapse into a tight visual blob.

## Section 3 — Action divergence (the headline; the heatmap)

For each frame: find its **k nearest neighbors in state space**, take the **std of their (normalized)
actions**, divide by the **global action std** → a unitless **0–1 ambiguity ratio**.
- **~0** = look-alike states → near-identical actions (consistent, learnable).
- **→1** = look-alike states → different actions (multimodal/divergent → the mushy, ~1 cm-off grasp).

Computed **twice**: once with neighbors in **visual (DINOv2)** space, once in **robot-state** space.

**The 2-D heatmap you asked for:** rows = episodes, columns = normalized time (0→100 % of the episode),
color = local action variance. You'll see *when* in the motion the divergence spikes (expect the
reach/grasp phase) and *which* episodes are worst (expect the upfront-camera block to light up in the
**visual** heatmap but not the **robot-state** heatmap).

## Section 4 — Transition diversity (the honest, actionable proxy)

True transition diversity (spread of next-state given state+action) is near-zero for a
position-controlled arm, so we measure the version the theory actually endorses as the safe-coverage
lever: **initial-state spread** — the variance of the **first-frame state** across episodes, per group.
Higher = more varied starting conditions = better coverage without raising action divergence.

---

## What to expect (the hypothesis this notebook tests)

1. **Visual-space action divergence is higher for the upfront-camera episodes** (occluded PCB → frames
   look identical but actions differ), and the heatmap localizes it to the grasp phase.
2. **Robot-state action divergence does *not* single out those episodes** — proving the problem is
   observation-space information loss (camera geometry), not inconsistent teleoperation.
3. Visual **state diversity** for the upfront group is *deceptively low* (collapsed embeddings), a
   reminder that raw "state diversity" is a coarse quality signal (paper's core caveat).

If (1) and (2) hold, you've converted "the camera couldn't see the PCB" into a measured result — and
the fix (wrist camera / better geometry) is aimed at the right axis.

---

## Next
Feeds the [06 doc](06_expand_data_multi_env.md) results log and the reading plan's CUPID step (CUPID is
this same "which data helps" question, made causal via influence functions). If the metrics confirm the
split, drop/refilm the flagged episodes before scaling to the full 8-env plan.
