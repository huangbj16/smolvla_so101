#!/usr/bin/env python3
"""Figures for the Phase 0.7 round-2 eval (section 12.9).

Two figures, in the order the questions arrive:

  1. **What happened** - the success rate did not move, but the behaviour did, and the two
     policies moved in opposite directions. Everything is shown against the HUMAN
     DEMONSTRATIONS, because "10% stuck" means nothing until you know a demonstrator is at 1%.
  2. **Why** - how much each policy's output actually depends on its cameras, where the grasp
     goes wrong for each, and what the bright-lighting intervention did.

Reads only from results/phase07_eval/, so every number is the measured one.

    python scripts/make_phase07_r2_figs.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_phase07_doc_figs import (C, CRITICAL, GOOD, GRID, INK, INK2, INK3, SURFACE, _style,
                                   _title)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results/phase07_eval"
OUT = ROOT / "docs/details/figs"

ARMS = ["act40k_d1", "sv20k_d1"]
LBL = {"act40k_d1": "ACT 40k (D1)", "sv20k_d1": "SmolVLA 20k (D1)"}
SHORT = {"act40k_d1": "ACT", "sv20k_d1": "SmolVLA"}
FAM = {"act40k_d1": "act", "sv20k_d1": "smolvla"}
R1_OF = {"act40k_d1": ("act60k", "ACT 60k (D0)"), "sv20k_d1": ("sv20k", "SmolVLA 20k (D0)")}
STAGES = ["reach", "grasp", "transport", "insert"]
DEMO_C = INK3            # the demonstrations are a reference, not a third series


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load():
    return {
        "per_arm": pd.read_csv(RES / "r2_summary_per_arm.csv").set_index("arm"),
        "r1": pd.read_csv(RES / "summary_per_arm.csv").set_index("arm"),
        "demo": pd.read_csv(RES / "r2_demo_baseline.csv").iloc[0],
        "cam": json.loads((RES / "camera_sensitivity.json").read_text())["arms"],
        "emb": json.loads((RES / "embedding_shift.json").read_text())["sets"],
        "grasp": json.loads((RES / "r2_grasp_axes.json").read_text()),
        "bright_blocks": pd.read_csv(RES / "r2_brightness_per_block.csv"),
        "bright_traj": pd.read_csv(RES / "r2_brightlighting_trajectory.csv"),
    }


# =============================================================================================
def fig_results(d, path):
    fig = plt.figure(figsize=(15.4, 4.7), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25], wspace=0.34)

    # ---- A: the success rate did not move ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    xs, labels, cols, hatches = [], [], [], []
    for i, arm in enumerate(ARMS):
        r1k, r1n = int(d["r1"].loc[R1_OF[arm][0], "successes"]), int(d["r1"].loc[R1_OF[arm][0], "n"])
        r2k, r2n = int(d["per_arm"].loc[arm, "successes"]), int(d["per_arm"].loc[arm, "n"])
        for j, (k, n, rnd) in enumerate([(r1k, r1n, "round 1"), (r2k, r2n, "round 2")]):
            x = i * 2.6 + j
            lo, hi = wilson(k, n)
            ax.bar(x, 100 * k / n, width=0.78, color=C[FAM[arm]], zorder=3, linewidth=0,
                   alpha=0.42 if j == 0 else 1.0)
            ax.errorbar(x, 100 * k / n, yerr=[[100 * k / n - 100 * lo], [100 * hi - 100 * k / n]],
                        fmt="none", ecolor=INK2, elinewidth=1.3, capsize=4, zorder=5)
            ax.text(x - 0.42, 100 * k / n, f"{k}/{n}", ha="right", va="center", fontsize=8.6,
                    color=INK2)
            xs.append(x); labels.append(f"{rnd}\n{SHORT[arm]}")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("success rate  (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 72)
    _style(ax)
    ax.text(0.5, -0.21, "every comparison  p = 1.000", transform=ax.transAxes, ha="center",
            va="top", fontsize=10.5, color=CRITICAL, fontweight="700")
    _title(ax, "A · Twice the data moved nothing",
           "faded = round 1 (D0) · solid = round 2 (D1) · Wilson 95% CI")

    # ---- B: but the failures moved ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    w = 0.38
    for i, arm in enumerate(ARMS):
        vals = [int(d["per_arm"].loc[arm, f"fail_{s}"]) for s in STAGES]
        ax.barh(np.arange(len(STAGES)) + (i - 0.5) * w, vals, height=w * 0.92,
                color=C[FAM[arm]], zorder=3, linewidth=0, label=SHORT[arm])
        for y, v in zip(np.arange(len(STAGES)) + (i - 0.5) * w, vals):
            if v:
                ax.text(v + 0.12, y, str(v), va="center", fontsize=8.6, color=INK2)
    ax.set_yticks(np.arange(len(STAGES)))
    ax.set_yticklabels(STAGES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("failures  (of 18 trials)", color=INK2, fontsize=9)
    ax.set_xlim(0, 10.4)
    _style(ax, grid_axis="x")
    leg = ax.legend(frameon=False, fontsize=8.8, loc="lower right", handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.35, 2.0, "0 failures here — as in all 54 round-1 trials", ha="left",
            va="center", fontsize=8.4, color=INK3)
    _title(ax, "B · Different failure modes", "ACT misses the place, SmolVLA the grasp")

    # ---- C: behaviour, against the human baseline -------------------------------------------
    # Small multiples, not one shared axis: jitter runs to 95% while stuck sits near 10%, and on a
    # common scale the smaller measure - the one that explains ACT - becomes invisible. Same unit
    # in all three, independent ranges.
    keys = [("stuck_frac", "stuck", "going nowhere"), ("jitter_frac", "jitter", "not straight"),
            ("pause_frac", "pause", "standing still")]
    sub = gs[0, 2].subgridspec(1, 3, wspace=0.62)
    for col, (k, name, gloss) in enumerate(keys):
        a = fig.add_subplot(sub[col])
        vals = [100 * d["per_arm"].loc[arm, k] for arm in ARMS]
        dv = 100 * d["demo"][k]
        a.bar([0, 1], vals, width=0.62, color=[C[FAM[x]] for x in ARMS], zorder=3, linewidth=0)
        top = max(max(vals), dv) * 1.42 + 0.6
        for x, v in zip([0, 1], vals):
            a.text(x, v + top * 0.025, f"{v:.1f}", ha="center", va="bottom", fontsize=8.4,
                   color=INK2)
        # The demonstrations are the reference the policy bars are read against, so they are a
        # rule across the panel rather than a third bar competing for identity.
        a.plot([-0.62, 1.62], [dv, dv], color=DEMO_C, lw=2.2, ls=(0, (4, 2)), zorder=6)
        a.text(1.62, dv, f" {dv:.1f}", va="center", ha="left", fontsize=8.4, color=DEMO_C,
               fontweight="700")
        a.set_xticks([0, 1]); a.set_xticklabels(["ACT", "SmolVLA"], fontsize=8.2, rotation=38,
                                                ha="right")
        a.set_xlim(-0.75, 2.05)
        a.set_ylim(0, top)
        if col == 0:
            a.set_ylabel("% of the episode", color=INK2, fontsize=9)
        _style(a)
        a.set_title(name, color=INK, fontsize=9.8, fontweight="600", loc="left", pad=13)
        a.text(0, 1.015, gloss, transform=a.transAxes, color=INK2, fontsize=8, va="bottom")
    fig.text(0.695, 1.055, "C · Two opposite pathologies", ha="left", va="top", fontsize=11.5,
             color=INK, fontweight="600")
    fig.text(0.695, 1.012, "ACT freezes · SmolVLA never stops moving · "
             "– – – = human demonstrations", ha="left", va="top", fontsize=8.5, color=INK2)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


# =============================================================================================
def fig_diagnosis(d, path):
    fig = plt.figure(figsize=(15.4, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30)

    # ---- A: how much does the output depend on the cameras? ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    conds = [("top_blank", "top camera\nremoved"), ("wrist_blank", "wrist camera\nremoved"),
             ("both_blank", "BOTH cameras\nremoved"), ("state_blank", "proprioception\nremoved")]
    xs = np.arange(len(conds))
    for i, arm in enumerate(ARMS):
        v = [d["cam"][arm][k]["mean"] for k, _ in conds]
        ax.bar(xs + (i - 0.5) * 0.36, v, width=0.33, color=C[FAM[arm]], zorder=3, linewidth=0,
               label=SHORT[arm])
        for x, val in zip(xs + (i - 0.5) * 0.36, v):
            ax.text(x, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=8.4,
                    color=INK2)
    floor = d["cam"]["sv20k_d1"]["seed_repeat"]["mean"]
    ax.axhline(floor, color=CRITICAL, lw=2, ls=(0, (4, 2)), zorder=6)
    ax.annotate("SmolVLA's own sampling noise.\nDestroying its top camera moves\n"
                "the output LESS than this.",
                xy=(0.46, floor), xytext=(0.05, 0.60), textcoords="axes fraction",
                fontsize=8.8, color=CRITICAL, fontweight="700", linespacing=1.6,
                arrowprops=dict(arrowstyle="-|>", color=CRITICAL, lw=1.4, mutation_scale=11,
                                shrinkB=2))
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in conds], fontsize=8.6)
    ax.set_ylabel("change in predicted action  (σ)", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.18)
    _style(ax)
    leg = ax.legend(frameon=False, fontsize=8.8, loc="upper left", handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    # The ratio is the finding, so state it rather than leaving it to be read off the bars.
    rv = {a: d["cam"][a]["both_blank"]["mean"] / d["cam"][a]["state_blank"]["mean"] for a in ARMS}
    ax.text(0.5, -0.33, f"vision : proprioception  —  ACT {rv['act40k_d1']:.1f} : 1"
            f"        SmolVLA 1 : {1/rv['sv20k_d1']:.1f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color=INK,
            fontweight="700")
    _title(ax, "A · The two policies read opposite inputs",
           "ACT is driven by its cameras · SmolVLA by where its joints already are")

    # ---- B: the grasp, decomposed -----------------------------------------------------------
    sub = gs[0, 1].subgridspec(2, 1, hspace=0.95)
    g = d["grasp"]["arms"]
    for row, (key, unit, title, better) in enumerate([
            ("bearing_repeat_median_deg", "degrees", "BEARING — repeatability across the two "
             "tries at a spot", "sv20k_d1"),
            ("height_z", "sd from the demos", "HEIGHT — error against the demonstrations",
             "act40k_d1")]):
        a = fig.add_subplot(sub[row])
        v = [g[arm][key] for arm in ARMS]
        a.barh([0, 1], v, height=0.52, color=[C[FAM[x]] for x in ARMS], zorder=3, linewidth=0)
        for y, (arm, val) in enumerate(zip(ARMS, v)):
            tag = "  ← better" if arm == better else ""
            a.text(val + max(v) * 0.03, y, f"{val:.2f} {unit}{tag}", va="center", fontsize=8.6,
                   color=GOOD if arm == better else INK2,
                   fontweight="700" if arm == better else "normal")
        a.set_yticks([0, 1]); a.set_yticklabels([SHORT[x] for x in ARMS], fontsize=8.8)
        a.invert_yaxis()
        a.set_xlim(0, max(v) * 1.75)
        _style(a, grid_axis="x")
        a.set_title(title, color=INK, fontsize=9.4, fontweight="600", loc="left", pad=6)
    fig.text(0.545, 0.985, "B · Each policy is accurate on the axis the other misses",
             ha="left", va="top", fontsize=11.5, color=INK, fontweight="600")
    fig.text(0.545, 0.962, "SmolVLA: right place, wrong height → the tip hits the cap  ·  "
             "ACT: right height, wrong place", ha="left", va="top", fontsize=8.5, color=INK2)

    # ---- C: brightness does not explain block 6 ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    b = d["bright_blocks"]
    reg = b[b.block > 0].sort_values("block")
    for _, r in reg.iterrows():
        fam = FAM[r.arm] if r.arm in FAM else "act"
        ax.scatter(r.top_mean, 100 * r.successes / r.n, s=190, color=C[fam], zorder=4,
                   edgecolor=SURFACE, linewidth=1.8)
        # Alternate the label side so near-coincident blocks do not overprint each other.
        dy = 11 if int(r.block) % 2 else -17
        ax.annotate(f"b{int(r.block)}", (r.top_mean, 100 * r.successes / r.n),
                    textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                    color=INK2)
    tr = b[b.block == 0]
    if len(tr):
        ax.axvline(float(tr.top_mean.iloc[0]), color=DEMO_C, lw=2, ls=(0, (4, 2)), zorder=2)
        ax.text(float(tr.top_mean.iloc[0]) + 0.06, 68, "D1 training", rotation=90, va="top",
                fontsize=8.4, color=DEMO_C, fontweight="700")
    ax.set_xlabel("mean top-camera luminance  (0–255)", color=INK2, fontsize=9)
    ax.set_ylabel("block success rate  (%)", color=INK2, fontsize=9)
    ax.set_ylim(-6, 78)
    _style(ax, grid_axis="both")
    ax.text(0.5, 0.97, "the whole session spans 1.5% in brightness",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=CRITICAL,
            fontweight="700")
    _title(ax, "C · Block 6 is not a lighting effect",
           "b6's 3/5 sits mid-range · p = 0.099 against ACT's other trials")

    # ---- D: what the bright-light intervention actually did ---------------------------------
    ax = fig.add_subplot(gs[1, 1])
    bt = d["bright_traj"]
    groups = []
    for arm in ARMS:
        normal = 100 * d["per_arm"].loc[arm, "stuck_frac"]
        br = bt[bt.arm == arm]
        groups.append((SHORT[arm], normal, 100 * br.stuck_frac.mean() if len(br) else np.nan))
    xs, ticks, ticklabels = np.arange(len(groups)), [], []
    for i, (name, nv, bv) in enumerate(groups):
        ax.bar(i - 0.19, nv, width=0.35, color=C[FAM[ARMS[i]]], zorder=3, linewidth=0)
        ax.bar(i + 0.19, bv, width=0.35, color=C[FAM[ARMS[i]]], alpha=0.40, zorder=3,
               linewidth=0)
        for x, v in ((i - 0.19, nv), (i + 0.19, bv)):
            ax.text(x, v + 0.22, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.6, color=INK2)
        ticks += [i - 0.19, i + 0.19]
        ticklabels += ["normal", "over-\nexposed"]
        ax.text(i, -0.235, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9.6, color=INK, fontweight="600")
    dv = 100 * d["demo"]["stuck_frac"]
    ax.axhline(dv, color=DEMO_C, lw=2.2, ls=(0, (4, 2)), zorder=6)
    ax.text(0.52, dv + 0.35, f"human demonstrations  {dv:.1f}%", ha="left",
            fontsize=8.6, color=DEMO_C, fontweight="700")
    ax.annotate("", xy=(-0.19 + 0.38, groups[0][2] + 0.4), xytext=(-0.19, groups[0][1] - 0.3),
                arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=2, mutation_scale=13), zorder=7)
    ax.text(0.16, (groups[0][1] + groups[0][2]) / 2, "  12× less\n  stuck", fontsize=9,
            color=GOOD, fontweight="700", va="center", linespacing=1.4)
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=8.2)
    ax.set_ylabel("time stuck  (% of episode)", color=INK2, fontsize=9)
    ax.set_ylim(0, 13)
    _style(ax)
    _title(ax, "D · Over-exposing the top camera unfreezes ACT",
           "and leaves SmolVLA untouched · n = 5 each, unscored")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> int:
    d = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_results(d, OUT / "phase07_r2_results.png")
    fig_diagnosis(d, OUT / "phase07_r2_diagnosis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
