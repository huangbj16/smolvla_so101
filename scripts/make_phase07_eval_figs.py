#!/usr/bin/env python3
"""Figures for the Phase 0.7 round-1 eval. Imported by analyze_phase07_eval.py.

Palette and mark specs follow the dataviz reference instance: categorical slots 1-3 for the
three policy families (validated all-pairs, light surface), the blue ordinal ramp for the
progress stages, status-good for success. Every bar is direct-labelled - slot 3 (aqua) sits
below 3:1 on the light surface, so the relief rule requires visible labels.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

C = {"act": "#2a78d6", "smolvla": "#eb6834", "pi05": "#1baf7a"}
STAGE_C = {"reach": "#86b6ef", "grasp": "#5598e7", "transport": "#2a78d6", "insert": "#184f95"}
SUCCESS_C = "#0ca30c"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#86857f"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
STAGES = ["reach", "grasp", "transport", "insert"]
ARMS = ["act40k", "act60k", "sv5k", "sv20k", "pi5k", "pi10k"]
ARM_LABEL = {"act40k": "ACT 40k", "act60k": "ACT 60k", "sv5k": "SmolVLA 5k",
             "sv20k": "SmolVLA 20k", "pi5k": "π0.5 5k", "pi10k": "π0.5 10k"}
ARM_POLICY = {"act40k": "act", "act60k": "act", "sv5k": "smolvla",
              "sv20k": "smolvla", "pi5k": "pi05", "pi10k": "pi05"}
POLICY_LABEL = {"act": "ACT", "smolvla": "SmolVLA", "pi05": "π0.5"}


def _style(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)


def _title(ax, title, subtitle=None):
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="600", loc="left", pad=14 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def fig_success(per_arm, df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.34})

    # --- (a) success rate, with the lucky trials split out and Wilson intervals
    ax = axes[0]
    y = np.arange(len(ARMS))[::-1]
    pa = per_arm.set_index("arm").loc[ARMS]
    clean = pa.success_rate_clean.to_numpy()
    lucky = pa.success_rate.to_numpy() - clean
    cols = [C[ARM_POLICY[a]] for a in ARMS]
    ax.barh(y, clean, height=0.6, color=cols, zorder=3)
    ax.barh(y, lucky, left=clean, height=0.6, color=cols, alpha=0.3, zorder=3,
            hatch="///", edgecolor=SURFACE, linewidth=2)
    lo = pa.success_rate.to_numpy() - pa.ci_lo.to_numpy()
    hi = pa.ci_hi.to_numpy() - pa.success_rate.to_numpy()
    ax.errorbar(pa.success_rate, y, xerr=[lo, hi], fmt="none", ecolor=INK3,
                elinewidth=1.4, capsize=4, zorder=4)
    for yi, a in zip(y, ARMS):
        r = pa.loc[a]
        lab = f"{int(r.successes)}/9" + (f"  ({int(r.lucky)} lucky)" if r.lucky else "")
        ax.text(r.ci_hi + 0.02, yi, lab, va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [ARM_LABEL[a] for a in ARMS])
    ax.set_xlim(0, 0.88)
    ax.set_xticks(np.arange(0, 0.76, 0.25), ["0%", "25%", "50%", "75%"])
    _style(ax, grid_axis="x")
    _title(ax, "Success rate, 9 trials per arm",
           "hatched = operator called it lucky · whiskers = 95% Wilson")

    # --- (b) progress: where each arm's 9 trials ended
    ax = axes[1]
    order = STAGES + ["success"]
    colors = [STAGE_C[s] for s in STAGES] + [SUCCESS_C]
    left = np.zeros(len(ARMS))
    for st, col in zip(order, colors):
        vals = np.array([(df[(df.arm == a)].fail_stage == st).sum() if st != "success"
                         else int(df[df.arm == a].success.sum()) for a in ARMS], dtype=float)
        ax.barh(y, vals, left=left, height=0.6, color=col, zorder=3,
                edgecolor=SURFACE, linewidth=2)
        for yi, v, l in zip(y, vals, left):
            if v > 0:
                ax.text(l + v / 2, yi, int(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if st in ("transport", "insert", "success") else INK,
                        fontweight="600")
        left += vals
    ax.set_yticks(y, [ARM_LABEL[a] for a in ARMS])
    ax.set_xlim(0, 9)
    ax.set_xticks(range(0, 10, 3))
    ax.set_xlabel("trials", color=INK2, fontsize=9)
    _style(ax, grid_axis="x")
    _title(ax, "Where the 9 trials ended",
           "no trial failed in transport")
    ax.legend(handles=[Patch(facecolor=c, label=s.capitalize()) for s, c in
                       zip(order, colors) if s != "transport"],
              loc="lower right", bbox_to_anchor=(1.0, -0.30), ncol=4, frameon=False,
              fontsize=9, labelcolor=INK2, handlelength=1.1, handleheight=1.1, columnspacing=1.2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_failure(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3})

    # --- (a) failure stage composition per policy: the three families fail differently
    ax = axes[0]
    pols = ["act", "smolvla", "pi05"]
    x = np.arange(len(STAGES[:2] + ["insert"]))
    labels = ["Reach", "Grasp", "Insert"]
    w = 0.26
    for i, p in enumerate(pols):
        s = df[df.policy == p]
        vals = [int((s.fail_stage == st).sum()) for st in ("reach", "grasp", "insert")]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.9, color=C[p], zorder=3,
                      label=POLICY_LABEL[p])
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.18, v, ha="center",
                        fontsize=9, color=INK2, fontweight="600")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("failures (of 18 trials)", color=INK2, fontsize=9)
    _style(ax)
    _title(ax, "Each family fails at a different stage",
           "π0.5 never gets past reach · ACT and SmolVLA fail on the grasp")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right",
              handlelength=1.1, handleheight=1.1)

    # --- (b) directional bias of the annotated misses
    ax = axes[1]
    dirs = ["left", "right", "top"]
    bot = np.zeros(len(dirs))
    for p in pols:
        vals = np.array([int(df[df.policy == p][f"off_{d}"].sum()) for d in dirs], dtype=float)
        ax.bar(np.arange(len(dirs)), vals, 0.52, bottom=bot, color=C[p], zorder=3,
               edgecolor=SURFACE, linewidth=2, label=POLICY_LABEL[p])
        for xi, (v, b) in enumerate(zip(vals, bot)):
            if v:
                ax.text(xi, b + v / 2, int(v), ha="center", va="center", fontsize=9,
                        color="#ffffff", fontweight="600")
        bot += vals
    ax.set_xticks(np.arange(len(dirs)), ["off left", "off right", "off top"])
    ax.set_ylim(0, 7)
    ax.set_ylabel("annotated misses", color=INK2, fontsize=9)
    _style(ax)
    _title(ax, "Misses skew left — but not for SmolVLA",
           "11 of 41 failures carry a direction in the notes")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_smoothness(per_arm, df, path):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.30})
    pa = per_arm.set_index("arm").loc[ARMS]
    x = np.arange(len(ARMS))
    cols = [C[ARM_POLICY[a]] for a in ARMS]
    # x carries the checkpoint, colour carries the family (named in the legend) - spelling
    # "SmolVLA" under two adjacent bars does not fit at this panel width.
    short = {"act40k": "40k", "act60k": "60k", "sv5k": "5k",
             "sv20k": "20k", "pi5k": "5k", "pi10k": "10k"}

    for ax, col, title, sub, unit in [
        (axes[0], "jerk", "Action jerk — the “shaky” note, measured",
         "std of the action's 2nd difference · joint-σ·s⁻²", ""),
        (axes[1], "terminal_motion", "Motion in the final second",
         "does the arm park once the task is done?", ""),
        (axes[2], "mean_duration_s", "Episode duration", "mean of 9 trials", " s"),
    ]:
        vals = pa[col].to_numpy()
        bars = ax.bar(x, vals, 0.58, color=cols, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025, f"{v:.0f}{unit}",
                    ha="center", fontsize=9, color=INK2, fontweight="600")
        ax.set_xticks(x, [short[a] for a in ARMS], fontsize=9)
        ax.set_ylim(0, max(vals) * 1.26)
        _style(ax)
        _title(ax, title, sub)

    axes[0].legend(handles=[Patch(facecolor=C[p], label=POLICY_LABEL[p])
                            for p in ("act", "smolvla", "pi05")],
                   loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=3, frameon=False,
                   fontsize=9, labelcolor=INK2, handlelength=1.1, handleheight=1.1)
    j5, j20 = pa.loc["sv5k", "jerk"], pa.loc["sv20k", "jerk"]
    axes[0].annotate(f"{j5 / j20:.1f}× jerkier\nthan 20k", xy=(2, j5 * 1.01),
                     xytext=(3.0, j5 * 1.07), fontsize=9, color=INK, fontweight="600",
                     ha="left", arrowprops=dict(arrowstyle="->", color=INK3, lw=1.3))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_position(per_spot, df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.26})

    # --- (a) per-spot success rate with intervals: is T8 really better?
    ax = axes[0]
    ps = per_spot.copy()
    x = np.arange(len(ps))
    vals = ps.success_rate.to_numpy()
    bars = ax.bar(x, vals, 0.6, color=["#184f95" if s == "T8" else "#86b6ef"
                                       for s in ps.tray_spot], zorder=3)
    lo = vals - ps.ci_lo.to_numpy()
    hi = ps.ci_hi.to_numpy() - vals
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK3, elinewidth=1.3,
                capsize=3.5, zorder=4)
    for b, v, k in zip(bars, vals, ps.successes):
        ax.text(b.get_x() + b.get_width() / 2, -0.055, f"{int(k)}/6", ha="center",
                fontsize=8.5, color=INK2)
    ax.set_xticks(x, ps.tray_spot)
    ax.set_ylim(-0.08, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
    _style(ax)
    _title(ax, "Per-position success rate",
           "every interval overlaps every other · 6 trials/spot · 95% Wilson")

    # --- (b) T8 trial by trial: who actually succeeded there
    ax = axes[1]
    t8 = df[df.tray_spot == "T8"].set_index("arm").loc[ARMS]
    y = np.arange(len(ARMS))[::-1]
    for yi, a in zip(y, ARMS):
        r = t8.loc[a]
        if r.success and not r.lucky:
            ax.barh(yi, 4, height=0.55, color=SUCCESS_C, zorder=3)
            ax.text(4.12, yi, "success", va="center", fontsize=9, color=INK2)
        elif r.success:
            ax.barh(yi, 4, height=0.55, color=SUCCESS_C, alpha=0.3, hatch="///",
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(4.12, yi, "lucky — kicked in", va="center", fontsize=9, color=INK2)
        else:
            ax.barh(yi, 0.12, height=0.55, color=STAGE_C["reach"], zorder=3)
            ax.text(0.3, yi, f"failed at {r.fail_stage}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [ARM_LABEL[a] for a in ARMS])
    ax.set_xlim(0, 9.2)
    ax.set_xticks([])
    _style(ax, grid_axis="x")
    _title(ax, "T8: easy for ACT, not for the VLAs",
           "both ACT arms seat it · all three VLA arms never reach it")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def render(df, per_arm, per_policy, per_spot, ts, figs):
    fig_success(per_arm, df, figs / "phase07_eval_success.png")
    fig_failure(df, figs / "phase07_eval_failure.png")
    fig_smoothness(per_arm, df, figs / "phase07_eval_smoothness.png")
    fig_position(per_spot, df, figs / "phase07_eval_position.png")
