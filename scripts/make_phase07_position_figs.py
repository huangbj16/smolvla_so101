#!/usr/bin/env python3
"""Per-position figures. Imported by analyze_phase07_positions.py.

Progress is an ordered magnitude on a fixed 0-4 scale, so the grid panels use one sequential
blue ramp shared across every panel - the whole point is comparing panels, which a per-panel
scale would destroy. Every cell is direct-labelled with its value.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

C = {"act": "#2a78d6", "smolvla": "#eb6834", "pi05": "#1baf7a"}
POLICY_LABEL = {"act": "ACT", "smolvla": "SmolVLA", "pi05": "π0.5"}
POLICIES = ["act", "smolvla", "pi05"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#86857f"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
# dataviz sequential blue, steps 100 -> 700.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)
SPOTS = [f"T{i}" for i in range(1, 10)]


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)


def _title(ax, title, subtitle=None, size=11.5):
    ax.set_title(title, color=INK, fontsize=size, fontweight="600", loc="left",
                 pad=14 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5,
                va="bottom")


def fig_grid(m, w, p_noshare, p_interaction, path):
    """The 3x3 tray laid out as it was recorded, once pooled and once per family."""
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.5), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.22})
    norm = Normalize(0, 4)
    panels = [("progress", "All three pooled", "6 trials/spot")] + [
        (f"progress_{p}", POLICY_LABEL[p], "2 trials/spot") for p in POLICIES]

    vals_by_spot = m.set_index("tray_spot")
    for ax, (col, title, sub) in zip(axes, panels):
        g = np.array([[vals_by_spot.loc[f"T{3 * r + c + 1}", col] for c in range(3)]
                      for r in range(3)])
        ax.imshow(g, cmap=CMAP, norm=norm)
        for r in range(3):
            for c in range(3):
                v = g[r, c]
                ax.text(c, r + 0.16, f"{v:.1f}", ha="center", va="center", fontsize=13,
                        fontweight="600", color="#ffffff" if v > 2.1 else INK)
                ax.text(c, r - 0.26, f"T{3 * r + c + 1}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if v > 2.1 else INK2)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        _title(ax, title, sub, size=11)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
                      fraction=0.018, pad=0.015, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["0 no reach", "1 reached", "2 grasped", "3 carried", "4 seated"],
                          fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.suptitle("Mean progress per tray spot — the families do not agree on which spots are hard",
                 x=0.085, y=1.10, ha="left", fontsize=12.5, fontweight="600", color=INK)
    fig.text(0.085, 1.035,
             f"Kendall's W = {w:.2f} across the three families (W = 1 would mean identical "
             f"rankings) · tray laid out in recording order",
             ha="left", fontsize=9, color=INK2)
    fig.text(0.085, 0.985,
             f"A concordance this low arises from 2-trial sampling noise alone only "
             f"{p_interaction:.1%} of the time when the families share one ordering — so the "
             f"disagreement is real.",
             ha="left", fontsize=9, color=INK2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_interaction(m, corr, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.26})

    # --- (a) ACT against SmolVLA, per position: the double dissociation
    ax = axes[0]
    xa, ys = m.progress_act.to_numpy(), m.progress_smolvla.to_numpy()
    ax.plot([-0.3, 4.3], [-0.3, 4.3], color=GRID, lw=1.5, zorder=1)
    ax.text(3.9, 4.05, "equal", color=INK3, fontsize=8.5, ha="right")
    ax.scatter(xa, ys, s=110, color="#2a78d6", zorder=3, edgecolor=SURFACE, linewidth=2)
    # Spots can coincide exactly (T1 and T4 are both ACT 1.0 / SmolVLA 2.5), so group the
    # labels rather than stacking two strings on one marker.
    groups: dict[tuple[float, float], list[str]] = {}
    for x, y, lab in zip(xa, ys, m.tray_spot):
        groups.setdefault((x, y), []).append(lab)
    for (x, y), labs in groups.items():
        txt = "·".join(labs)
        hi = any(l in ("T5", "T8") for l in labs)
        ax.annotate(txt, (x, y), xytext=(0, 13 if not hi else -20),
                    textcoords="offset points", ha="center", fontsize=10 if hi else 9,
                    fontweight="600" if hi else "normal", color=INK if hi else INK2)
    for s, note in (("T8", "ACT's best,\nSmolVLA's worst"), ("T5", "SmolVLA's best,\nACT's worst")):
        r = m[m.tray_spot == s].iloc[0]
        ax.annotate(note, (r.progress_act, r.progress_smolvla),
                    xytext=(-78 if s == "T8" else 28, 8), textcoords="offset points",
                    fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=INK3, lw=1.1))
    ax.set_xlim(-0.3, 4.4), ax.set_ylim(-0.4, 4.6)
    ax.set_xlabel("ACT mean progress at that spot", color=INK2, fontsize=9)
    ax.set_ylabel("SmolVLA mean progress", color=INK2, fontsize=9)
    _style(ax, grid_axis="both")
    _title(ax, "Where ACT does well, SmolVLA does badly",
           "9 tray spots · Spearman ρ = −0.65 (p = 0.07)")

    # --- (b) the candidate mechanism: demonstration divergence during the reach
    ax = axes[1]
    for p in POLICIES:
        ax.scatter(m.div_reach, m[f"progress_{p}"], s=85, color=C[p], zorder=3,
                   edgecolor=SURFACE, linewidth=1.8, label=POLICY_LABEL[p])
        z = np.polyfit(m.div_reach, m[f"progress_{p}"], 1)
        xs = np.linspace(m.div_reach.min(), m.div_reach.max(), 10)
        ax.plot(xs, np.polyval(z, xs), color=C[p], lw=2, alpha=0.5, zorder=2)
    rows = corr.set_index("metric").loc["div_reach"]
    ax.set_xlabel("divergence between the 5 demonstrations, reach phase  (joint-σ)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("mean progress at that spot", color=INK2, fontsize=9)
    ax.set_ylim(-0.4, 4.8)
    _style(ax, grid_axis="both")
    _title(ax, "The same demos help one family, hurt the other",
           f"ACT ρ = {rows.progress_act:+.2f} (p = {rows.p_pooled_act:.2f}) · "
           f"SmolVLA ρ = {rows.progress_smolvla:+.2f} (p = {rows.p_pooled_smolvla:.2f})")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left",
              handlelength=1.1, handleheight=1.1, ncol=3)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def render(m, corr, w, p_noshare, p_interaction, figs):
    fig_grid(m, w, p_noshare, p_interaction, figs / "phase07_position_grid.png")
    fig_interaction(m, corr, figs / "phase07_position_interaction.png")
