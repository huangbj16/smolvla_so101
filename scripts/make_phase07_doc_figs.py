#!/usr/bin/env python3
"""Three explanatory figures for phase07_portable_rig.md, each replacing a block of prose.

  1. phase07_r1_schedule.png   - the mirrored 12-block eval schedule (doc 10.3)
  2. phase07_grasp_detector.png - why the gripper-trace detector was rewritten (doc 12.5)
  3. phase07_lineage.png        - which dataset trained what, and what is still pending

Palette and styling are shared with make_phase07_eval_figs.py / make_phase07_position_figs.py so
the doc's figures read as one set: categorical slots for the three policy families, the sequential
blue ramp for magnitudes, status colours only for state.

Figure 2 reads the real datasets, so its annotations cannot drift away from the data the way a
hand-drawn diagram would. It also asserts the claim it illustrates - that no global threshold
separates the hold plateau from the approach opening - and fails loudly if a future dataset makes
that false.

    python scripts/make_phase07_doc_figs.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase07_eval import CACHE, frames, grasp_seated_index, turning_points
from check_d1_divergence import D0, D1

# --- the shared tokens, identical to the other two phase07 figure scripts ---------------------
C = {"act": "#2a78d6", "smolvla": "#eb6834", "pi05": "#1baf7a"}
POLICY_LABEL = {"act": "ACT", "smolvla": "SmolVLA", "pi05": "π0.5"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#86857f"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
BLUE_MID, BLUE_DARK = "#3987e5", "#184f95"

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs/details/figs"


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)


def _title(ax, title, subtitle=None, size=11.5):
    ax.set_title(title, color=INK, fontsize=size, fontweight="600", loc="left",
                 pad=15 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


# =============================================================================================
# 1 - the schedule
# =============================================================================================
# Arm -> (family, the two blocks it runs, the spot order in each). Transcribed from doc 10.3.
SCHEDULE = [
    ("act40k",  "act",     (1, 12), ("T4 T2 T8 T5 T6", "T3 T7 T9 T1")),
    ("act60k",  "act",     (2, 11), ("T5 T2 T6 T4 T8", "T9 T3 T1 T7")),
    ("sv5k",    "smolvla", (3, 10), ("T4 T8 T6 T5 T2", "T9 T3 T7 T1")),
    ("sv20k",   "smolvla", (4, 9),  ("T4 T6 T8 T2 T5", "T9 T3 T7 T1")),
    ("pi5k",    "pi05",    (5, 8),  ("T4 T2 T8 T6 T5", "T3 T9 T7 T1")),
    ("pi10k",   "pi05",    (6, 7),  ("T8 T2 T4 T6 T5", "T1 T7 T3 T9")),
]
HALF_A = ["T2", "T4", "T5", "T6", "T8"]


def fig_schedule(path: pathlib.Path) -> None:
    fig, (ax, axg) = plt.subplots(
        1, 2, figsize=(14.0, 4.6), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [3.5, 1], "wspace": 0.16})

    ys = {arm: len(SCHEDULE) - 1 - i for i, (arm, *_) in enumerate(SCHEDULE)}
    for arm, fam, (b1, b2), spots in SCHEDULE:
        y, col = ys[arm], C[fam]
        # The connector is the point of the figure: it is symmetric about 6.5 for every arm.
        ax.plot([b1, b2], [y, y], color=col, lw=2, alpha=0.28, zorder=1,
                solid_capstyle="round")
        for b, sp in zip((b1, b2), spots):
            ax.scatter([b], [y], s=215, color=col, zorder=3,
                       edgecolor=SURFACE, linewidth=2)
            ax.text(b, y, str(b), color="white", fontsize=8.5 if b < 10 else 7.4,
                    fontweight="700", ha="center", va="center", zorder=4)
            # The innermost pair (blocks 6 and 7) is one unit apart, so centring both spot
            # lists would overlap them; push each outward from the midline instead.
            ha = "center" if abs(b - 6.5) > 1.2 else ("right" if b < 6.5 else "left")
            dx = 0 if ha == "center" else (-0.22 if ha == "right" else 0.22)
            ax.text(b + dx, y - 0.34, sp, color=INK3, fontsize=7.2, ha=ha, va="top",
                    family="monospace", zorder=4)

    ax.axvline(6.5, color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(6.5, len(SCHEDULE) - 0.02, "every arm's mean block = 6.5", color=INK2,
            fontsize=8.5, ha="center", va="bottom", style="italic")

    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([f"{arm}" for arm, *_ in SCHEDULE], fontsize=9.5, color=INK)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.3, 12.7)
    ax.set_ylim(-0.75, len(SCHEDULE) + 0.25)
    ax.set_xlabel("block, in the order it was run", color=INK2, fontsize=9)
    _style(ax, grid_axis="x")
    _title(ax, "The mirror: each arm runs once early and once late",
           "12 blocks · 54 trials · 9 per arm. Pairs of checkpoints are adjacent "
           "(1–2, 3–4, …), so the checkpoint question is settled across minutes, "
           "not across the afternoon")

    # The 3x3 tray, coloured by which half each spot belongs to.
    for r in range(3):
        for c in range(3):
            spot = f"T{3 * r + c + 1}"
            in_a = spot in HALF_A
            axg.add_patch(FancyBboxPatch(
                (c, 2 - r), 0.86, 0.86, boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=BLUE_MID if in_a else SURFACE, edgecolor=BLUE_DARK,
                linewidth=1.6, zorder=2))
            axg.text(c + 0.43, 2 - r + 0.43, spot, ha="center", va="center", fontsize=10,
                     fontweight="600", color="white" if in_a else BLUE_DARK, zorder=3)
    axg.set_xlim(-0.1, 2.96)
    axg.set_ylim(-0.1, 2.96)
    axg.set_aspect("equal")
    axg.axis("off")
    _title(axg, "Half A (5 spots) · half B (4)",
           "each arm sees every spot exactly once")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


# =============================================================================================
# 2 - the grasp detector
# =============================================================================================
def _grips(repo: str) -> dict[int, np.ndarray]:
    fr = frames(CACHE / repo)
    st = np.stack(fr["observation.state"].to_numpy()).astype(float)
    ep = fr["episode_index"].to_numpy()
    return {int(e): st[np.where(ep == e)[0], 5] for e in np.unique(ep)}


def _peaks(x: np.ndarray) -> tuple[float, float, list[int]]:
    span = float(x.max()) - float(np.percentile(x, 5))
    rest = float(np.percentile(x, 5))
    return span, rest, [i for i, k in turning_points(x, 0.18 * span) if k == "max"]


def fig_detector(path: pathlib.Path) -> None:
    d0, d1 = _grips(D0), _grips(D1)

    # Per episode, a global threshold expressed as a fraction of span would have to sit ABOVE the
    # hold plateau and BELOW the approach peak. That is an interval per episode; a single global
    # threshold exists only if all 100 intervals share a point.
    lo, hi = [], []
    for g in (d0, d1):
        for x in g.values():
            span, rest, mx = _peaks(x)
            if len(mx) < 2:
                continue
            hold = float(x[mx[0]:mx[-1]].min())
            lo.append((hold - rest) / span)                      # must be above this
            hi.append((float(x[mx[0]]) - rest) / span)           # must be below this
    lo, hi = np.array(lo), np.array(hi)
    assert lo.size == 100, f"expected 100 episodes, got {lo.size}"
    assert lo.max() > hi.min(), (
        "the ranges no longer overlap - a global threshold would now work, so the figure's "
        f"claim is stale: max hold {lo.max():.3f} <= min approach peak {hi.min():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.17})

    # --- panel 1: the structure the detector has to find -------------------------------------
    ax = axes[0]
    x = d0[0]
    gi, ri = grasp_seated_index(x)
    span, rest, mx = _peaks(x)
    t = np.arange(len(x)) / 30.0
    ax.plot(t, x, color=BLUE_DARK, lw=2, zorder=3)
    ax.axhline(float(x[mx[0]:mx[-1]].min()), color=INK3, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(t[0], float(x[mx[0]:mx[-1]].min()) + span * 0.04, "hold", color=INK2,
            fontsize=8.5, va="bottom", ha="left")
    # grasp and seated are only ~8 s apart here, so the two labels are staggered vertically
    # rather than both sitting above the axes where they would collide.
    for i, lab, dy in ((gi, "grasp", 0.04), (ri, "seated", 0.16)):
        ax.axvline(t[i], color=GOOD, lw=1.6, zorder=4)
        ax.text(t[i], rest - span * dy, f" {lab} ", color=GOOD, fontsize=8.5,
                fontweight="600", va="top", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    for i, lab in ((mx[0], "open to\napproach"), (mx[-1], "open to\nrelease")):
        ax.scatter([t[i]], [x[i]], s=55, color=BLUE_MID, edgecolor=SURFACE, linewidth=1.6,
                   zorder=5)
        ax.text(t[i], x[i] + span * 0.05, lab, color=INK2, fontsize=7.8, ha="center",
                va="bottom", linespacing=1.15)
    ax.set_ylim(rest - span * 0.30, x.max() + span * 0.28)
    ax.set_xlabel("seconds", color=INK2, fontsize=9)
    ax.set_ylabel("gripper opening", color=INK2, fontsize=9)
    _style(ax)
    _title(ax, "What is being detected", "D0 episode 0 · two openings bracket the carry")

    # --- panel 2: the episode that broke it --------------------------------------------------
    ax = axes[1]
    x = d1[39]
    span, rest, mx = _peaks(x)
    t = np.arange(len(x)) / 30.0
    half = len(x) // 2
    old = (int(np.argmax(x[:half])), half + int(np.argmax(x[half:])))
    new_gi, new_ri = grasp_seated_index(x)
    ax.plot(t, x, color=BLUE_DARK, lw=2, zorder=3)
    ax.axvline(t[half], color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(t[half], rest + span * 0.30, " midpoint", color=INK2, fontsize=8,
            ha="left", va="center")
    for i in old:
        ax.scatter([t[i]], [x[i]], s=130, marker="X", color=CRITICAL, edgecolor=SURFACE,
                   linewidth=1.6, zorder=6)
    # One caption, in the empty top-left quadrant: both old picks sit on the release plateau,
    # so anything placed near them collides with the trace or with the midpoint rule.
    ax.text(0.03, 0.97, "both old picks land in the one\nrelease opening, so the grasp pose\n"
                        "came out as the fixture pose",
            transform=ax.transAxes, color=CRITICAL, fontsize=7.8, linespacing=1.35,
            ha="left", va="top", zorder=7)
    for i, lab, dy in ((new_gi, "grasp", 0.04), (new_ri, "seated", 0.16)):
        ax.axvline(t[i], color=GOOD, lw=1.6, zorder=4)
        ax.text(t[i], rest - span * dy, f" {lab} ", color=GOOD, fontsize=8.2,
                fontweight="600", ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    ax.set_ylim(rest - span * 0.34, x.max() + span * 0.30)
    ax.set_xlabel("seconds", color=INK2, fontsize=9)
    _style(ax)
    _title(ax, "Failure 1 — the midpoint assumption",
           f"D1 ep 39 · {len(x)} frames · release opens at {old[0]}")

    # --- panel 3: why no global threshold replaces it ----------------------------------------
    ax = axes[2]
    order = np.argsort(lo)
    yy = np.arange(100)
    ax.barh(yy, (hi - lo)[order], left=lo[order], height=0.86, color=BLUE_MID, alpha=0.55,
            zorder=2, linewidth=0)
    # The tightest ceiling sits BELOW the tightest floor, which is exactly what "no global
    # threshold" means - so the two lines are labelled as the bounds they are, not as a band.
    floor, ceil = lo.max(), hi.min()
    ax.axvline(ceil, color=CRITICAL, lw=1.5, zorder=4)
    ax.axvline(floor, color=CRITICAL, lw=1.5, zorder=4)
    ax.annotate("", xy=(floor, 106), xytext=(ceil, 106), zorder=5,
                arrowprops=dict(arrowstyle="<|-|>", color=CRITICAL, lw=1.2,
                                mutation_scale=9))
    ax.text(0.5 * (floor + ceil), 110,
            f"lowest ceiling {ceil:.2f} < highest floor {floor:.2f}\nso no single value works",
            color=CRITICAL, fontsize=8, fontweight="600", ha="center", va="bottom",
            linespacing=1.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-2, 136)
    ax.set_yticks([])
    ax.set_xlabel("threshold, as a fraction of the episode's span", color=INK2, fontsize=9)
    _style(ax, grid_axis="x")
    _title(ax, "Failure 2 — no global threshold exists",
           "one bar per episode \u00b7 100 episodes")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}  [highest floor {floor:.2f} > lowest ceiling {ceil:.2f}, "
          "so no global threshold exists]")


# =============================================================================================
# 3 - dataset lineage
# =============================================================================================
# (column, row, label, sublabel, state) - state picks the colour, so status is never decorative.
DONE, ACTIVE, PENDING = "done", "active", "pending"
STATE_COLOR = {DONE: GOOD, ACTIVE: WARNING, PENDING: INK3}


def fig_lineage(path: pathlib.Path) -> None:
    """Three self-contained lanes, left to right, with no arrow ever crossing another.

    An earlier version drew the causal links between lanes (findings -> D1, D2 -> round 2) and
    the result was unreadable: four long diagonals over the boxes. Those links are one sentence
    each, so they live in the subtitle and the lane labels instead, and the figure keeps the one
    job a figure is better at - showing which data fed which training run.
    """
    fig, ax = plt.subplots(figsize=(13.6, 5.2), facecolor=SURFACE)

    W, H, GAP = 0.86, 0.56, 0.20
    lanes = [
        ("Round 1", 2.55, [
            ("D0 · 50 ep", "35,141 fr · 9 spot blocks", DONE),
            ("D0 trimmed", "29,462 fr · both ends cut", DONE),
            ("3 policies trained", "ACT 60k · SmolVLA 20k\nπ0.5 10k", DONE),
            ("Round 1 eval", "6 arms · 54 trials · best 33%", DONE),
        ]),
        ("Round 2", 1.45, [
            ("D1 · 50 ep", "38,389 fr · randomised", DONE),
            ("D1 trimmed", "33,337 fr · 2 s tail kept", DONE),
            ("2 policies continued", "ACT 40k done\nSmolVLA 10.3k of 20k", ACTIVE),
            ("Round 2 eval", "per position, per family", PENDING),
        ]),
        ("Planned", 0.35, [
            ("D1-perturb · 30 ep", "recovery · appends to D1\nas ep 50–79", PENDING),
            ("D2 · 130 ep", "stage split\n50 grasp, 50 insert", PENDING),
            ("Train D1 vs D2", "matched on operator\nhours, not episodes", PENDING),
            ("", "", None),
        ]),
    ]
    # Which cells actually follow from the one on their left. D1-perturb and D2 are independent
    # datasets sitting in the same lane, so no arrow joins them.
    no_arrow = {("Planned", 1)}

    for lane_name, y, cells in lanes:
        ax.text(-0.16, y + H / 2, lane_name, ha="right", va="center", fontsize=10,
                fontweight="600", color=INK2, rotation=90)
        for ci, (label, sub, state) in enumerate(cells):
            if state is None:
                continue
            cx = ci * (W + GAP)
            col = STATE_COLOR[state]
            ax.add_patch(FancyBboxPatch(
                (cx, y), W, H, boxstyle="round,pad=0,rounding_size=0.04",
                facecolor=SURFACE, edgecolor=col, linewidth=1.9,
                linestyle="solid" if state != PENDING else (0, (4, 3)), zorder=3))
            ax.add_patch(FancyBboxPatch(
                (cx, y + H - 0.05), W, 0.05, boxstyle="round,pad=0,rounding_size=0.008",
                facecolor=col, edgecolor="none", zorder=4))
            ax.text(cx + W / 2, y + H - 0.17, label, ha="center", va="center",
                    fontsize=8.8, fontweight="600", color=INK, zorder=5)
            ax.text(cx + W / 2, y + 0.155, sub, ha="center", va="center", fontsize=7.2,
                    color=INK2, linespacing=1.3, zorder=5)
            if ci and cells[ci - 1][2] is not None and (lane_name, ci) not in no_arrow:
                ax.add_patch(FancyArrowPatch(
                    (cx - GAP, y + H / 2), (cx, y + H / 2), arrowstyle="-|>",
                    mutation_scale=11, color=INK3, lw=1.3, shrinkA=1, shrinkB=1, zorder=2))

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                          markerfacecolor=STATE_COLOR[s], markeredgecolor="none", label=l)
               for s, l in ((DONE, "done"), (ACTIVE, "running"), (PENDING, "planned"))]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
                    ncol=3, handletextpad=0.4, columnspacing=1.4,
                    bbox_to_anchor=(1.0, -0.06))
    for txt in leg.get_texts():
        txt.set_color(INK2)

    ax.set_xlim(-0.42, 4 * W + 3 * GAP + 0.04)
    ax.set_ylim(0.05, 3.42)
    ax.axis("off")
    ax.set_title("Which dataset trained what", color=INK, fontsize=12.5, fontweight="600",
                 loc="left", pad=20)
    ax.text(0, 1.015,
            "Round 1's findings set D1's design. Round 2 continues round 1's weights rather "
            "than retraining, so its policies have seen D0 and D1 both.",
            transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_schedule(OUT / "phase07_r1_schedule.png")
    fig_detector(OUT / "phase07_grasp_detector.png")
    fig_lineage(OUT / "phase07_lineage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
