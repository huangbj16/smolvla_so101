#!/usr/bin/env python3
"""Figures for the two offline metrics measured by `eval_checkpoint_metrics.py`.

Three panels, answering three questions in order:

  A. Does **chunk jerk**, measured offline on held-out observations, reproduce the jerk the robot
     actually produced? The pre-registered prediction is a ratio - SmolVLA 5k ~2x SmolVLA 20k,
     matching the 2.0x measured on the robot in section 11.3.
  B. Does **per-stage held-out loss** separate the SmolVLA pair where the pooled number does not?
     The prediction names **grasp** and **insert**, where tolerance is tightest. Faceted by family,
     because an L1 loss and a flow-matching MSE are different quantities - a shared y-axis across
     families would invite exactly the comparison section 10.7a says cannot be made.
  C. Which of the two offline numbers tracks what the robot did? Rollout mean progress against each,
     within family, since that is the only comparison the units support.

    python scripts/make_phase07_metric_figs.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_phase07_doc_figs import C, GRID, INK, INK2, INK3, POLICY_LABEL, SURFACE, _style, _title

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/phase07_eval/checkpoint_metrics.json"
PER_ARM = ROOT / "results/phase07_eval/summary_per_arm.csv"
OUT = ROOT / "docs/details/figs"

STAGES = ["reach", "grasp", "transport", "insert"]
ARMS = ["act40k", "act60k", "sv5k", "sv20k", "pi5k", "pi10k"]
ARM_LABEL = {"act40k": "ACT 40k", "act60k": "ACT 60k", "sv5k": "SmolVLA 5k",
             "sv20k": "SmolVLA 20k", "pi5k": "π0.5 5k", "pi10k": "π0.5 10k"}
FAMILY = {"act40k": "act", "act60k": "act", "sv5k": "smolvla", "sv20k": "smolvla",
          "pi5k": "pi05", "pi10k": "pi05"}
PAIRS = [("act", ["act40k", "act60k"]), ("smolvla", ["sv5k", "sv20k"]), ("pi05", ["pi5k", "pi10k"])]


def load():
    m = json.loads(METRICS.read_text())["arms"]
    # pi0.5 10k was not measured: at 67 min an arm it is 6x the cost of any other and the run was
    # stopped after pi5k. Everything below drops arms that are absent rather than assuming six.
    global ARMS, PAIRS
    ARMS = [a for a in ARMS if a in m]
    PAIRS = [(f, [a for a in arms if a in m]) for f, arms in PAIRS]
    PAIRS = [(f, arms) for f, arms in PAIRS if arms]
    roll = {}
    for r in csv.DictReader(PER_ARM.open()):
        roll[r["arm"]] = {"jerk": float(r["jerk"]), "progress": float(r["mean_progress"]),
                          "successes": int(r["successes"])}
    return m, roll


def fig_metrics(m, roll, path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(15.0, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.56, wspace=0.26)

    # ---- A: offline chunk jerk against the jerk the robot produced -------------------------
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(ARMS))
    off = np.array([m[a]["jerk"]["mean"] for a in ARMS])
    err = np.array([m[a]["jerk"]["std"] for a in ARMS])
    rb = np.array([roll[a]["jerk"] for a in ARMS])
    cols = [C[FAMILY[a]] for a in ARMS]

    ax.bar(x - 0.19, off, width=0.36, color=cols, zorder=3, linewidth=0)
    ax.errorbar(x - 0.19, off, yerr=err, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3,
                zorder=4)
    ax.bar(x + 0.19, rb, width=0.36, color=cols, alpha=0.32, zorder=3, linewidth=0)
    for xi, (o, r) in enumerate(zip(off, rb)):
        ax.text(xi - 0.19, o, f"{o:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.text(xi + 0.19, r, f"{r:.1f}", ha="center", va="bottom", fontsize=8, color=INK3)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("jerk  (normalised units · s⁻²)", color=INK2, fontsize=9)
    ax.set_ylim(0, max(off.max(), rb.max()) * 1.26)
    _style(ax)
    ax.text(0.985, 0.97, "solid = predicted chunk, measured offline\n"
                         "faded = executed rollout (§11.1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2,
            linespacing=1.4)
    _title(ax, "A · Chunk jerk measured offline, against the jerk the robot produced",
           "one pass over the held-out episodes per checkpoint · 3 repeats, fixed seeds")

    # The pre-registered test, called out explicitly.
    r_off = m["sv5k"]["jerk"]["mean"] / m["sv20k"]["jerk"]["mean"]
    r_rob = roll["sv5k"]["jerk"] / roll["sv20k"]["jerk"]
    y_ann = off[3] * 1.30
    ax.annotate("", xy=(2 - 0.19, y_ann), xytext=(3 - 0.19, y_ann), zorder=6,
                arrowprops=dict(arrowstyle="<|-|>", color=C["smolvla"], lw=1.3,
                                mutation_scale=9, shrinkA=2, shrinkB=2))
    # Parked in the empty band above the pi0.5 bars rather than over the SmolVLA 5k bar.
    ax.text(3.42, y_ann * 1.55, f"the pre-registered test\npredicted ~2×\n"
                                f"offline {r_off:.2f}×  ·  robot {r_rob:.2f}×",
            ha="left", va="center", fontsize=8.5, color=C["smolvla"], fontweight="600",
            linespacing=1.5)

    # ---- C: which offline number tracks the robot? ------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for fam, arms in PAIRS:
        xs = [m[a]["jerk"]["mean"] for a in arms]
        ys = [roll[a]["progress"] for a in arms]
        ax.plot(xs, ys, color=C[fam], lw=1.6, alpha=0.5, zorder=2)
        ax.scatter(xs, ys, s=95, color=C[fam], zorder=4, edgecolor=SURFACE, linewidth=1.8,
                   label=POLICY_LABEL[fam])
        for a, xx, yy in zip(arms, xs, ys):
            ax.annotate(ARM_LABEL[a].split()[-1], (xx, yy), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.6, color=INK2)
    ax.set_xlabel("offline chunk jerk", color=INK2, fontsize=9)
    ax.set_ylabel("rollout mean progress", color=INK2, fontsize=9)
    _style(ax, grid_axis="both")
    ax.margins(0.14)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3,
                    handletextpad=0.3, columnspacing=0.9, bbox_to_anchor=(0.5, -0.17))
    for t in leg.get_texts():
        t.set_color(INK2)
    _title(ax, "C · Jerk vs what the robot did",
           "lines join one family's two checkpoints")

    # ---- B: per-stage held-out loss, faceted by family ---------------------------------------
    xs = np.arange(len(STAGES))
    for col, (fam, arms) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[1, col])
        for a, ls, mk in zip(arms, ["-", "--"], ["o", "s"]):
            ys = [m[a][f"loss_{s}"]["mean"] for s in STAGES]
            es = [m[a][f"loss_{s}"]["std"] for s in STAGES]
            ax.errorbar(xs, ys, yerr=es, color=C[fam], lw=2, ls=ls, marker=mk, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.4, capsize=3, elinewidth=1,
                        label=ARM_LABEL[a], zorder=3)
        pooled = [m[a]["loss_pooled"]["mean"] for a in arms]
        for p, ls in zip(pooled, ["-", "--"]):
            ax.axhline(p, color=INK3, lw=1, ls=ls, alpha=0.55, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(STAGES, fontsize=8.5)
        ax.set_xlim(-0.35, len(STAGES) - 0.65)
        if col == 0:
            ax.set_ylabel("held-out loss", color=INK2, fontsize=9)
        _style(ax)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="best", handletextpad=0.5)
        for t in leg.get_texts():
            t.set_color(INK2)
        _title(ax, f"B{col + 1} · {POLICY_LABEL[fam]} per stage",
               "grey lines = each arm's pooled loss" if col == 0 else None)

    fig.text(0.5, 0.487, "B · Held-out loss split by task stage — one y-axis per family, because an "
                         "L1 loss and a flow-matching MSE are different quantities (§10.7a)",
             ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="600")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> int:
    if not METRICS.is_file():
        raise SystemExit(f"{METRICS} not found - run scripts/eval_checkpoint_metrics.py --all-round1")
    m, roll = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_metrics(m, roll, OUT / "phase07_offline_metrics.png")

    # The numbers the doc quotes, printed so they are copied rather than retyped.
    print("\n  arm        offline jerk      rollout jerk   pooled loss     grasp     insert")
    for a in ARMS:
        j, p = m[a]["jerk"], m[a]["loss_pooled"]
        print(f"  {a:9s}  {j['mean']:6.2f} ± {j['std']:4.2f}   {roll[a]['jerk']:11.2f}   "
              f"{p['mean']:.4f}    {m[a]['loss_grasp']['mean']:.4f}  "
              f"{m[a]['loss_insert']['mean']:.4f}")
    r_off = m["sv5k"]["jerk"]["mean"] / m["sv20k"]["jerk"]["mean"]
    r_rob = roll["sv5k"]["jerk"] / roll["sv20k"]["jerk"]
    print(f"\n  pre-registered test: SmolVLA 5k / 20k jerk = {r_off:.2f}x offline, "
          f"{r_rob:.2f}x on the robot (predicted ~2x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
