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

A second figure covers the round-2 selection (section 12.6): the four D1 SmolVLA checkpoints, where
jerk falls and loss rises at every step. It is drawn separately rather than as a fourth panel because
its x-axis is training steps within one run, not a set of arms, and its jerk is normalised by D1's
action sigma - not comparable to the first figure's numbers.

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
from make_phase07_doc_figs import (C, CRITICAL, GOOD, GRID, INK, INK2, INK3, POLICY_LABEL,
                                   SURFACE, _style, _title)

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/phase07_eval/checkpoint_metrics.json"
PER_ARM = ROOT / "results/phase07_eval/summary_per_arm.csv"
# The training run writes this under the gitignored outputs/; a copy is committed beside the
# round-1 metrics so the figure is reproducible from a fresh clone. Prefer the copy.
D1_ACT_METRICS = next(
    (q for q in (ROOT / "results/phase07_eval/checkpoint_metrics_d1_act.json",
                 ROOT / "outputs/train/act_tray_cylinder_s1000_d1/checkpoint_metrics.json")
     if q.is_file()),
    ROOT / "results/phase07_eval/checkpoint_metrics_d1_act.json")
D1_METRICS = next(
    (q for q in (ROOT / "results/phase07_eval/checkpoint_metrics_d1_smolvla.json",
                 ROOT / "outputs/train/smolvla_tray_cylinder_s1000_d1_resumed"
                        "/checkpoint_metrics.json") if q.is_file()),
    ROOT / "results/phase07_eval/checkpoint_metrics_d1_smolvla.json")
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


def _divergence_panel(ax, rows, fam, title, subtitle, assert_loss_rises):
    """One family's D1 checkpoints, both metrics INDEXED to the first checkpoint.

    Indexed rather than drawn on twin y-axes. A dual-axis chart lets the author slide two
    unrelated scales until the lines cross wherever they like, so the reader cannot tell whether
    a crossing means anything - it is the single most misleading chart form. Percent change from
    a common base puts both on one axis, where "jerk falls while loss rises" is the literal shape
    of the two lines and the magnitudes are comparable.
    """
    steps = [s for s, _ in rows]
    jerk = np.array([v["jerk"]["mean"] for _, v in rows])
    jerr = np.array([v["jerk"]["std"] for _, v in rows])
    loss = np.array([v["loss_pooled"]["mean"] for _, v in rows])
    lerr = np.array([v["loss_pooled"]["std"] for _, v in rows])

    assert all(a > b for a, b in zip(jerk, jerk[1:])), f"{fam} jerk no longer falls: {jerk}"
    if assert_loss_rises:
        assert all(a < b for a, b in zip(loss, loss[1:])), f"{fam} loss no longer rises: {loss}"

    ks = np.array(steps) / 1000.0
    pj, pl = 100 * (jerk / jerk[0] - 1), 100 * (loss / loss[0] - 1)
    ej, el = 100 * jerr / jerk[0], 100 * lerr / loss[0]

    ax.axhline(0, color=GRID, lw=1.2, zorder=1)
    ax.errorbar(ks, pj, yerr=ej, color=C[fam], lw=2.2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.6, capsize=3, zorder=4,
                label="chunk jerk")
    ax.errorbar(ks, pl, yerr=el, color=INK3, lw=2.2, ls="--", marker="s", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.6, capsize=3, zorder=4,
                label="held-out loss")
    # Direct labels on the final point: identity is never carried by colour alone.
    for y, lbl, col in ((pj[-1], "jerk", C[fam]), (pl[-1], "loss", INK3)):
        ax.annotate(f"{lbl}  {y:+.0f}%", (ks[-1], y), textcoords="offset points", xytext=(17, 0),
                    va="center", fontsize=9, color=col, fontweight="700")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{int(k)}k" for k in ks])
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel(f"% change from {steps[0] // 1000}k", color=INK2, fontsize=9)
    ax.set_xlim(ks[0] - 0.45 * (ks[1] - ks[0]), ks[-1] + 1.15 * (ks[-1] - ks[-2]))
    _style(ax)

    sel = int(np.argmin(jerk))
    ax.scatter([ks[sel]], [pj[sel]], s=300, facecolor="none", edgecolor=GOOD, linewidth=2.2,
               zorder=6)
    lpick = int(np.argmin(loss))
    if lpick != sel:
        ax.scatter([ks[lpick]], [pl[lpick]], s=300, facecolor="none", edgecolor=CRITICAL,
                   linewidth=2.2, zorder=6)
    _title(ax, title, subtitle)
    return jerk, jerr, loss, lerr


def fig_d1_selection(path: pathlib.Path) -> None:
    """Section 12.8's selection, both families: jerk falls for both, the loss does not agree."""
    def rows(p):
        return sorted((int(k), v) for k, v in json.loads(p.read_text())["results"].items())

    sv, act = rows(D1_METRICS), rows(D1_ACT_METRICS)

    fig = plt.figure(figsize=(17.4, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 1.0], wspace=0.55)

    j_sv, je_sv, l_sv, le_sv = _divergence_panel(
        fig.add_subplot(gs[0, 0]), sv, "smolvla",
        "A · SmolVLA on D1 — the metrics disagree",
        "jerk ↓ and loss ↑ at every step · ρ = −1.00", assert_loss_rises=True)

    j_ac, je_ac, l_ac, le_ac = _divergence_panel(
        fig.add_subplot(gs[0, 1]), act, "act",
        "B · ACT on D1 — only jerk moves",
        "jerk ↓ at every step · loss wanders 2.5%, non-monotonically", assert_loss_rises=False)

    # The contrast is the point: for ACT the loss MINIMUM lands on the same checkpoint, so jerk
    # overrules nothing there. Say so, rather than letting two similar-looking panels imply the
    # inversion is general.
    fig.text(0.40, -0.015,
             "green ring = selected (lowest jerk)        red ring = what the loss curve would pick",
             ha="center", va="top", fontsize=9, color=INK2, fontweight="600")
    same = int(np.argmin(l_ac)) == int(np.argmin(j_ac))
    fig.text(0.40, -0.09, "No inversion for ACT: its loss minimum is the same checkpoint, so jerk "
             "breaks a tie rather than overruling one.\nThe §11.3a inversion is so far a SmolVLA "
             "phenomenon — seen twice there, never on ACT." if same else "",
             ha="center", va="top", fontsize=8.6, color=INK3, linespacing=1.6)

    # ---- C: how much milder than round 1 (SmolVLA, the only family with a round-1 pair) -------
    ax = fig.add_subplot(gs[0, 2])
    r1 = json.loads(METRICS.read_text())["arms"]
    labels = ["jerk 5k→20k\n(ratio)", "pooled loss 5k→20k\n(% change)"]
    d0 = [r1["sv5k"]["jerk"]["mean"] / r1["sv20k"]["jerk"]["mean"],
          100 * (r1["sv20k"]["loss_pooled"]["mean"] / r1["sv5k"]["loss_pooled"]["mean"] - 1)]
    d1 = [j_sv[0] / j_sv[-1], 100 * (l_sv[-1] / l_sv[0] - 1)]

    # Two different units on one panel, so each pair is normalised to its own round-1 bar and the
    # real values are printed on the bars. The comparison is "how much milder", nothing else.
    for i, (a, b) in enumerate(zip(d0, d1)):
        ax.bar(i - 0.2, 1.0, width=0.38, color=INK3, alpha=0.45, zorder=3, linewidth=0)
        ax.bar(i + 0.2, b / a, width=0.38, color=C["smolvla"], zorder=3, linewidth=0)
        fmt = (lambda v: f"{v:.2f}×") if i == 0 else (lambda v: f"+{v:.0f}%")
        ax.text(i - 0.2, 1.02, fmt(a), ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(i + 0.2, b / a + 0.02, fmt(b), ha="center", va="bottom", fontsize=9,
                color=C["smolvla"], fontweight="700")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("relative to round 1", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    _style(ax)
    ax.text(0.985, 0.955, "grey = round 1 (D0, fresh, lr 1e-4)\norange = round 2 (D1, warm, lr 1e-5)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2,
            linespacing=1.4)
    _title(ax, "C · Round 2 drifts far less",
           "SmolVLA — the only family with a round-1 pair")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main() -> int:
    if not METRICS.is_file():
        raise SystemExit(f"{METRICS} not found - run scripts/eval_checkpoint_metrics.py --all-round1")
    m, roll = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_metrics(m, roll, OUT / "phase07_offline_metrics.png")
    if D1_METRICS.is_file() and D1_ACT_METRICS.is_file():
        fig_d1_selection(OUT / "phase07_d1_selection.png")
    else:
        print(f"  skipped phase07_d1_selection.png - {D1_METRICS} not found")

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
