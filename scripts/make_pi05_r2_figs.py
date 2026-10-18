#!/usr/bin/env python3
"""Figures for the pi0.5 D0+D1 80k run, read at 35,000 steps.

    python scripts/make_pi05_r2_figs.py

Reads outputs/train/pi05_d0d1_80k/checkpoint_metrics.json (written by eval_checkpoint_metrics.py)
and writes docs/details/figs/phase07_pi05_r2.png.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_phase07_doc_figs import C, CRITICAL, GOOD, GRID, INK, INK2, INK3, SURFACE, _style, _title

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/details/figs"

# Sequential ramp over an ORDERED dimension (training step), one hue light->dark, built on the
# project's pi0.5 green. Validated: adjacent normal-vision dE 17.7, worst CVD dE 15.9. The light
# end sits under 3:1 on the surface, so every bar carries a direct label as relief.
STEP_RAMP = {10000: "#8fdcc0", 20000: "#1baf7a", 30000: "#0b6245"}
JERK_FLOOR = 0.02   # worst repeat-to-repeat std seen across the measured arms

# Held-out loss every 2,500 steps, read off the run's wandb chart (eval/eval_loss). Only the three
# decade checkpoints are SAVED, so the 15k minimum is a point we cannot evaluate on the robot.
WANDB = {2500: 0.0758, 5000: 0.0740, 7500: 0.0677, 10000: 0.0662, 12500: 0.0685,
         15000: 0.0650, 17500: 0.0664, 20000: 0.0672, 22500: 0.0694, 25000: 0.0697,
         27500: 0.0698, 30000: 0.0692, 32500: 0.0736, 35000: 0.0700}

# Round-2 reference points, both measured with this same script on D1's five held-out episodes.
# Different holdout from pi0.5's ten, so the LOSSES are not comparable across families - the jerk
# is, because it is a property of the predicted chunk rather than of the target.
REF_JERK = {"ACT 40k": (11.97, C["act"]), "SmolVLA 20k": (36.04, C["smolvla"])}
STAGES = ["reach", "grasp", "transport", "insert"]


DEFAULT_JSON = ROOT / "outputs/train/pi05_d0d1_80k/checkpoint_metrics.json"


def load(p: pathlib.Path) -> dict[int, dict]:
    if not p.is_file():
        raise SystemExit(f"missing {p} - run eval_checkpoint_metrics.py first")
    return {int(k): v for k, v in json.loads(p.read_text())["results"].items()}


def panel_loss_curve(ax, res):
    xs = sorted(WANDB)
    ys = [WANDB[x] for x in xs]
    ax.plot(xs, ys, color=C["pi05"], lw=2, zorder=3)

    lo = min(WANDB, key=WANDB.get)
    ax.axvspan(lo, max(xs), color=CRITICAL, alpha=0.055, lw=0, zorder=0)
    ax.scatter([lo], [WANDB[lo]], s=70, color=CRITICAL, zorder=5,
               edgecolor=SURFACE, linewidth=1.6)
    ax.annotate(f"minimum {WANDB[lo]:.4f} at {lo // 1000}k\nno checkpoint saved here",
                (lo, WANDB[lo]), xytext=(10, 4), textcoords="offset points",
                fontsize=8, color=CRITICAL, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=CRITICAL, lw=0.9,
                                shrinkA=2, shrinkB=4))

    for s in sorted(res):
        if s not in WANDB:
            continue
        ax.scatter([s], [WANDB[s]], s=85, color=STEP_RAMP[s], zorder=6,
                   edgecolor=INK2, linewidth=1.2)
        ax.annotate(f"{s // 1000}k", (s, WANDB[s]), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)

    ax.text(0.985, 0.95, "shaded = past the minimum", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=CRITICAL)
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel("held-out loss", color=INK2, fontsize=9)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_xticklabels(["0", "10k", "20k", "30k", "40k"])
    _style(ax)
    _title(ax, "Held-out loss bottoms at 15k, then climbs",
           "wandb, all 10 held-out episodes · filled markers are the saved checkpoints")


def panel_jerk(ax, res):
    xs = sorted(res)
    ys = [res[s]["jerk"]["mean"] for s in xs]
    es = [res[s]["jerk"]["std"] for s in xs]

    for name, (v, col) in REF_JERK.items():
        ax.axhline(v, color=col, lw=1.4, ls=(0, (5, 3)), alpha=0.85, zorder=1)
        ax.text(0.015, v, f"{name}  {v:.1f}", color=col, fontsize=8, va="bottom",
                ha="left", zorder=5, transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))

    ax.errorbar(xs, ys, yerr=es, color=C["pi05"], lw=2, marker="o", ms=9,
                markeredgecolor=SURFACE, markeredgewidth=1.6, capsize=4,
                elinewidth=1.2, ecolor=INK3, zorder=4)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)

    d = (ys[-1] - ys[0]) / ys[0] * 100
    # Top-right: the bottom-left corner is where the ACT reference label lives.
    span = f"{xs[0] // 1000}k → {xs[-1] // 1000}k"
    ax.text(0.985, 0.06, f"{span}   {d:+.1f}%", transform=ax.transAxes, ha="right",
            fontsize=9.5, color=GOOD if d < -2 else CRITICAL if d > 2 else INK2,
            fontweight="600", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel("chunk jerk  (normalised · s⁻²)", color=INK2, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x // 1000}k" for x in xs])
    ax.set_xlim(xs[0] - 3500, xs[-1] + 3500)
    _style(ax)
    _title(ax, "Chunk jerk — the selector that worked in round 1",
           "mean ± std over 2 seeds · dashed = the two policies run on the robot")


def panel_stages(ax, res):
    xs = sorted(res)
    w = 0.26
    base = np.arange(len(STAGES))
    for i, s in enumerate(xs):
        vals = [res[s][f"loss_{k}"]["mean"] for k in STAGES]
        errs = [res[s][f"loss_{k}"]["std"] for k in STAGES]
        pos = base + (i - 1) * (w + 0.02)
        ax.bar(pos, vals, w, color=STEP_RAMP[s], label=f"{s // 1000}k",
               edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1,
                    capsize=3, zorder=4)
        for p, v, e in zip(pos, vals, errs):
            ax.text(p, v + e, f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                    color=INK2)

    ax.set_xticks(base)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel("held-out loss", color=INK2, fontsize=9)
    ax.set_ylim(0, max(res[s][f"loss_{k}"]["mean"] for s in xs for k in STAGES) * 1.22)
    leg = ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    _style(ax)
    _title(ax, "Where the loss sits, and where it moved",
           "D1's five held-out episodes — the set ACT and SmolVLA were measured on")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=DEFAULT_JSON)
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "phase07_pi05_r2.png")
    a = ap.parse_args()
    res = load(a.json)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.9), facecolor=SURFACE)
    panel_loss_curve(axes[0], res)
    panel_jerk(axes[1], res)
    panel_stages(axes[2], res)
    for ax, tag in zip(axes, "ABC"):
        ax.text(-0.09, 1.13, tag, transform=ax.transAxes, fontsize=12,
                fontweight="700", color=INK3)
    fig.tight_layout(w_pad=3.2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
