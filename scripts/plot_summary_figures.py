"""Regenerate the two aggregate-ACC figures used in the README.

Usage:  python scripts/plot_summary_figures.py

The per-variable grid figures in figures/ are produced by the plotting scripts in
ACC_RMSE_Eval/ and Power_Spectra/. This script regenerates only the two figures
that average across variables, because those must exclude
mean_sea_level_pressure to stay consistent with the tables in README.md and
docs/RESULTS.md -- the climatology fallback documented under Known issues pins
MSLP's ACC near 0.999 and inflates any mean that includes it.

Colours match the per-variable overlays: 6-layer blue, 48-layer red.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

C48, C6 = "tab:red", "tab:blue"
INK, MUTED = "#1a1a1a", "#6b6b6b"
EXCLUDE = {"mean_sea_level_pressure"}
SKILL_THRESHOLD = 0.6


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def mean_curve(data, variables):
    """Mean ACC across variables, excluding the ones with invalid climatology."""
    acc = np.array(data["acc"])
    keep = [i for i, v in enumerate(variables) if v not in EXCLUDE]
    return acc[:, keep].mean(axis=1)


def style(ax):
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=10)


def crossing(days, curve, threshold=SKILL_THRESHOLD):
    """Linearly interpolated lead time at which the curve falls below threshold."""
    below = np.where(curve < threshold)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    x0, x1 = days[i - 1], days[i]
    y0, y1 = curve[i - 1], curve[i]
    return x0 + (y0 - threshold) * (x1 - x0) / (y0 - y1)


def plot_overlay(m48, m6, variables):
    """Mean ACC to 15 days, 48-layer against the 6-layer ablation."""
    days = np.arange(1, len(m48["acc"]) + 1) * 6 / 24.0
    c48, c6 = mean_curve(m48, variables), mean_curve(m6, variables)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(SKILL_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(f"skill threshold, ACC = {SKILL_THRESHOLD}", xy=(days[-1], SKILL_THRESHOLD),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                fontsize=9, color=MUTED)

    ax.plot(days, c6, color=C6, linewidth=2, label="6 layers", zorder=2)
    ax.plot(days, c48, color=C48, linewidth=2, label="48 layers", zorder=3)

    for curve, colour, label in ((c48, C48, "48 layers"), (c6, C6, "6 layers")):
        x = crossing(days, curve)
        if x is not None:
            ax.plot([x], [SKILL_THRESHOLD], "o", color=colour, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=4)
            ax.annotate(f"{label}: {x:.1f} d", xy=(x, SKILL_THRESHOLD),
                        xytext=(6, 14 if colour == C48 else -20),
                        textcoords="offset points", fontsize=10, color=INK)

    ax.set_xlabel("Forecast lead time (days)", fontsize=11, color=INK)
    ax.set_ylabel("Mean anomaly correlation", fontsize=11, color=INK)
    ax.set_title("Mean ACC across 19 variables — depth ablation",
                 fontsize=13, color=INK, pad=12)
    ax.set_xlim(0, days[-1])
    ax.set_ylim(0, 1.02)
    style(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    fig.tight_layout()

    out = os.path.join(FIGURES, "acc_average_48L_vs_6L.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_7day(m48, variables):
    """Mean ACC over the first 7 days, 48-layer only."""
    acc = np.array(m48["acc"])
    days_all = np.arange(1, len(acc) + 1) * 6 / 24.0
    mask = days_all <= 7
    days = days_all[mask]
    curve = mean_curve(m48, variables)[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(SKILL_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(f"skill threshold, ACC = {SKILL_THRESHOLD}", xy=(0, SKILL_THRESHOLD),
                xytext=(4, 6), textcoords="offset points", ha="left",
                fontsize=9, color=MUTED)
    ax.plot(days, curve, color=C48, linewidth=2, zorder=3)

    for d in (1, 3, 5, 7):
        i = int(np.argmin(np.abs(days - d)))
        ax.plot([days[i]], [curve[i]], "o", color=C48, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        ax.annotate(f"{curve[i]:.2f}", xy=(days[i], curve[i]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10, color=INK)

    ax.set_xlabel("Forecast lead time (days)", fontsize=11, color=INK)
    ax.set_ylabel("Mean anomaly correlation", fontsize=11, color=INK)
    ax.set_title("HEAL-ViT 48 layers — mean ACC across 19 variables, first 7 days",
                 fontsize=13, color=INK, pad=12)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.02)
    style(ax)
    fig.tight_layout()

    out = os.path.join(FIGURES, "acc_7day_average.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    m48, m6 = load("metrics_48L.json"), load("metrics_6L.json")
    variables = m48["metadata"]["variables"]
    plot_overlay(m48, m6, variables)
    plot_7day(m48, variables)


if __name__ == "__main__":
    main()
