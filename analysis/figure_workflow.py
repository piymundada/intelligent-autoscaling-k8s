#!/usr/bin/env python3
"""Figure 3.3.1: the research workflow, drawn so that each stage IS a thesis section.

Stage headers carry the Chapter 3 section number and the boxes beneath carry the subsection
numbers, so the figure and the contents page cannot drift apart. Every box TITLE is the exact
section heading; the grey line under it is explanatory detail only and is not part of the heading.
Chapters 4 and 5 walk the same stages in the same order, see
.github/thesis_extracted/CHAPTER_STRUCTURE.md.

House style, matched to the hand-drawn SVGs in docs/ (figure_4_1_ml_pipeline.svg and siblings):
  font        Helvetica, Arial, sans-serif
  headings    #1f1f1f, weight 600
  secondary   #5F5E5A
  connectors  #7a7a74
  box pairs   pastel fill with a saturated same-hue stroke, taken from the existing palette

Writes both PNG (for Word) and SVG (hand-editable, text kept as text). Re-running overwrites
BOTH, so save any hand-edited SVG under a different name first.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

# ── house palette, every pair already used elsewhere in docs/ ────────────────
INK, SECONDARY, CONNECTOR = "#1f1f1f", "#5F5E5A", "#7a7a74"
BLUE = ("#378ADD", "#E6F1FB")
AMBER = ("#BA7517", "#FAEEDA")
GREEN = ("#639922", "#EAF3DE")
TEAL = ("#1D9E75", "#E1F5EE")
PURPLE = ("#534AB7", "#EEEDFE")

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

# (section number, exact section heading, accent pair, [(number, heading, detail), ...])
STAGES = [
    ("3.4", "Data Collection from the Spring Boot Microservice on Kubernetes", BLUE, [
        ("3.4.1", "Application Under Test", "Spring Boot service with CPU,\nheap and GC endpoints"),
        ("3.4.2", "Deployment Environments", "Minikube (local) and\nAmazon EKS (cloud)"),
        ("3.4.3", "Workload Generation and\nCapacity Calibration", "steady, bursty, spike, GC"),
        ("3.4.4", "Workload, Infrastructure and\nJVM Metric Collection", "three layers, 30 s sampling"),
    ]),
    ("3.5", "Dataset Preparation for the Predictive Model", AMBER, [
        ("3.5.1", "Feature Selection", "10 of 17 collected metrics"),
        ("3.5.2", "Exploratory Data Analysis", "feature ranges and correlation\nwith the forecast target"),
        ("3.5.3", "Preprocessing, Windowing and\nTime-Ordered Splitting", "min-max scaling, 10-step windows"),
    ]),
    ("3.6", "Modelling: Forecaster and Autoscaling Strategies", GREEN, [
        ("3.6.1", "LSTM Workload Forecaster", "60 s prediction horizon"),
        ("3.6.2", "Baseline: Kubernetes HPA", "CPU and memory targets"),
        ("3.6.3", "JVM-Aware Predictive\nAutoscaler", "forecast plus JVM-pressure floor"),
        ("3.6.4", "RPS-Only Ablation Arm", "floor disabled"),
    ]),
    ("3.7", "Experimental Execution in Two Environments", TEAL, [
        ("3.7.1", "Run Procedure", "ten steps, repeated per run"),
        ("3.7.2", "Local Execution", "Minikube campaign"),
        ("3.7.3", "Cloud Execution", "Amazon EKS, two isolated clusters"),
        ("3.7.4", "Environment-Specific\nConfiguration", "per-environment calibration"),
    ]),
    ("3.8", "Evaluation and Comparison", PURPLE, [
        ("3.8.1", "Forecast Quality Metrics", "MAE, RMSE, naive baselines"),
        ("3.8.2", "Autoscaling Evaluation Metrics", "responsiveness, SLA,\nutilisation, cost"),
        ("3.8.3", "Data Analysis Approach", "feature ablation and lead-lag"),
    ]),
]

X0, X1 = 1.5, 98.5
HEADER_H = 6.4
SUB_H = 15.0
ARROW_H = 5.2
PAD = 0.9          # gap between adjacent subsection boxes
HEAD_GAP = 0.7     # gap between a stage header and its subsection row
ROUND = 1.1

# One accent per stage (chosen 2026-07-30), or the single blue of figure_4_1_ml_pipeline.svg.
# Every pair below is already used elsewhere in docs/, so nothing new enters the palette.
# Run with --both to render the mono alternative alongside for comparison.
MULTI_HUE = True


def stack_height():
    return (len(STAGES) * (HEADER_H + HEAD_GAP + SUB_H)
            + (len(STAGES) - 1) * ARROW_H)


def draw(outfile="docs/figure_3_3_workflow"):
    H = stack_height() + 2.4
    fig_w = 11.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * (H / (X1 - X0)) * 0.94))
    ax.set_xlim(0, 100)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = 1.2
    for i, (num, heading, pair, subs) in enumerate(STAGES):
        accent, pastel = pair if MULTI_HUE else BLUE
        # stage header: saturated accent, white text, mirroring the numbered badges
        # in figure_4_1_ml_pipeline.svg
        ax.add_patch(FancyBboxPatch(
            (X0, y), X1 - X0, HEADER_H,
            boxstyle=f"round,pad=0,rounding_size={ROUND}",
            linewidth=0, facecolor=accent,
        ))
        ax.text((X0 + X1) / 2, y + HEADER_H / 2, f"{num}   {heading}",
                ha="center", va="center", color="white", fontsize=13,
                fontweight="semibold")
        y += HEADER_H + HEAD_GAP

        n = len(subs)
        w = (X1 - X0 - PAD * (n - 1)) / n
        for j, (snum, stitle, detail) in enumerate(subs):
            x = X0 + j * (w + PAD)
            ax.add_patch(FancyBboxPatch(
                (x, y), w, SUB_H,
                boxstyle=f"round,pad=0,rounding_size={ROUND}",
                linewidth=1.5, edgecolor=accent, facecolor=pastel,
            ))
            cx = x + w / 2
            ax.text(cx, y + 2.8, snum, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=accent)
            ax.text(cx, y + 6.9, stitle, ha="center", va="center",
                    fontsize=11, fontweight="semibold", color=INK, linespacing=1.4)
            ax.text(cx, y + SUB_H - 3.1, detail, ha="center", va="center",
                    fontsize=9.2, color=SECONDARY, linespacing=1.4)
        y += SUB_H

        # connector: thin shaft with a filled head, in the house connector grey
        if i < len(STAGES) - 1:
            top, bot = y + 0.9, y + ARROW_H - 0.9
            ax.add_line(Line2D([50, 50], [top, bot - 1.4], color=CONNECTOR,
                               linewidth=1.6, solid_capstyle="butt"))
            ax.add_patch(Polygon([[48.0, bot - 1.5], [52.0, bot - 1.5], [50, bot]],
                                 closed=True, facecolor=CONNECTOR, linewidth=0))
            y += ARROW_H

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{outfile}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{outfile}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outfile}.png and .svg")


if __name__ == "__main__":
    import sys
    if "--both" in sys.argv:
        MULTI_HUE = True
        draw("docs/figure_3_3_workflow")
        MULTI_HUE = False
        draw("docs/figure_3_3_workflow_mono")
    else:
        draw()
