#!/usr/bin/env python3
"""Figure 5.11.1: SLA violations by arm, EKS against Minikube.

Regenerated 2026-07-31. The previous render (docs/figure_5_9_eks_vs_minikube.png) had two defects:

  1. STALE DATA. It read results/c1/n2_final.json, whose Minikube RPS-only aggregates are the
     pre-correction n = 2 means (spike 28.86, bursty 29.67). Applying the container-image dagger
     rule uniformly, as Table 5.5.1 now does, leaves those arms at n = 1 and 26.8 on both
     scenarios. The old figure therefore contradicted the corrected table.
  2. WRONG LEGEND. It coloured the EKS bars by arm (red, green, blue) while the legend claimed
     "EKS = red". The arm is already on the x-axis, so colouring by arm carried no information and
     broke the legend. Two environments now get two colours.

Values are taken from the corrected Tables 5.5.1, 5.6.1 and 5.6.3 rather than re-aggregated, so
the figure and the tables cannot drift.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

MK, EKS = "#B0AFA6", "#378ADD"     # house neutral and house blue
INK, SECONDARY = "#1f1f1f", "#5F5E5A"

ARMS = ["HPA", "JVM-aware", "RPS-only"]

# (Minikube, EKS); None means the arm was not run in that environment for that scenario
DATA = {
    "Spike":  {"Minikube": [15.9, 17.5, 26.8], "EKS": [19.2, 17.8, 20.2]},
    "Bursty": {"Minikube": [1.6, 2.4, 26.8],   "EKS": [8.7, 2.9, None]},
}


def draw(outfile="docs/figure_5_11_1_eks_vs_minikube"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(ARMS))
    w = 0.36

    for ax, (scn, d) in zip(axes, DATA.items()):
        for off, env, col in ((-w / 2, "Minikube", MK), (w / 2, "EKS", EKS)):
            vals = d[env]
            plotted = [0 if v is None else v for v in vals]
            bars = ax.bar(x + off, plotted, w, color=col, label=env,
                          edgecolor="white", linewidth=0.6)
            for xi, (b, v) in enumerate(zip(bars, vals)):
                if v is None:
                    ax.text(x[xi] + off, 0.6, "not run", ha="center", va="bottom",
                            fontsize=8.5, color=SECONDARY, rotation=90)
                else:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS)
        ax.set_ylabel("SLA violations (% of sampled intervals)", fontsize=9.5)
        ax.set_title(scn, fontsize=11, fontweight="semibold", color=INK)
        ax.set_ylim(0, 32)
        ax.grid(axis="y", color="#E5E4E0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5)

    fig.tight_layout(pad=1.0)
    fig.savefig(f"{outfile}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{outfile}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outfile}.png and .svg")


if __name__ == "__main__":
    import json
    stale = json.load(open("results/c1/n2_final.json"))
    print("stale values still in results/c1/n2_final.json (NOT used here):")
    for scn in ("SPIKE", "BURSTY"):
        v = stale[scn]["RPS-only"]
        print(f"   {scn:7} RPS-only n={v.get('n')} violpct={v['violpct']:.2f}")
    print("corrected values used, from Table 5.5.1: spike 26.8 (n=1), bursty 26.8 (n=1)")
    draw()
