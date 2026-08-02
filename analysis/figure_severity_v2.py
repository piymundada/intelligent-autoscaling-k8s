#!/usr/bin/env python3
"""Figure 5.8.3: cumulative SLA-breach severity by scenario and arm, Minikube.

Regenerated 2026-07-31. The previous render (docs/figure_5_4_severity.png) was built from
results/c1/n2_final.json, whose Minikube RPS-only rows are pre-correction n = 2 means that
average a superseded container-image run with its corrected repeat. Applying the dagger rule
uniformly, as Table 5.5.1 now does, changes one bar materially:

    spike  RPS-only   263.1 (mean of 397.1 and 129.1)  ->  129.1   (corrected run only)
    bursty RPS-only  1536.5 (mean of 1535.7 and 1537.3) -> 1537.3  (rounds to 1,537 either way)

The old figure also labelled each panel with a single n, which is now wrong: on spike the HPA and
JVM-aware arms rest on two runs while RPS-only rests on one. The n is therefore given per bar.

Arm colours match Figure 5.10.1 so that a reader moving between the two ablation figures sees the
same arm in the same colour. Values are hard-coded from the corrected tables rather than
re-aggregated, so the figure and the tables cannot drift.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

INK = "#1f1f1f"
COL = {"HPA": "#E24B4A", "JVM-aware": "#2ECC71", "RPS-only": "#378ADD"}

# scenario -> arm -> (severity in seconds, n)
DATA = {
    "Spike":  {"HPA": (94.8, 2), "JVM-aware": (18.0, 2), "RPS-only": (129.1, 1)},
    "Bursty": {"HPA": (16.9, 1), "JVM-aware": (17.0, 1), "RPS-only": (1537.3, 1)},
}


def draw(outfile="docs/figure_5_8_3_severity"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (scn, arms) in zip(axes, DATA.items()):
        names = list(arms)
        vals = [arms[a][0] for a in names]
        x = np.arange(len(names))
        bars = ax.bar(x, vals, 0.6, color=[COL[a] for a in names],
                      edgecolor="white", linewidth=0.6)
        top = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=10, fontweight="semibold",
                    color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}\n(n = {arms[a][1]})" for a in names], fontsize=9.5)
        ax.set_ylabel("cumulative breach severity (s above 300 ms)", fontsize=9.5)
        ax.set_title(scn, fontsize=11, fontweight="semibold", color=INK)
        ax.set_ylim(0, top * 1.15)
        ax.grid(axis="y", color="#E5E4E0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.savefig(f"{outfile}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{outfile}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outfile}.png and .svg")


if __name__ == "__main__":
    import json
    stale = json.load(open("results/c1/n2_final.json"))
    print("stale severity in results/c1/n2_final.json (NOT used here):")
    for scn in ("SPIKE", "BURSTY"):
        v = stale[scn]["RPS-only"]
        print(f"   {scn:7} RPS-only n={v.get('n')} severity={v['sev']:.1f}")
    print("corrected: spike 129.1 (n=1), bursty 1537.3 (n=1)")
    draw()
