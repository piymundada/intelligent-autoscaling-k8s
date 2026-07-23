"""
EKS campaign thesis figures: mirrors analysis/figures_c1.py (Minikube) in style,
drawn from the FINAL EKS data (max=6, control-interval=15s, EKS-retrained LSTM).

  fig_eks_sla            RQ4: SLA violations per scenario, HPA vs JVM-aware (n=2)
  fig_eks_idle           RQ1/RQ5: 3h idle, HPA pinned at 6 vs predictive 1
  fig_eks_cost           RQ1: replica-hours (cost) per scenario + 3h idle
  fig_eks_ablation       RQ2: 3-way HPA / RPS-only / JVM on spike + gc
  fig_eks_spike_timeline RQ4: spike p95 over time, HPA vs JVM-aware vs SLA

Usage:  /usr/bin/python3 analysis/figures_eks.py
"""
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/eks/figures"
HPA, JVM, RPS, SLA = "#e74c3c", "#2ecc71", "#3498db", "#f39c12"
DPI = 150
SLA_MS = 300
STEP = 30
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def m(label):
    return pd.read_csv(f"data/{label}_merged.csv", index_col="timestamp", parse_dates=True).sort_index()

def viol_pct(d): return 100 * (d["response_time_p95_ms"] > SLA_MS).sum() / len(d)
def repl_hours(d): return d["replica_count"].sum() * STEP / 3600


# fig 1: SLA violations per scenario (RQ4), n=2 with error bars
def fig_sla():
    scns = ["bursty", "spike", "gc", "steady"]
    x = np.arange(len(scns)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for arm, label, col, off in [("hpa", "HPA", HPA, -w/2), ("pred", "JVM-aware", JVM, w/2)]:
        means, errs = [], []
        for scn in scns:
            vals = [viol_pct(m(f"{scn}_m6_run{r}_{arm}")) for r in (1, 2)
                    if os.path.exists(f"data/{scn}_m6_run{r}_{arm}_merged.csv")]
            means.append(np.mean(vals)); errs.append(np.std(vals))
        bars = ax.bar(x + off, means, w, yerr=errs, capsize=3, color=col, label=label)
        for i, v in enumerate(means):
            ax.text(x[i] + off, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([s.title() for s in scns])
    ax.set_ylabel("SLA violations  (% of intervals with p95 > 300 ms)")
    ax.legend()
    ax.set_title("EKS: JVM-aware predictive has ≤ HPA SLA violations in every scenario (n=2)")
    save(fig, "fig_eks_sla.png")


# fig 2: 3h idle scale-in (RQ1/RQ5)
def fig_idle():
    h, p = m("idle_3h_hpa"), m("idle_3h_pred")
    th = (h.index - h.index[0]).total_seconds() / 60
    tp = (p.index - p.index[0]).total_seconds() / 60
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.step(th, h["replica_count"], where="post", color=HPA, lw=2.5, label="HPA replicas")
    ax1.step(tp, p["replica_count"], where="post", color=JVM, lw=2, label="JVM-aware replicas")
    ax1.set_xlabel("minutes (15-min spike load, then 3 h idle)")
    ax1.set_ylabel("replicas"); ax1.set_ylim(0, 7)
    ax2 = ax1.twinx()
    ax2.plot(th, h["cpu_usage_percent"], color="#34495e", lw=1, alpha=0.5, label="HPA CPU %")
    ax2.set_ylabel("CPU %", color="#34495e"); ax2.set_ylim(0, 100)
    l1, l2 = ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    ax1.legend(l1[0] + l2[0], l1[1] + l2[1], loc="center right")
    ax1.set_title("EKS 3 h idle: HPA pinned at 6 replicas; JVM-aware returns to 1 "
                  "(20.4 → 5.7 replica-hours, −72%)")
    save(fig, "fig_eks_idle.png")


# fig 3: cost (replica-hours) (RQ1)
def fig_cost():
    scns = ["bursty", "spike", "gc", "steady"]
    x = np.arange(len(scns)); w = 0.38
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    for arm, label, col, off in [("hpa", "HPA", HPA, -w/2), ("pred", "JVM-aware", JVM, w/2)]:
        vals = [np.mean([repl_hours(m(f"{s}_m6_run{r}_{arm}")) for r in (1, 2)
                if os.path.exists(f"data/{s}_m6_run{r}_{arm}_merged.csv")]) for s in scns]
        axL.bar(x + off, vals, w, color=col, label=label)
        for i, v in enumerate(vals): axL.text(x[i] + off, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels([s.title() for s in scns])
    axL.set_ylabel("replica-hours (per run)"); axL.legend(); axL.set_title("Per-scenario cost (load+observe)")
    # 3h idle on the right
    hi, pi = repl_hours(m("idle_3h_hpa")), repl_hours(m("idle_3h_pred"))
    axR.bar(["HPA", "JVM"], [hi, pi], color=[HPA, JVM])
    for i, v in enumerate([hi, pi]): axR.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
    axR.set_ylabel("replica-hours"); axR.set_title("3 h idle (−72%)")
    fig.suptitle("EKS cost: JVM-aware uses fewer replica-hours; the idle gap is the headline", fontweight="bold")
    save(fig, "fig_eks_cost.png")


# fig 4: 3-way ablation (RQ2)
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scn in zip(axes, ["spike", "gc"]):
        arms = [("HPA", f"{scn}_m6_run1_hpa", HPA),
                ("RPS-only", f"{scn}_rpsonly_run1", RPS),
                ("JVM-aware", f"{scn}_m6_run1_pred", JVM)]
        names, vios, cols = [], [], []
        for label, lab, col in arms:
            path = f"data/{lab}_merged.csv"
            if not os.path.exists(path): continue
            names.append(label); vios.append(viol_pct(m(lab))); cols.append(col)
        ax.bar(names, vios, color=cols)
        for i, v in enumerate(vios): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"{scn.title()} — SLA violations %"); ax.set_ylabel("p95 > 300 ms (% intervals)")
        ax.set_ylim(0, max(vios) * 1.25 + 1)
    fig.suptitle("EKS ablation: the JVM floor helps on spike; on gc the forecast alone already meets SLA",
                 fontweight="bold")
    save(fig, "fig_eks_ablation.png")


# fig 5: spike p95 over time (RQ4)
def fig_spike_timeline():
    h, p = m("spike_m6_run1_hpa"), m("spike_m6_run1_pred")
    th = (h.index - h.index[0]).total_seconds() / 60
    tp = (p.index - p.index[0]).total_seconds() / 60
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax2 = ax1.twinx()
    ax2.fill_between(th, h["request_rate_rps"], color="gray", alpha=0.12, zorder=0)
    ax2.set_ylabel("request rate (RPS)", color="gray")
    ax1.plot(th, h["response_time_p95_ms"], color=HPA, lw=1.8, label="HPA p95", zorder=3)
    ax1.plot(tp, p["response_time_p95_ms"], color=JVM, lw=1.8, label="JVM-aware p95", zorder=3)
    ax1.axhline(SLA_MS, color=SLA, ls="--", lw=1.5, label="SLA 300 ms", zorder=2)
    ax1.set_yscale("log"); ax1.set_xlabel("minutes into load")
    ax1.set_ylabel("p95 response time (ms, log)")
    ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
    ax1.legend(loc="upper right")
    ax1.set_title("EKS spike: JVM-aware p95 stays at/under HPA's across the surges")
    save(fig, "fig_eks_spike_timeline.png")


if __name__ == "__main__":
    print("Generating EKS figures -> results/eks/figures/")
    fig_sla(); fig_idle(); fig_cost(); fig_ablation(); fig_spike_timeline()
    print("done.")
