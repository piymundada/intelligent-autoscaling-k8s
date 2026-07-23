"""
EKS vs Minikube cross-environment comparison (spike + bursty, the two scenarios both
campaigns share). Minikube numbers from results/c1/n2_final.json; EKS computed from
the merged CSVs. Produces a table + results/eks/figures/fig_eks_vs_minikube.png.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLA_MS, STEP = 300, 30
HPA, JVM, RPS = "#e74c3c", "#2ecc71", "#3498db"
mk = json.load(open("results/c1/n2_final.json"))


def eks(scn, arm):
    """mean over available repeats of (violpct, severity, replica-hours)."""
    paths = [f"data/{scn}_m6_run{r}_{'hpa' if arm=='HPA' else 'pred'}_merged.csv" for r in (1, 2)]
    if arm == "RPS-only":
        paths = [f"data/{scn}_rpsonly_run1_merged.csv"]
    V, S, R = [], [], []
    for p in paths:
        if not os.path.exists(p): continue
        d = pd.read_csv(p); p95 = d["response_time_p95_ms"]
        V.append(100 * (p95 > SLA_MS).sum() / len(d))
        S.append(np.maximum(p95 - SLA_MS, 0).sum() / 1000)
        R.append(d["replica_count"].sum() * STEP / 3600)
    return (np.mean(V), np.mean(S), np.mean(R)) if V else (np.nan, np.nan, np.nan)

out = []
def pr(s=""): out.append(s); print(s)

pr("="*88)
pr("EKS vs MINIKUBE  (SLA violation %, breach severity, replica-hours)")
pr("  Minikube: max=4, rps/pod≈35, p95med≈15ms (strong pods)")
pr("  EKS:      max=6, rps/pod≈18, p95med higher (½-capacity pods, real EC2)")
pr("="*88)
for SCN in ["SPIKE", "BURSTY"]:
    scn = SCN.lower()
    pr(f"\n{SCN}")
    pr(f"  {'arm':9s} | {'Minikube viol%':>14s} {'EKS viol%':>10s} | {'MK sev':>7s} {'EKS sev':>8s} | {'MK rh':>6s} {'EKS rh':>7s}")
    pr("  " + "-"*70)
    for arm in ["HPA", "JVM-aware", "RPS-only"]:
        mkd = mk[SCN][arm]
        ev, es, er = eks(scn, "HPA" if arm == "HPA" else ("RPS-only" if arm == "RPS-only" else "JVM"))
        pr(f"  {arm:9s} | {mkd['violpct']:14.1f} {ev:10.1f} | {mkd['sev']:7.0f} {es:8.0f} | {mkd['rh']:6.2f} {er:7.2f}")

pr("\nKEY CROSS-ENVIRONMENT FINDINGS:")
pr("  1. Idle pathology REPRODUCES & strengthens: HPA pinned at 4 (MK) / 6 (EKS) for 3h.")
pr("  2. SLA: predictive beats HPA in BOTH envs — via severity on MK (shallower breaches),")
pr("     via violation COUNT on EKS (fewer breaches).")
pr("  3. Ablation diverges: on MK the floor is load-bearing (RPS-only SLA collapses,")
pr("     sev 263/1536); on EKS RPS-only stays competitive — the retrained forecast +")
pr("     15s loop is more self-sufficient, so the floor's marginal value is smaller.")

# figure: grouped bars, Minikube vs EKS, SLA viol% per arm (spike+bursty)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
arms = ["HPA", "JVM-aware", "RPS-only"]; cols = [HPA, JVM, RPS]
for ax, SCN in zip(axes, ["SPIKE", "BURSTY"]):
    scn = SCN.lower()
    mkv = [mk[SCN][a]["violpct"] for a in arms]
    ekv = [eks(scn, "HPA" if a == "HPA" else ("RPS-only" if a == "RPS-only" else "JVM"))[0] for a in arms]
    x = np.arange(len(arms)); w = 0.38
    ax.bar(x - w/2, mkv, w, color="#bdc3c7", label="Minikube")
    ax.bar(x + w/2, ekv, w, color=cols, label="EKS")
    for i in range(len(arms)):
        ax.text(x[i]-w/2, mkv[i], f"{mkv[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i]+w/2, ekv[i], f"{ekv[i]:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=10)
    ax.set_ylabel("SLA violations %"); ax.set_title(f"{SCN.title()}")
    ax.legend()
fig.suptitle("EKS vs Minikube: SLA violations by arm (EKS bars coloured by arm; grey = Minikube)",
             fontweight="bold")
os.makedirs("results/eks/figures", exist_ok=True)
fig.tight_layout(); fig.savefig("results/eks/figures/fig_eks_vs_minikube.png", dpi=150, bbox_inches="tight")
pr("\nsaved -> results/eks/figures/fig_eks_vs_minikube.png")
open("results/eks/eks_vs_minikube.txt", "w").write("\n".join(out))
pr("saved -> results/eks/eks_vs_minikube.txt")
