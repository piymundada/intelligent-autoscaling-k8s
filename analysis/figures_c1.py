"""
Thesis figures for the C1 n=2 campaign: drawn from the FINAL data so the figures
match the README results tables (the old data/figures/* were pre-campaign).

Each figure maps to a research question and the JSON/CSV it is drawn from:
  fig_c1_severity        RQ4/RQ5: SLA-breach severity, 3 arms, n=2 error bars
  fig_c1_spike_timeline  RQ1/RQ4: replica timeline HPA vs JVM-aware (proactive)
  fig_c1_idle_scalein    RQ1/RQ5: 3h idle, HPA pinned at 4 vs CPU ~3%
  fig_c1_ablation        RQ2: floor-on vs floor-off collapse, both scenarios
  fig_c1_forecast        RQ3: LSTM vs persistence vs moving-average
  fig_c1_leadlag         RQ2: JVM signals leading p95 latency

Usage:  /usr/bin/python3 analysis/figures_c1.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/c1/figures"
HPA, JVM, RPS = "#e74c3c", "#2ecc71", "#3498db"
SLA = "#f39c12"
DPI = 150
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


def aligned(r):
    df = pd.read_csv(f"data/{r}_merged.csv", index_col="timestamp", parse_dates=True).sort_index()
    a = df["request_rate_rps"].notna()
    return df.loc[a.idxmax(): a[::-1].idxmax()]


# fig 1: severity (RQ4/RQ5)
def fig_severity():
    n2 = json.load(open("results/c1/n2_final.json"))
    arms = ["HPA", "JVM-aware", "RPS-only"]
    colors = [HPA, JVM, RPS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scen in zip(axes, ["SPIKE", "BURSTY"]):
        sev = [n2[scen][a]["sev"] for a in arms]
        ax.bar(arms, sev, color=colors)
        for i, v in enumerate(sev):
            ax.text(i, v, f" {v:.0f}", ha="center", va="bottom", fontweight="bold")
        # arms may differ in n (e.g. bursty: HPA/JVM n=1, RPS-only n=2), show the range, not one arm's n
        ns = sorted({n2[scen][a]["n"] for a in arms})
        ntxt = f"n={ns[0]}" if len(ns) == 1 else f"n={ns[0]}–{ns[-1]}"
        ax.set_title(f"{scen.title()} — SLA-breach severity ({ntxt})")
        ax.set_ylabel("severity index  Σ(ms over 300)/1000")
        ax.set_ylim(0, max(sev) * 1.18)
    fig.suptitle("Breach severity: JVM-aware breaches are far shallower; floor-off collapses",
                 fontweight="bold")
    save(fig, "fig_c1_severity.png")


# fig 2: spike replica timeline (RQ1/RQ4)
def fig_spike_timeline():
    h, p = aligned("hpa_spike_run4"), aligned("pred_spike_run4")
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    t_h = (h.index - h.index[0]).total_seconds() / 60
    t_p = (p.index - p.index[0]).total_seconds() / 60
    ax1.step(t_h, h["replica_count"], where="post", color=HPA, lw=2, label="HPA replicas")
    ax1.step(t_p, p["replica_count"], where="post", color=JVM, lw=2, label="JVM-aware replicas")
    ax1.set_xlabel("minutes into load"); ax1.set_ylabel("replicas"); ax1.set_ylim(0, 4.5)
    ax2 = ax1.twinx()
    ax2.plot(t_h, h["request_rate_rps"], color="gray", alpha=0.4, lw=1, label="RPS (HPA run)")
    ax2.set_ylabel("request rate (RPS)", color="gray")
    ax1.legend(loc="upper left"); ax1.set_title("Spike: replica response to surges (HPA vs JVM-aware)")
    save(fig, "fig_c1_spike_timeline.png")


# fig 2b: spike p95 latency over time (RQ4/RQ5)
def fig_spike_latency():
    h, p = aligned("hpa_spike_run4"), aligned("pred_spike_run4")
    th = (h.index - h.index[0]).total_seconds() / 60
    tp = (p.index - p.index[0]).total_seconds() / 60
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    # faint RPS context on a back axis
    ax2 = ax1.twinx()
    ax2.fill_between(th, h["request_rate_rps"], color="gray", alpha=0.12, zorder=0)
    ax2.set_ylabel("request rate (RPS)", color="gray")
    ax2.set_ylim(0, max(h["request_rate_rps"].max(), p["request_rate_rps"].max()) * 1.05)
    # p95 on log scale (median ~16 ms vs spikes ~40 s)
    ax1.plot(th, h["response_time_p95_ms"], color=HPA, lw=1.8, label="HPA p95", zorder=3)
    ax1.plot(tp, p["response_time_p95_ms"], color=JVM, lw=1.8, label="JVM-aware p95", zorder=3)
    ax1.axhline(300, color=SLA, ls="--", lw=1.5, label="SLA 300 ms", zorder=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("minutes into load")
    ax1.set_ylabel("p95 response time (ms, log scale)")
    ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
    ax1.legend(loc="upper right")
    ax1.set_title("Spike: HPA breaches the SLA deeply on every surge; JVM-aware stays controlled")
    save(fig, "fig_c1_spike_latency.png")


# fig 3: 3h idle scale-in (RQ1/RQ5)
def fig_idle():
    df = pd.read_csv("data/hpa_idle_obs_3h.csv", index_col="timestamp", parse_dates=True).sort_index()
    t = (df.index - df.index[0]).total_seconds() / 60
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.step(t, df["replica_count"], where="post", color=HPA, lw=2.5, label="HPA replicas")
    ax1.axhline(1, color=JVM, ls="--", lw=1.5, label="predictive returns to 1 (every cooldown)")
    ax1.set_xlabel("minutes of zero load"); ax1.set_ylabel("replicas", color=HPA)
    ax1.set_ylim(0, 4.5)
    ax2 = ax1.twinx()
    ax2.plot(t, df["cpu_usage_percent"], color="#34495e", lw=1.2, label="CPU %")
    ax2.axhline(60, color=SLA, ls=":", lw=1, label="HPA CPU target 60%")
    ax2.set_ylabel("CPU %", color="#34495e"); ax2.set_ylim(0, 70)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="center right")
    ax1.set_title("HPA holds 4 idle replicas for 3 h at ~3% CPU — committed heap blocks scale-in")
    save(fig, "fig_c1_idle_scalein.png")


# fig 4: ablation collapse (RQ2)
def fig_ablation():
    n2 = json.load(open("results/c1/n2_final.json"))
    arms = ["HPA", "JVM-aware", "RPS-only"]
    colors = [HPA, JVM, RPS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scen in zip(axes, ["SPIKE", "BURSTY"]):
        vio = [n2[scen][a]["viol"] for a in arms]
        ax.bar(arms, vio, color=colors)
        for i, v in enumerate(vio):
            ax.text(i, v, f" {v:.0f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"{scen.title()} — SLA violations (count)")
        ax.set_ylabel("p95 > 300 ms samples")
        ax.set_ylim(0, max(vio) * 1.18)
    fig.suptitle("Ablation: removing the JVM floor (RPS-only) collapses SLA — floor is load-bearing",
                 fontweight="bold")
    save(fig, "fig_c1_ablation.png")


# fig 5: forecast quality (RQ3)
def fig_forecast():
    fq = json.load(open("results/forecast_quality.json"))["models"]
    models = ["lstm", "persistence", "moving_average"]
    labels = ["LSTM", "Persistence", "Moving avg"]
    rmse = [fq[m]["rmse_rps"] for m in models]
    mae = [fq[m]["mae_rps"] for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, mae, w, label="MAE (RPS)", color="#95a5a6")
    ax.bar(x + w / 2, rmse, w, label="RMSE (RPS)", color=JVM)
    for i in range(len(models)):
        ax.text(x[i] - w / 2, mae[i], f"{mae[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + w / 2, rmse[i], f"{rmse[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("error (RPS)")
    ax.legend(); ax.set_title("LSTM forecast vs naive baselines (60 s horizon)")
    save(fig, "fig_c1_forecast.png")


# fig 6: JVM lead-lag (RQ2)
def fig_leadlag():
    ll = json.load(open("results/feature_ablation.json"))["lead_lag_vs_p95"]
    feats = ["jvm_gc_avg_pause_ms", "cpu_usage_percent", "jvm_heap_used_bytes",
             "jvm_threads_live", "request_rate_rps"]
    feats = [f for f in feats if f in ll]
    lags = [ll[f]["peak_lag_seconds"] for f in feats]
    corr = [abs(ll[f]["peak_corr"]) for f in feats]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(feats, lags, color=[JVM if "jvm" in f or "cpu" in f else "#95a5a6" for f in feats])
    for i, (lg, c) in enumerate(zip(lags, corr)):
        ax.text(lg, i, f"  +{lg:.0f}s (r={c:.2f})", va="center", fontsize=8)
    ax.set_xlabel("seconds the signal LEADS p95 latency")
    ax.set_title("JVM/saturation signals lead latency degradation (basis for the scale-up floor)")
    ax.invert_yaxis()
    save(fig, "fig_c1_leadlag.png")


if __name__ == "__main__":
    print("Generating C1 campaign figures -> results/c1/figures/")
    fig_severity()
    fig_spike_timeline()
    fig_spike_latency()
    fig_idle()
    fig_ablation()
    fig_forecast()
    fig_leadlag()
    print("done.")
