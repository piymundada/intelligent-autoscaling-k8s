"""
Generate all thesis figures from collected experiment data.

Produces:
  fig1_replica_timeline.png    replicas over time: HPA vs predictive
  fig2_latency_comparison.png  p95/p99 response time comparison
  fig3_sla_violations.png      SLA violation periods highlighted
  fig4_resource_usage.png      CPU + memory usage comparison
  fig5_jvm_metrics.png         GC pause, heap usage, thread count
  fig6_lstm_training.png       training/validation loss curves
  fig7_lstm_predictions.png    predicted vs actual RPS on test set

Usage:
    python visualise.py \
        --hpa-metrics    ../data/experiment_hpa_spike.csv \
        --pred-metrics   ../data/experiment_predictive_spike.csv \
        --model-results  ../ml/models/lstm_forecaster_results.json \
        --test-preds     ../ml/models/lstm_forecaster_test_preds.npy \
        --test-targets   ../ml/models/lstm_forecaster_test_targets.npy \
        --output-dir     results/figures/ \
        --sla-ms         300
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

STYLE = {
    "hpa_color": "#e74c3c",
    "pred_color": "#2ecc71",
    "sla_color": "#f39c12",
    "jvm_color": "#9b59b6",
    "figsize": (12, 5),
    "dpi": 150,
}
plt.rcParams["font.family"] = "DejaVu Sans"


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_csv(path):
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df.sort_index()


def fig1_replica_timeline(hpa_df, pred_df, out_dir):
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(hpa_df.index, hpa_df["replica_count"], color=STYLE["hpa_color"],
            label="HPA (reactive)", linewidth=1.5)
    ax.plot(pred_df.index, pred_df["replica_count"], color=STYLE["pred_color"],
            label="Predictive (LSTM)", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pod Replicas")
    ax.set_title("Replica Count Over Time: HPA vs Predictive Autoscaler")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)
    save(fig, os.path.join(out_dir, "fig1_replica_timeline.png"))


def fig2_latency_comparison(hpa_df, pred_df, out_dir, sla_ms):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, label, color in [
        (axes[0], hpa_df, "HPA (reactive)", STYLE["hpa_color"]),
        (axes[1], pred_df, "Predictive (LSTM)", STYLE["pred_color"]),
    ]:
        if "response_time_p95_ms" in df:
            ax.plot(df.index, df["response_time_p95_ms"], color=color,
                    label="p95", linewidth=1.2)
        if "response_time_p99_ms" in df:
            ax.plot(df.index, df["response_time_p99_ms"], color=color,
                    label="p99", linewidth=1.2, linestyle=":")
        ax.axhline(sla_ms, color=STYLE["sla_color"], linestyle="--",
                   linewidth=1.5, label=f"SLA ({sla_ms} ms)")
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.set_ylabel("Response Time (ms)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Response Time (p95 / p99) Comparison", fontsize=13)
    save(fig, os.path.join(out_dir, "fig2_latency_comparison.png"))


def fig3_sla_violations(hpa_df, pred_df, out_dir, sla_ms):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, df, label, color in [
        (axes[0], hpa_df, "HPA (reactive)", STYLE["hpa_color"]),
        (axes[1], pred_df, "Predictive (LSTM)", STYLE["pred_color"]),
    ]:
        if "response_time_p95_ms" not in df.columns:
            continue
        p95 = df["response_time_p95_ms"].fillna(0)
        ax.fill_between(df.index, p95, sla_ms,
                        where=(p95 > sla_ms), color="red", alpha=0.3,
                        label="SLA violation")
        ax.plot(df.index, p95, color=color, linewidth=1.0, label="p95 latency")
        ax.axhline(sla_ms, color=STYLE["sla_color"], linestyle="--",
                   linewidth=1.2, label=f"SLA ({sla_ms} ms)")
        violations = (p95 > sla_ms).sum()
        ax.set_title(f"{label} — {violations} SLA violations ({100*violations/len(p95):.1f}%)")
        ax.set_ylabel("Response Time (ms)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time")
    fig.suptitle("SLA Violation Periods (p95 > 300 ms)", fontsize=13)
    save(fig, os.path.join(out_dir, "fig3_sla_violations.png"))


def fig4_resource_usage(hpa_df, pred_df, out_dir):
    # cpu_usage_percent is from Micrometer process_cpu_usage * 100
    # (cAdvisor / memory_usage_bytes not available in this lab setup)
    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    col = "cpu_usage_percent"
    if col in hpa_df.columns:
        ax.plot(hpa_df.index, hpa_df[col],
                color=STYLE["hpa_color"], label="HPA", linewidth=1.2)
    if col in pred_df.columns:
        ax.plot(pred_df.index, pred_df[col],
                color=STYLE["pred_color"], label="Predictive",
                linewidth=1.2, linestyle="--")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("CPU Utilisation: HPA vs Predictive Autoscaler")
    save(fig, os.path.join(out_dir, "fig4_resource_usage.png"))


def fig5_jvm_metrics(pred_df, out_dir):
    jvm_cols = [
        ("jvm_heap_utilisation", "Heap Utilisation (ratio)", 1),
        ("jvm_gc_avg_pause_ms", "GC Avg Pause (ms)", 1),
        ("jvm_threads_live", "Live Threads", 1),
    ]
    available = [(c, l, s) for c, l, s in jvm_cols if c in pred_df.columns]
    if not available:
        print("  No JVM metrics available in pred_df — skipping fig5")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, (col, ylabel, scale) in zip(axes, available):
        ax.plot(pred_df.index, pred_df[col] * scale,
                color=STYLE["jvm_color"], linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time")
    fig.suptitle("JVM Runtime Metrics During Predictive Experiment", fontsize=13)
    save(fig, os.path.join(out_dir, "fig5_jvm_metrics.png"))


def fig6_lstm_training(model_results_path, out_dir):
    if not model_results_path or not os.path.exists(model_results_path):
        print("  No model results file — skipping fig6")
        return
    with open(model_results_path) as f:
        results = json.load(f)
    hist = results.get("history", {})
    if not hist:
        return

    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", color="#3498db")
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#e74c3c")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["val_mae"], label="Val MAE", color="#2ecc71")
    axes[1].plot(epochs, hist["val_rmse"], label="Val RMSE", color="#9b59b6")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Validation MAE & RMSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    save(fig, os.path.join(out_dir, "fig6_lstm_training.png"))


def fig7_lstm_predictions(preds_path, targets_path, out_dir):
    if not preds_path or not os.path.exists(preds_path):
        print("  No predictions file — skipping fig7")
        return
    preds = np.load(preds_path)
    targets = np.load(targets_path)
    n = len(preds)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(x, targets, color="#3498db", label="Actual RPS", linewidth=1.2)
    ax.plot(x, preds, color="#e74c3c", label="LSTM Predicted RPS",
            linewidth=1.2, linestyle="--")
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ax.set_xlabel("Test Timestep")
    ax.set_ylabel("Request Rate (RPS, normalised)")
    ax.set_title(f"LSTM: Predicted vs Actual — MAE={mae:.4f}, RMSE={rmse:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, os.path.join(out_dir, "fig7_lstm_predictions.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--hpa-metrics", required=True)
    parser.add_argument("--pred-metrics", required=True)
    parser.add_argument("--model-results", default=None)
    parser.add_argument("--test-preds", default=None)
    parser.add_argument("--test-targets", default=None)
    parser.add_argument("--output-dir", default="results/figures/")
    parser.add_argument("--sla-ms", type=float, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hpa_df = load_csv(args.hpa_metrics)
    pred_df = load_csv(args.pred_metrics)

    print("Generating figures...")
    fig1_replica_timeline(hpa_df, pred_df, args.output_dir)
    fig2_latency_comparison(hpa_df, pred_df, args.output_dir, args.sla_ms)
    fig3_sla_violations(hpa_df, pred_df, args.output_dir, args.sla_ms)
    fig4_resource_usage(hpa_df, pred_df, args.output_dir)
    fig5_jvm_metrics(pred_df, args.output_dir)
    fig6_lstm_training(args.model_results, args.output_dir)
    fig7_lstm_predictions(args.test_preds, args.test_targets, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
