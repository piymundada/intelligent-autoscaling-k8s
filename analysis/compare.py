"""
Compare HPA baseline vs predictive autoscaler across all evaluation metrics
defined in Section 8.7 of the thesis:

  1. Scaling responsiveness: scale-out/in event counts, max/avg replicas
  2. Performance & SLA: p95/p99 latency, error rate, SLA violations
  3. Resource utilisation: CPU usage, replica-hours
  4. Prediction quality: MAE, RMSE (from LSTM training results JSON)

Usage:
    python3.11 compare.py \
        --hpa-csv      ../data/hpa_bursty_run1_merged.csv \
        --pred-csv     ../ml/logs/predictive_autoscaler.csv \
        --pred-metrics ../data/pred_bursty_run1_merged.csv \
        --model-results ../ml/models/lstm_forecaster_results.json \
        --output-dir   results/bursty/
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running from either analysis/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from config import SLA_THRESHOLD_MS, SCRAPE_INTERVAL_S


def load_csv(path: str, align_to_load: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    df = df.sort_index()
    if align_to_load and "request_rate_rps" in df.columns:
        # Trim to the load-active window (first to last sample with RPS data) so
        # runs whose Prometheus collection windows differ in padding are compared
        # over the same load shape: otherwise replica-hours are not comparable.
        active = df["request_rate_rps"].notna()
        if active.any():
            df = df.loc[active.idxmax(): active[::-1].idxmax()]
    return df


def compute_scaling_responsiveness(df: pd.DataFrame, label: str) -> dict:
    """Count scale-out / scale-in events and summarise replica behaviour."""
    replicas = df["replica_count"].dropna()
    delta = replicas.diff().fillna(0)
    scale_outs = delta[delta > 0]
    scale_ins = delta[delta < 0]

    return {
        f"{label}_scale_out_count": len(scale_outs),
        f"{label}_scale_in_count": len(scale_ins),
        f"{label}_total_scaling_events": len(scale_outs) + len(scale_ins),
        f"{label}_max_replicas": int(replicas.max()),
        f"{label}_avg_replicas": round(float(replicas.mean()), 2),
        f"{label}_replica_std": round(float(replicas.std()), 2),
    }


def compute_sla_metrics(df: pd.DataFrame, label: str, sla_ms: float) -> dict:
    """Compute latency percentiles, SLA violations, error rate and throughput."""
    p95 = df["response_time_p95_ms"].dropna() if "response_time_p95_ms" in df.columns else pd.Series(dtype=float)
    p99 = df["response_time_p99_ms"].dropna() if "response_time_p99_ms" in df.columns else p95
    total = len(p95)
    violations = int((p95 > sla_ms).sum())

    return {
        f"{label}_p95_mean_ms": round(float(p95.mean()), 2) if total else None,
        f"{label}_p95_max_ms": round(float(p95.max()), 2) if total else None,
        f"{label}_p99_mean_ms": round(float(p99.mean()), 2) if len(p99) else None,
        f"{label}_sla_violation_count": violations,
        f"{label}_sla_violation_pct": round(100 * violations / total, 2) if total else 0,
        f"{label}_error_rate_mean": (
            round(float(df["error_rate"].mean()), 4) if "error_rate" in df.columns else None
        ),
        f"{label}_throughput_mean_rps": (
            round(float(df["throughput_rps"].mean()), 2) if "throughput_rps" in df.columns else None
        ),
    }


def compute_resource_metrics(df: pd.DataFrame, interval_s: int, label: str) -> dict:
    """Compute replica-hours and CPU utilisation (percent, from Micrometer)."""
    replicas = df["replica_count"].dropna()
    duration_hours = len(replicas) * interval_s / 3600
    replica_hours = float((replicas * (interval_s / 3600)).sum())

    # cpu_usage_percent is from process_cpu_usage * 100 (Micrometer)
    cpu = df["cpu_usage_percent"].dropna() if "cpu_usage_percent" in df.columns else pd.Series(dtype=float)

    result = {
        f"{label}_replica_hours": round(replica_hours, 4),
        f"{label}_duration_hours": round(duration_hours, 4),
        f"{label}_avg_replicas": round(float(replicas.mean()), 2),
    }
    if len(cpu):
        result[f"{label}_cpu_mean_pct"] = round(float(cpu.mean()), 2)
        result[f"{label}_cpu_max_pct"] = round(float(cpu.max()), 2)
    return result


def compare(args: argparse.Namespace) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)
    results: dict = {}

    hpa_df = load_csv(args.hpa_csv, args.align_to_load)
    pred_df = load_csv(args.pred_metrics, args.align_to_load)
    if args.align_to_load:
        print(f"Aligned to load-active window: "
              f"HPA {len(hpa_df)} rows | predictive {len(pred_df)} rows")

    # 1. Scaling responsiveness
    results.update(compute_scaling_responsiveness(hpa_df, "hpa"))
    results.update(compute_scaling_responsiveness(pred_df, "pred"))

    # 2. SLA / performance
    results.update(compute_sla_metrics(hpa_df, "hpa", args.sla_threshold_ms))
    results.update(compute_sla_metrics(pred_df, "pred", args.sla_threshold_ms))

    # 3. Resource utilisation
    results.update(compute_resource_metrics(hpa_df, SCRAPE_INTERVAL_S, "hpa"))
    results.update(compute_resource_metrics(pred_df, SCRAPE_INTERVAL_S, "pred"))

    # 4. Prediction quality (from LSTM training JSON)
    if args.model_results and os.path.exists(args.model_results):
        with open(args.model_results) as f:
            ml = json.load(f)
        results["lstm_test_mae"] = ml.get("test_mae")
        results["lstm_test_rmse"] = ml.get("test_rmse")
        results["lstm_test_mse"] = ml.get("test_mse")
        results["lstm_epochs_trained"] = ml.get("epochs_trained")

    # Delta / improvement calculations
    def improvement(key_pred: str, key_hpa: str, negate: bool = False) -> float:
        p = results.get(key_pred) or 0
        h = results.get(key_hpa) or 0
        if h == 0:
            return 0.0
        change = ((p - h) / h) * 100
        return round(-change if negate else change, 2)

    results["improvement_sla_violations_pct"] = improvement(
        "pred_sla_violation_count", "hpa_sla_violation_count", negate=True
    )
    results["improvement_p95_latency_pct"] = improvement(
        "pred_p95_mean_ms", "hpa_p95_mean_ms", negate=True
    )
    results["improvement_replica_hours_pct"] = improvement(
        "pred_replica_hours", "hpa_replica_hours", negate=True
    )
    results["improvement_scaling_events_pct"] = improvement(
        "pred_total_scaling_events", "hpa_total_scaling_events", negate=True
    )

    # Persist and print
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    print(f"\nFull results -> {results_path}")
    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 65)
    print("COMPARISON RESULTS")
    print("=" * 65)
    categories = [
        ("Scaling Responsiveness", [
            "scale_out_count", "scale_in_count",
            "total_scaling_events", "max_replicas", "avg_replicas",
        ]),
        ("SLA & Performance", [
            "p95_mean_ms", "p95_max_ms", "p99_mean_ms",
            "sla_violation_count", "sla_violation_pct",
        ]),
        ("Resource Utilisation", [
            "replica_hours", "cpu_mean_pct", "cpu_max_pct",
        ]),
    ]
    for cat, keys in categories:
        print(f"\n{cat}:")
        print(f"  {'Metric':<35} {'HPA':>12} {'Predictive':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12}")
        for k in keys:
            hpa_v = results.get(f"hpa_{k}", "-")
            pred_v = results.get(f"pred_{k}", "-")
            print(f"  {k:<35} {str(hpa_v):>12} {str(pred_v):>12}")

    print("\nImprovements (positive = predictive is better):")
    for k, v in results.items():
        if k.startswith("improvement_") and v is not None:
            label = k.replace("improvement_", "").replace("_", " ")
            print(f"  {label:<35}: {v:>+.1f}%")

    if "lstm_test_mae" in results:
        print(f"\nLSTM Prediction Quality:")
        print(f"  MAE  : {results['lstm_test_mae']:.6f}")
        print(f"  RMSE : {results['lstm_test_rmse']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare HPA vs predictive autoscaler")
    parser.add_argument("--hpa-csv", required=True,
                        help="Merged metrics CSV from HPA experiment (e.g. hpa_bursty_run1_merged.csv)")
    parser.add_argument("--pred-csv", default=None,
                        help="Predictive autoscaler decision log CSV (optional)")
    parser.add_argument("--pred-metrics", required=True,
                        help="Merged metrics CSV from predictive experiment")
    parser.add_argument("--model-results", default=None,
                        help="LSTM training results JSON (lstm_forecaster_results.json)")
    parser.add_argument("--sla-threshold-ms", type=float, default=SLA_THRESHOLD_MS,
                        help=f"p95 SLA threshold in ms (default {SLA_THRESHOLD_MS})")
    parser.add_argument("--align-to-load", action="store_true",
                        help="Trim both CSVs to their load-active window (first to "
                             "last row with RPS data) so collection-window padding "
                             "does not skew replica-hours")
    parser.add_argument("--output-dir", default="results/")
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
