"""
Merge Locust stats_history.csv (latency percentiles) with Prometheus CSV (JVM + infra metrics).

Locust records p50/p95/p99 from the client side every few seconds.
Prometheus records JVM + infra metrics every 30s.
This script resamples both to 30s intervals and joins them into one dataset.

Usage:
    python3.11 merge_locust_prometheus.py \
        --locust-csv  data/hpa_steady_run1_stats_history.csv \
        --prom-csv    data/hpa_steady_run1.csv \
        --output      data/hpa_steady_run1_merged.csv
"""

import argparse
import pandas as pd


def load_locust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only the Aggregated row (all endpoints combined)
    df = df[df["Name"] == "Aggregated"].copy()

    # Convert Unix timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Rename columns to match thesis metric names
    df = df.rename(columns={
        "Requests/s":     "request_rate_rps",
        "50%":            "response_time_p50_ms",
        "95%":            "response_time_p95_ms",
        "99%":            "response_time_p99_ms",
        "100%":           "response_time_max_ms",
        "Failures/s":     "failure_rate_rps",
        "User Count":     "user_count",
    })

    keep = ["request_rate_rps", "response_time_p50_ms", "response_time_p95_ms",
            "response_time_p99_ms", "response_time_max_ms", "failure_rate_rps"]
    df = df[[c for c in keep if c in df.columns]]

    # Convert N/A strings to NaN
    df = df.replace("N/A", float("nan"))
    df = df.apply(pd.to_numeric, errors="coerce")

    # Resample to 30s to align with Prometheus
    df = df.resample("30s").mean()

    return df


def load_prometheus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    df.index = df.index.tz_localize("UTC") if df.index.tzinfo is None else df.index
    return df.sort_index()


def _snap_to_30s_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Floor the index to the 30s boundary so Locust (:00/:30, midnight-anchored)
    and Prometheus (query_range points at :15/:45) share one grid. Without this
    the outer join interleaves the two grids and produces rows with alternating
    NaNs that later get dropped. Floor (not round) is used because Prometheus
    points fall midway between 30s boundaries; rounding creates ties that collapse
    adjacent points and halve the resolution, whereas floor maps every point to a
    distinct 30s slot and preserves full 30s sampling."""
    df = df.copy()
    df.index = df.index.floor("30s")
    df = df.groupby(level=0).mean()
    return df.sort_index()


def merge(locust_path: str, prom_path: str, output: str, warmup_minutes: int = 5):
    locust_df = _snap_to_30s_grid(load_locust(locust_path))
    prom_df   = _snap_to_30s_grid(load_prometheus(prom_path))

    print(f"Locust rows: {len(locust_df)} | Prometheus rows: {len(prom_df)}")

    # Outer join: Locust has ground-truth latency; Prometheus has JVM/infra
    # Drop p95/p99 from Prometheus (they are empty without histograms) before joining
    for col in ["response_time_p95_ms", "response_time_p99_ms", "response_time_max_ms"]:
        if col in prom_df.columns:
            prom_df = prom_df.drop(columns=[col])

    # Drop duplicate columns from Prometheus that Locust already provides
    for col in ["request_rate_rps", "throughput_rps"]:
        if col in prom_df.columns:
            prom_df = prom_df.drop(columns=[col])

    merged = locust_df.join(prom_df, how="outer")
    merged = merged.sort_index().ffill(limit=2).bfill(limit=1)

    # Drop warm-up rows. This is the single warm-up authority for the pipeline;
    # preprocess.py should be run with --warmup-minutes 0 to avoid double-stripping.
    if warmup_minutes > 0:
        cutoff = merged.index.min() + pd.Timedelta(minutes=warmup_minutes)
        merged = merged[merged.index >= cutoff]

    # Drop rows with all-NaN (usually at edges)
    merged = merged.dropna(how="all")

    merged.index.name = "timestamp"
    merged.to_csv(output)

    print(f"Merged: {len(merged)} rows x {len(merged.columns)} columns -> {output}")
    print(f"\nLatency summary (client-side from Locust):")
    if "response_time_p95_ms" in merged.columns:
        p95 = merged["response_time_p95_ms"].dropna()
        print(f"  p95 mean={p95.mean():.1f}ms  max={p95.max():.1f}ms  "
              f"violations(>300ms)={int((p95 > 300).sum())}/{len(p95)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locust-csv",  required=True)
    parser.add_argument("--prom-csv",    required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--warmup-minutes", type=int, default=5,
                        help="Warm-up minutes to drop from the start (default 5)")
    args = parser.parse_args()
    merge(args.locust_csv, args.prom_csv, args.output, args.warmup_minutes)


if __name__ == "__main__":
    main()
