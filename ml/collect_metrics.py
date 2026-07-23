"""
Collect experiment metrics from Prometheus and export to CSV.

Run after each experiment scenario to capture the dataset for LSTM training.
Queries all three metric layers defined in Section 8.4.1 of the thesis:
  - Workload layer  : request rate, response time (max), error rate, throughput
  - Infrastructure  : CPU usage (%), replica count
  - JVM layer       : GC pause time/frequency, heap usage, thread count

Note: p95/p99 latency percentiles are sourced from Locust stats_history.csv
(client-side) via merge_locust_prometheus.py, because Spring Boot 3.2's
Observation API bypasses MeterFilter so histogram_quantile returns no data.

Usage:
    python3.11 collect_metrics.py \
        --prometheus-url http://localhost:9090 \
        --start "2024-01-01T10:00:00Z" \
        --end   "2024-01-01T10:30:00Z" \
        --step  30 \
        --output data/hpa_steady_run1.csv
"""

import argparse
from datetime import datetime, timezone

import pandas as pd
import requests

from config import PROMETHEUS_QUERIES


def query_range(
    prometheus_url: str,
    query: str,
    start: str,
    end: str,
    step: int,
) -> pd.Series:
    resp = requests.get(
        f"{prometheus_url}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus error: {data}")
    results = data["data"]["result"]
    if not results:
        return pd.Series(dtype=float, name="value")
    values = results[0]["values"]
    series = pd.Series(
        {
            datetime.fromtimestamp(float(ts), tz=timezone.utc): float(val)
            for ts, val in values
        }
    )
    return series


def collect(
    prometheus_url: str,
    start: str,
    end: str,
    step: int,
    output: str,
) -> None:
    print(f"Collecting metrics from {prometheus_url} [{start} -> {end}] step={step}s")
    frames = {}
    for metric_name, query in PROMETHEUS_QUERIES.items():
        print(f"  querying: {metric_name} ...", end=" ", flush=True)
        try:
            series = query_range(prometheus_url, query, start, end, step)
            frames[metric_name] = series
            print(f"OK ({len(series)} points)")
        except Exception as exc:
            print(f"WARN: {exc}")
            frames[metric_name] = pd.Series(dtype=float)

    df = pd.DataFrame(frames)
    df.index.name = "timestamp"
    df = df.sort_index()

    # Forward-fill gaps up to 2 missing points (≤ 1 minute at 30s interval)
    df = df.ffill(limit=2)

    # Derived metrics
    df["jvm_gc_avg_pause_ms"] = (
        df["jvm_gc_pause_seconds_sum"]
        / df["jvm_gc_pause_count"].replace(0, float("nan"))
    ) * 1000

    df["jvm_heap_utilisation"] = (
        df["jvm_heap_used_bytes"]
        / df["jvm_heap_committed_bytes"].replace(0, float("nan"))
    )

    # Fallback: if p95 histogram unavailable, copy max response time as proxy
    if "response_time_p95_ms" in df.columns and df["response_time_p95_ms"].isna().all():
        if "response_time_max_ms" in df.columns:
            df["response_time_p95_ms"] = df["response_time_max_ms"]
            print("  NOTE: p95 unavailable — using max response time as p95 proxy")

    df.to_csv(output)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns -> {output}")
    print(df.describe().to_string())


def main():
    parser = argparse.ArgumentParser(description="Collect Prometheus metrics to CSV")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                        help="Prometheus base URL (default: port-forwarded localhost:9090)")
    parser.add_argument("--start", required=True,
                        help="ISO8601 UTC start time, e.g. 2024-01-01T10:00:00Z")
    parser.add_argument("--end", required=True,
                        help="ISO8601 UTC end time")
    parser.add_argument("--step", type=int, default=30,
                        help="Scrape step in seconds (default 30)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    args = parser.parse_args()
    collect(args.prometheus_url, args.start, args.end, args.step, args.output)


if __name__ == "__main__":
    main()
