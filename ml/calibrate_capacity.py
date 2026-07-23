"""
Calibrate RPS_PER_REPLICA and validate the SLA threshold from baseline data.

The predictive autoscaler converts predicted RPS to replicas via RPS_PER_REPLICA
(config.py). If that constant is wrong, the autoscaler systematically over- or
under-scales and the comparison against HPA is meaningless. This script derives a
data-driven value from a steady-load baseline run instead of guessing.

Method:
  - Keep only intervals where the SLA was met (p95 <= SLA_THRESHOLD_MS), i.e.
    the pods were actually keeping up.
  - Compute per-interval per-replica throughput = request_rate_rps / replica_count.
  - Report a conservative percentile (default p75) as the sustainable RPS/replica.
    Using a percentile (not the max) leaves headroom and ignores lucky outliers.

It also prints the p95 latency distribution so the SLA target (§8.7) can be
confirmed against what the system actually delivers under normal load.

Usage:
    python ml/calibrate_capacity.py --input data/hpa_steady_run1_merged.csv
    python ml/calibrate_capacity.py --input data/hpa_steady_run1_merged.csv --percentile 75
"""

import argparse

import numpy as np
import pandas as pd

from config import SLA_THRESHOLD_MS


def calibrate(path: str, percentile: float) -> None:
    df = pd.read_csv(path)

    required = {"request_rate_rps", "response_time_p95_ms", "replica_count"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    df = df[list(required)].dropna()
    df = df[(df["replica_count"] > 0) & (df["request_rate_rps"] > 0)]
    if df.empty:
        raise SystemExit("No usable rows after cleaning.")

    p95 = df["response_time_p95_ms"]
    print("=" * 70)
    print(f"CAPACITY CALIBRATION  ({path})")
    print("=" * 70)
    print(f"  intervals analysed     : {len(df)}")
    print(f"  p95 latency  min/med/max: "
          f"{p95.min():.0f} / {p95.median():.0f} / {p95.max():.0f} ms")
    print(f"  SLA threshold (config) : {SLA_THRESHOLD_MS} ms")
    sla_ok = df[p95 <= SLA_THRESHOLD_MS]
    print(f"  intervals within SLA   : {len(sla_ok)}/{len(df)} "
          f"({100 * len(sla_ok) / len(df):.0f}%)")

    if sla_ok.empty:
        print("\n  WARNING: NO interval met the SLA — the threshold may be too")
        print("  strict, or the baseline is under-provisioned. Falling back to")
        print("  all intervals for the capacity estimate.")
        sla_ok = df

    per_replica = sla_ok["request_rate_rps"] / sla_ok["replica_count"]
    value = float(np.percentile(per_replica, percentile))

    print("-" * 70)
    print(f"  per-replica RPS p{int(percentile):<2d}     : {value:.1f} "
          f"(median {per_replica.median():.1f}, max {per_replica.max():.1f})")
    print("-" * 70)
    print(f"  -> set RPS_PER_REPLICA = {value:.0f}  in ml/config.py")
    print(f"     (or pass --rps-per-replica {value:.0f} to ml/autoscaler.py)")
    if p95.median() < SLA_THRESHOLD_MS * 0.5:
        print(f"\n  NOTE: median p95 ({p95.median():.0f} ms) is well under the SLA — "
              f"the {SLA_THRESHOLD_MS} ms target looks realistic.")
    elif p95.median() > SLA_THRESHOLD_MS:
        print(f"\n  NOTE: median p95 ({p95.median():.0f} ms) already exceeds the SLA — "
              f"consider raising the threshold or the baseline replica count.")


def main():
    ap = argparse.ArgumentParser(description="Calibrate RPS_PER_REPLICA from baseline data")
    ap.add_argument("--input", required=True,
                    help="A steady-load merged CSV (e.g. data/hpa_steady_run1_merged.csv)")
    ap.add_argument("--percentile", type=float, default=75.0,
                    help="Percentile of per-replica RPS to report (default 75 = conservative)")
    args = ap.parse_args()
    calibrate(args.input, args.percentile)


if __name__ == "__main__":
    main()
