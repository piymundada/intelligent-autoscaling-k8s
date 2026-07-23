"""
Backtest the JVM-pressure scale-up floor on EXISTING experiment data.

For each real LSTM run, replay every 30s interval through the JVM floor logic and
estimate, WITHOUT running a live experiment:
  - how often the floor would trigger
  - how many trigger intervals coincide with an actual SLA breach (p95 > 300ms)
    -> "coverage": breaches the floor would have added capacity for
  - extra replica-cycles the floor adds (cost delta)
  - false-positive triggers (fire when p95 was already fine)

This lets us tune thresholds before committing to 30-min live runs. It is a
counterfactual estimate (it cannot model the system's latency response to the
extra pods), but it shows whether the floor fires at the right moments.
"""

import argparse
import glob
import os

import pandas as pd

from config import SLA_THRESHOLD_MS, jvm_floor_bump

SLA_MS = SLA_THRESHOLD_MS


def backtest(paths, gc_high, gc_severe, cpu_high, max_replicas=4, lookahead=1):
    rows = []
    for path in sorted(paths):
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        need = {"jvm_gc_avg_pause_ms", "cpu_usage_percent",
                "response_time_p95_ms", "replica_count"}
        if not need.issubset(df.columns):
            continue
        df = df[list(need)].dropna()
        if df.empty:
            continue

        p95 = df["response_time_p95_ms"].values
        gc = df["jvm_gc_avg_pause_ms"].values
        cpu = df["cpu_usage_percent"].values
        rep = df["replica_count"].round().astype(int).values
        n = len(df)

        triggers = 0
        extra_replica_cycles = 0
        covered_breaches = 0   # breach intervals "covered" by a trigger at t or within lookahead before
        total_breaches = 0
        useful_triggers = 0    # trigger where a breach occurs at t or within next `lookahead` intervals
        false_pos = 0          # trigger with no breach now or soon

        breach = p95 > SLA_MS
        total_breaches = int(breach.sum())

        # mark, for each interval, whether the floor fires
        fires = [jvm_floor_bump(gc[i], cpu[i], gc_high, gc_severe, cpu_high) > 0
                 for i in range(n)]

        for i in range(n):
            bump = jvm_floor_bump(gc[i], cpu[i], gc_high, gc_severe, cpu_high)
            applied_bump = min(bump, max(0, max_replicas - rep[i]))
            if bump > 0:
                triggers += 1
                extra_replica_cycles += applied_bump
                # useful if a breach happens now or within the next `lookahead` intervals
                soon = any(breach[j] for j in range(i, min(n, i + 1 + lookahead)))
                if soon:
                    useful_triggers += 1
                else:
                    false_pos += 1

        # coverage: breaches that had a trigger at t or within `lookahead` BEFORE
        for i in range(n):
            if breach[i]:
                preceded = any(fires[j] for j in range(max(0, i - lookahead), i + 1))
                if preceded:
                    covered_breaches += 1

        name = os.path.basename(path).replace("_merged.csv", "")
        base_replica_cycles = int(df["replica_count"].round().sum())
        rows.append({
            "run": name,
            "intervals": len(df),
            "breaches": total_breaches,
            "triggers": triggers,
            "covered_breaches": covered_breaches,
            "false_pos": false_pos,
            "extra_replica_cycles": extra_replica_cycles,
            "base_replica_cycles": base_replica_cycles,
            "cost_increase_pct": round(extra_replica_cycles / base_replica_cycles * 100, 1)
                                  if base_replica_cycles else 0,
            "breach_coverage_pct": round(covered_breaches / total_breaches * 100, 1)
                                    if total_breaches else 0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/experiment2/lstm_*_merged.csv")
    ap.add_argument("--gc-high", type=float, default=10.0)
    ap.add_argument("--gc-severe", type=float, default=20.0)
    ap.add_argument("--cpu-high", type=float, default=70.0)
    ap.add_argument("--lookahead", type=int, default=1,
                    help="Intervals of proactive credit (1 = trigger counts if breach "
                         "occurs at t or t+1). 30s each.")
    args = ap.parse_args()

    paths = glob.glob(args.glob)
    rows = backtest(paths, args.gc_high, args.gc_severe, args.cpu_high,
                    lookahead=args.lookahead)

    print("=" * 100)
    print(f"JVM FLOOR BACKTEST  (gc_high={args.gc_high}ms gc_severe={args.gc_severe}ms "
          f"cpu_high={args.cpu_high}%)")
    print("=" * 100)
    print(f"{'Run':<22}{'Intv':>5}{'Breach':>7}{'Trig':>6}{'Covered':>8}"
          f"{'FalsePos':>9}{'ExtraRep':>9}{'Cost+%':>8}{'Coverage%':>10}")
    print("-" * 100)
    tot = {k: 0 for k in ["intervals", "breaches", "triggers", "covered_breaches",
                          "false_pos", "extra_replica_cycles", "base_replica_cycles"]}
    for r in rows:
        print(f"{r['run']:<22}{r['intervals']:>5}{r['breaches']:>7}{r['triggers']:>6}"
              f"{r['covered_breaches']:>8}{r['false_pos']:>9}{r['extra_replica_cycles']:>9}"
              f"{r['cost_increase_pct']:>7.1f}%{r['breach_coverage_pct']:>9.1f}%")
        for k in tot:
            tot[k] += r[k]

    print("-" * 100)
    cov = tot["covered_breaches"] / tot["breaches"] * 100 if tot["breaches"] else 0
    cost = tot["extra_replica_cycles"] / tot["base_replica_cycles"] * 100 if tot["base_replica_cycles"] else 0
    fp = tot["false_pos"] / tot["triggers"] * 100 if tot["triggers"] else 0
    print(f"{'TOTAL':<22}{tot['intervals']:>5}{tot['breaches']:>7}{tot['triggers']:>6}"
          f"{tot['covered_breaches']:>8}{tot['false_pos']:>9}{tot['extra_replica_cycles']:>9}"
          f"{cost:>7.1f}%{cov:>9.1f}%")
    print(f"\n  Breach coverage: {cov:.0f}% of SLA breaches occur at intervals where the "
          f"floor would add capacity")
    print(f"  Cost increase:   +{cost:.0f}% replica-cycles")
    print(f"  False-positive rate: {fp:.0f}% of triggers fire when p95 was already OK")
    print(f"\n  Interpretation: high coverage + low cost increase = good thresholds.")
    print(f"  (Counterfactual — does not model latency improvement from the added pods.)")


if __name__ == "__main__":
    main()
