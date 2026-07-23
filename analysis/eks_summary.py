"""
EKS campaign summary: aggregates the full EKS dataset into the per-RQ results.
Offline (reads merged CSVs only). Usage: python3 analysis/eks_summary.py
"""
import glob, json, os
import numpy as np
import pandas as pd

SLA = 300          # ms (config.SLA_THRESHOLD_MS)
STEP_S = 30        # collect_metrics --step

def load(path):
    d = pd.read_csv(path)
    return d

def run_metrics(path):
    """Per-run metrics from one merged CSV."""
    d = load(path)
    p95 = d['response_time_p95_ms']
    rps = d['request_rate_rps']
    rep = d['replica_count']
    # error fraction (locust failures / requests)
    fr = d.get('failure_rate_rps')
    err = (fr.sum() / rps.sum() * 100) if fr is not None and rps.sum() > 0 else float('nan')
    return dict(
        p95_med=p95.median(),
        p95_mean=p95.mean(),
        sla_viol_pct=100 * (p95 > SLA).sum() / len(d),
        err_pct=err,
        mean_repl=rep.mean(),
        replica_hours=rep.sum() * STEP_S / 3600,
        scaling_events=int((rep.diff().abs() > 0).sum()),
        idle_tail=rep.tail(8).mean(),
        n=len(d),
    )

def agg(paths):
    """Mean ± std of run metrics across repeats."""
    ms = [run_metrics(p) for p in paths if os.path.exists(p)]
    if not ms: return None
    keys = ms[0].keys()
    return {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms])) for k in keys}, len(ms)

out = []
def pr(s=""): out.append(s); print(s)

pr("="*82)
pr("EKS CAMPAIGN RESULTS  (max=6, control-interval=15s, EKS-retrained LSTM)")
pr("="*82)

# ---- RQ1/RQ4/RQ5: per-scenario HPA vs JVM-aware predictive (n=2) ----
pr("\n[A] HPA vs JVM-aware predictive — per scenario (mean over n runs)\n")
pr(f"{'scenario':8s} {'arm':4s} | {'p95_med':>7s} {'SLA_viol%':>9s} {'err%':>5s} {'repl_hrs':>8s} {'scale_ev':>8s} {'idle':>5s}  n")
pr("-"*78)
for scn in ['bursty','spike','gc','steady']:
    for arm,label in [('hpa','HPA'),('pred','JVM')]:
        a = agg(sorted(glob.glob(f'data/{scn}_m6_run*_{arm}_merged.csv')))
        if not a: continue
        m,n = a
        pr(f"{scn:8s} {label:4s} | {m['p95_med'][0]:7.0f} {m['sla_viol_pct'][0]:9.1f} {m['err_pct'][0]:5.1f} {m['replica_hours'][0]:8.2f} {m['scaling_events'][0]:8.1f} {m['idle_tail'][0]:5.1f}  {n}")
    pr("")

# ---- RQ2: 3-way ablation (HPA vs RPS-only vs JVM) on spike + gc ----
pr("[B] RQ2 ablation — HPA vs RPS-only vs JVM (n=1)\n")
pr(f"{'scenario':8s} {'arm':9s} | {'p95_med':>7s} {'SLA_viol%':>9s} {'repl_hrs':>8s} {'idle':>5s}")
pr("-"*55)
for scn in ['spike','gc']:
    for label,path in [('HPA',f'data/{scn}_m6_run1_hpa_merged.csv'),
                       ('RPS-only',f'data/{scn}_rpsonly_run1_merged.csv'),
                       ('JVM',f'data/{scn}_m6_run1_pred_merged.csv')]:
        if not os.path.exists(path): continue
        m = run_metrics(path)
        pr(f"{scn:8s} {label:9s} | {m['p95_med']:7.0f} {m['sla_viol_pct']:9.1f} {m['replica_hours']:8.2f} {m['idle_tail']:5.1f}")
    pr("")

# ---- RQ1/RQ5: 3h idle cost gap ----
pr("[C] RQ1/RQ5 — 3-hour idle cost gap\n")
for label,path in [('HPA',f'data/idle_3h_hpa_merged.csv'),('JVM',f'data/idle_3h_pred_merged.csv')]:
    if not os.path.exists(path): continue
    m = run_metrics(path)
    pr(f"  {label:4s}: replica-hours={m['replica_hours']:5.1f}  mean_repl={m['mean_repl']:.2f}  idle_tail={m['idle_tail']:.1f}")
try:
    h = run_metrics('data/idle_3h_hpa_merged.csv')['replica_hours']
    p = run_metrics('data/idle_3h_pred_merged.csv')['replica_hours']
    pr(f"  -> predictive saves {100*(h-p)/h:.0f}% of replica-hours ({h:.1f} -> {p:.1f}); {h/p:.1f}x gap")
except Exception: pass

# ---- RQ3: forecast quality ----
pr("\n[D] RQ3 — EKS LSTM forecast quality\n")
r = json.load(open('ml/models/lstm_forecaster_eks_results.json'))
pr(f"  test MAE={r['test_mae']:.4f}  RMSE={r['test_rmse']:.4f} (normalised)  epochs={r['epochs_trained']}")
try:
    X=np.load('data/processed_eks/X_test.npy'); y=np.load('data/processed_eks/y_test.npy')
    pr_=np.load('ml/models/lstm_forecaster_eks_test_preds.npy')
    pr(f"  pred-vs-actual correlation (held-out test): {np.corrcoef(pr_, y)[0,1]:.3f}")
except Exception as e:
    pr(f"  (corr: {e})")

pr("\n"+"="*82)
os.makedirs('results', exist_ok=True)
open('results/eks_analysis_summary.txt','w').write("\n".join(out))
pr("saved -> results/eks_analysis_summary.txt")
