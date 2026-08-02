"""
JVM feature ablation for the LSTM forecaster (RQ2, which metrics are informative).

Two analyses:

1. Forecast ablation: retrains the LSTM on the SAME sequence windows and
   normalisation, once with all 10 features and once with the JVM columns
   removed (workload + infra only). Trains each arm over several seeds and
   reports mean +/- std test error in RPS, so the JVM contribution to forecast
   accuracy is isolated from training variance.

2. Lead-lag correlation: on the merged baseline CSVs, correlates each feature
   at time t with p95 latency at t+k (k = 0..6 steps of 30 s). A feature whose
   correlation peaks at k > 0 is a LEADING indicator of latency degradation,
   the core JVM-awareness claim (thesis §5.10).

Usage:
    python3 analysis/feature_ablation.py \
        --data-dir data/processed \
        --inputs data/hpa_bursty_run1_merged.csv data/hpa_spike_run2_merged.csv \
        --output results/feature_ablation.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from lstm_model import LSTMForecaster  # noqa: E402

JVM_COLS = [
    "jvm_heap_used_bytes",
    "jvm_heap_utilisation",
    "jvm_gc_avg_pause_ms",
    "jvm_gc_pause_count",
    "jvm_threads_live",
]
MAX_LAG_STEPS = 6  # 6 x 30 s = 3 min ahead


def train_once(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = LSTMForecaster(
        input_size=X_tr.shape[2], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=32, shuffle=False)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)
    X_te_t = torch.tensor(X_te)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(100):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).numpy()
    return preds


def lead_lag(csv_paths, feature_names):
    """Mean Pearson correlation of feature(t) vs p95(t+k), averaged across runs."""
    out = {}
    for col in feature_names:
        per_lag = []
        for k in range(MAX_LAG_STEPS + 1):
            corrs = []
            for path in csv_paths:
                df = pd.read_csv(path)
                if col not in df.columns or "response_time_p95_ms" not in df.columns:
                    continue
                a = df[col]
                b = df["response_time_p95_ms"].shift(-k)
                pair = pd.concat([a, b], axis=1).dropna()
                if len(pair) > 10 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                    corrs.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            per_lag.append(float(np.mean(corrs)) if corrs else None)
        valid = [(k, c) for k, c in enumerate(per_lag) if c is not None]
        if not valid:
            continue
        best_k, best_c = max(valid, key=lambda kc: abs(kc[1]))
        out[col] = {
            "corr_by_lag": per_lag,
            "peak_lag_steps": best_k,
            "peak_lag_seconds": best_k * 30,
            "peak_corr": best_c,
            "leading_indicator": best_k > 0,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="JVM feature ablation + lead-lag analysis")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Merged baseline CSVs for the lead-lag analysis")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="results/feature_ablation.json")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "scaler.json")) as f:
        scaler = json.load(f)
    feature_names = scaler["feature_names"]
    target_idx = scaler["target_idx"]
    cfg = scaler.get("model_config",
                     {"hidden_size": 16, "num_layers": 1, "dropout": 0.2})
    lo, hi = scaler["data_min_"][target_idx], scaler["data_max_"][target_idx]
    rps_scale = hi - lo

    splits = {s: (np.load(os.path.join(args.data_dir, f"X_{s}.npy")),
                  np.load(os.path.join(args.data_dir, f"y_{s}.npy")))
              for s in ("train", "val", "test")}

    jvm_idx = [feature_names.index(c) for c in JVM_COLS if c in feature_names]
    keep_idx = [i for i in range(len(feature_names)) if i not in jvm_idx]
    arms = {
        "all_features": list(range(len(feature_names))),
        "no_jvm_features": keep_idx,
    }
    print(f"Arms: all_features={len(feature_names)} cols | "
          f"no_jvm_features={len(keep_idx)} cols (dropped: {[feature_names[i] for i in jvm_idx]})")

    y_te = splits["test"][1]
    results = {"model_config": cfg, "seeds": args.seeds, "arms": {}}
    for arm, idx in arms.items():
        maes, rmses = [], []
        for seed in range(args.seeds):
            preds = train_once(
                splits["train"][0][:, :, idx], splits["train"][1],
                splits["val"][0][:, :, idx], splits["val"][1],
                splits["test"][0][:, :, idx], y_te, cfg, seed,
            )
            maes.append(np.mean(np.abs(preds - y_te)) * rps_scale)
            rmses.append(np.sqrt(np.mean((preds - y_te) ** 2)) * rps_scale)
        results["arms"][arm] = {
            "features": [feature_names[i] for i in idx],
            "test_mae_rps_mean": float(np.mean(maes)),
            "test_mae_rps_std": float(np.std(maes)),
            "test_rmse_rps_mean": float(np.mean(rmses)),
            "test_rmse_rps_std": float(np.std(rmses)),
        }
        print(f"  {arm:<18} MAE={np.mean(maes):6.2f}±{np.std(maes):4.2f} RPS | "
              f"RMSE={np.mean(rmses):6.2f}±{np.std(rmses):4.2f} RPS  ({args.seeds} seeds)")

    full = results["arms"]["all_features"]["test_rmse_rps_mean"]
    nojvm = results["arms"]["no_jvm_features"]["test_rmse_rps_mean"]
    results["jvm_rmse_improvement_pct"] = (nojvm - full) / nojvm * 100
    print(f"\nJVM features change forecast RMSE by {results['jvm_rmse_improvement_pct']:+.1f}% "
          f"(positive = JVM features help)")

    print("\nLead-lag vs p95 latency (peak |corr| and its lag):")
    results["lead_lag_vs_p95"] = lead_lag(args.inputs, feature_names)
    for col, r in sorted(results["lead_lag_vs_p95"].items(),
                         key=lambda kv: -abs(kv[1]["peak_corr"])):
        flag = "LEADING" if r["leading_indicator"] else "coincident"
        print(f"  {col:<26} peak corr {r['peak_corr']:+.3f} at +{r['peak_lag_seconds']}s ({flag})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
