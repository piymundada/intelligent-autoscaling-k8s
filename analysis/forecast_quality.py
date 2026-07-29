"""
Forecast quality analysis (thesis §3.8, prediction quality).

Reports LSTM test-set error in BOTH normalised units and denormalised RPS,
and compares against two naive baselines computed on the same test windows:

  - Persistence: predict y(t+h) = last observed RPS in the lookback window
  - Moving average: predict y(t+h) = mean RPS over the lookback window

The naive baselines justify the LSTM choice (thesis §3.6.2 specifies
last-value persistence and moving-average as the comparison baselines).

Usage:
    python3 analysis/forecast_quality.py \
        --data-dir data/processed \
        --model-prefix ml/models/lstm_forecaster \
        --output results/forecast_quality.json
"""

import argparse
import json
import os

import numpy as np


def denorm(values: np.ndarray, scaler: dict, idx: int) -> np.ndarray:
    lo = scaler["data_min_"][idx]
    hi = scaler["data_max_"][idx]
    return values * (hi - lo) + lo


def errors(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return {"mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(description="LSTM forecast quality vs naive baselines")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-prefix", default="ml/models/lstm_forecaster")
    parser.add_argument("--output", default="results/forecast_quality.json")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "scaler.json")) as f:
        scaler = json.load(f)
    target_idx = scaler["target_idx"]

    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    lstm_preds = np.load(f"{args.model_prefix}_test_preds.npy")
    targets = np.load(f"{args.model_prefix}_test_targets.npy")

    baselines = {
        "lstm": lstm_preds,
        "persistence": X_test[:, -1, target_idx],
        "moving_average": X_test[:, :, target_idx].mean(axis=1),
    }

    results = {
        "n_test_samples": int(len(targets)),
        "target": scaler["target_col"],
        "horizon_steps": scaler["pred_horizon"],
        "horizon_seconds": scaler["pred_horizon"] * 30,
        "models": {},
    }
    targets_rps = denorm(targets, scaler, target_idx)
    results["test_rps_range"] = [float(targets_rps.min()), float(targets_rps.max())]

    print(f"Test samples: {len(targets)} | horizon: {results['horizon_seconds']}s | "
          f"target RPS range: {targets_rps.min():.1f}-{targets_rps.max():.1f}")
    print(f"\n  {'Model':<16} {'MAE (norm)':>11} {'RMSE (norm)':>12} {'MAE (RPS)':>10} {'RMSE (RPS)':>11}")
    for name, preds in baselines.items():
        norm = errors(preds, targets)
        rps = errors(denorm(preds, scaler, target_idx), targets_rps)
        results["models"][name] = {
            "mae_normalised": norm["mae"], "rmse_normalised": norm["rmse"],
            "mae_rps": rps["mae"], "rmse_rps": rps["rmse"],
        }
        print(f"  {name:<16} {norm['mae']:>11.4f} {norm['rmse']:>12.4f} "
              f"{rps['mae']:>10.2f} {rps['rmse']:>11.2f}")

    lstm_rmse = results["models"]["lstm"]["rmse_rps"]
    for base in ("persistence", "moving_average"):
        base_rmse = results["models"][base]["rmse_rps"]
        improvement = (base_rmse - lstm_rmse) / base_rmse * 100
        results["models"][base]["lstm_rmse_improvement_pct"] = improvement
        print(f"\nLSTM RMSE improvement vs {base}: {improvement:+.1f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
