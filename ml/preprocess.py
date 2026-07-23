"""
Preprocess collected CSV metrics for LSTM training.

Steps:
  1. Load one or more merged experiment CSVs (from merge_locust_prometheus.py)
  2. Drop the warm-up period (first 5 minutes, already stripped by merge script,
     but applied again as a safety guard)
  3. Drop all-NaN columns (e.g. memory_usage_bytes when cAdvisor is absent)
  4. Normalise all features to [0,1] using Min-Max scaling
  5. Create sliding-window sequences for LSTM input
  6. Split into train / validation / test using time-ordered splits (no leakage)
  7. Save processed arrays and scaler params for later inference

Usage:
    python3.11 preprocess.py \
        --inputs data/hpa_steady_run1_merged.csv \
                 data/hpa_steady_run2_merged.csv \
                 data/hpa_steady_run3_merged.csv \
        --output-dir data/processed/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import FEATURE_COLS, WARMUP_MINUTES, TARGET_COL, SEQ_LEN, PRED_HORIZON


def load_and_clean(csv_paths, warmup_minutes: int = WARMUP_MINUTES) -> pd.DataFrame:
    """Load one or more CSVs, drop warm-up, keep only available feature columns.

    Note: merge_locust_prometheus.py already strips the warm-up, so pass
    warmup_minutes=0 here to avoid double-stripping (10 min lost per run)."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if warmup_minutes > 0:
            cutoff = df.index.min() + pd.Timedelta(minutes=warmup_minutes)
            df = df[df.index >= cutoff]
        frames.append(df)

    combined = pd.concat(frames).sort_index()

    # Keep only feature columns that are present in this dataset
    available = [c for c in FEATURE_COLS if c in combined.columns]
    combined = combined[available].copy()

    # Drop columns that are entirely NaN (e.g. memory_usage_bytes without cAdvisor)
    before = len(combined.columns)
    combined = combined.dropna(axis=1, how="all")
    dropped = before - len(combined.columns)
    if dropped:
        print(f"  Dropped {dropped} all-NaN column(s).")

    # Fill remaining NaNs with 0 (do NOT drop rows). Several features are
    # legitimately NaN at idle/low load: jvm_gc_avg_pause_ms when no GC fires,
    # response_time_*/request_rate_rps when there's no traffic. Dropping those rows
    # discards exactly the idle samples the model must learn from (scale-in), and the
    # live autoscaler already coerces NaN->0 (np.nan_to_num) at inference, so filling
    # here keeps training consistent with inference instead of throwing data away.
    combined = combined.fillna(0.0)
    return combined


def make_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_horizon: int,
    target_idx: int,
):
    """
    Slide a window over `data` to produce (X, y) pairs.

    X shape : (n_samples, seq_len, n_features)
    y shape : (n_samples,), target value `pred_horizon` steps ahead
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len + pred_horizon - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def time_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Chronological train / val / test split (no shuffling to avoid data leakage)."""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def preprocess(
    input_paths,
    output_dir: str,
    seq_len: int = SEQ_LEN,
    pred_horizon: int = PRED_HORIZON,
    target_col: str = TARGET_COL,
    warmup_minutes: int = WARMUP_MINUTES,
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean(input_paths, warmup_minutes)
    print(f"Loaded {len(df)} timesteps, {len(df.columns)} features: {list(df.columns)}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    target_idx = list(df.columns).index(target_col)

    # Normalise to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    # Persist scaler params for inference-time denormalisation
    scaler_params = {
        "feature_names": list(df.columns),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "min_": scaler.min_.tolist(),
        "target_col": target_col,
        "target_idx": target_idx,
        "seq_len": seq_len,
        "pred_horizon": pred_horizon,
    }
    scaler_path = os.path.join(output_dir, "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  Scaler saved -> {scaler_path}")

    X, y = make_sequences(scaled, seq_len, pred_horizon, target_idx)
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    for name, arr in splits.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  Saved {name}: {arr.shape} -> {path}")

    print(f"\nPreprocessing complete.")
    return scaler_params


def main():
    parser = argparse.ArgumentParser(description="Preprocess metrics CSVs for LSTM training")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more merged CSV files (hpa_*_merged.csv)")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Directory to write .npy splits and scaler.json")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN,
                        help=f"LSTM lookback window in timesteps (default {SEQ_LEN} = 5 min at 30s)")
    parser.add_argument("--pred-horizon", type=int, default=PRED_HORIZON,
                        help=f"Steps ahead to predict (default {PRED_HORIZON} = 30 s ahead)")
    parser.add_argument("--target", default=TARGET_COL,
                        help=f"Column to predict (default: {TARGET_COL})")
    parser.add_argument("--warmup-minutes", type=int, default=WARMUP_MINUTES,
                        help=f"Warm-up minutes to drop (default {WARMUP_MINUTES}). "
                             f"Use 0 when the merge step already stripped warm-up.")
    args = parser.parse_args()
    preprocess(args.inputs, args.output_dir, args.seq_len, args.pred_horizon,
               args.target, args.warmup_minutes)


if __name__ == "__main__":
    main()
