"""
Train the LSTM workload forecasting model.

Usage:
    python train.py \
        --data-dir data/processed/ \
        --model-out models/lstm_forecaster.pt \
        --epochs 100 \
        --batch-size 32 \
        --hidden-size 64 \
        --num-layers 2 \
        --dropout 0.2 \
        --lr 0.001

After training, reports MAE and RMSE on the held-out test split.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import LSTMForecaster


def load_split(data_dir: str, split: str):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    return torch.tensor(X), torch.tensor(y)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            n += len(y_batch)
    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    mae = float(np.mean(np.abs(preds_np - targets_np)))
    rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    return total_loss / n, mae, rmse, preds_np, targets_np


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, y_train = load_split(args.data_dir, "train")
    X_val,   y_val   = load_split(args.data_dir, "val")
    X_test,  y_test  = load_split(args.data_dir, "test")

    input_size = X_train.shape[2]
    print(f"Input size: {input_size} features | seq_len: {X_train.shape[1]}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size, shuffle=False,
    )

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Reduce LR on plateau to avoid oscillation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 20
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    print(f"\nTraining for up to {args.epochs} epochs (early stop patience={early_stop_patience})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
            n += len(y_batch)

        train_loss = epoch_loss / n
        val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_out)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    test_loss, test_mae, test_rmse, preds, targets = evaluate(model, test_loader, criterion, device)

    print("\n=== Test Set Results ===")
    print(f"  MSE  : {test_loss:.6f}")
    print(f"  MAE  : {test_mae:.6f}")
    print(f"  RMSE : {test_rmse:.6f}")

    # Save training history and test metrics
    results = {
        "test_mse": test_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
        "model_config": {
            "input_size": input_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
    }
    results_path = args.model_out.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nModel saved -> {args.model_out}")
    print(f"Results saved -> {results_path}")

    # Inject model_config into scaler.json so autoscaler.py reconstructs the LSTM
    # with the exact hyperparameters used here (not its hard-coded 64/2 defaults).
    scaler_path = os.path.join(args.data_dir, "scaler.json")
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            scaler_params = json.load(f)
        scaler_params["model_config"] = results["model_config"]
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        print(f"model_config written -> {scaler_path}")

    # Save test predictions for analysis scripts
    np.save(args.model_out.replace(".pt", "_test_preds.npy"), preds)
    np.save(args.model_out.replace(".pt", "_test_targets.npy"), targets)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM workload forecaster")
    parser.add_argument("--data-dir", default="data/processed/")
    parser.add_argument("--model-out", default="models/lstm_forecaster.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
