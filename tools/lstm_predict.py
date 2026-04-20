"""LSTM invocation-count predictor for traffic_pattern CSVs.

Reproduces the LSTM from Vahidinia et al. 2023 (Layer 2):
  - 5 hidden LSTM layers, 32 neurons each
  - Dropout 0.5
  - Linear output (1 neuron)
  - MSE loss, Adam optimizer

Usage:
    python tools/lstm_predict.py datasets/traffic_pattern/Java_APIG-S_*.csv
    python tools/lstm_predict.py datasets/traffic_pattern/*.csv --window 10 --epochs 100
    python tools/lstm_predict.py datasets/traffic_pattern/*.csv --output-dir outputs/lstm
    python tools/lstm_predict.py datasets/traffic_pattern/*.csv --hourly --plot
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """LSTM model matching Vahidinia et al. 2023 Layer 2 architecture.

    1 LSTM layer (32 units) + 4 FC layers (32 units each, ReLU) + linear output.
    Dropout 0.5 applied after the LSTM layer.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take output of last timestep
        out = self.dropout(out[:, -1, :])
        return self.fc_layers(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_fill(csv_path: str) -> pd.DataFrame:
    """Load traffic CSV and fill missing minutes with count=0."""
    df = pd.read_csv(csv_path)
    # Build full minute range and fill gaps with 0 invocations
    minutes = np.arange(df["minute"].min(), df["minute"].max() + 1)
    full = pd.DataFrame({"minute": minutes})
    merged = full.merge(df, on="minute", how="left")
    merged["count"] = merged["count"].fillna(0).astype(float)
    merged["function_id"] = merged["function_id"].ffill().bfill()
    return merged


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level data to hourly: sum count."""
    df = df.copy()
    df["hour"] = df["minute"] // 60
    hourly = df.groupby("hour").agg(
        count=("count", "sum"),
        function_id=("function_id", "first"),
    ).reset_index()
    return hourly


def create_sequences(values: np.ndarray, window: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM.

    Returns (X, y) where X[i] = values[i:i+window], y[i] = values[i+window].
    """
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def normalize(train: np.ndarray, test: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Min-max normalize using train statistics."""
    vmin = train.min()
    vmax = train.max()
    scale = vmax - vmin if vmax != vmin else 1.0
    return (train - vmin) / scale, (test - vmin) / scale, vmin, scale


def denormalize(arr: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return arr * scale + vmin


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------

class TrainResult:
    """Holds training outputs: predictions, actuals, loss history."""

    def __init__(self, train_pred, test_pred, train_actual, test_actual,
                 test_mse, train_losses, val_losses):
        self.train_pred = train_pred
        self.test_pred = test_pred
        self.train_actual = train_actual
        self.test_actual = test_actual
        self.test_mse = test_mse
        self.train_losses = train_losses
        self.val_losses = val_losses


def train_and_predict(
    values: np.ndarray,
    window: int = 10,
    train_ratio: float = 0.7,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
    hidden_size: int = 32,
) -> TrainResult:
    """Train LSTM and return predictions for both train and test."""
    # Split before sequencing (no data leakage)
    split_idx = int(len(values) * train_ratio)
    train_raw = values[:split_idx]
    test_raw = values[split_idx:]

    # Normalize
    train_norm, test_norm, vmin, scale = normalize(train_raw, test_raw)

    # Full normalized series for sequencing
    full_norm = np.concatenate([train_norm, test_norm])

    # Create sequences from full series
    X_all, y_all = create_sequences(full_norm, window)

    # Split sequences: train sequences end where train data ends
    # A sequence at index i uses values[i:i+window] to predict values[i+window]
    # Train sequences: those where target index (i+window) < split_idx
    train_seq_end = split_idx - window  # last train sequence index
    if train_seq_end <= 0:
        raise ValueError(
            f"Not enough data: {len(values)} points with window={window} "
            f"and train_ratio={train_ratio}"
        )

    X_train_all = X_all[:train_seq_end]
    y_train_all = y_all[:train_seq_end]
    X_test = X_all[train_seq_end:]
    y_test = y_all[train_seq_end:]

    # Split 10% of training data for validation (from the end, preserving time order)
    val_size = max(1, int(len(X_train_all) * 0.1))
    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    # To tensors
    X_train_t = torch.tensor(X_train).unsqueeze(-1).to(device)  # (N, W, 1)
    y_train_t = torch.tensor(y_train).to(device)
    X_val_t = torch.tensor(X_val).unsqueeze(-1).to(device)
    y_val_t = torch.tensor(y_val).to(device)
    X_test_t = torch.tensor(X_test).unsqueeze(-1).to(device)
    y_test_t = torch.tensor(y_test).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = LSTMPredictor(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training with early stopping
    best_val_loss = float("inf")
    wait = 0
    best_state = None
    train_losses = []
    val_losses = []


    pbar = tqdm(range(epochs), desc="  Training", unit="ep", leave=True)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train_t)

        # Validation on held-out 10% of training data
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        pbar.set_postfix(train_mse=f"{epoch_loss:.6f}", val_mse=f"{val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                pbar.set_postfix(train_mse=f"{epoch_loss:.6f}",
                                 val_mse=f"{val_loss:.6f}", early_stop=True)
                pbar.close()
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    # Predict on full train (train+val) and test
    X_train_all_t = torch.tensor(X_train_all).unsqueeze(-1).to(device)
    with torch.no_grad():
        train_pred_norm = model(X_train_all_t).cpu().numpy()
        test_pred_norm = model(X_test_t).cpu().numpy()

    # Denormalize
    train_pred = denormalize(train_pred_norm, vmin, scale)
    test_pred = denormalize(test_pred_norm, vmin, scale)
    train_actual = denormalize(y_train_all, vmin, scale)
    test_actual = denormalize(y_test, vmin, scale)

    # Test MSE in original scale
    test_mse = float(np.mean((test_pred - test_actual) ** 2))

    return TrainResult(
        train_pred, test_pred, train_actual, test_actual,
        test_mse, train_losses, val_losses,
    )


def _val_mse_only(
    values: np.ndarray, window: int, train_ratio: float,
    epochs: int, batch_size: int, lr: float, patience: int,
    device: str, hidden_size: int,
) -> float:
    """Train and return only the best validation MSE (for grid search)."""
    split_idx = int(len(values) * train_ratio)
    train_raw = values[:split_idx]
    test_raw = values[split_idx:]
    train_norm, test_norm, vmin, scale = normalize(train_raw, test_raw)
    full_norm = np.concatenate([train_norm, test_norm])
    X_all, y_all = create_sequences(full_norm, window)

    train_seq_end = split_idx - window
    if train_seq_end <= 0:
        return float("inf")

    X_train_all = X_all[:train_seq_end]
    y_train_all = y_all[:train_seq_end]

    val_size = max(1, int(len(X_train_all) * 0.1))
    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    X_train_t = torch.tensor(X_train).unsqueeze(-1).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_val_t = torch.tensor(X_val).unsqueeze(-1).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LSTMPredictor(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_val


def grid_search(
    values: np.ndarray, train_ratio: float, epochs: int,
    batch_size: int, patience: int, device: str, hourly: bool,
    hidden_size: int = 32,
) -> dict:
    """Search best hyperparameters (window, lr) and return the best combo.

    Returns dict with keys: window, lr, val_mse.
    """
    if hourly:
        windows = [3, 6, 12, 24]
    else:
        windows = [5, 10, 24, 48]
    lrs = [1e-2, 1e-3, 1e-4]



    combos = [(w, l) for w in windows for l in lrs]
    total = len(combos)

    best = {"val_mse": float("inf")}
    pbar = tqdm(combos, desc="  Grid search", unit="combo", leave=True)
    for w, l in pbar:
        try:
            val_mse = _val_mse_only(
                values, window=w, train_ratio=train_ratio,
                epochs=epochs, batch_size=batch_size, lr=l,
                patience=patience, device=device, hidden_size=hidden_size,
            )
        except (ValueError, RuntimeError):
            val_mse = float("inf")

        is_best = val_mse < best["val_mse"]
        if is_best:
            best = {"window": w, "lr": l, "val_mse": val_mse}

        pbar.set_postfix(window=w, lr=f"{l:.0e}", val_mse=f"{val_mse:.6f}",
                         best_mse=f"{best['val_mse']:.6f}")

    print(f"  Best: window={best['window']} lr={best['lr']:.0e}"
          f" val_mse={best['val_mse']:.6f}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_csv(csv_path: str, output_dir: str, window: int, train_ratio: float,
                epochs: int, batch_size: int, lr: float, patience: int,
                device: str, hourly: bool = False,
                do_grid_search: bool = False, hidden_size: int = 32) -> str | None:
    """Process a single CSV: train LSTM, save predictions."""
    name = Path(csv_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {name}" + (" [hourly]" if hourly else ""))
    print(f"{'='*60}")

    df_minutes = load_and_fill(csv_path)

    if hourly:
        df_hourly = aggregate_hourly(df_minutes)
        values = df_hourly["count"].values
        print(f"  Aggregated: {len(df_minutes)} minutes -> {len(df_hourly)} hours")
    else:
        values = df_minutes["count"].values

    if len(values) < window + 10:
        print(f"  Skipping: only {len(values)} data points (need > {window + 10})")
        return None

    # Grid search for best hyperparameters
    if do_grid_search:
        best = grid_search(
            values, train_ratio=train_ratio, epochs=epochs,
            batch_size=batch_size, patience=patience, device=device,
            hourly=hourly, hidden_size=hidden_size,
        )
        window = best["window"]
        lr = best["lr"]

    result = train_and_predict(
        values, window=window, train_ratio=train_ratio, epochs=epochs,
        batch_size=batch_size, lr=lr, patience=patience, device=device,
        hidden_size=hidden_size,
    )
    train_pred = result.train_pred
    test_pred = result.test_pred

    print(f"  Test MSE: {result.test_mse:.4f}")

    # Build output — always minute-level rows
    split_idx = int(len(values) * train_ratio)

    out = df_minutes.copy()
    out["predicted_count"] = np.nan
    out["phase"] = ""

    if hourly:
        # Output 1 row per hour — same format as aggregate trace CSV
        # (minute = hour * 60, count = hourly actual, predicted_count = hourly predicted)
        out = df_hourly.copy()
        out["minute"] = out["hour"] * 60  # hour start in minutes
        out["predicted_count"] = np.nan
        out["phase"] = ""

        # Mark phases
        out.loc[:split_idx - 1, "phase"] = "train"
        out.loc[split_idx:, "phase"] = "test"

        # Fill predictions (offset by window)
        train_start = window
        train_end = train_start + len(train_pred)
        test_start = train_end
        test_end = test_start + len(test_pred)

        out.loc[train_start:train_end - 1, "predicted_count"] = train_pred
        out.loc[test_start:test_end - 1, "predicted_count"] = test_pred

        out = out[["minute", "function_id", "count", "predicted_count", "phase"]]
    else:
        # Mark phases on all rows
        out.loc[:split_idx - 1, "phase"] = "train"
        out.loc[split_idx:, "phase"] = "test"

        # Fill predictions (offset by window)
        train_start = window
        train_end = train_start + len(train_pred)
        test_start = train_end
        test_end = test_start + len(test_pred)

        out.loc[train_start:train_end - 1, "predicted_count"] = train_pred
        out.loc[test_start:test_end - 1, "predicted_count"] = test_pred

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_lstm_pred.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Rows: {len(out)} (train_pred={len(train_pred)}, test_pred={len(test_pred)})")

    return out_path, result


def plot_predictions(csv_path: str, output_dir: str, hourly: bool = False,
                     plot_range: tuple[float, float] | None = None,
                     plot_style: str = "scatter") -> None:
    """Plot actual vs predicted invocation counts from a result CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    name = Path(csv_path).stem.replace("_lstm_pred", "")

    if hourly:
        # CSV already has 1 row per hour with hourly counts
        plot_df = df.copy()
        plot_df["hour"] = plot_df["minute"] // 60
        x_col = "hour"
        x_divisor = 24.0  # hours -> days
    else:
        plot_df = df
        x_col = "minute"
        x_divisor = 1440.0  # minutes -> days

    # Convert x values to days for plotting
    plot_df = plot_df.copy()
    plot_df["_x_days"] = plot_df[x_col] / x_divisor
    x_col = "_x_days"
    x_label = "Day"

    has_pred = plot_df["predicted_count"].notna()
    train_mask = has_pred & (plot_df["phase"] == "train")
    test_mask = has_pred & (plot_df["phase"] == "test")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Filter out zero-count actuals for cleaner scatter plots
    nonzero = plot_df["count"] > 0

    def _draw(ax, x, y, color, label, small=False):
        if plot_style == "scatter":
            ax.scatter(x, y, color=color, alpha=0.5 if small else 0.6,
                       s=4 if small else 8, label=label)
        else:
            ax.plot(x, y, color=color, alpha=0.6, linewidth=0.5 if small else 0.8,
                    label=label)

    # --- Top: full view (always shows entire dataset) ---
    ax = axes[0]
    actual_mask = nonzero if plot_style == "scatter" else plot_df.index
    _draw(ax, plot_df.loc[actual_mask, x_col], plot_df.loc[actual_mask, "count"],
          "#2196F3", "Actual", small=True)
    _draw(ax, plot_df.loc[train_mask, x_col], plot_df.loc[train_mask, "predicted_count"],
          "#4CAF50", "Predicted (train)", small=True)
    _draw(ax, plot_df.loc[test_mask, x_col], plot_df.loc[test_mask, "predicted_count"],
          "#F44336", "Predicted (test)", small=True)

    # Train/test split line
    split_x = plot_df.loc[test_mask, x_col].min() if test_mask.any() else None
    if split_x is not None:
        ax.axvline(x=split_x, color="gray", linestyle="--", linewidth=0.8,
                   label="Train/Test split")

    ax.set_ylabel("Invocation Count")
    ax.set_title(f"LSTM Predictions — {name}" + (" (hourly)" if hourly else ""))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom: zoom view (plot_range or test phase) ---
    ax2 = axes[1]
    if plot_range is not None:
        day_min, day_max = plot_range
        zoom_mask = (plot_df[x_col] >= day_min) & (plot_df[x_col] <= day_max)
        zoom_df = plot_df[zoom_mask]
    elif test_mask.any():
        zoom_df = plot_df[test_mask | (nonzero & (plot_df[x_col] >= plot_df.loc[test_mask, x_col].min()))]
    else:
        zoom_df = plot_df

    zoom_nonzero = zoom_df["count"] > 0
    zoom_has_pred = zoom_df["predicted_count"].notna()
    zoom_actual = zoom_nonzero if plot_style == "scatter" else zoom_df.index.isin(zoom_df.index)

    _draw(ax2, zoom_df.loc[zoom_actual, x_col], zoom_df.loc[zoom_actual, "count"],
          "#2196F3", "Actual")
    _draw(ax2, zoom_df.loc[zoom_has_pred, x_col], zoom_df.loc[zoom_has_pred, "predicted_count"],
          "#F44336", "Predicted")
    ax2.legend(loc="upper right", fontsize=8)

    zoom_title = f"Zoom: Day {plot_range[0]}-{plot_range[1]}" if plot_range else "Test Phase (zoom)"
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Invocation Count")
    ax2.set_title(zoom_title)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{name}_lstm_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {plot_path}")


def plot_training_curves(train_losses: list[float], val_losses: list[float],
                        name: str, output_dir: str) -> None:
    """Plot training and validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#4CAF50", linewidth=1.2, label="Train MSE")
    ax.plot(epochs, val_losses, color="#F44336", linewidth=1.2, label="Validation MSE")

    # Mark best epoch
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8,
               label=f"Best epoch ({best_epoch}, val={best_val:.6f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Curves — {name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{name}_training_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LSTM invocation-count predictor (Vahidinia et al. 2023 Layer 2)")
    parser.add_argument("csv_files", nargs="+", help="Input CSV file(s) from traffic_pattern/")
    parser.add_argument("--output-dir", default="outputs/lstm",
                        help="Directory for output CSVs (default: outputs/lstm)")
    parser.add_argument("--window", type=int, default=10,
                        help="Sliding window size (default: 10)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Train/test split ratio (default: 0.7)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--plot", nargs="?", const="all", default=None, metavar="RANGE",
                        help="Generate plots. Optional day range: --plot 5-10 (default: all)")
    parser.add_argument("--hourly", action="store_true",
                        help="Aggregate to hourly before training; plot by hours, output in minutes")
    parser.add_argument("--plot-style", choices=["scatter", "line"], default="scatter",
                        help="Plot style: scatter or line (default: scatter)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Grid search for best hyperparameters per CSV")
    parser.add_argument("--hidden-size", type=int, default=32,
                        help="LSTM hidden size (default: 32)")
    args = parser.parse_args()

    results = []
    for csv_path in args.csv_files:
        ret = process_csv(
            csv_path, args.output_dir, args.window, args.train_ratio,
            args.epochs, args.batch_size, args.lr, args.patience, args.device,
            hourly=args.hourly, do_grid_search=args.grid_search,
            hidden_size=args.hidden_size,
        )
        if ret is not None:
            results.append((csv_path, ret[0], ret[1]))

    if args.plot is not None:
        plot_range = None
        if args.plot != "all":
            parts = args.plot.split("-", 1)
            plot_range = (float(parts[0]), float(parts[1]))

        print("\nGenerating plots...")
        for csv_path, out_path, result in results:
            name = Path(csv_path).stem
            plot_predictions(out_path, args.output_dir, hourly=args.hourly,
                             plot_range=plot_range, plot_style=args.plot_style)
            plot_training_curves(
                result.train_losses, result.val_losses,
                name, args.output_dir,
            )

    print(f"\nDone. All results in {args.output_dir}/")


if __name__ == "__main__":
    main()
