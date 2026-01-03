# train_predictor.py
# Seq2seq RPS predictor with variational (locked) dropout + MC-dropout uncertainty.
# Author: you + a helpful AI
#
# Usage:
#   python train_predictor.py --data_path data.csv --input_n 30 --output_n 30
#
# Notes:
# - Expects columns: <target_col>, hour_of_day_sin, hour_of_day_cos, day_of_week_sin, day_of_week_cos
# - Uses sliding windows to build (X_in -> y_future) examples.
# - Variational (locked) dropout across time; MC-dropout is used at eval to produce mean/std bands.

import argparse
import math
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StandardScaler:
    """Simple standardizer with inverse-transform."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


def build_windows(
    data: np.ndarray,
    target_index: int,
    input_n: int,
    output_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: shape [T, D] with D features inc. target.
    Returns:
      X: [N, input_n, D]
      Y: [N, output_n]  (only the target is predicted)
    """
    T, D = data.shape
    N = T - input_n - output_n + 1
    if N <= 0:
        raise ValueError("Not enough rows for the requested input/output windows.")
    X = np.zeros((N, input_n, D), dtype=np.float32)
    Y = np.zeros((N, output_n), dtype=np.float32)
    for i in range(N):
        X[i] = data[i:i+input_n, :]
        Y[i] = data[i+input_n:i+input_n+output_n, target_index]
    return X, Y


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


# ----------------------------
# Variational (Locked) Dropout
# ----------------------------

class LockedDropout(nn.Module):
    """
    Apply the same dropout mask across all timesteps (variational dropout).
    If x is [B, T, D], one mask of [B, 1, D] is sampled and broadcast across T.
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if x.dim() == 3:
            B, T, D = x.size()
            mask = x.new_empty(B, 1, D).bernoulli_(1 - self.p).div_(1 - self.p)
            return x * mask
        elif x.dim() == 2:
            B, D = x.size()
            mask = x.new_empty(B, D).bernoulli_(1 - self.p).div_(1 - self.p)
            return x * mask
        else:
            return x  # fallback


# ----------------------------
# Model
# ----------------------------

class Seq2SeqRPS(nn.Module):
    """
    Encoder: LSTM over input window (features include target and 4 cyclical features).
    Decoder: LSTMCell autoregressive over horizon, input is previous target (scaled).
    Variational dropout is applied to encoder inputs and between LSTM layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_in: float = 0.1,
        dropout_hidden: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lockdrop_in = LockedDropout(dropout_in)
        # Dropout between LSTM layers (PyTorch applies this on the outputs of intermediate layers).
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
        )

        # Single-layer decoder cell for simplicity.
        self.decoder_cell = nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.lockdrop_dec_in = LockedDropout(dropout_hidden)

        self.readout = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,                 # [B, Tin, D]
        y: torch.Tensor = None,          # [B, Tout] (optional for teacher forcing)
        teacher_forcing_ratio: float = 0.5,
        out_steps: int = 30,
        target_in_x_index: int = 0
    ) -> torch.Tensor:
        B, Tin, D = x.size()
        x = self.lockdrop_in(x)
        enc_out, (h, c) = self.encoder(x)        # h,c: [num_layers, B, H]
        # Initialize decoder from top encoder layer
        h_t = h[-1]                               # [B, H]
        c_t = c[-1]                               # [B, H]

        # Decoder input starts as the last seen target from the input window
        dec_in = x[:, -1, target_in_x_index].unsqueeze(-1)  # [B, 1]
        dec_in = self.lockdrop_dec_in(dec_in)               # locked mask across steps

        outputs = []
        for t in range(out_steps):
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            step_out = self.readout(h_t)          # [B, 1]
            outputs.append(step_out)

            use_tf = (self.training and y is not None
                      and random.random() < teacher_forcing_ratio)
            next_in = (y[:, t:t+1] if use_tf else step_out.detach())
            # keep the *same* mask across timesteps for variational effect
            dec_in = self.lockdrop_dec_in(next_in)

        return torch.cat(outputs, dim=1)          # [B, Tout]


# ----------------------------
# Training / Evaluation
# ----------------------------

def train_one_epoch(model, loader, optimizer, device, target_scaler, args):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(
            xb,
            yb,
            teacher_forcing_ratio=args.teacher_forcing,
            out_steps=yb.size(1),
            target_in_x_index=args.target_in_x_index,
        )
        loss = nn.functional.mse_loss(preds, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(
            xb, None, teacher_forcing_ratio=0.0,
            out_steps=yb.size(1), target_in_x_index=args.target_in_x_index
        )
        mse_sum += nn.functional.mse_loss(preds, yb, reduction='sum').item()
        mae_sum += nn.functional.l1_loss(preds, yb, reduction='sum').item()
        n += yb.numel()
    rmse = math.sqrt(mse_sum / n)
    mae = mae_sum / n
    return rmse, mae


@torch.no_grad()
def predict_with_uncertainty(model, loader, device, mc_samples: int):
    """
    Returns mean and std over MC samples.
    - Keeps dropout active by setting model.train() but disables grads.
    """
    model.train()  # keep dropout ON
    all_means, all_stds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        T_out = yb.size(1)
        samples = []
        for _ in range(mc_samples):
            preds = model(
                xb, None, teacher_forcing_ratio=0.0,
                out_steps=T_out, target_in_x_index=args.target_in_x_index
            )
            samples.append(preds.unsqueeze(0))  # [1, B, T]
        stack = torch.cat(samples, dim=0)       # [S, B, T]
        mean = stack.mean(dim=0)                # [B, T]
        std = stack.std(dim=0)                  # [B, T]
        all_means.append(mean.cpu())
        all_stds.append(std.cpu())
    return torch.cat(all_means), torch.cat(all_stds)  # [N, T], [N, T]


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="invocation_rate",
                        help="Name of the target (invocation rate) column.")
    parser.add_argument("--input_n", type=int, default=30)
    parser.add_argument("--output_n", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)

    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_in", type=float, default=0.1)
    parser.add_argument("--dropout_hidden", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--teacher_forcing", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mc_samples", type=int, default=50,
                        help="Number of MC-dropout samples for uncertainty.")
    parser.add_argument("--save_path", type=str, default="model.pt")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in CSV.")
    for req in ["hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos"]:
        if req not in df.columns:
            raise ValueError(f"Missing required feature column: {req}")

    # Order features: target first, then the four cyclical features, then any extras if present
    base_feats = [args.target_col,
                  "hour_of_day_sin", "hour_of_day_cos",
                  "day_of_week_sin", "day_of_week_cos"]
    extras = [c for c in df.columns if c not in base_feats and c != "date"]
    cols = base_feats + extras
    data = df[cols].astype(np.float32).values
    target_in_x_index = cols.index(args.target_col)
    args.target_in_x_index = target_in_x_index  # stash for eval

    # Split (temporal)
    T = len(data)
    n_test = int(T * args.test_split)
    n_val = int(T * args.val_split)
    n_train = T - n_val - n_test
    if n_train <= args.input_n + args.output_n:
        raise ValueError("Training split too small for chosen window sizes.")

    data_train = data[:n_train]
    data_val   = data[n_train - (args.input_n + args.output_n): n_train + n_val]  # allow windows to cross boundary
    data_test  = data[n_train + n_val - (args.input_n + args.output_n):]

    # Standardize per-feature using train stats
    feat_scaler = StandardScaler().fit(data_train)
    data_train_s = feat_scaler.transform(data_train)
    data_val_s   = feat_scaler.transform(data_val)
    data_test_s  = feat_scaler.transform(data_test)

    # Build windows
    Xtr, Ytr = build_windows(data_train_s, target_in_x_index, args.input_n, args.output_n)
    Xva, Yva = build_windows(data_val_s, target_in_x_index, args.input_n, args.output_n)
    Xte, Yte = build_windows(data_test_s, target_in_x_index, args.input_n, args.output_n)

    train_ds = TimeSeriesWindowDataset(Xtr, Ytr)
    val_ds   = TimeSeriesWindowDataset(Xva, Yva)
    test_ds  = TimeSeriesWindowDataset(Xte, Yte)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = Seq2SeqRPS(
        input_dim=Xtr.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_in=args.dropout_in,
        dropout_hidden=args.dropout_hidden
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    patience, bad = 8, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, device, None, args)
        val_rmse, val_mae = evaluate(model, val_dl, device)
        print(f"Epoch {epoch:03d} | train_mse: {tr_loss:.6f} | val_RMSE: {val_rmse:.6f} | val_MAE: {val_mae:.6f}")

        score = val_rmse
        if score < best_val - 1e-5:
            best_val = score
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "cols": cols,
                "input_n": args.input_n,
                "output_n": args.output_n,
                "feat_scaler_mean": feat_scaler.mean_,
                "feat_scaler_std": feat_scaler.std_,
                "target_in_x_index": target_in_x_index,
                "args": vars(args)
            }, args.save_path)
            print(f"  Saved checkpoint to {args.save_path}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Reload best
    if os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    # Final evaluation with MC-dropout
    mean_preds, std_preds = predict_with_uncertainty(model, test_dl, device, args.mc_samples)
    # Compute point metrics using means
    with torch.no_grad():
        model.eval()  # for point estimate (no dropout); optional
        rmse, mae = evaluate(model, test_dl, device)
    print(f"Test point-estimate  RMSE: {rmse:.6f} | MAE: {mae:.6f}")

    # Optionally write predictions to CSV
    out_csv = "test_predictions_with_uncertainty.csv"
    # For alignment, we’ll store per-sample horizon in wide format
    idx = []
    rows = []
    for i in range(mean_preds.shape[0]):
        row = {}
        for t in range(mean_preds.shape[1]):
            row[f"mean_t{t+1}"] = float(mean_preds[i, t].item())
            row[f"std_t{t+1}"]  = float(std_preds[i, t].item())
        rows.append(row)
        idx.append(i)
    out_df = pd.DataFrame(rows, index=idx)
    out_df.to_csv(out_csv)
    print(f"Wrote MC-dropout predictions to {out_csv}")
