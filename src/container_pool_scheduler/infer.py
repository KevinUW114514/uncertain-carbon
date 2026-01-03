# infer_predictor.py
# Uncertainty-aware inference with MC-dropout + denormalized (RPS) outputs.

import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Reusable bits (match training)
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class StandardScaler:
    def __init__(self, mean_=None, std_=None):
        self.mean_ = mean_
        self.std_ = std_

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / (self.std_ + 1e-8)

def build_windows(
    data: np.ndarray,
    target_index: int,
    input_n: int,
    output_n: int
) -> Tuple[np.ndarray, np.ndarray]:
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

class LockedDropout(nn.Module):
    """Variational (locked) dropout: same mask across timesteps."""
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
        return x

class Seq2SeqRPS(nn.Module):
    """Encoder LSTM over input window, decoder LSTMCell autoregressive."""
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_in: float = 0.1,
        dropout_hidden: float = 0.2
    ):
        super().__init__()
        self.lockdrop_in = LockedDropout(dropout_in)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.lockdrop_dec_in = LockedDropout(dropout_hidden)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,                 # [B, Tin, D]
        y: torch.Tensor = None,          # [B, Tout] (unused here)
        teacher_forcing_ratio: float = 0.0,
        out_steps: int = 30,
        target_in_x_index: int = 0
    ) -> torch.Tensor:
        x = self.lockdrop_in(x)
        _, (h, c) = self.encoder(x)
        h_t, c_t = h[-1], c[-1]                         # [B, H]

        dec_in = x[:, -1, target_in_x_index].unsqueeze(-1)  # [B, 1]
        dec_in = self.lockdrop_dec_in(dec_in)

        outs = []
        for _ in range(out_steps):
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            step = self.readout(h_t)
            outs.append(step)
            dec_in = self.lockdrop_dec_in(step.detach())
        return torch.cat(outs, dim=1)                   # [B, Tout]


# ----------------------------
# Inference helpers
# ----------------------------


@torch.no_grad()
def mc_predict(model, loader, device, mc_samples: int, target_in_x_index: int, out_steps: int):
    """Run MC-dropout: keep model.train() so dropout is active; disable grads."""
    model.train()
    means, stds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        steps = out_steps if out_steps is not None else yb.size(1)
        samples = []
        for _ in range(mc_samples):
            preds = model(
                xb, None, teacher_forcing_ratio=0.0,
                out_steps=steps, target_in_x_index=target_in_x_index
            )
            samples.append(preds.unsqueeze(0))
        stack = torch.cat(samples, dim=0)   # [S, B, T]
        means.append(stack.mean(0).cpu())
        stds.append(stack.std(0).cpu())
    return torch.cat(means), torch.cat(stds)  # [N, T], [N, T]

def slice_last_n_windows(X: np.ndarray, Y: np.ndarray, n: int):
    if n <= 0 or n > len(X):
        n = 1
    return X[-n:], Y[-n:]


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC-dropout inference for RPS predictor")
    parser.add_argument("--data_path", type=str, required=True, help="CSV with features/target")
    parser.add_argument("--ckpt_path", type=str, default="model.pt", help="Checkpoint from training")
    parser.add_argument("--target_col", type=str, default=None,
                        help="Override target column name; defaults to checkpoint")
    parser.add_argument("--input_n", type=int, default=None, help="Override input window length")
    parser.add_argument("--output_n", type=int, default=None, help="Override output horizon")
    parser.add_argument("--mc_samples", type=int, default=50, help="MC samples for uncertainty")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--windows", type=int, default=1, help="Number of latest windows to forecast")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_csv", type=str, default="inference_predictions_with_uncertainty.csv")
    parser.add_argument("--include_z", action="store_true",
                        help="Also write standardized mean/std (z-units) to CSV")
    parser.add_argument("--no_intervals", action="store_true",
                        help="Do not include 95% intervals in RPS units")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Inference device: {device}")

    # ---- Load checkpoint ----
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)

    saved_cols = ckpt.get("cols")
    saved_input_n = ckpt.get("input_n")
    saved_output_n = ckpt.get("output_n")
    saved_target_idx = ckpt.get("target_in_x_index")
    saved_args = ckpt.get("args", {})
    saved_target_col = saved_args.get("target_col", "rps")

    target_col = args.target_col or saved_target_col
    input_n = args.input_n if args.input_n is not None else saved_input_n
    output_n = args.output_n if args.output_n is not None else saved_output_n

    # ---- Load data ----
    required = ["hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos"]
    df = pd.read_csv(args.data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required feature column: {r}")

    base_feats = [target_col] + required
    extras = [c for c in df.columns if c not in base_feats and c != "timestamp"]
    cols = base_feats + extras
    if saved_cols is not None and cols != saved_cols:
        missing_from_saved = [c for c in saved_cols if c not in cols]
        if missing_from_saved:
            raise ValueError(f"Your CSV lacks columns present during training: {missing_from_saved}")
        cols = saved_cols  # use exact training order

    data = df[cols].astype(np.float32).values
    target_in_x_index = cols.index(target_col)

    # ---- Restore scaler & transform ----
    feat_scaler = StandardScaler(ckpt["feat_scaler_mean"], ckpt["feat_scaler_std"])
    data_s = feat_scaler.transform(data)
    print(data[0])
    input("debug")

    # ---- Windows ----
    X, Y_dummy = build_windows(data_s, target_in_x_index, input_n, output_n)
    X, Y_dummy = slice_last_n_windows(X, Y_dummy, args.windows)
    dl = DataLoader(TimeSeriesWindowDataset(X, Y_dummy), batch_size=args.batch_size, shuffle=False)

    # ---- Rebuild model ----
    hidden_size = saved_args.get("hidden_size", 128)
    num_layers = saved_args.get("num_layers", 2)
    dropout_in = saved_args.get("dropout_in", 0.1)
    dropout_hidden = saved_args.get("dropout_hidden", 0.2)

    model = Seq2SeqRPS(
        input_dim=X.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_in=dropout_in,
        dropout_hidden=dropout_hidden
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ---- MC-dropout inference (standardized units) ----
    means_z, stds_z = mc_predict(model, dl, device, args.mc_samples, target_in_x_index, out_steps=output_n)

    # ---- Denormalize to RPS ----
    target_mean = float(ckpt["feat_scaler_mean"][0, target_in_x_index])
    target_std  = float(ckpt["feat_scaler_std"][0, target_in_x_index])

    means_rps = means_z.numpy() * target_std + target_mean
    stds_rps  = stds_z.numpy() * target_std

    # ---- Save CSV ----
    rows = {"mean_rps": [], "std_rps": [], "lo95_rps": [], "hi95_rps": [], "mean_z": [], "std_z": []}
    include_intervals = not args.no_intervals
    include_z = args.include_z

    for i in range(means_rps.shape[0]):
        for t in range(means_rps.shape[1]):
            # RPS units
            m = float(means_rps[i, t]); s = float(stds_rps[i, t])
            rows["mean_rps"].append(m)
            rows["std_rps"].append(s)
            if include_intervals:
                rows["lo95_rps"].append(m - 1.96 * s)
                rows["hi95_rps"].append(m + 1.96 * s)

            # Optional: standardized outputs
            if include_z:
                rows["mean_z"] = float(means_z[i, t].item())
                rows["std_z"]  = float(stds_z[i, t].item())

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions to {args.out_csv}")

    # Quick print for the most recent window
    last_mean = means_rps[-1]
    last_std  = stds_rps[-1]
    print("Most recent window (first 5 steps in RPS):")
    for k in range(min(5, output_n)):
        print(f" t+{k+1:02d}: mean={last_mean[k]:.2f}, std={last_std[k]:.2f}")
