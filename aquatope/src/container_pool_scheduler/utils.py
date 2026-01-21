import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.encoder_decoder_dropout import *
from torch.utils.data import DataLoader
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(SCHED_DIR))

import huawei_data as data


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"

    return torch.device(device)


from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import torch
import torch.nn as nn
import pandas as pd

import torch

def knn_conditioned_residuals(
    resid_hist: torch.Tensor,          # [m]
    feat_hist: torch.Tensor,           # [m, d]
    feat_t: torch.Tensor,              # [d]
    k: int = 15,
):
    """
    Return a subset of residuals (and corresponding indices) chosen as K nearest neighbors
    in feature space. Uses L1 distance for stability.
    """
    m = resid_hist.numel()
    if m == 0:
        return resid_hist

    k = min(k, m)
    # L1 distance; robust and simple
    d = (feat_hist - feat_t.unsqueeze(0)).abs().sum(dim=1)  # [m]
    nn_idx = torch.topk(-d, k=k).indices  # pick smallest distances
    return resid_hist[nn_idx]

import math

def sample_residuals_smoothed(
    resid: torch.Tensor,               # [k]
    num_samples: int,
    lam: float = 0.1,                  # recency decay within selected set (0 = uniform)
    noise: str = "studentt",           # "gaussian" or "studentt"
    df: float = 5.0,                   # for student-t
    bandwidth_scale: float = 0.5,      # 0.3–0.8 typical
    min_bandwidth: float = 1e-6,
):
    """
    Smoothed bootstrap: pick residual with (optional) recency weights, then add noise
    with robust bandwidth (MAD). Produces continuous residual samples.
    """
    k = resid.numel()
    if k == 0:
        return resid.new_zeros((num_samples,))

    # Recency weights: assume resid is ordered oldest->newest; newest gets highest weight.
    # If your selection doesn't preserve time order, set lam=0 or reorder before calling.
    ages = torch.arange(k-1, -1, -1, device=resid.device, dtype=torch.float32)  # 0 newest
    w = torch.exp(-lam * ages)
    w = w / w.sum()

    # Bootstrap indices
    bidx = torch.multinomial(w, num_samples=num_samples, replacement=True)
    base = resid[bidx]

    # Robust bandwidth via MAD
    med = resid.median()
    mad = (resid - med).abs().median()
    robust_std = 1.4826 * mad
    h = max(float(bandwidth_scale) * float(robust_std), float(min_bandwidth))
    h = resid.new_tensor(h)

    if noise == "gaussian":
        eps = torch.randn(num_samples, device=resid.device, dtype=resid.dtype) * h
    else:
        eps = torch.distributions.StudentT(df=df, loc=0.0, scale=h).sample((num_samples,))
        eps = eps.to(device=resid.device, dtype=resid.dtype)

    return base + eps



def sample_continuous_residuals(
    resid_window: torch.Tensor,     # shape [m], signed residuals
    num_samples: int,
    lam: float = 0.25,              # recency decay (0 -> uniform)
    bandwidth: float = None,        # if None, use robust default
    min_bandwidth: float = 1e-6,
):
    """
    Continuous residual sampler via weighted KDE / smoothed bootstrap:
      r = r_i + eps, eps ~ N(0, h^2), i sampled with recency weights.

    resid_window is assumed oldest->newest or newest->oldest; we treat the last element as newest.
    """
    m = resid_window.numel()
    assert m > 0

    # Recency weights: newest gets most weight
    ages = torch.arange(m-1, -1, -1, device=resid_window.device, dtype=torch.float32)  # 0 newest
    w = torch.exp(-lam * ages)
    w = w / w.sum()

    # Choose bandwidth h
    if bandwidth is None:
        med = resid_window.median()
        mad = (resid_window - med).abs().median()
        # robust std approx: 1.4826 * MAD
        robust_std = 1.4826 * mad
        bandwidth = 0.5 * robust_std  # tune 0.3-0.8 depending on smoothness desired

    h = max(float(bandwidth), min_bandwidth)

    # Sample base residual indices
    idx = torch.multinomial(w, num_samples=num_samples, replacement=True)
    base = resid_window[idx]

    # Smooth with Gaussian noise
    eps = torch.randn(num_samples, device=resid_window.device, dtype=resid_window.dtype) * h
    return base + eps


# @torch.no_grad()
# def inference_conformal(
#     datasets: dict,
#     model: nn.Module,
#     mc_dropout: bool = False,
#     dropout_passes: int = 30,          # model stochastic passes if mc_dropout=True
#     window_size: int = 20,             # sliding window size
#     alpha: float = 0.50,               # 90% CI => alpha=0.10
#     mc_samples: int = 300,             # MC samples drawn from residual bootstrap
#     agg: str = "median",                 # "mean" or "median"
# ):
#     device = get_device()

#     # IMPORTANT: inference loader must be ordered (shuffle=False / sequential sampler)
#     valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

#     model.to(device)
#     model.eval()

#     if mc_dropout:
#         model = model.apply(dropout_on)   # enable dropout layers
#     else:
#         model = model.apply(dropout_off)

#     json_data = read_json_params("train_invocation_rate_normalization.json")
#     train_mu, train_sigma = json_data["mu"], json_data["sigma"]
#     sigma, mu = train_sigma, train_mu

#     # One batch containing all data
#     (x, y, last_x, first_y, first_x, idx) = next(iter(valid_loader))
#     x, y = x.to(device), y.to(device)

#     # Ensure idx is 1D CPU tensor for sorting/validation
#     idx_cpu = idx.detach().cpu().view(-1)
#     # Enforce increasing order (and reorder tensors if needed)
#     if not torch.all(idx_cpu[:-1] <= idx_cpu[1:]):
#         sort_perm = torch.argsort(idx_cpu)
#         idx_cpu = idx_cpu[sort_perm]
#         x = x[sort_perm]
#         y = y[sort_perm]
#         # If you use these later, keep them aligned too:
#         last_x = last_x[sort_perm]
#         first_y = first_y[sort_perm]
#         first_x = first_x[sort_perm]

#     # Point prediction (optionally MC-dropout over the model)
#     # Your model expects: model((x, y[:, 0, 1:]))
#     cond = y[:, 0, 1:].to(device)

#     # if mc_dropout:
#     #     preds = []
#     #     for _ in range(dropout_passes):
#     #         res = model((x, cond))               # shape [N, 1] (likely)
#     #         preds.append(res)
#     #     res = torch.stack(preds, dim=0).mean(dim=0)  # mean over dropout passes
#     #     # model epistemic variance proxy (on res scale)
#     #     model_var = torch.stack(preds, dim=0).var(dim=0).squeeze(-1)
#     # else:
#     res = model((x, cond))
#     model_var = torch.zeros(res.shape[0], device=device)

#     # Denormalize to your target space
#     point_pred = (res.squeeze(-1) + x[:, -1, 0]) * train_sigma + train_mu
#     target = ((y[:, 0, 0] + x[:, -1, 0]) * train_sigma + train_mu).squeeze(-1)

#     # Sequential rolling conformal
#     n = point_pred.shape[0]
#     lower = torch.empty(n, device=device)
#     upper = torch.empty(n, device=device)
#     refined = torch.empty(n, device=device)

#     # windows store signed residuals: r_t = target_t - pred_t
#     signed_resid_hist = []

#     def conformal_q(abs_resids: torch.Tensor, alpha: float) -> torch.Tensor:
#         """
#         Conformal quantile with 'higher' interpolation:
#         q = abs_resids[k-1] where k = ceil((m+1)*(1-alpha)), m=len(abs_resids)
#         """
#         m = abs_resids.numel()
#         if m == 0:
#             return torch.tensor(0.0, device=abs_resids.device)
#         k = int(math.ceil((m + 1) * (1 - alpha)))
#         k = min(max(k, 1), m)
#         # torch.kthvalue is 1-indexed
#         return abs_resids.kthvalue(k).values

#     for t in range(n):
#         # Use last window_size residuals (strictly from the past)
#         past = signed_resid_hist[-window_size:]
#         if len(past) == 0:
#             q = torch.tensor(0.0, device=device)
#             bias = torch.tensor(0.0, device=device)
#         else:
#             past_tensor = torch.tensor(past, device=device, dtype=point_pred.dtype)
#             q = conformal_q(past_tensor.abs(), alpha=alpha)
#             bias = past_tensor.mean()  # empirical bias over the window

#         # 90% CI around current point prediction (symmetric)
#         lo = point_pred[t] - q
#         hi = point_pred[t] + q
#         lower[t], upper[t] = lo, hi

#         # Monte Carlo refinement:
#         # - bootstrap signed residuals from the window (captures bias/asymmetry)
#         # - add to current point prediction
#         # - clip to conformal CI
#         if len(past) == 0 or mc_samples <= 0:
#             refined[t] = point_pred[t]
#         else:
#             ####################################################################
#             #  Uniform bootstrap (discrete residuals)
#             # past_tensor = torch.tensor(past, device=device, dtype=point_pred.dtype)
#             # # bootstrap indices
#             # bidx = torch.randint(low=0, high=past_tensor.numel(), size=(mc_samples,), device=device)
#             # boot = past_tensor[bidx]

#             # # Optionally blend in bias directly (stabilizes with tiny windows)
#             # # sample = point_pred[t] + boot  is usually enough
#             # samples = point_pred[t] + boot

#             # # Clip samples to CI (so refinement cannot “escape” the conformal guarantee band)
#             # samples = torch.clamp(samples, min=lo, max=hi)

#             # if agg == "median":
#             #     refined[t] = samples.median()
#             # else:
#             #     refined[t] = samples.mean()
#             ####################################################################
#             ####################################################################
#             # Direct residual sampling with KDE smoothing
#             past = torch.tensor(past, device=device)  # signed residuals, most recent last
#             # print(f"past: {past}")
#             # input("debug")
#             m = past.numel()
#             print(f"mean of past residuals: {past.mean().item()}, std: {past.std().item()}")
#             input()

#             # Exponential recency weights: newest gets largest weight
#             lam = 0.1  # tune; larger => more recent emphasis
#             ages = torch.arange(m-1, -1, -1, device=device, dtype=point_pred.dtype)  # 0 for newest
#             w = torch.exp(-lam * ages)
#             w = w / w.sum()

#             # Sample residual indices with those weights
#             bidx = torch.multinomial(w, num_samples=mc_samples, replacement=True)
#             boot = past[bidx]
            
#             # print(f"boot: {boot}")
#             # input("debug")
#             # print(f"point_pred[t]: {point_pred[t]}")
#             # input("debug")
#             samples = point_pred[t] + boot
#             # h = 0.1 * (hi - lo)  # small fraction of CI width
#             med = past.median()
#             mad = (past - med).abs().median()
#             robust_std = 1.4826 * mad
#             h = 0.5 * robust_std  # start here; tune 0.3–0.8
#             h = torch.clamp(h, min=1e-6)
#             samples = samples + torch.randn_like(samples) * h
#             # samples = torch.clamp(samples, min=lo, max=hi)

#             refined[t] = samples.mean()  # or samples.mean()
#             ####################################################################
            

#         # After producing prediction for t, update residual history with the *observed* residual at t
#         signed_resid_hist.append((target[t] - point_pred[t]).detach().item())

#     # Error metrics (use refined prediction as your "final")
#     final_pred = refined
#     error_rates = torch.abs(final_pred - target) / target
#     smape_rate = smape(target.detach().cpu().numpy(), final_pred.detach().cpu().numpy())

#     # Summary stats
#     # (mean/var of model output is not very meaningful with one batch unless you define it;
#     # here we report mean/var of prediction errors and average model variance proxy)
#     mean_abs_err = torch.mean(torch.abs(final_pred - target))
#     mean_model_var = model_var.mean()

#     calc_percentile_stats(
#         error_rates.detach().cpu().numpy(),
#         (torch.abs(final_pred - target) / target).sum().item() / len(final_pred),
#         "inference_results_uncertain"
#     )

#     s = (
#         f"[inference_conformal] "
#         f"mean_abs_err: {mean_abs_err.item():.6f}, "
#         f"mean_model_var: {mean_model_var.item():.6f}, "
#         f"smape_rate: {smape_rate}"
#     )
#     with open("inference_results_uncertain.log", "a") as f:
#         f.write(s + "\n")
#     print(s)

#     to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)

#     df = pd.DataFrame(
#         {
#             "idx": to_1d(idx_cpu),
#             "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
#             "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),
#             "point_pred": to_1d(point_pred),
#             "pred_refined": to_1d(final_pred),
#             "target": to_1d(target),
#             "ci_lower": to_1d(lower),
#             "ci_upper": to_1d(upper),
#             "error_rate_refined_pct": to_1d(error_rates) * 100,
#         }
#     )
#     df.to_csv("inference_results_uncertain.csv", index=False)

#     # Return useful objects
#     return {
#         "point_pred": point_pred,
#         "pred_refined": final_pred,
#         "target": target,
#         "ci_lower": lower,
#         "ci_upper": upper,
#         "smape": smape_rate,
#         "mean_abs_err": mean_abs_err,
#         "mean_model_var": mean_model_var,
#         "df": df,
#     }

def ew_robust_scale(past_resid: torch.Tensor, lam: float = 0.05):
    """
    Exponentially weighted robust scale using weighted MAD.
    lam: higher => more emphasis on most recent residuals.
    """
    m = past_resid.numel()
    ages = torch.arange(m-1, -1, -1, device=past_resid.device, dtype=torch.float32)  # 0 newest
    w = torch.exp(-lam * ages)
    w = w / w.sum()

    # weighted median (approx via sorting)
    srt, idx = torch.sort(past_resid)
    w_srt = w[idx]
    cdf = torch.cumsum(w_srt, dim=0)
    med = srt[(cdf >= 0.5).nonzero(as_tuple=False)[0].item()]

    abs_dev = (past_resid - med).abs()
    srt2, idx2 = torch.sort(abs_dev)
    w2 = w[idx2]
    cdf2 = torch.cumsum(w2, dim=0)
    mad = srt2[(cdf2 >= 0.5).nonzero(as_tuple=False)[0].item()]

    robust_std = 1.4826 * mad
    return med, robust_std

import numpy as np
import matplotlib.pyplot as plt

def plot_like_example(
    df,
    x_col="idx",                     # or a datetime column
    y_true_col="target",
    y_pred_col="pred_refined",       # refined / post-processed prediction
    y_pred_raw_col="point_pred",     # pre-refined prediction
    lo_col="ci_lower",
    hi_col="ci_upper",
    title="Traffic volume prediction with uncertainty bounds",
    ylabel="Traffic volume",
    xlabel="Time",
    use_2sigma=True,
):
    x = df[x_col].to_numpy()
    y_true = df[y_true_col].to_numpy()
    y_pred = df[y_pred_col].to_numpy()
    y_pred_raw = df[y_pred_raw_col].to_numpy()
    lo = df[lo_col].to_numpy()
    hi = df[hi_col].to_numpy()

    # Treat conformal half-width as "sigma" for visualization
    sigma = 0.5 * (hi - lo)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Lighter band: ±2σ (around refined prediction)
    if use_2sigma:
        ax.fill_between(
            x,
            y_pred - 2.0 * sigma,
            y_pred + 2.0 * sigma,
            alpha=0.15,
            label=r"Uncertainty, $\hat{Y}_{ref}\pm 2\sigma$",
            linewidth=0,
        )

    # Darker band: ±σ
    ax.fill_between(
        x,
        y_pred - sigma,
        y_pred + sigma,
        alpha=0.30,
        label=r"Uncertainty, $\hat{Y}_{ref}\pm \sigma$",
        linewidth=0,
    )

    # Lines
    ax.plot(x, y_true, linestyle="--", linewidth=2, label=r"True, $Y$")
    ax.plot(x, y_pred_raw, linestyle=":", linewidth=2,
            label=r"Pred. (pre-refine), $\hat{Y}_{raw}$")
    ax.plot(x, y_pred, linewidth=2,
            label=r"Pred. (refined), $\hat{Y}_{ref}$")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("prediction_with_uncertainty.png", dpi=350)
    print("Saved plot to prediction_with_uncertainty.png")
    plt.show()


# Example usage (after your inference call):
# out = inference_conformal(...)
# plot_like_example(out["df"])

def robust_median_mad(x: torch.Tensor, eps: float = 1e-12):
    """
    Robust location/scale:
      loc = median(x)
      scale = 1.4826 * MAD(x)
    """
    if x.numel() == 0:
        return x.new_tensor(0.0), x.new_tensor(0.0)
    loc = x.median()
    mad = (x - loc).abs().median()
    scale = x.new_tensor(1.4826) * mad
    scale = torch.clamp(scale, min=eps)
    return loc, scale

# @torch.no_grad()
# def inference_conformal(
#     datasets: dict,
#     model: nn.Module,
#     mc_dropout: bool = False,
#     dropout_passes: int = 30,          # only used if mc_dropout=True
#     window_size: int = 30,             # rolling history size
#     k_neighbors: int = 10,             # KNN subset size for regime conditioning (<= window_size)
#     alpha: float = 0.1,               # 85% CI => alpha=0.15
#     mc_samples: int = 300,             # number of predictive samples
#     agg: str = "median",               # "median" (MAE-optimal), "mean", "trimmed_mean"
#     lam_recency: float = 0.00,          # recency decay inside selected set (keep 0 unless you keep time order)
#     noise: str = "studentt",           # "studentt" or "gaussian"
#     studentt_df: float = 4,          # df for Student-t noise
#     bandwidth_scale: float = 0.35,      # 0.3–0.8 typical
#     clamp_to_ci: bool = True,          # keep refined samples within conformal CI
# ):
#     """
#     Rolling conformal intervals + regime-conditioned residual sampling for point refinement.

#     Regime features (default):
#       - x_last = x[t, -1, 0]
#       - x_trend = x[t, -1, 0] - x[t, -2, 0]  (0 if not available)

#     Refinement for MAE:
#       - bias_l1 = median(residuals | regime)
#       - samples = point_pred[t] + bias_l1 + (bootstrapped centered residual + noise)
#       - refined[t] = median(samples)  (MAE-optimal)
#     """

#     device = get_device()

#     # IMPORTANT: inference loader must be ordered (shuffle=False / sequential sampler)
#     valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

#     model.to(device)
#     model.eval()

#     if mc_dropout:
#         model = model.apply(dropout_on)
#     else:
#         model = model.apply(dropout_off)

#     # normalization params
#     json_data = read_json_params("train_invocation_rate_normalization.json")
#     train_mu, train_sigma = json_data["mu"], json_data["sigma"]
#     sigma, mu = train_sigma, train_mu

#     # one batch containing all data
#     (x, y, last_x, first_y, first_x, idx) = next(iter(valid_loader))
#     x, y = x.to(device), y.to(device)

#     # Ensure idx is 1D CPU tensor for sorting/validation
#     idx_cpu = idx.detach().cpu().view(-1)
#     if not torch.all(idx_cpu[:-1] <= idx_cpu[1:]):
#         sort_perm = torch.argsort(idx_cpu)
#         idx_cpu = idx_cpu[sort_perm]
#         x = x[sort_perm]
#         y = y[sort_perm]
#         last_x = last_x[sort_perm]
#         first_y = first_y[sort_perm]
#         first_x = first_x[sort_perm]

#     # model conditional input
#     cond = y[:, 0, 1:].to(device)

#     # point prediction (optionally average MC-dropout passes)
#     # if mc_dropout:
#     #     preds = []
#     #     for _ in range(dropout_passes):
#     #         preds.append(model((x, cond)))
#     #     res_stack = torch.stack(preds, dim=0)                 # [S, N, 1]
#     #     res = res_stack.mean(dim=0)                           # [N, 1]
#     #     model_var = res_stack.var(dim=0).squeeze(-1)          # [N]
#     # else:
#     res = model((x, cond))
#     model_var = torch.zeros(res.shape[0], device=device)

#     # denormalize
#     point_pred = (res.squeeze(-1) + x[:, -1, 0]) * train_sigma + train_mu
#     target = ((y[:, 0, 0] + x[:, -1, 0]) * train_sigma + train_mu).squeeze(-1)
#     # for i in range(300, 350):
#     #     print(f"point_pred[{i}]: {point_pred[i].item()}, target[{i}]: {target[i].item()}")
#     # input("debug")

#     n = point_pred.shape[0]
#     ci_lower = torch.empty(n, device=device)
#     ci_upper = torch.empty(n, device=device)
#     refined = torch.empty(n, device=device)

#     # histories for conditioning
#     resid_hist = []   # list[float] signed residuals (target - point_pred)
#     feat_hist = []    # list[torch.Tensor] regime features (on device)

#     # -------- helpers --------
#     def conformal_q(abs_resids: torch.Tensor, alpha_: float) -> torch.Tensor:
#         """
#         Conformal quantile with 'higher' interpolation:
#           k = ceil((m+1)*(1-alpha)), q = k-th smallest abs residual
#         """
#         m = abs_resids.numel()
#         if m == 0:
#             return torch.tensor(0.0, device=abs_resids.device, dtype=abs_resids.dtype)
#         k = int(math.ceil((m + 1) * (1 - alpha_)))
#         k = min(max(k, 1), m)
#         return abs_resids.kthvalue(k).values

#     def knn_conditioned_residuals(
#         resid_window: torch.Tensor,   # [m]
#         feat_window: torch.Tensor,    # [m, d]
#         feat_t: torch.Tensor,         # [d]
#         k: int,
#     ) -> torch.Tensor:
#         m = resid_window.numel()
#         if m == 0:
#             return resid_window
#         k = min(k, m)
#         # robust L1 distance
#         d = (feat_window - feat_t.unsqueeze(0)).abs().sum(dim=1)  # [m]
#         nn_idx = torch.topk(-d, k=k).indices
#         return resid_window[nn_idx]

#     def sample_residuals_smoothed(
#         resid: torch.Tensor,         # [k]
#         num_samples: int,
#         lam: float,
#         noise_kind: str,
#         df: float,
#         bw_scale: float,
#         min_bw: float = 1e-6
#     ) -> torch.Tensor:
#         k = resid.numel()
#         if k == 0:
#             return resid.new_zeros((num_samples,))

#         # Recency weights assume resid ordered oldest->newest; if not, keep lam=0
#         if lam > 0:
#             ages = torch.arange(k-1, -1, -1, device=resid.device, dtype=torch.float32)  # 0 newest
#             w = torch.exp(-lam * ages)
#             w = w / w.sum()
#             bidx = torch.multinomial(w, num_samples=num_samples, replacement=True)
#         else:
#             bidx = torch.randint(low=0, high=k, size=(num_samples,), device=resid.device)

#         base = resid[bidx]

#         # robust bandwidth
#         # med = resid.median()
#         # mad = (resid - med).abs().median()
#         # robust_std = 1.4826 * mad
#         med, robust_std = ew_robust_scale(resid)
#         h = torch.clamp(bw_scale * robust_std, min=min_bw)

#         if noise_kind == "gaussian":
#             eps = torch.randn(num_samples, device=resid.device, dtype=resid.dtype) * h
#         else:
#             eps = torch.distributions.StudentT(df=df, loc=0.0, scale=h).sample((num_samples,))
#             eps = eps.to(device=resid.device, dtype=resid.dtype)

#         return base + eps * 1.3

#     def aggregate_samples(samples: torch.Tensor, how: str) -> torch.Tensor:
#         if how == "median":
#             return samples.median()
#         if how == "trimmed_mean":
#             srt, _ = torch.sort(samples)
#             ktrim = max(1, int(0.1 * samples.numel()))
#             return srt[ktrim:-ktrim].mean()
#         return samples.mean()
#     # -------------------------

#     for t in range(n):
#         # rolling window from the past
#         if len(resid_hist) == 0:
#             q = point_pred.new_tensor(0.0)
#             past_resid = None
#         else:
#             start = max(0, len(resid_hist) - window_size)
#             past_resid = torch.tensor(resid_hist[start:], device=device, dtype=point_pred.dtype)  # [m]
#             q = conformal_q(past_resid.abs(), alpha)

#         # print(f"past_resid (t={t}): mean {past_resid.mean().item() if past_resid is not None else 'N/A'}, std {past_resid.std().item() if past_resid is not None else 'N/A'}")
#         # input("debug")
            
#         # symmetric conformal interval around point forecast
#         lo = point_pred[t] - q
#         hi = point_pred[t] + q
#         ci_lower[t], ci_upper[t] = lo, hi

#         # regime feature at time t
#         x_last = x[t, -1, 0]
#         x_trend = x[t, -1, 0] - x[t, -2, 0] if x.shape[1] >= 2 else x_last.new_tensor(0.0)
#         feat_t = torch.stack([x_last, x_trend])  # [2]

#         # refinement
#         if past_resid is None or past_resid.numel() < 5 or mc_samples <= 0:
#             refined[t] = point_pred[t]
#         else:
#             past_feat = torch.stack(feat_hist[max(0, len(feat_hist) - window_size):], dim=0)\
#                             .to(device=device, dtype=point_pred.dtype)  # [m, 2]

#             # KNN conditioned subset
#             K = min(k_neighbors, past_resid.numel())
#             cond_resid = knn_conditioned_residuals(past_resid, past_feat, feat_t, k=K)  # [K]

#             # MAE-optimal bias (L1) is median residual of conditioned set
#             bias_l1 = cond_resid.median()
#             center = point_pred[t] + bias_l1

#             # sample around center: use centered residuals
#             cond_centered = cond_resid - bias_l1
#             r_samp = sample_residuals_smoothed(
#                 resid=cond_centered,
#                 num_samples=mc_samples,
#                 lam=lam_recency,            # keep 0 unless your conditioned residuals are time-ordered
#                 noise_kind=noise,
#                 df=studentt_df,
#                 bw_scale=bandwidth_scale,
#             )

#             samples = center + r_samp

#             if clamp_to_ci:
#                 samples = torch.clamp(samples, min=lo, max=hi)

#             refined[t] = aggregate_samples(samples, agg)

#         # update histories with observed residual from point forecast
#         r_t = (target[t] - point_pred[t]).detach()
#         resid_hist.append(r_t.item())
#         feat_hist.append(feat_t.detach())

#     final_pred = refined
#     error_rates = torch.abs(final_pred - target) / target
#     # print(f"final_pred[153]: {final_pred[153].item()}, target[153]: {target[153].item()}")
#     # print(f"error_rates[153]: {error_rates[153].item()}")
#     # print(f"actual error: {torch.abs(final_pred[153] - target[153]).item() / target[153].item()}")
#     # input("debug")
#     smape_rate = smape(target.detach().cpu().numpy(), final_pred.detach().cpu().numpy())

#     mean_abs_err = torch.mean(torch.abs(final_pred - target))
#     mean_model_var = model_var.mean()

#     calc_percentile_stats(
#         error_rates.detach().cpu().numpy(),
#         (torch.abs(final_pred - target) / target).sum().item() / len(final_pred),
#         "inference_results_uncertain"
#     )

#     s = (
#         f"[inference_conformal_regime_conditioned] "
#         f"mean_abs_err: {mean_abs_err.item():.6f}, "
#         f"mean_model_var: {mean_model_var.item():.6f}, "
#         f"smape_rate: {smape_rate}"
#     )
#     with open("inference_results_uncertain.log", "a") as f:
#         f.write(s + "\n")
#     print(s)

#     to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)

#     df = pd.DataFrame(
#         {
#             "idx": to_1d(idx_cpu),
#             "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
#             "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),
#             "point_pred": to_1d(point_pred),
#             "pred_refined": to_1d(final_pred),
#             "target": to_1d(target),
#             "ci_lower": to_1d(ci_lower),
#             "ci_upper": to_1d(ci_upper),
#             "error_rate_refined_pct": to_1d(error_rates) * 100,
#         }
#     )
#     df.to_csv("inference_results_uncertain.csv", index=False)

#     plot_like_example(df[:100])

#     return {
#         "point_pred": point_pred,
#         "pred_refined": final_pred,
#         "target": target,
#         "ci_lower": ci_lower,
#         "ci_upper": ci_upper,
#         "smape": smape_rate,
#         "mean_abs_err": mean_abs_err,
#         "mean_model_var": mean_model_var,
#         "df": df,
#     }

import pickle

# @torch.no_grad()
# def inference_conformal(
#     datasets: dict,
#     model: nn.Module,
#     mc_dropout: bool = False,
#     dropout_passes: int = 30,          # only used if mc_dropout=True
#     window_size: int = 2,             # rolling history size
#     k_neighbors: int = 2,             # KNN subset size for regime conditioning (<= window_size)
#     alpha: float = 0.1,               # 85% CI => alpha=0.15
#     mc_samples: int = 300,             # number of predictive samples
#     agg: str = "median",               # "median" (MAE-optimal), "mean", "trimmed_mean"
#     lam_recency: float = 0.1,          # recency decay inside selected set (keep 0 unless you keep time order)
#     noise: str = "gaussian",           # "studentt" or "gaussian"
#     studentt_df: float = 4,          # df for Student-t noise
#     bandwidth_scale: float = 0.1,      # 0.3–0.8 typical
#     clamp_to_ci: bool = True,          # keep refined samples within conformal CI
# ):
#     """
#     Rolling conformal intervals + regime-conditioned residual sampling for point refinement.

#     Regime features (default):
#       - x_last = x[t, -1, 0]
#       - x_trend = x[t, -1, 0] - x[t, -2, 0]  (0 if not available)

#     Refinement for MAE:
#       - bias_l1 = median(residuals | regime)
#       - samples = point_pred[t] + bias_l1 + (bootstrapped centered residual + noise)
#       - refined[t] = median(samples)  (MAE-optimal)
#     """

#     data_object = []
    
#     device = get_device()

#     # IMPORTANT: inference loader must be ordered (shuffle=False / sequential sampler)
#     valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

#     model.to(device)
#     model.eval()

#     if mc_dropout:
#         model = model.apply(dropout_on)
#     else:
#         model = model.apply(dropout_off)

#     # normalization params
#     json_data = read_json_params("train_invocation_rate_normalization.json")
#     train_mu, train_sigma = json_data["mu"], json_data["sigma"]
#     sigma, mu = train_sigma, train_mu

#     # one batch containing all data
#     (x, y, last_x, first_y, first_x, idx, hours) = next(iter(valid_loader))
#     x, y = x.to(device), y.to(device)
#     print(f"x shape: {x.shape}, y shape: {y.shape}")

#     # Ensure idx is 1D CPU tensor for sorting/validation
#     idx_cpu = idx.detach().cpu().view(-1)
#     if not torch.all(idx_cpu[:-1] <= idx_cpu[1:]):
#         sort_perm = torch.argsort(idx_cpu)
#         idx_cpu = idx_cpu[sort_perm]
#         x = x[sort_perm]
#         y = y[sort_perm]
#         last_x = last_x[sort_perm]
#         first_y = first_y[sort_perm]
#         first_x = first_x[sort_perm]

#     # model conditional input
#     cond = y[:, 0, 1:].to(device)

#     # point prediction (optionally average MC-dropout passes)
#     # if mc_dropout:
#     #     preds = []
#     #     for _ in range(dropout_passes):
#     #         preds.append(model((x, cond)))
#     #     res_stack = torch.stack(preds, dim=0)                 # [S, N, 1]
#     #     res = res_stack.mean(dim=0)                           # [N, 1]
#     #     model_var = res_stack.var(dim=0).squeeze(-1)          # [N]
#     # else:
#     res = model((x, cond))
#     model_var = torch.zeros(res.shape[0], device=device)

#     # denormalize
#     point_pred = (res.squeeze(-1) + x[:, -1, 0]) * train_sigma + train_mu
#     target = ((y[:, 0, 0] + x[:, -1, 0]) * train_sigma + train_mu).squeeze(-1)
    
#     point_pred = (res.squeeze(-1)) * train_sigma + train_mu
#     target = ((y[:, 0, 0]) * train_sigma + train_mu).squeeze(-1)


#     point_pred = point_pred.clamp(min=0.0)
#     # pickle.dump(point_pred.detach().cpu().numpy(), open("point_pred.pkl", "wb"))
    
#     # point_pred = res.squeeze(-1) * train_sigma + train_mu
#     # target = (y[:, 0, 0] * train_sigma + train_mu).squeeze(-1)
    
#     # point_pred = (res.squeeze(-1) + x[:, -1, 0]) 
#     # target = ((y[:, 0, 0] + x[:, -1, 0])).squeeze(-1)
    
#     # point_pred = (res.squeeze(-1))
#     # target = ((y[:, 0, 0])).squeeze(-1)
    
#     # for i in range(300, 350):
#     #     print(f"point_pred[{i}]: {point_pred[i].item()}, target[{i}]: {target[i].item()}")
#     # input("debug")

#     n = point_pred.shape[0]
#     ci_lower = torch.empty(n, device=device)
#     ci_upper = torch.empty(n, device=device)
#     refined = torch.empty(n, device=device)

#     # histories for conditioning
#     resid_hist = []   # list[float] signed residuals (target - point_pred)
#     feat_hist = []    # list[torch.Tensor] regime features (on device)

#     # -------- helpers --------
#     def conformal_q(abs_resids: torch.Tensor, alpha_: float) -> torch.Tensor:
#         """
#         Conformal quantile with 'higher' interpolation:
#           k = ceil((m+1)*(1-alpha)), q = k-th smallest abs residual
#         """
#         m = abs_resids.numel()
#         if m == 0:
#             return torch.tensor(0.0, device=abs_resids.device, dtype=abs_resids.dtype)
#         k = int(math.ceil((m + 1) * (1 - alpha_)))
#         k = min(max(k, 1), m)
#         return abs_resids.kthvalue(k).values

#     def knn_conditioned_residuals(
#         resid_window: torch.Tensor,   # [m]
#         feat_window: torch.Tensor,    # [m, d]
#         feat_t: torch.Tensor,         # [d]
#         k: int,
#     ) -> torch.Tensor:
#         m = resid_window.numel()
#         if m == 0:
#             return resid_window
#         k = min(k, m)
#         # robust L1 distance
#         d = (feat_window - feat_t.unsqueeze(0)).abs().sum(dim=1)  # [m]
#         nn_idx = torch.topk(-d, k=k).indices
#         # return resid_window[nn_idx]
#         return resid_window[k:]

#     def sample_residuals_smoothed(
#         resid: torch.Tensor,         # [k]
#         num_samples: int,
#         lam: float,
#         noise_kind: str,
#         df: float,
#         bw_scale: float,
#         min_bw: float = 1e-6
#     ) -> torch.Tensor:
#         k = resid.numel()
#         if k == 0:
#             return resid.new_zeros((num_samples,))

#         # Recency weights assume resid ordered oldest->newest; if not, keep lam=0
#         if lam > 0:
#             ages = torch.arange(k-1, -1, -1, device=resid.device, dtype=torch.float32)  # 0 newest
#             w = torch.exp(-lam * ages)
#             w = w / w.sum()
#             bidx = torch.multinomial(w, num_samples=num_samples, replacement=True)
#         else:
#             bidx = torch.randint(low=0, high=k, size=(num_samples,), device=resid.device)

#         base = resid[bidx]

#         # robust bandwidth
#         # med = resid.median()
#         # mad = (resid - med).abs().median()
#         # robust_std = 1.4826 * mad
#         med, robust_std = ew_robust_scale(resid)
#         h = torch.clamp(bw_scale * robust_std, min=min_bw)

#         if noise_kind == "gaussian":
#             eps = torch.randn(num_samples, device=resid.device, dtype=resid.dtype) * h
#         else:
#             eps = torch.distributions.StudentT(df=df, loc=0.0, scale=h).sample((num_samples,))
#             eps = eps.to(device=resid.device, dtype=resid.dtype)

#         return base + eps

#     def aggregate_samples(samples: torch.Tensor, how: str) -> torch.Tensor:
#         if how == "median":
#             return samples.median()
#         if how == "trimmed_mean":
#             srt, _ = torch.sort(samples)
#             ktrim = max(1, int(0.1 * samples.numel()))
#             return srt[ktrim:-ktrim].mean()
#         return samples.mean()
#     # -------------------------
    

#     pickle_path = "rate_data.pkl"
#     pickle_file = open(pickle_path, "wb")
    
#     for t in range(n):
#         # -------- rolling window from the past (signed residuals) --------
#         if len(resid_hist) < 5 or mc_samples <= 0:
#             past_resid = None
#         else:
#             start = max(0, len(resid_hist) - window_size)
#             past_resid = torch.tensor(
#                 resid_hist[start:], device=device, dtype=point_pred.dtype
#             )  # [m]

#         # -------- regime feature at time t --------
#         x_last = x[t, -1, 0]
#         x_trend = x[t, -1, 0] - x[t, -2, 0] if x.shape[1] >= 2 else x_last.new_tensor(0.0)
#         feat_t = torch.stack([x_last, x_trend])  # [2]

#         # -------- build predictive samples via residual bootstrap --------
#         if past_resid is None:
#             # warm start: no history -> degenerate distribution
#             samples = point_pred[t].repeat(1)  # [1]
#         else:
#             past_feat = torch.stack(
#                 feat_hist[max(0, len(feat_hist) - window_size):], dim=0
#             ).to(device=device, dtype=point_pred.dtype)  # [m, 2]

#             # KNN conditioned residuals
#             K = min(k_neighbors, past_resid.numel())
#             cond_resid = knn_conditioned_residuals(past_resid, past_feat, feat_t, k=K)  # [K]
#             cond_resid = torch.tensor(resid_hist[-window_size:], device=device, dtype=point_pred.dtype)
#             # print(f"resid_hist: {resid_hist}")
#             # input("debug")

#             # center choice:
#             # - If you want the bootstrap to inherit conditional bias, keep bias_l1
#             # - If you want "pure" bootstrap around point_pred, set bias_l1=0
#             bias_l1 = cond_resid.median()
#             center = point_pred[t] + bias_l1 * 1.

#             # bootstrap residuals around the conditional center
#             cond_centered = cond_resid - bias_l1  # centered residuals

#             # draw residual samples (bootstrap + optional smoothing noise)
#             r_samp = sample_residuals_smoothed(
#                 resid=cond_centered,
#                 num_samples=mc_samples,
#                 lam=lam_recency,
#                 noise_kind=noise,
#                 df=studentt_df,
#                 bw_scale=bandwidth_scale,
#             )  # [mc_samples]
#             # print(f"r_samp stats (t={t}): mean {r_samp.mean().item()}, std {r_samp.std().item()}")        
            
#             samples = center + r_samp  # [mc_samples]
            
#             # print(f"t={t}, hour={hours[t].item()}")
#             # input("debug")

#         # -------- quantile interval from samples (asymmetric) --------
#         # Note: torch.quantile expects a Tensor input; ensure samples is [S]
#         lo = torch.quantile(samples, alpha / 2, interpolation="linear")
#         hi = torch.quantile(samples, 1 - alpha / 2, interpolation="linear")

#         # safety: enforce ordering
#         lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)

#         ci_lower[t], ci_upper[t] = lo, hi

#         # -------- point refinement from the SAME samples --------
#         # For MAE-optimal center, median is a good default.
#         refined[t] = aggregate_samples(samples, agg)

#         # -------- update histories with observed residual from POINT forecast --------
#         r_t = (target[t] - point_pred[t]).detach()
#         resid_hist.append(r_t.item())
#         feat_hist.append(feat_t.detach())
        
#         if past_resid is not None:
#             data_object.append({
#                 "hour": hours[t].item(),
#                 "mc_samples": samples.detach().cpu().numpy(),
#                 "original_prediction": point_pred[t].item(),
#                 "target": target[t].item(),
#                 "refined_prediction": refined[t].item(),
#             })

#     pickle.dump(
#         data_object,
#         pickle_file,
#         protocol=pickle.HIGHEST_PROTOCOL
#     )
#     pickle_file.close()
#     final_pred = refined
#     final_pred = final_pred.clamp(min=0.0)

#     error_rates = torch.abs(final_pred - target) / target
#     # print(f"final_pred[153]: {final_pred[153].item()}, target[153]: {target[153].item()}")
#     # print(f"error_rates[153]: {error_rates[153].item()}")
#     # print(f"actual error: {torch.abs(final_pred[153] - target[153]).item() / target[153].item()}")
#     # input("debug")
#     smape_rate = smape(target.detach().cpu().numpy(), final_pred.detach().cpu().numpy())

#     mean_abs_err = torch.mean(torch.abs(final_pred - target))
#     mean_model_var = model_var.mean()
    
#     print(f"max target: {target.max().item()}, min target: {target.min().item()}")

#     calc_percentile_stats(
#         error_rates.detach().cpu().numpy(),
#         (torch.abs(final_pred - target) / target).sum().item() / len(final_pred),
#         "inference_results_uncertain"
#     )

#     s = (
#         f"[inference_conformal_regime_conditioned] "
#         f"mean_abs_err: {mean_abs_err.item():.6f}, "
#         f"mean_model_var: {mean_model_var.item():.6f}, "
#         f"smape_rate: {smape_rate}"
#     )
#     with open("inference_results_uncertain.log", "a") as f:
#         f.write(s + "\n")
#     print(s)

#     to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)

#     df = pd.DataFrame(
#         {
#             "idx": to_1d(idx_cpu),
#             "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
#             "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),
#             # "x_last_hour": to_1d(x[:, -1, 0]),
#             # "x_start_hour": to_1d(x[:, 0, 0]),
#             "point_pred": to_1d(point_pred),
#             "pred_refined": to_1d(final_pred),
#             "target": to_1d(target),
#             "ci_lower": to_1d(ci_lower),
#             "ci_upper": to_1d(ci_upper),
#             "error_rate_refined_pct": to_1d(error_rates) * 100,
#         }
#     )
#     df.to_csv("inference_results_uncertain.csv", index=False)

#     plot_like_example(df)

#     return {
#         "point_pred": point_pred,
#         "pred_refined": final_pred,
#         "target": target,
#         "ci_lower": ci_lower,
#         "ci_upper": ci_upper,
#         "smape": smape_rate,
#         "mean_abs_err": mean_abs_err,
#         "mean_model_var": mean_model_var,
#         "df": df,
#     }



import math
import pickle
from typing import Dict, Optional

import pandas as pd
import torch
from torch import nn

# horizontal
# -----------------------------
# Main function
# -----------------------------
@torch.no_grad()
def inference_conformal_horizontal(
    datasets: dict,
    model: nn.Module,
    # NOTE: window_size here is now used for "conditioning residual set"
    window_size: int = 2,
    k_neighbors: int = 2,
    alpha: float = 0.1,                # miscoverage; 90% CI => alpha=0.1
    mc_samples: int = 300,
    agg: str = "median",               # "median", "mean", "trimmed_mean"
    lam_recency: float = 0.0,          # recency decay weights inside conditioning window
    noise: str = "gaussian",           # "studentt" or "gaussian"
    studentt_df: float = 4.0,
    bandwidth_scale: float = 0.3,      # typical 0.3–0.8
    clamp_to_ci: bool = True,
    # NEW: decouple bias tracking from uncertainty estimation
    bias_window: int = 1,              # keep 1 if that gives best point accuracy
    scale_window: int = 72,            # longer history for uncertainty (e.g., 72 hours)
    min_history: int = 8,              # minimum history before generating non-degenerate samples
    use_knn: bool = True,              # actually use regime conditioning (was disabled in your code)
    min_bw: float = 1e-6,
    # Optional: if your model output is a delta relative to last x, set use_delta=True
    use_delta: bool = False,
):
    """
    Rolling conformal interval + regime-conditioned residual sampling.

    Key fix vs your original:
      - Keep bias_window=1 for best point tracking.
      - Estimate uncertainty scale from scale_window (or longer) so window_size=1 does NOT collapse variance.
      - Fix KNN conditioning (was overwritten/disabled).
      - Use conformal symmetric CI (center ± q_hat) and optionally clamp samples to it.
    """

    # -----------------------------
    # External project hooks you already have in your repo:
    #   - get_device()
    #   - data.get_dataloaders(...)
    #   - read_json_params(...)
    #   - smape(...)
    #   - calc_percentile_stats(...)
    #   - plot_like_example(...)
    # -----------------------------
    device = get_device()

    # IMPORTANT: inference loader must be ordered (shuffle=False / sequential sampler)
    valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

    model.to(device)
    model.eval()
    
    model = model.apply(dropout_off)

    # normalization params
    json_data = read_json_params("train_invocation_rate_normalization.json")
    train_mu, train_sigma = json_data["mu"], json_data["sigma"]

    # one batch containing all data
    (x, y, last_x, first_y, first_x, idx, hours) = next(iter(valid_loader))
    x, y = x.to(device), y.to(device)

    # Ensure idx is 1D CPU tensor for sorting/validation
    idx_cpu = idx.detach().cpu().view(-1)
    if not torch.all(idx_cpu[:-1] <= idx_cpu[1:]):
        sort_perm = torch.argsort(idx_cpu)
        idx_cpu = idx_cpu[sort_perm]
        x = x[sort_perm]
        y = y[sort_perm]
        last_x = last_x[sort_perm]
        first_y = first_y[sort_perm]
        first_x = first_x[sort_perm]
        hours = hours[sort_perm]

    # model conditional input
    cond = y[:, 0, 1:].to(device)

    # -----------------------------
    # Point prediction
    # -----------------------------
    res = model((x, cond))
    model_var = torch.zeros(res.shape[0], device=device)

    # Denormalize
    # model predicts absolute normalized target
    point_pred = (res.squeeze(-1)) * train_sigma + train_mu
    target = ((y[:, 0, 0]) * train_sigma + train_mu).squeeze(-1)

    point_pred = point_pred.clamp(min=0.0)

    n = point_pred.shape[0]
    ci_lower = torch.empty(n, device=device)
    ci_upper = torch.empty(n, device=device)
    refined = torch.empty(n, device=device)

    # histories for conditioning
    resid_hist = []   # list[float] signed residuals (target - point_pred)
    feat_hist = []    # list[torch.Tensor] regime features
    
    centers = []

    # -----------------------------
    # Helpers
    # -----------------------------
    def conformal_q(abs_resids: torch.Tensor, alpha_: float) -> torch.Tensor:
        """
        Conformal quantile with 'higher' interpolation:
          k = ceil((m+1)*(1-alpha)), q = k-th smallest abs residual
        """
        m = abs_resids.numel()
        if m == 0:
            return torch.tensor(0.0, device=abs_resids.device, dtype=abs_resids.dtype)
        k = int(math.ceil((m + 1) * (1 - alpha_)))
        k = min(max(k, 1), m)
        return abs_resids.kthvalue(k).values

    def build_regime_feature(t: int) -> torch.Tensor:
        # Default regime feature: last value + short trend
        x_last = x[t, -1, 0]
        x_trend = x[t, -1, 0] - x[t, -2, 0] if x.shape[1] >= 2 else x_last.new_tensor(0.0)
        return torch.stack([x_last, x_trend])  # [2]

    def knn_conditioned_residuals(
        resid_window: torch.Tensor,   # [m]
        feat_window: torch.Tensor,    # [m, d]
        feat_t: torch.Tensor,         # [d]
        k: int,
    ) -> torch.Tensor:
        """
        Return the k nearest residuals in regime-feature space.
        """
        m = resid_window.numel()
        if m == 0:
            return resid_window
        k = min(k, m)
        d = (feat_window - feat_t.unsqueeze(0)).abs().sum(dim=1)  # [m], L1 distance
        nn_idx = torch.topk(-d, k=k).indices
        return resid_window[nn_idx]

    def sample_deviations(
        deviations: torch.Tensor,     # [k], typically centered residuals
        num_samples: int,
        lam: float,
        noise_kind: str,
        df: float,
        scale_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bootstrap deviations (possibly degenerate) + add smooth noise with externally estimated scale_std.
        This is the key to avoid collapse when window_size=1.
        """
        k = deviations.numel()
        if k == 0 or num_samples <= 0:
            return deviations.new_zeros((max(1, num_samples),))

        # Recency weights assume deviations ordered oldest->newest
        if lam > 0 and k > 1:
            ages = torch.arange(k - 1, -1, -1, device=deviations.device, dtype=torch.float32)  # 0 newest
            w = torch.exp(-lam * ages)
            w = w / w.sum()
            bidx = torch.multinomial(w, num_samples=num_samples, replacement=True)
        else:
            bidx = torch.randint(low=0, high=k, size=(num_samples,), device=deviations.device)

        base = deviations[bidx]

        # bandwidth derived from longer history scale
        h = torch.clamp(bandwidth_scale * scale_std, min=min_bw)

        if noise_kind == "gaussian":
            eps = torch.randn(num_samples, device=deviations.device, dtype=deviations.dtype) * h
        else:
            eps = torch.distributions.StudentT(df=df, loc=0.0, scale=h).sample((num_samples,))
            eps = eps.to(device=deviations.device, dtype=deviations.dtype)

        return base + eps

    def aggregate_samples(samples: torch.Tensor, how: str) -> torch.Tensor:
        if how == "median":
            return samples.median()
        if how == "trimmed_mean":
            srt, _ = torch.sort(samples)
            ktrim = max(1, int(0.1 * samples.numel()))
            if samples.numel() <= 2 * ktrim:
                return samples.mean()
            return srt[ktrim:-ktrim].mean()
        return samples.mean()

    # -----------------------------
    # store samples for later analysis
    # -----------------------------
    data_object = []

    pickle_path = "rate_data.pkl"
    with open(pickle_path, "wb") as pickle_file:

        # -----------------------------
        # Main loop
        # -----------------------------
        for t in range(n):
            feat_t = build_regime_feature(t)
            feat_hist.append(feat_t.detach())

            # Need history to do anything non-degenerate
            if len(resid_hist) < min_history or mc_samples <= 0:
                center = point_pred[t]
                samples = center.repeat(1)  # degenerate
                # CI also degenerate
                ci_lower[t] = center
                ci_upper[t] = center
                refined[t] = center
            else:
                # -----------------------------
                # Build residual tensors
                # -----------------------------
                resid_all = torch.tensor(resid_hist, device=device, dtype=point_pred.dtype)

                # Bias estimate from short window (keep this small; bias_window=1 if best)
                bw = min(bias_window, resid_all.numel())
                bias_set = resid_all[-bw:]
                bias_hat = bias_set.median()  # robust
                center = point_pred[t] #+ bias_hat
                centers.append(center.item())
                
                # Uncertainty scale from longer window (prevents collapse at window_size=1)
                sw = min(scale_window, resid_all.numel())
                scale_set = resid_all[-sw:]
                # Remove slow location for scale; robust
                loc_s, scale_std = robust_median_mad(scale_set)

                # Conditioning residual set (regime-conditioned if enabled)
                cw = min(window_size, resid_all.numel())
                cond_resid_full = resid_all[-cw:]  # oldest->newest within the window
                if use_knn and cw >= 2:
                    past_feat = torch.stack(feat_hist[-cw:], dim=0).to(device=device, dtype=point_pred.dtype)
                    K = min(k_neighbors, cw)
                    cond_resid = knn_conditioned_residuals(cond_resid_full, past_feat, feat_t, k=K)
                else:
                    cond_resid = cond_resid_full

                # Deviations around 0 (do NOT re-introduce bias here)
                # If cond_resid has one element, deviations will be 0; sampling still gets variance via scale_std.
                cond_loc = cond_resid.median()
                deviations = cond_resid - cond_loc

                # Monte-Carlo deviations + smooth noise using longer-horizon scale_std
                dev_samp = sample_deviations(
                    deviations=deviations,
                    num_samples=mc_samples,
                    lam=lam_recency,
                    noise_kind=noise,
                    df=studentt_df,
                    scale_std=scale_std,
                )  # [mc_samples]

                samples = center + dev_samp 
                # samples = center + torch.zeros_like(dev_samp)
                samples = samples.clamp(min=0.0)

                # -----------------------------
                # Conformal CI from absolute residuals (symmetric)
                # -----------------------------
                abs_resids = (scale_set - loc_s).abs()
                qhat = conformal_q(abs_resids, alpha_=alpha)

                lo = (center - qhat).clamp(min=0.0)
                hi = (center + qhat).clamp(min=0.0)
                lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
                ci_lower[t], ci_upper[t] = lo, hi

                # Optionally clamp samples to CI (stabilizes tails)
                if clamp_to_ci:
                    samples = torch.clamp(samples, min=lo, max=hi)

                # Point refinement from samples
                refined_t = aggregate_samples(samples, agg)
                if clamp_to_ci:
                    refined_t = torch.clamp(refined_t, min=lo, max=hi)
                refined[t] = refined_t

                # Save per-step object for later analysis
                data_object.append({
                    "hour": float(hours[t].item()),
                    "mc_samples": samples.detach().cpu().numpy(),
                    "original_prediction": float(point_pred[t].item()),
                    "target": float(target[t].item()),
                    "refined_prediction": float(refined[t].item()),
                    "ci_lower": float(ci_lower[t].item()),
                    "ci_upper": float(ci_upper[t].item()),
                    "bias_hat": float(bias_hat.item()),
                    "scale_std": float(scale_std.item()),
                })

            # -----------------------------
            # Update histories with observed residual from POINT forecast
            # -----------------------------
            r_t = (target[t] - point_pred[t]).detach()
            resid_hist.append(r_t.item())

        pickle.dump(data_object, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    final_pred = refined.clamp(min=0.0)

    # -----------------------------
    # Metrics / outputs (same as your original structure)
    # -----------------------------
    error_rates = torch.abs(final_pred - target) / torch.clamp(target, min=1e-9)

    smape_rate = smape(target.detach().cpu().numpy(), final_pred.detach().cpu().numpy())
    mean_abs_err = torch.mean(torch.abs(final_pred - target))
    mean_model_var = model_var.mean()

    print(f"max target: {target.max().item()}, min target: {target.min().item()}")
    calc_percentile_stats(
        error_rates.detach().cpu().numpy(),
        (torch.abs(final_pred - target) / torch.clamp(target, min=1e-9)).mean().item(),
        "inference_results_uncertain",
    )

    s = (
        f"[inference_conformal_decoupled] "
        f"mean_abs_err: {mean_abs_err.item():.6f}, "
        f"mean_model_var: {mean_model_var.item():.6f}, "
        f"smape_rate: {smape_rate}"
    )
    with open("inference_results_uncertain.log", "a") as f:
        f.write(s + "\n")
    print(s)

    to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)

    df = pd.DataFrame(
        {
            "idx": to_1d(idx_cpu),
            "x_last_hour": to_1d(x[:, -1, 0] * train_sigma + train_mu),
            "x_start_hour": to_1d(x[:, 0, 0] * train_sigma + train_mu),
            "point_pred": to_1d(point_pred),
            "pred_refined": to_1d(final_pred),
            "target": to_1d(target),
            "ci_lower": to_1d(ci_lower),
            "ci_upper": to_1d(ci_upper),
            "error_rate_refined_pct": to_1d(error_rates) * 100,
        }
    )
    df.to_csv("inference_results_uncertain.csv", index=False)
    
    print(f"=" * 80)
    residuals = torch.tensor(centers, dtype=target.dtype, device=target.device) - target[len(target) - len(centers):]
    print(f"residual means: {torch.mean(residuals).item()}")
    exit()
    
    print("data itself's stats:")
    std_y = target.std().item()
    print(f"target std: {std_y}")
    variance_y = target.var().item()
    print(f"target variance: {variance_y}")
    print(f"=" * 80)
    print("point of prediction vs target stats:")
    std_r = (point_pred - target).std().item()
    print(f"residual std: {std_r}")
    variance_r = (point_pred - target).var().item()
    print(f"residual variance: {variance_r}")
    print(f"=" * 80)
    print("residual mean:")
    mean_r = (point_pred - target).mean().item()
    print(f"residual mean: {mean_r}")
    print(f"scale_window={scale_window}, bias_window={bias_window}, cond_window={window_size} used for uncertainty estimation.")
    print("refined prediction vs target stats:")
    std_r2 = (final_pred - target).std().item()
    print(f"residual std: {std_r2}")
    variance_r2 = (final_pred - target).var().item()
    print(f"residual variance: {variance_r2}")
    print(f"=" * 80)
    
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # residuals: 1D array-like

    result = acorr_ljungbox((point_pred - target).detach().cpu().numpy(), lags=[10], return_df=True)
    print(result)

    plot_like_example(df)
    

    return df, data_object

from collections import deque

# vertical
@torch.no_grad()
def inference_conformal_vertical(
    datasets: dict,
    model: nn.Module,
    # NOTE: window_size here is now used for "conditioning residual set"
    window_size: int = 2,
    k_neighbors: int = 2,
    alpha: float = 0.1,                # miscoverage; 90% CI => alpha=0.1
    mc_samples: int = 300,
    agg: str = "median",               # "median", "mean", "trimmed_mean"
    lam_recency: float = 0.0,          # recency decay weights inside conditioning window
    noise: str = "gaussian",           # "studentt" or "gaussian"
    studentt_df: float = 4.0,
    bandwidth_scale: float = 0.3,      # typical 0.3–0.8
    clamp_to_ci: bool = True,
    # NEW: decouple bias tracking from uncertainty estimation
    bias_window: int = 15,             
    scale_window: int = 72,            # longer history for uncertainty (e.g., 72 hours)
    min_history: int = 8,              # minimum history before generating non-degenerate samples
    use_knn: bool = True,              # actually use regime conditioning (was disabled in your code)
    min_bw: float = 1e-6,
    # Optional: if your model output is a delta relative to last x, set use_delta=True
    use_delta: bool = False,
):
    """
    Rolling conformal interval + regime-conditioned residual sampling.

    Key fix vs your original:
      - Keep bias_window=1 for best point tracking.
      - Estimate uncertainty scale from scale_window (or longer) so window_size=1 does NOT collapse variance.
      - Fix KNN conditioning (was overwritten/disabled).
      - Use conformal symmetric CI (center ± q_hat) and optionally clamp samples to it.
    """

    # -----------------------------
    # External project hooks you already have in your repo:
    #   - get_device()
    #   - data.get_dataloaders(...)
    #   - read_json_params(...)
    #   - smape(...)
    #   - calc_percentile_stats(...)
    #   - plot_like_example(...)
    # -----------------------------
    device = get_device()

    # IMPORTANT: inference loader must be ordered (shuffle=False / sequential sampler)
    valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

    model.to(device)
    model.eval()
    
    model = model.apply(dropout_off)

    # normalization params
    json_data = read_json_params("train_invocation_rate_normalization.json")
    train_mu, train_sigma = json_data["mu"], json_data["sigma"]

    # one batch containing all data
    (x, y, last_x, first_y, first_x, idx, hours) = next(iter(valid_loader))
    x, y = x.to(device), y.to(device)

    # Ensure idx is 1D CPU tensor for sorting/validation
    idx_cpu = idx.detach().cpu().view(-1)
    if not torch.all(idx_cpu[:-1] <= idx_cpu[1:]):
        sort_perm = torch.argsort(idx_cpu)
        idx_cpu = idx_cpu[sort_perm]
        x = x[sort_perm]
        y = y[sort_perm]
        last_x = last_x[sort_perm]
        first_y = first_y[sort_perm]
        first_x = first_x[sort_perm]
        hours = hours[sort_perm]

    # model conditional input
    cond = y[:, 0, 1:].to(device)

    # -----------------------------
    # Point prediction
    # -----------------------------
    res = model((x, cond))
    model_var = torch.zeros(res.shape[0], device=device)

    # Denormalize
    # model predicts absolute normalized target
    point_pred = (res.squeeze(-1)) * train_sigma + train_mu
    target = ((y[:, 0, 0]) * train_sigma + train_mu).squeeze(-1)

    point_pred = point_pred.clamp(min=0.0)

    n = point_pred.shape[0]
    ci_lower = torch.empty(n, device=device)
    ci_upper = torch.empty(n, device=device)
    refined = torch.empty(n, device=device)

    # histories for conditioning
    resid_hist = []   # list[float] signed residuals (target - point_pred)
    feat_hist = []    # list[torch.Tensor] regime features
    bias_bufs = {h: deque(maxlen=bias_window) for h in range(24)}
    # Per-hour residual buffers for sampling/scale (use longer window than bias)
    # size = scale_window (or you can choose a separate param if you want)
    resid_bufs = {h: deque(maxlen=scale_window) for h in range(24)}
    feat_bufs  = {h: deque(maxlen=scale_window) for h in range(24)}  # optional but needed if you want KNN within-hour
    centers = []

    # -----------------------------
    # Helpers
    # -----------------------------
    def conformal_q(abs_resids: torch.Tensor, alpha_: float) -> torch.Tensor:
        """
        Conformal quantile with 'higher' interpolation:
          k = ceil((m+1)*(1-alpha)), q = k-th smallest abs residual
        """
        m = abs_resids.numel()
        if m == 0:
            return torch.tensor(0.0, device=abs_resids.device, dtype=abs_resids.dtype)
        k = int(math.ceil((m + 1) * (1 - alpha_)))
        k = min(max(k, 1), m)
        return abs_resids.kthvalue(k).values

    def build_regime_feature(t: int) -> torch.Tensor:
        # Default regime feature: last value + short trend
        x_last = x[t, -1, 0]
        x_trend = x[t, -1, 0] - x[t, -2, 0] if x.shape[1] >= 2 else x_last.new_tensor(0.0)
        return torch.stack([x_last, x_trend])  # [2]

    def knn_conditioned_residuals(
        resid_window: torch.Tensor,   # [m]
        feat_window: torch.Tensor,    # [m, d]
        feat_t: torch.Tensor,         # [d]
        k: int,
    ) -> torch.Tensor:
        """
        Return the k nearest residuals in regime-feature space.
        """
        m = resid_window.numel()
        if m == 0:
            return resid_window
        k = min(k, m)
        d = (feat_window - feat_t.unsqueeze(0)).abs().sum(dim=1)  # [m], L1 distance
        nn_idx = torch.topk(-d, k=k).indices
        return resid_window[nn_idx]

    def sample_deviations(
        deviations: torch.Tensor,     # [k], typically centered residuals
        num_samples: int,
        lam: float,
        noise_kind: str,
        df: float,
        scale_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bootstrap deviations (possibly degenerate) + add smooth noise with externally estimated scale_std.
        This is the key to avoid collapse when window_size=1.
        """
        k = deviations.numel()
        if k == 0 or num_samples <= 0:
            return deviations.new_zeros((max(1, num_samples),))

        # Recency weights assume deviations ordered oldest->newest
        if lam > 0 and k > 1:
            ages = torch.arange(k - 1, -1, -1, device=deviations.device, dtype=torch.float32)  # 0 newest
            w = torch.exp(-lam * ages)
            w = w / w.sum()
            bidx = torch.multinomial(w, num_samples=num_samples, replacement=True)
        else:
            bidx = torch.randint(low=0, high=k, size=(num_samples,), device=deviations.device)

        base = deviations[bidx]

        # bandwidth derived from longer history scale
        h = torch.clamp(bandwidth_scale * scale_std, min=min_bw)

        if noise_kind == "gaussian":
            eps = torch.randn(num_samples, device=deviations.device, dtype=deviations.dtype) * h
        else:
            eps = torch.distributions.StudentT(df=df, loc=0.0, scale=h).sample((num_samples,))
            eps = eps.to(device=deviations.device, dtype=deviations.dtype)

        return base + eps

    def aggregate_samples(samples: torch.Tensor, how: str) -> torch.Tensor:
        if how == "median":
            return samples.median()
        if how == "trimmed_mean":
            srt, _ = torch.sort(samples)
            ktrim = max(1, int(0.1 * samples.numel()))
            if samples.numel() <= 2 * ktrim:
                return samples.mean()
            return srt[ktrim:-ktrim].mean()
        return samples.mean()

    # -----------------------------
    # store samples for later analysis
    # -----------------------------
    data_object = []
    dev_samps = []
    

    pickle_path = "rate_data.pkl"
    with open(pickle_path, "wb") as pickle_file:

        # -----------------------------
        # Main loop
        # -----------------------------
        for t in range(n):
            # feat_t = build_regime_feature(t)
            # feat_hist.append(feat_t.detach())
            
            hour_bucket = int(hours[t].item()) % 24  # <-- compute once

            feat_t = build_regime_feature(t)
            feat_hist.append(feat_t.detach())        # keep global if you still want it


            # Need history to do anything non-degenerate
            if len(resid_bufs[hour_bucket]) < min_history or mc_samples <= 0:
                center = point_pred[t]
                samples = center.repeat(1)  # degenerate
                # CI also degenerate
                ci_lower[t] = center
                ci_upper[t] = center
                refined[t] = center
            else:
                # -----------------------------
                # Build residual tensors
                # -----------------------------
                # Build residual tensors from THIS hour-of-day bucket only
                resid_hour = torch.tensor(list(resid_bufs[hour_bucket]), device=device, dtype=point_pred.dtype)

                # Bias estimate from short window (keep this small; bias_window=1 if best)
                # --- Hour-of-day specific bias (24 rolling windows) ---
                # Map timestamp to hour-of-day bucket.
                # If `hours[t]` is already 0..23 you're good; if it's a running hour index,
                # modulo makes it periodic by day.
                hour_bucket = int(hours[t].item()) % 24

                buf = bias_bufs[hour_bucket]
                if len(buf) > 0:
                    bias_set = torch.tensor(list(buf), device=device, dtype=point_pred.dtype)
                    bias_hat = bias_set.median()  # robust per-hour rolling median
                    bias_hat = torch.tensor(resid_hist[-1:], device=device, dtype=point_pred.dtype)
                else:
                    bias_hat = point_pred.new_tensor(0.0)

                center = point_pred[t] + bias_hat
                centers.append(center.item())

                # Uncertainty scale from longer window within-hour
                sw = min(scale_window, resid_hour.numel())
                scale_set = resid_hour[-sw:]
                loc_s, scale_std = robust_median_mad(scale_set)

                # Conditioning residual set within-hour (regime-conditioned if enabled)
                cw = min(window_size, resid_hour.numel())
                cond_resid_full = resid_hour[-cw:]  # oldest->newest within the within-hour window

                if use_knn and cw >= 2:
                    # KNN within the same hour bucket
                    past_feat = torch.stack(list(feat_bufs[hour_bucket])[-cw:], dim=0).to(device=device, dtype=point_pred.dtype)
                    K = min(k_neighbors, cw)
                    cond_resid = knn_conditioned_residuals(cond_resid_full, past_feat, feat_t, k=K)
                else:
                    cond_resid = cond_resid_full

                # Deviations around 0 (do NOT re-introduce bias here)
                # If cond_resid has one element, deviations will be 0; sampling still gets variance via scale_std.
                cond_loc = cond_resid.median()
                deviations = cond_resid - cond_loc

                # Monte-Carlo deviations + smooth noise using longer-horizon scale_std
                dev_samp = sample_deviations(
                    deviations=deviations,
                    num_samples=mc_samples,
                    lam=lam_recency,
                    noise_kind=noise,
                    df=studentt_df,
                    scale_std=scale_std,
                )  # [mc_samples]
                
                dev_samps.append(dev_samp.cpu().numpy())

                samples = center + dev_samp 
                # samples = center + torch.zeros_like(dev_samp)
                samples = samples.clamp(min=0.0)

                # -----------------------------
                # Conformal CI from absolute residuals (symmetric)
                # -----------------------------
                abs_resids = (scale_set - loc_s).abs()
                qhat = conformal_q(abs_resids, alpha_=alpha)

                lo = (center - qhat).clamp(min=0.0)
                hi = (center + qhat).clamp(min=0.0)
                lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
                ci_lower[t], ci_upper[t] = lo, hi

                # Optionally clamp samples to CI (stabilizes tails)
                if clamp_to_ci:
                    samples = torch.clamp(samples, min=lo, max=hi)

                # Point refinement from samples
                refined_t = aggregate_samples(samples, agg)
                if clamp_to_ci:
                    refined_t = torch.clamp(refined_t, min=lo, max=hi)
                refined[t] = refined_t

                # Save per-step object for later analysis
                data_object.append({
                    "hour": float(hours[t].item()),
                    "mc_samples": samples.detach().cpu().numpy(),
                    "original_prediction": float(point_pred[t].item()),
                    "target": float(target[t].item()),
                    "refined_prediction": float(refined[t].item()),
                    "ci_lower": float(ci_lower[t].item()),
                    "ci_upper": float(ci_upper[t].item()),
                    "bias_hat": float(bias_hat.item()),
                    "scale_std": float(scale_std.item()),
                })

            # -----------------------------
            # Update histories with observed residual from POINT forecast
            # -----------------------------
            r_t = (target[t] - point_pred[t]).detach()
            r_item = float(r_t.item())

            # keep global history if you still want it for logging/other uses
            resid_hist.append(r_item)

            # update hour-of-day buffers
            bias_bufs[hour_bucket].append(r_item)     # size = 15 (bias)
            resid_bufs[hour_bucket].append(r_item)    # size = scale_window (sampling/scale)
            feat_bufs[hour_bucket].append(feat_t.detach().cpu())  # store feature aligned to residual (on CPU to save GPU mem)


        pickle.dump(data_object, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    final_pred = refined.clamp(min=0.0)

    # -----------------------------
    # Metrics / outputs (same as your original structure)
    # -----------------------------
    error_rates = torch.abs(final_pred - target) / torch.clamp(target, min=1e-9)

    smape_rate = smape(target.detach().cpu().numpy(), final_pred.detach().cpu().numpy())
    mean_abs_err = torch.mean(torch.abs(final_pred - target))
    mean_model_var = model_var.mean()

    print(f"max target: {target.max().item()}, min target: {target.min().item()}")
    calc_percentile_stats(
        error_rates.detach().cpu().numpy(),
        (torch.abs(final_pred - target) / torch.clamp(target, min=1e-9)).mean().item(),
        "inference_results_uncertain",
    )

    s = (
        f"[inference_conformal_decoupled] "
        f"mean_abs_err: {mean_abs_err.item():.6f}, "
        f"mean_model_var: {mean_model_var.item():.6f}, "
        f"smape_rate: {smape_rate}"
    )
    with open("inference_results_uncertain.log", "a") as f:
        f.write(s + "\n")
    print(s)

    to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)

    df = pd.DataFrame(
        {
            "idx": to_1d(idx_cpu),
            "x_last_hour": to_1d(x[:, -1, 0] * train_sigma + train_mu),
            "x_start_hour": to_1d(x[:, 0, 0] * train_sigma + train_mu),
            "point_pred": to_1d(point_pred),
            "pred_refined": to_1d(final_pred),
            "target": to_1d(target),
            "ci_lower": to_1d(ci_lower),
            "ci_upper": to_1d(ci_upper),
            "error_rate_refined_pct": to_1d(error_rates) * 100,
        }
    )
    df.to_csv("inference_results_uncertain.csv", index=False)
    
    # print(dev_samps)
    # input("debug")
       
    print(f"=" * 80)
    print(f"center residual means: {np.mean(np.array(centers) - target.detach().cpu().numpy()[len(target) - len(centers):])}")
    result = acorr_ljungbox((centers - target.detach().cpu().numpy()[len(target) - len(centers):]), lags=[1, 2, 10], return_df=True)
    print(result)
    # input("debug")
    
    print(f"=" * 80)
    print("data itself's stats:")
    std_y = target.std().item()
    print(f"target std: {std_y}")
    variance_y = target.var().item()
    print(f"target variance: {variance_y}")
    print(f"=" * 80)
    print("point of prediction vs target stats:")
    std_r = (point_pred - target).std().item()
    print(f"residual std: {std_r}")
    variance_r = (point_pred - target).var().item()
    print(f"residual variance: {variance_r}")
    print(f"=" * 80)
    print("residual mean:")
    mean_r = (point_pred - target).mean().item()
    print(f"residual mean: {mean_r}")
    print(f"scale_window={scale_window}, bias_window={bias_window}, cond_window={window_size} used for uncertainty estimation.")
    print("refined prediction vs target stats:")
    print(f"refined residual mean: {(final_pred - target).mean().item()}")
    std_r2 = (final_pred - target).std().item()
    print(f"residual std: {std_r2}")
    variance_r2 = (final_pred - target).var().item()
    print(f"residual variance: {variance_r2}")
    print(f"=" * 80)
    
    
    # residuals: 1D array-like

    result = acorr_ljungbox((final_pred - target).detach().cpu().numpy(), lags=[1, 2, 10], return_df=True)
    print(result)

    plot_like_example(df)
    

    return df, data_object



def simple_plot(data):
    plt.close()
    plt.plot(data)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Array Plot")
    plt.savefig("autocorrelation_residual_test.png")

# @torch.no_grad()
# def inference_conformal(
#     datasets: dict,
#     model: nn.Module,
#     mc_dropout: bool = False,
#     dropout_passes: int = 30,          # only used if mc_dropout=True
#     window_size: int = 20,             # window for z-quantile calibration
#     alpha: float = 0.10,               # 90% CI => alpha=0.10
#     mc_samples: int = 300,             # MC samples for refinement
#     agg: str = "median",               # "median" (MAE-opt), "mean", "trimmed_mean"
#     # Volatility / normalization
#     beta_vol: float = 0.9,            # EWMA update for |residual| (higher reacts faster)
#     eps_scale: float = 1e-6,           # numerical stability
#     # Sampling
#     noise_kind: str = "studentt",      # "studentt" or "gaussian"
#     studentt_df: float = 4.0,          # df for Student-t noise
#     bandwidth_scale: float = 0.5,      # multiplier for sigma_t used as sampling noise scale
#     clamp_to_ci: bool = True,          # keep refined samples within conformal CI
#     # Optional: if you want recency-weighted bootstrap of residuals (time-ordered)
#     lam_sample: float = 0.0,           # 0 disables; otherwise exp(-lam*age) weights residual indices
# ):
#     """
#     Balanced implementation:
#       1) Normalized conformal: build CI using z_t = |r_t| / sigma_t where sigma_t is EWMA(|r|).
#          CI half-width at t: q_t = q_z * sigma_t
#       2) Volatility-aware sampling for point refinement:
#          - MAE-optimal center correction uses median of recent residuals (optional but recommended)
#          - sample residuals from recent window (time-ordered) with optional recency weights
#          - add Student-t or Gaussian noise with scale proportional to sigma_t
#          - aggregate via median (MAE-minimizing) or other aggregator
#       3) Enforces ordered indices (sorts by idx if needed)
#     """

#     device = get_device()
#     valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

#     model.to(device)
#     model.eval()

#     if mc_dropout:
#         model = model.apply(dropout_on)
#     else:
#         model = model.apply(dropout_off)

#     # normalization params
#     json_data = read_json_params("train_invocation_rate_normalization.json")
#     train_mu, train_sigma = json_data["mu"], json_data["sigma"]
#     sigma, mu = train_sigma, train_mu

#     # One batch with all data
#     (x, y, last_x, first_y, first_x, idx) = next(iter(valid_loader))
#     x, y = x.to(device), y.to(device)

#     # Ensure idx ordered, and reorder tensors if needed
#     idx_cpu = idx.detach().cpu().view(-1)
#     if idx_cpu.numel() > 1 and not torch.all(idx_cpu[:-1] <= idx_cpu[1:]):
#         sort_perm = torch.argsort(idx_cpu)
#         idx_cpu = idx_cpu[sort_perm]
#         x = x[sort_perm]
#         y = y[sort_perm]
#         last_x = last_x[sort_perm]
#         first_y = first_y[sort_perm]
#         first_x = first_x[sort_perm]

#     # Model conditional input
#     cond = y[:, 0, 1:].to(device)

#     # Point prediction (optionally average MC-dropout passes)
#     # if mc_dropout:
#     #     preds = []
#     #     for _ in range(dropout_passes):
#     #         preds.append(model((x, cond)))
#     #     res_stack = torch.stack(preds, dim=0)                # [S, N, 1]
#     #     res = res_stack.mean(dim=0)                          # [N, 1]
#     #     model_var = res_stack.var(dim=0).squeeze(-1)         # [N]
#     # else:
#     res = model((x, cond))
#     model_var = torch.zeros(res.shape[0], device=device)

#     # Denormalize to target space
#     point_pred = (res.squeeze(-1) + x[:, -1, 0]) * train_sigma + train_mu
#     target = ((y[:, 0, 0] + x[:, -1, 0]) * train_sigma + train_mu).squeeze(-1)

#     n = point_pred.shape[0]
#     ci_lower = torch.empty(n, device=device)
#     ci_upper = torch.empty(n, device=device)
#     refined = torch.empty(n, device=device)

#     # Histories
#     resid_hist = []   # signed residuals r_t = target - point_pred  (floats)
#     z_hist = []       # normalized abs residuals z_t = |r_t| / sigma_t  (floats)

#     # EWMA volatility state: sigma_t ≈ EWMA(|r|)
#     abs_resid_ewma = None

#     # ---------- helpers ----------
#     def conformal_q(values: torch.Tensor, alpha_: float) -> torch.Tensor:
#         """
#         Conformal quantile with 'higher' interpolation:
#           k = ceil((m+1)*(1-alpha)), q = k-th smallest value
#         """
#         m = values.numel()
#         if m == 0:
#             return torch.tensor(0.0, device=values.device, dtype=values.dtype)
#         k = int(math.ceil((m + 1) * (1 - alpha_)))
#         k = min(max(k, 1), m)
#         return values.kthvalue(k).values

#     def aggregate_samples(samples: torch.Tensor, how: str) -> torch.Tensor:
#         if how == "median":
#             return samples.median()
#         if how == "trimmed_mean":
#             srt, _ = torch.sort(samples)
#             ktrim = max(1, int(0.1 * samples.numel()))
#             return srt[ktrim:-ktrim].mean()
#         return samples.mean()

#     def sample_from_past_residuals(
#         past_resid: torch.Tensor,         # [m] signed residuals, time-ordered oldest->newest
#         num_samples: int,
#         lam: float,
#     ) -> torch.Tensor:
#         m = past_resid.numel()
#         if m == 0:
#             return past_resid.new_zeros((num_samples,))

#         if lam > 0:
#             ages = torch.arange(m-1, -1, -1, device=past_resid.device, dtype=torch.float32)  # 0 newest
#             w = torch.exp(-lam * ages)
#             w = w / w.sum()
#             idx = torch.multinomial(w, num_samples=num_samples, replacement=True)
#         else:
#             idx = torch.randint(low=0, high=m, size=(num_samples,), device=past_resid.device)

#         return past_resid[idx]
#     # ----------------------------

#     for t in range(n):
#         # Current volatility estimate (sigma_t). If none yet, start conservatively.
#         if abs_resid_ewma is None:
#             sigma_t = point_pred.new_tensor(1.0)
#         else:
#             sigma_t = abs_resid_ewma.clamp(min=eps_scale)

#         # Compute q_z from normalized residual history
#         if len(z_hist) == 0:
#             qz = point_pred.new_tensor(0.0)
#         else:
#             z_tensor = torch.tensor(z_hist[-window_size:], device=device, dtype=point_pred.dtype)
#             qz = conformal_q(z_tensor, alpha)

#         # Normalized conformal half-width
#         q_t = qz * sigma_t

#         # CI in target space
#         lo = point_pred[t] - q_t
#         hi = point_pred[t] + q_t
#         ci_lower[t], ci_upper[t] = lo, hi

#         # ---- Volatility-aware sampling refinement ----
#         if len(resid_hist) < 5 or mc_samples <= 0:
#             refined[t] = point_pred[t]
#         else:
#             # past residual window (time-ordered)
#             start = max(0, len(resid_hist) - window_size)
#             past = torch.tensor(resid_hist[start:], device=device, dtype=point_pred.dtype)  # [m]

#             # MAE-optimal bias: median residual in the past window
#             bias_l1 = past.median()

#             # Center residuals for sampling (so we apply bias deterministically)
#             past_centered = past - bias_l1

#             # Bootstrap centered residuals (optionally recency-weighted)
#             boot = sample_from_past_residuals(past_centered, mc_samples, lam=lam_sample)

#             # Add volatility-aware continuous noise with scale tied to current sigma_t
#             h = torch.clamp(bandwidth_scale * sigma_t, min=eps_scale)  # sigma_t already target-scale
#             if noise_kind == "gaussian":
#                 eps = torch.randn(mc_samples, device=device, dtype=point_pred.dtype) * h
#             else:
#                 eps = torch.distributions.StudentT(df=studentt_df, loc=0.0, scale=h).sample((mc_samples,))
#                 eps = eps.to(device=device, dtype=point_pred.dtype)

#             center = point_pred[t] + bias_l1
#             samples = center + boot + eps

#             if clamp_to_ci:
#                 samples = torch.clamp(samples, min=lo, max=hi)

#             refined[t] = aggregate_samples(samples, agg)

#         # ---- Update histories with observed residual at t (based on point_pred) ----
#         r_t = (target[t] - point_pred[t]).detach()
#         abs_r = r_t.abs()

#         # Update EWMA(|r|)
#         if abs_resid_ewma is None:
#             abs_resid_ewma = abs_r
#         else:
#             abs_resid_ewma = (1 - beta_vol) * abs_resid_ewma + beta_vol * abs_r

#         sigma_next = abs_resid_ewma.clamp(min=eps_scale)

#         # Store residual history (signed) and normalized abs residual z
#         resid_hist.append(r_t.item())
#         z_hist.append((abs_r / sigma_next).item())

#     # Metrics
#     final_pred = refined
#     error_rates = torch.abs(final_pred - target) / target
#     smape_rate = smape(target.detach().cpu().numpy(), final_pred.detach().cpu().numpy())

#     mean_abs_err = torch.mean(torch.abs(final_pred - target))
#     mean_model_var = model_var.mean()

#     calc_percentile_stats(
#         error_rates.detach().cpu().numpy(),
#         (torch.abs(final_pred - target) / target).sum().item() / len(final_pred),
#         "inference_results_uncertain"
#     )

#     s = (
#         f"[inference_conformal_normalized_volaware] "
#         f"mean_abs_err: {mean_abs_err.item():.6f}, "
#         f"mean_model_var: {mean_model_var.item():.6f}, "
#         f"smape_rate: {smape_rate}"
#     )
#     with open("inference_results_uncertain.log", "a") as f:
#         f.write(s + "\n")
#     print(s)

#     to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)

#     df = pd.DataFrame(
#         {
#             "idx": to_1d(idx_cpu),
#             "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
#             "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),
#             "point_pred": to_1d(point_pred),
#             "pred_refined": to_1d(final_pred),
#             "target": to_1d(target),
#             "ci_lower": to_1d(ci_lower),
#             "ci_upper": to_1d(ci_upper),
#             "error_rate_refined_pct": to_1d(error_rates) * 100,
#         }
#     )
#     df.to_csv("inference_results_uncertain.csv", index=False)

#     return {
#         "point_pred": point_pred,
#         "pred_refined": final_pred,
#         "target": target,
#         "ci_lower": ci_lower,
#         "ci_upper": ci_upper,
#         "smape": smape_rate,
#         "mean_abs_err": mean_abs_err,
#         "mean_model_var": mean_model_var,
#         "df": df,
#     }






def calc_percentile_stats(error_rates, overall_error_rate: float, path):
    print(f"max error rate: {np.max(error_rates)}")
    percentiles = [25, 50, 75, 90, 95, 99, 99.9, 99.99]
    percentile_values = np.percentile(error_rates, percentiles)

    lines = []
    lines.append("=" * 80)
    lines.append(f"Overall mean error rate: {overall_error_rate:.2%}")
    lines.append("=" * 80)
    for p, val in zip(percentiles, percentile_values):
        lines.append(f"{p}th percentile error rate: {val:.2%}")

    log_text = "\n".join(lines)

    # Print to console
    print(log_text)

    # Save to file
    with open(f"{path}.log", "w") as f:
        f.write(log_text + "\n")


def train_encoder_decoder(
    device: str,
    model: nn.Module,
    datasets: dict,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_tqdm: bool = True,
) -> Tuple[nn.Module, dict]:
    stats_data = json.load(open("train_invocation_rate_normalization.json", "r"))
    sigma = stats_data["sigma"]
    mu = stats_data["mu"]
    
    model.to(device)
    optimiser = optim.Adam(
        lr=learning_rate, params=model.parameters(), weight_decay=3e-4
    )
    dataloaders = data.get_dataloaders(datasets=datasets, train_batch_size=batch_size)

    loss_fn = F.mse_loss
    losses = {"train": [], "valid": []}

    # Wrap epochs in a progress iterator, but DON'T overwrite num_epochs
    if use_tqdm:
        from tqdm.auto import tqdm

        epoch_iter = tqdm(
            range(num_epochs), leave=True, disable=not use_tqdm, dynamic_ncols=True
        )
    else:
        epoch_iter = range(num_epochs)

    total_train = len(dataloaders["train"].dataset)
    x = dataloaders["train"].dataset[0]
    best_loss, best_idx = float("inf"), -1
    
    for epoch in epoch_iter:
        model.train()

        running_train_loss = 0.0
        samples_seen = 0

        for i, (x, y, last_x, first_y, first_x, idx, hour) in enumerate(dataloaders["train"]):
            x, y = x.to(device), y.to(device)
            # print(f"idx: {idx}")
            # print(f"x: {(x[0]) * sigma + mu}")
            # print(f"x[0].shape: {x[0].shape}")
            # print(f"x.shape: {x.shape}")
            # print(f"y[0]: {(y[0] + x[0][-1]) * sigma + mu}")
            # print(f"first_y: {first_y[0] * sigma + mu}")
            # print(f"last_x: {last_x[0] * sigma + mu}")
            # print(f"first_x: {first_x[0] }")            
            # input("debug")
            
            out = model(x)
            optimiser.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimiser.step()

            # track running average (nice stable metric per epoch)
            bs = x.size(0)
            # running_train_loss += loss.item() * bs
            running_train_loss += torch.abs(out - y).sum().item()
            samples_seen += bs

            # record step-wise loss if you want to keep your current log structure
            # step = i * batch_size + bs
            # losses["train"].append([epoch * total_train + step, loss.item()])

        # end of epoch: compute validation ONCE
        valid_loss = lstm_evaluate(device, model, dataloaders["valid"], samples_seen)[
            "loss"
        ]
        # avg_train_loss = running_train_loss / max(1, samples_seen) * 100
        avg_train_loss = loss

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_idx = epoch
            print("=== New best model found! Saving... ===")
            print(f"=== Valid loss: {best_loss:.4f} at epoch {best_idx} ===")

        # store a point for the epoch (align step at end of epoch)
        # losses["valid"].append([epoch * total_train + samples_seen, valid_loss])

        # update UI ONCE per epoch
        if use_tqdm:
            epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            epoch_iter.set_postfix(
                train_loss=f"{avg_train_loss:.4f}%", valid_loss=f"{valid_loss:.4f}%"
            )
            # epoch_iter.refresh()  # optional
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train loss={avg_train_loss:.4f} | valid loss={valid_loss:.4f}"
            )

    return model, losses


def lstm_evaluate(
    device: str, model: nn.Module, valid_loader: DataLoader, samples_seen: int
):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i, (x, y, _, _, _, idx, hour) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        # loss = torch.abs(out - y).sum() / samples_seen * 100

    return {"loss": np.float32(loss.cpu().detach().numpy())}


from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_prediction_network(
    device: str,
    datasets: dict,
    prediction_network: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_tqdm: bool = True,
) -> Tuple[nn.Module, dict]:
    stats_data = json.load(open("train_invocation_rate_normalization.json", "r"))
    sigma = stats_data["sigma"]
    mu = stats_data["mu"]
    
    dataloaders = data.get_dataloaders(datasets=datasets, train_batch_size=batch_size)
    prediction_network.to(device)

    optimiser = optim.Adam(
        lr=learning_rate, params=prediction_network.model.parameters()
    )
    loss_fn = F.mse_loss
    losses = {"train": [], "valid": []}

    if use_tqdm:
        from tqdm.auto import tqdm

        epoch_iter = tqdm(range(num_epochs), leave=True, disable=not use_tqdm, dynamic_ncols=True)
    else:
        epoch_iter = range(num_epochs)

    # total_train = len(dataloaders["train"].dataset)

    best_loss, best_idx = float("inf"), -1

    for epoch in epoch_iter:
        prediction_network.train()
        running_train_loss = 0.0
        samples_seen = 0

        # ---- training loop ----
        for i, (x, y, last_x, first_y, first_x, idx, hour) in enumerate(dataloaders["train"]):
            x, y = x.to(device), y.to(device)
            # print(f"idx: {idx}")
            # print(f"x: {(x[0]) * sigma + mu}")
            # print(f"x[0].shape: {x[0].shape}")
            # print(f"x.shape: {x.shape}")
            # print(f"y: {y[0]}")
            # print(f"y[0]: {(y[0][0] + x[0][-1])[0] * sigma + mu}")
            # print(f"first_y: {first_y[0] * sigma + mu}")
            # print(f"last_x: {last_x[0] * sigma + mu}")
            # print(f"first_x: {first_x[0] }")
            # input("debug")
            
            out = prediction_network((x, y[:, 0, 1:]))

            optimiser.zero_grad()
            loss = loss_fn(out, y[:, :, 0])
            loss.backward()
            optimiser.step()

            bs = x.size(0)
            # running_train_loss += loss.item() * bs
            running_train_loss += torch.abs(out - y[:, :, 0]).sum().item()
            samples_seen += bs

            # step = i * batch_size + bs
            # losses["train"].append([epoch * total_train + step, loss.item()])

        # ---- validation after each epoch ----
        valid_loss = evaluate_prediction_network(
            device, prediction_network, dataloaders["valid"], samples_seen
        )
        # avg_train_loss = running_train_loss / max(1, samples_seen) * 100
        avg_train_loss = loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_idx = epoch
            print("=== New best model found! Saving... ===")
            print(f"=== Valid loss: {best_loss:.4f} at epoch {best_idx} ===")
        # losses["valid"].append([epoch * total_train + samples_seen, valid_loss])

        # ---- update tqdm ONCE per epoch ----
        if use_tqdm:
            epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            epoch_iter.set_postfix(
                train_loss=f"{avg_train_loss:.4f}%", valid_loss=f"{valid_loss:.4f}%"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train loss={avg_train_loss:.4f} | valid loss={valid_loss:.4f}"
            )

    return prediction_network, losses


def evaluate_prediction_network(
    device: str, model: nn.Module, valid_loader: DataLoader, samples_seen: int
):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i, (x, y, _, _, _, idx, hour) in enumerate(valid_loader):
        break
    x, y = x.to(device), y.to(device)
    out = model((x, y[:, 0, 1:]))
    # loss = loss_fn(out, y[:, :, 0])
    # loss = torch.abs(out - y[:, :, 0]).sum() / samples_seen * 100
    loss = loss_fn(out, y[:, :, 0])

    return np.float32(loss.cpu().detach().numpy())


def dropout_on(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.train()


def dropout_off(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.eval()


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true)

    # Avoid division by zero
    mask = denominator == 0
    denominator[mask] = 1

    return 200 * np.mean(numerator / denominator)

# Original inference function commented out
# @torch.no_grad()
# def inference(
#     datasets: dict, model: nn.Module, mc_dropout: bool = False, batch_size: int = 1
# ):

#     device = get_device()
#     valid_loader = data.get_dataloaders(datasets=datasets)["inference"]
#     model.to(device)

#     if mc_dropout:
#         model = model.apply(dropout_on)
#     else:
#         model = model.apply(dropout_off)

#     json_data = read_json_params("train_invocation_rate_normalization.json")
#     train_mu, train_sigma = json_data["mu"], json_data["sigma"]
#     sigma, mu = train_sigma, train_mu

#     for i, (x, y, last_x, first_y, first_x, idx) in enumerate(valid_loader):
#         x, y = x.to(device), y.to(device)
#         # print(f"idx: {idx}")
#         # print(f"x: {(x[0]) * sigma + mu}")
#         # print(f"x[0].shape: {x[0].shape}")
#         # print(f"x.shape: {x.shape}")
#         # print(f"y: {y[0]}")
#         # print(f"y[0]: {(y[0][0] + x[0][-1])[0] * sigma + mu}")
#         # print(f"first_y: {first_y[0] * sigma + mu}")
#         # print(f"last_x: {last_x[0] * sigma + mu}")
#         # print(f"first_x: {first_x[0] }")
#         # input("debug")
#         x, y = x.to(device), y.to(device)
        
#         # res = []
#         # for _ in range(batch_size):
#         #     res.extend(model((x, y[:, 0, 1:])).cpu().detach().tolist())
        
#         res = model((x, y[:, 0, 1:]))
        
#         # res = torch.tensor(res, dtype=torch.double)
#         mean = torch.mean(torch.tensor(res)).to(device)
#         var = torch.var(torch.tensor(res))

#         predicted = (res.squeeze(-1)) * train_sigma + train_mu
#         target = (
#             ((y[:, 0, 0]) * train_sigma + train_mu).to(device).squeeze(-1)
#         )
        
#         # predicted = (res.squeeze(-1) + x[:, -1, 0])
#         # predicted = (res.squeeze(-1))
#         # print(f"res.shape: {res.shape}")
#         # print(f"predicted.shape: {predicted.shape}")
#         # print(f"x[:, -1, 0].shape: {x[:, -1, 0].shape}")
#         # input("debug")
#         # print(f"predicted: {predicted[0]}")
#         # print(f"predicted.shape: {predicted.shape}")
#         # print(f"y_shape: {y.shape}")
#         # print(f"x_shape: {x.shape}")
#         # input("debug")
    
#         # target = (
#         #     ((y[:, 0, 0])).to(device).squeeze(-1)
#         # )
        
#         # print(f"target.shape: {target.shape}")
#         # print(f"target: {target[0]}")
#         # input("debug")
        
#         error_rates = torch.abs(predicted - target) / target 
#         # error_rate = (torch.abs(predicted - target) / target).sum() / len(x_hour) * 100
#         smape_rate = smape(target.cpu().numpy(), predicted.cpu().numpy())

#         calc_percentile_stats(
#             error_rates.cpu().numpy(),
#             (torch.abs(predicted - target) / target).sum() / len(predicted),
#             "inference_results"
#         )

#         s = f"[inference] mean: {mean}, var: {var}, smape_rate: {smape_rate}"
#         with open(f"inference_results.log", "a") as f:
#             f.write(s + "\n")
#         print(s)    
        
#         # print(f"[inference] predicted: {predicted}, target: {target}, error: {predicted - target}")

#         # print(f"x_hour.shape: {x_hour.shape}, y_hour.shape: {y_hour.shape}")
#         print(f"predicted.shape: {predicted.shape}, target.shape: {target.shape}, error_rates.shape: {error_rates.shape}")
#         print(f"x.shape: {x.shape}, y.shape: {y.shape}")
#         print(f"predicted.shape: {predicted.shape}, target.shape: {target.shape}, error_rates.shape: {error_rates.shape}")
#         to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)
        
#         print(f"max target: {target.max().item()}, min target: {target.min().item()}")

#         df = pd.DataFrame(
#             {
#                 "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
#                 "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),
#                 "predicted": to_1d(predicted),
#                 "target": to_1d(target),
#                 "error_rates": to_1d(error_rates) * 100,
#             }
#         )

#         df.to_csv(f"inference_results.csv", index=False)

#     return mean, var
    # return {"loss": np.float32(loss.cpu().detach().numpy())}
    
    
@torch.no_grad()
def inference(
    datasets: dict, model: nn.Module, not_used1: bool = False, not_used2: int = 1
):

    device = get_device()
    valid_loader = data.get_dataloaders(datasets=datasets)["inference"]
    model.to(device)

    json_data = read_json_params("train_invocation_rate_normalization.json")
    train_mu, train_sigma = json_data["mu"], json_data["sigma"]
    sigma, mu = train_sigma, train_mu

    for i, (x, y, _, _, _, _, _) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        x, y = x.to(device), y.to(device)
        
        res = model((x, y[:, 0, 1:]))
        # print(f"y.shape: {y.shape}")

        predicted = (res.squeeze(-1)) * train_sigma + train_mu
        target = (
            ((y[:, :, 0]) * train_sigma + train_mu).to(device).squeeze(-1)
        )
        
        error_rates = torch.abs(predicted - target) / target 
        smape_rate = smape(target.cpu().numpy(), predicted.cpu().numpy())

        calc_percentile_stats(
            error_rates.cpu().numpy(),
            (torch.abs(predicted - target) / target).sum() / len(predicted),
            "inference_results"
        )

        s = f"[inference] smape_rate: {smape_rate}"
        with open(f"inference_results.log", "a") as f:
            f.write(s + "\n")
        print(s)    
    
        to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)
        
        print(f"max target: {target.max().item()}, min target: {target.min().item()}")

        df = pd.DataFrame(
            {
                "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
                "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),
                "predicted": to_1d(predicted),
                "target": to_1d(target),
                "error_rates": to_1d(error_rates) * 100,
            }
        )

        df.to_csv(f"inference_results.csv", index=False)



import json
import numpy as np
import torch
from scipy.stats import genpareto

def write_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

@torch.no_grad()
def collect_oos_signed_residuals(
    datasets: dict,
    model,
    loader_key: str = "calibration",
    normalization_json: str = "train_invocation_rate_normalization.json",
    max_batches: int | None = None,
):
    """
    Collect out-of-sample signed residuals r = y - yhat in de-normalized units.
    """
    device = get_device()
    loader = data.get_dataloaders(datasets=datasets)[loader_key]

    model = model.to(device)
    model.eval()

    norm = read_json_params(normalization_json)
    mu, sigma = float(norm["mu"]), float(norm["sigma"])

    rs = []

    for i, (x, y, _, _, _, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        res = model((x, y[:, 0, 1:]))
        yhat = (res.squeeze(-1)) * sigma + mu
        yt = (y[:, 0, 0]) * sigma + mu
        yt = yt.squeeze(-1)

        r = (yt - yhat).detach().cpu().numpy().reshape(-1)
        rs.append(r)

    r = np.concatenate(rs, axis=0).astype(np.float64)
    r = r[np.isfinite(r)]
    return r

def fit_signed_evt_profile(
    r: np.ndarray,
    u_quantile: float = 0.95,
    min_tail_each: int = 50,
):
    """
    Fit a *signed* EVT mixture:
      bulk: signed residuals with |r| <= u
      pos tail: exceedances of r > u  (fit GPD on r-u)
      neg tail: exceedances of r < -u (fit GPD on (-r)-u)

    Returns:
      profile dict and bulk_signed array.
    """
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]

    if r.size < 1000:
        print(f"[warn] Only {r.size} residuals. Tail fits may be noisy.")

    # symmetric threshold on |r|
    u = float(np.quantile(np.abs(r), u_quantile))
    if u <= 0:
        raise ValueError("Computed u<=0; residuals appear degenerate.")

    # bulk (signed)
    bulk_signed = r[np.abs(r) <= u].astype(np.float32)
    if bulk_signed.size == 0:
        raise ValueError("Bulk pool is empty; lower u_quantile.")

    # pos tail exceedances
    pos_exc = r[r > u] - u
    neg_exc = (-r[r < -u]) - u  # magnitude exceedance on negative side

    n = r.size
    n_pos = pos_exc.size
    n_neg = neg_exc.size

    p_pos = float(n_pos / n)
    p_neg = float(n_neg / n)
    p_bulk = float(1.0 - (p_pos + p_neg))

    if n_pos < min_tail_each or n_neg < min_tail_each:
        raise ValueError(
            f"Too few tail samples: n_pos={n_pos}, n_neg={n_neg} at u={u:.6f} "
            f"(quantile={u_quantile}). Collect more calibration data or lower u_quantile."
        )

    # Fit GPD to exceedances, loc fixed at 0
    xi_pos, loc, beta_pos = genpareto.fit(pos_exc, floc=0.0)
    xi_neg, loc, beta_neg = genpareto.fit(neg_exc, floc=0.0)

    beta_pos = float(beta_pos)
    beta_neg = float(beta_neg)
    if beta_pos <= 0 or beta_neg <= 0:
        raise ValueError("Bad GPD fit: beta <= 0. Try a different u_quantile or more data.")

    profile = {
        "u": float(u),
        "u_quantile": float(u_quantile),

        "p_bulk": float(p_bulk),
        "p_pos": float(p_pos),
        "p_neg": float(p_neg),

        "xi_pos": float(xi_pos),
        "beta_pos": float(beta_pos),

        "xi_neg": float(xi_neg),
        "beta_neg": float(beta_neg),

        "n_total": int(n),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "bulk_size": int(bulk_signed.size),
    }
    return profile, bulk_signed

def build_and_save_signed_evt_profile(
    datasets: dict,
    model,
    loader_key: str = "calibration",
    normalization_json: str = "train_invocation_rate_normalization.json",
    u_quantile: float = 0.95,
    min_tail_each: int = 50,
    out_json: str = "evt_signed.json",
    out_bulk: str = "evt_bulk_signed.npy",
):
    r = collect_oos_signed_residuals(
        datasets=datasets,
        model=model,
        loader_key=loader_key,
        normalization_json=normalization_json,
    )

    # helpful diagnostics
    print("Residual percentiles:", np.quantile(r, [0.01, 0.05, 0.5, 0.95, 0.99]))
    print("Abs residual percentiles:", np.quantile(np.abs(r), [0.5, 0.9, 0.95, 0.99]))

    profile, bulk_signed = fit_signed_evt_profile(
        r=r,
        u_quantile=u_quantile,
        min_tail_each=min_tail_each,
    )

    write_json(out_json, profile)
    np.save(out_bulk, bulk_signed)

    print("[Saved signed EVT profile]")
    print(" ", out_json, "->", profile)
    print(" ", out_bulk, "bulk_signed shape:", bulk_signed.shape, "max|bulk|:", float(np.max(np.abs(bulk_signed))))


import numpy as np
import torch

def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_signed_evt_profile(evt_json_path: str, bulk_npy_path: str):
    prof = read_json(evt_json_path)
    bulk = np.load(bulk_npy_path).astype(np.float32)
    if bulk.ndim != 1 or bulk.size == 0:
        raise ValueError("bulk_signed must be a non-empty 1D array.")

    # minimal validation
    for k in ["u", "p_bulk", "p_pos", "p_neg", "xi_pos", "beta_pos", "xi_neg", "beta_neg"]:
        if k not in prof:
            raise ValueError(f"Missing key {k} in {evt_json_path}")

    # enforce probabilities sum ~ 1
    s = prof["p_bulk"] + prof["p_pos"] + prof["p_neg"]
    if not (0.999 <= s <= 1.001):
        raise ValueError(f"Probabilities do not sum to 1: {s}")

    prof["bulk_signed"] = bulk
    return prof

def gpd_sample_exceedance(u_unif: torch.Tensor, xi: float, beta: float) -> torch.Tensor:
    eps = 1e-12
    u_unif = torch.clamp(u_unif, eps, 1.0 - eps)
    if abs(xi) < 1e-8:
        return -beta * torch.log1p(-u_unif)
    return (beta / xi) * (torch.pow(1.0 - u_unif, -xi) - 1.0)

@torch.no_grad()
def mc_signed_evt_predict(
    yhat: torch.Tensor,
    profile: dict,
    K: int = 10000,
    quantiles=(0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99),
    clamp_nonneg: bool = True,
):
    """
    Draw predictive samples:
      y = yhat + r
    where r is sampled from:
      bulk (signed, |r|<=u) with prob p_bulk
      pos tail (r = +(u + Zpos)) with prob p_pos
      neg tail (r = -(u + Zneg)) with prob p_neg

    Returns: qvals, diag, mc_mean, mc_std
    """
    device = yhat.device
    B = yhat.shape[0]

    u = float(profile["u"])
    p_bulk = float(profile["p_bulk"])
    p_pos = float(profile["p_pos"])
    p_neg = float(profile["p_neg"])

    xi_pos = float(profile["xi_pos"])
    beta_pos = float(profile["beta_pos"])
    xi_neg = float(profile["xi_neg"])
    beta_neg = float(profile["beta_neg"])

    bulk_signed = torch.from_numpy(profile["bulk_signed"]).to(device=device)

    # component selection
    U = torch.rand((K, B), device=device)
    comp_bulk = U < p_bulk
    comp_pos = (U >= p_bulk) & (U < p_bulk + p_pos)
    comp_neg = U >= (p_bulk + p_pos)

    # bulk residuals (signed)
    idx = torch.randint(low=0, high=bulk_signed.numel(), size=(K, B), device=device)
    r_bulk = bulk_signed[idx]  # signed

    # tail residuals
    z_pos = gpd_sample_exceedance(torch.rand((K, B), device=device), xi=xi_pos, beta=beta_pos)
    r_pos = u + z_pos

    z_neg = gpd_sample_exceedance(torch.rand((K, B), device=device), xi=xi_neg, beta=beta_neg)
    r_neg = -(u + z_neg)

    # assemble residuals
    r = torch.zeros((K, B), device=device)
    r = torch.where(comp_bulk, r_bulk, r)
    r = torch.where(comp_pos, r_pos, r)
    r = torch.where(comp_neg, r_neg, r)

    # predictive samples
    y_samples = yhat.unsqueeze(0) + r
    if clamp_nonneg:
        y_samples = torch.clamp(y_samples, min=0.0)

    # summary stats
    mc_mean = y_samples.mean(dim=0)
    mc_std = y_samples.std(dim=0)

    # quantiles
    qvals = {q: torch.quantile(y_samples, q, dim=0) for q in quantiles}

    diag = {
        "comp_bulk_rate": comp_bulk.float().mean(dim=0),
        "comp_pos_rate": comp_pos.float().mean(dim=0),
        "comp_neg_rate": comp_neg.float().mean(dim=0),
        "r_mean": r.mean(dim=0),
        "r_abs_mean": torch.abs(r).mean(dim=0),
    }
    return qvals, diag, mc_mean, mc_std


import pandas as pd
import numpy as np
import torch
from torch import nn

@torch.no_grad()
def inference_with_signed_EVT_MC(
    datasets: dict,
    model: nn.Module,
    evt_json_path: str = "evt_signed.json",
    evt_bulk_path: str = "evt_bulk_signed.npy",
    K: int = 10000,
):
    device = get_device()
    valid_loader = data.get_dataloaders(datasets=datasets)["inference"]

    model.to(device)
    model.eval()

    # normalization params
    json_data = read_json_params("train_invocation_rate_normalization.json")
    train_mu, train_sigma = float(json_data["mu"]), float(json_data["sigma"])
    sigma, mu = train_sigma, train_mu

    profile = load_signed_evt_profile(evt_json_path, evt_bulk_path)
    print("[Loaded EVT signed profile]", profile)

    all_rows = []
    smape_base_list = []
    smape_mc_list = []
    cov90_list = []
    cov98_list = []

    for i, (x, y, _, _, _, _) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)

        # baseline inference
        res = model((x, y[:, 0, 1:]))
        predicted = (res.squeeze(-1)) * sigma + mu
        target = ((y[:, 0, 0]) * sigma + mu).squeeze(-1)

        # MC sampling (signed mixture)
        qvals, diag, mc_mean, mc_std = mc_signed_evt_predict(
            yhat=predicted,
            profile=profile,
            K=K,
            quantiles=(0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99),
            clamp_nonneg=True,
        )

        # metrics
        smape_base = smape(target.detach().cpu().numpy(), predicted.detach().cpu().numpy())
        smape_mc = smape(target.detach().cpu().numpy(), mc_mean.detach().cpu().numpy())
        smape_base_list.append(float(smape_base))
        smape_mc_list.append(float(smape_mc))

        inside_90 = ((target >= qvals[0.05]) & (target <= qvals[0.95])).float().mean().item()
        inside_98 = ((target >= qvals[0.01]) & (target <= qvals[0.99])).float().mean().item()
        cov90_list.append(inside_90)
        cov98_list.append(inside_98)

        # log
        print(f"[inference] batch={i} SMAPE baseline={smape_base:.4f}  MCmean={smape_mc:.4f}  cov90={inside_90:.3f} cov98={inside_98:.3f}")

        # dataframe
        to_1d = lambda t: t.detach().cpu().numpy().reshape(-1)
        denom = torch.clamp(target, min=1e-12)
        error_rates_pct = (torch.abs(predicted - target) / denom) * 100.0
        error_rates_pct_mc = (torch.abs(mc_mean - target) / denom) * 100.0

        df_batch = pd.DataFrame(
            {
                "x_last_hour": to_1d(x[:, -1, 0] * sigma + mu),
                "x_start_hour": to_1d(x[:, 0, 0] * sigma + mu),

                "predicted": to_1d(predicted),
                "mc_mean": to_1d(mc_mean),
                "mc_std": to_1d(mc_std),

                "target": to_1d(target),

                "error_pct_baseline": to_1d(error_rates_pct),
                "error_pct_mcmean": to_1d(error_rates_pct_mc),

                "p01": to_1d(qvals[0.01]),
                "p05": to_1d(qvals[0.05]),
                "p10": to_1d(qvals[0.10]),
                "p50": to_1d(qvals[0.50]),
                "p90": to_1d(qvals[0.90]),
                "p95": to_1d(qvals[0.95]),
                "p99": to_1d(qvals[0.99]),

                # diagnostics
                "comp_bulk_rate": to_1d(diag["comp_bulk_rate"]),
                "comp_pos_rate": to_1d(diag["comp_pos_rate"]),
                "comp_neg_rate": to_1d(diag["comp_neg_rate"]),
                "r_mean": to_1d(diag["r_mean"]),
                "r_abs_mean": to_1d(diag["r_abs_mean"]),
            }
        )
        df_batch["batch_id"] = i
        df_batch["batch_cov90"] = inside_90
        df_batch["batch_cov98"] = inside_98

        all_rows.append(df_batch)

    df = pd.concat(all_rows, axis=0, ignore_index=True)
    df.to_csv("inference_results_signed_evt_mc.csv", index=False)

    print("[summary]")
    print("  mean SMAPE baseline:", float(np.mean(smape_base_list)))
    print("  mean SMAPE mc_mean:", float(np.mean(smape_mc_list)))
    print("  mean cov90:", float(np.mean(cov90_list)))
    print("  mean cov98:", float(np.mean(cov98_list)))
    print("  wrote:", "inference_results_signed_evt_mc.csv")




def save(model: nn.Module, name: str, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    model_path = Path(path) / "{}.pt".format(name)
    torch.save(model, model_path)
    print(f"PyTorch model saved at {model_path}")


def read_json_params(path):
    with open(path) as json_file:
        params = json.load(json_file)
    return params
