#!/usr/bin/env python3
"""
Compute 300 hourly energy samples from Monte-Carlo samples of *next-hour average* request rate,
using historical intra-hour variance from per-minute ground-truth CSV.

UPDATED ALIGNMENT:
- Your predictor consumes the previous 24 hours and predicts the "25th hour" relative to the
  start of that 24-hour window. Operationally, if the pickle key corresponds to the FIRST hour
  of the 24-hour history window, then the prediction target hour is:

      target_hour = hour_key + 24

This matches: hours [hour_key ... hour_key+23] are inputs, hour_key+24 is predicted.

If instead your pickle key corresponds to the LAST hour of the history window (i.e., the 24th),
then target_hour would be hour_key + 1. This script implements the +24 rule per your statement.

Inputs:
  1) requests.csv with columns: day,time,invocation_rate
     - time is Unix timestamp in seconds (minute granularity)
  2) mc_samples.pkl: dict-like {hour_key: array_like(shape=(300,))} of predicted avg rate samples

Outputs:
  - hourly_ground_truth_energy.csv
  - energy_samples_tidy.csv                 (variance-corrected)
  - energy_samples_tidy_uncorrected.csv     (no variance correction)
  - energy_summary.csv                      (per-hour summary, corrected, merged with ground truth)
"""


import argparse
import pickle
import sys
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_pred - y_true)

    # avoid division by zero
    mask = denominator != 0

    return 200 * np.mean(diff[mask] / denominator[mask])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/valid.csv", help="Per-minute ground-truth CSV")
    ap.add_argument("--pkl", default="../rate_data.pkl", help="Pickle file with MC samples")
    ap.add_argument("--a", type=float, default=0.01, help="Quadratic coefficient a")
    ap.add_argument("--b", type=float, default=0.5, help="Linear coefficient b")
    ap.add_argument("--c", type=float, default=10.0, help="Constant coefficient c")
    ap.add_argument("--out_pkl_corr", default="energy_samples_2d_corrected.pkl",
                    help="Output pickle (2D array) for variance-corrected energy")
    ap.add_argument("--out_pkl_uncorr", default="energy_samples_2d_uncorrected.pkl",
                    help="Output pickle (2D array) for uncorrected energy")
    ap.add_argument("--start_hour", type=int, default=24,
                    help="Hour id corresponding to MC row 0 (hour_key = start_hour + i)")


    # Key correction: prediction horizon in HOURS from the pickle key to target hour
    ap.add_argument(
        "--horizon_hours",
        type=int,
        default=24,
        help="Target hour = hour_key + horizon_hours. For 24h-history predicting 25th hour, use 24.",
    )

    # Variance choice
    ap.add_argument(
        "--population_var",
        action="store_true",
        help="Use population variance (ddof=0) for sigma2; default is ddof=0 already.",
    )

    return ap.parse_args()


def energy_quadratic(r: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return a * r**2 + b * r + c



import sys
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np


def compute_energy_arrays(
    *,
    mc_arr: np.ndarray,
    target_hours: Union[np.ndarray, Iterable[int]],
    hourly_stats,  # expects a pandas DataFrame-like with .index and .loc[hour, "sigma2"]
    a: float,
    b: float,
    c: float,
    energy_quadratic: Callable[[np.ndarray, float, float, float], np.ndarray],
    warn_missing_sigma2: bool = True,
    return_missing_hours: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, set]]:
    """
    Compute uncorrected and variance-corrected energy arrays for Monte Carlo samples.

    For each i (row) and sample s:
      - E_uncorr[i, s] = 60 * energy_quadratic(mu_samples[s], a, b, c)
      - E_corr[i, s]   = 60 * (a * (mu_samples[s]^2 + sigma2_hour) + b * mu_samples[s] + c)

    sigma2_hour is pulled from hourly_stats.loc[target_hour, "sigma2"] if available; else 0.

    Parameters
    ----------
    mc_arr:
        Array of shape (N, S) containing Monte Carlo samples of mu for each target hour.
    target_hours:
        Length-N sequence/array of target hours corresponding to rows of mc_arr.
    hourly_stats:
        Pandas DataFrame-like object indexed by hour with column "sigma2".
    a, b, c:
        Coefficients for the quadratic energy model.
    energy_quadratic:
        Function f(mu, a, b, c) -> energy per minute (or unit consistent with your model),
        vectorized over mu.
    warn_missing_sigma2:
        If True, prints a warning to stderr when sigma2 is missing for some hours.
    return_missing_hours:
        If True, also returns a set of target hours for which sigma2 was missing.

    Returns
    -------
    (E_uncorr, E_corr) or (E_uncorr, E_corr, missing_sigma2_hours)

    Notes
    -----
    - Uses sigma2=0.0 when missing.
    - Multiplies by 60.0 as in your original code.
    """
    mc_arr = np.asarray(mc_arr, dtype=float)
    target_hours = np.asarray(list(target_hours))

    if mc_arr.ndim != 2:
        raise ValueError(f"mc_arr must be 2D (N,S). Got shape {mc_arr.shape}.")
    N, S = mc_arr.shape
    if target_hours.shape[0] != N:
        raise ValueError(
            f"target_hours length must match mc_arr rows N={N}. Got {target_hours.shape[0]}."
        )

    # Pre-allocate
    E_uncorr = np.empty((N, S), dtype=float)
    E_corr = np.empty((N, S), dtype=float)

    missing_sigma2_hours: set = set()

    for i in range(N):
        target_hour = target_hours[i]
        mu_samples = mc_arr[i, :]  # (S,)

        # Uncorrected: 60*f(mu)
        E_uncorr[i, :] = 60.0 * energy_quadratic(mu_samples, a, b, c)

        # Corrected: add sigma2 for target hour (scalar applied across samples)
        if target_hour in hourly_stats.index:
            sigma2 = float(hourly_stats.loc[target_hour, "sigma2"])
        else:
            sigma2 = 0.0
            missing_sigma2_hours.add(int(target_hour))

        E_corr[i, :] = 60.0 * (a * (mu_samples**2 + sigma2) + b * mu_samples + c)

    if warn_missing_sigma2 and missing_sigma2_hours:
        print(
            f"Warning: missing historical variance for {len(missing_sigma2_hours)} target hours; "
            f"used sigma2=0. Examples: {sorted(list(missing_sigma2_hours))[:5]}",
            file=sys.stderr,
        )

    if return_missing_hours:
        return E_uncorr, E_corr, missing_sigma2_hours
    return E_uncorr, E_corr

def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return np.mean(np.abs(y_true - y_pred))

class CI_data:
    def __init__(self, pred_ci_list, actual_ci_list, ramp_list, samples_list, pred_refined_list, smape_before, smape_after, verification_code):
        self.predicted_ci_list = pred_ci_list
        self.actual_ci_list = actual_ci_list
        self.predicted_refined_list = pred_refined_list
        self.ramp_list = ramp_list
        self.samples_list = samples_list
        self.smape_before = smape_before
        self.smape_after = smape_after
        self.verification_code = verification_code

def scale_to_range(arr, n):
    arr = np.asarray(arr, dtype=float)
    min_val = arr.min()
    max_val = arr.max()

    if min_val == max_val:
        raise ValueError("Cannot scale an array with all identical values.")

    return (arr - min_val) / (max_val - min_val) * n
            
def change_CI_length(CI_data_obj, length):
    CI_data_obj.predicted_ci_list = CI_data_obj.predicted_ci_list[:length]
    CI_data_obj.actual_ci_list = CI_data_obj.actual_ci_list[:length]
    CI_data_obj.predicted_refined_list = CI_data_obj.predicted_refined_list[:length]
    CI_data_obj.ramp_list = CI_data_obj.ramp_list[:length]
    CI_data_obj.samples_list = CI_data_obj.samples_list[:length]
    
    return CI_data_obj


def residual_seasonal_ar_correction(
    y_true,
    y_pred,
    lags=(1, 2, 24),
    min_train=200,
    lb_lags=(1, 2, 24),
):
    """
    Correct base predictions using a seasonal AR model fitted on residuals.

    Parameters
    ----------
    y_true : array-like
        Ground-truth time series y_t (time-ordered).
    y_pred : array-like
        Base model predictions \hat{y}_t aligned with y_true.
    lags : tuple[int]
        Residual AR lags to model. Include 24 for daily seasonality in hourly data.
    min_train : int
        Minimum number of points before starting walk-forward correction.
        Must be > max(lags). Increase for stability.
    lb_lags : tuple[int]
        Lags to use in Ljung–Box test for corrected residuals.

    Returns
    -------
    result : dict
        Contains corrected residuals, corrected predictions, and Ljung–Box table.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same length"

    e = y_true - y_pred  # residuals
    T = len(e)

    max_lag = int(max(lags))
    start_t = max(min_train, max_lag + 5)

    # Predicted residuals (one-step-ahead) for each t
    e_hat = np.full(T, np.nan, dtype=float)

    for t in range(start_t, T):
        hist = pd.Series(e[:t])  # residuals up to t-1
        # Fit seasonal AR on residuals
        ar = AutoReg(hist, lags=list(lags), old_names=False).fit()
        pred_t = ar.predict(start=t, end=t)  # predict e_t using only past info
        e_hat[t] = float(pred_t.iloc[0])

    mask = ~np.isnan(e_hat)
    y_pred_adj = y_pred[mask] + e_hat[mask]          # corrected prediction
    e_corrected = y_true[mask] - y_pred_adj          # corrected residual

    lb = acorr_ljungbox(e_corrected, lags=list(lb_lags), return_df=True)

    return {
        "start_index": start_t,
        "mask": mask,
        "y_pred_adj": y_pred_adj,
        "e_corrected": e_corrected,
        "mean_corrected_residual": float(np.mean(e_corrected)),
        "lb_table": lb,
    }

        
def main(region: str, days: int, window_size: int, DATA_PATH: str = "data", is_vertical: bool = False, is_refined: bool = False):
    print(f"=" * 80)
    print(f"Processing region: {region}")
    args = parse_args()

    csv_path = Path(args.csv)
    pkl_path = Path(args.pkl)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not pkl_path.exists():
        print(f"Pickle not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    a, b, c = -0.157702, 79.9895, 3050.61
    # c, b, a = 9151.83, 239.969, -0.473106
    H = int(args.horizon_hours)
    
    infer_df = pd.read_csv("/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/debug_24h_inference.csv", usecols=["hour"])
    infer_df["hour"] = pd.to_numeric(infer_df["hour"], errors="coerce")
    infer_df = infer_df.dropna(subset=["hour"]).copy()
    infer_hours = set(infer_df["hour"].astype(int).tolist())

    print(f"Inference hour count: {len(infer_hours)}", file=sys.stderr)
    print(f"Inference hour range: {min(infer_hours)}..{max(infer_hours)}", file=sys.stderr)


    # ----------------------------
    # 1) Load per-minute ground truth
    # ----------------------------
    df = pd.read_csv(csv_path)
    required_cols = {"day", "time", "invocation_rate"}
    if not required_cols.issubset(df.columns):
        print(f"CSV must contain columns {sorted(required_cols)}; found {sorted(df.columns)}", file=sys.stderr)
        sys.exit(2)

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["invocation_rate"] = pd.to_numeric(df["invocation_rate"], errors="coerce")
    df = df.dropna(subset=["time", "invocation_rate"]).copy()

    # Hour bucket from Unix seconds
    df["hour"] = (df["time"] // 3600).astype(int)
    # Keep only minutes that belong to inference hours
    df = df[df["hour"].isin(infer_hours)].copy()

    df.to_csv("debug_minute_level.csv", index=False)
    # input("[debug] saved debug_minute_level.csv")

    # ----------------------------
    # 2) Hourly stats from history: mean and variance of minute rates
    # ----------------------------
    ddof = 0  # population variance recommended for sigma^2 term
    if not args.population_var:
        ddof = 0
        
    df["invocation_rate"] = scale_to_range(df["invocation_rate"].astype(float), 150)

    hourly_stats = (
        df.groupby("hour")["invocation_rate"]
        .agg(mu="mean", sigma2=lambda x: x.var(ddof=ddof))
        .reset_index()
        .set_index("hour")
    )

    # ----------------------------
    # 3) Ground-truth hourly energy from minute-level sum (for validation)
    # ----------------------------
    df["energy_minute"] = energy_quadratic(df["invocation_rate"].to_numpy(), a, b, c)
    gt_hourly_energy = (
        df.groupby("hour", as_index=False)
          .agg(
              true_hourly_energy=("energy_minute", "sum"),
              avg_invocation_rate=("invocation_rate", "mean"),
          )
    )
    
    gt_hourly_energy.to_csv("debug_hourly_ground_truth_energy.csv", index=False)
    # input("debug")

    # ----------------------------
    # 4) Load Monte-Carlo samples
    # ----------------------------
    with open(pkl_path, "rb") as fp:
      mc_obj = pickle.load(fp)
      rate_data_length = len(mc_obj)
      print(f"len(mc_obj) = {len(mc_obj)}")
    # print(mc_obj.keys())

    arr = [x["mc_samples"] for x in mc_obj]
    original_rate_preds = [x["original_prediction"] for x in mc_obj]
    refined_rate_preds = [x["refined_prediction"] for x in mc_obj]
    target_hours = [x["hour"] for x in mc_obj]
    mc_arr = np.array(arr, dtype=float)

    gt_hourly_energy = gt_hourly_energy[gt_hourly_energy["hour"].isin(target_hours)].copy()
    gt_hourly_energy.to_csv("hourly_ground_truth_energy.csv", index=False)
    print(f"len (mc_obj) = {len(mc_obj)}")
    print(f"len gt_hourly_energy = {len(gt_hourly_energy)}")
    
    point_of_estimation_rates_mc = np.mean(mc_arr, axis=1)
    gt_rate = gt_hourly_energy.set_index("hour").loc[target_hours, "avg_invocation_rate"].to_numpy()
    rate_score = smape(gt_rate, point_of_estimation_rates_mc)
    print(f"Point-estimation prediction rate SMAPE: {rate_score:.2f}%")
    
    if mc_arr.ndim != 2:
        print(f"Expected 2-D MC array, got shape {mc_arr.shape}", file=sys.stderr)
        sys.exit(2)
    
    N, S = mc_arr.shape
    if S != 300:
        print(f"Warning: MC sample count is {S} (expected 300). Proceeding.", file=sys.stderr)

    # ----------------------------
    # 5) Compute energy samples per hour
    # ----------------------------
    # ----------------------------
    # 5) Compute energy samples per hour as 2-D arrays (N, S)
    # ----------------------------

    # Pre-allocate
    E_uncorr, E_corr = compute_energy_arrays(
        mc_arr=mc_arr,
        target_hours=np.array(target_hours),
        hourly_stats=hourly_stats,
        a=a,
        b=b,
        c=c,
        energy_quadratic=energy_quadratic,
    )

    original_point_pred_rate = np.array(original_rate_preds).reshape(-1, 1)
    original_point_pred_rate = np.ones(mc_arr.shape) * original_point_pred_rate
    p_ori_E_uncorr, p_ori_E_corr = compute_energy_arrays(
        mc_arr=original_point_pred_rate,
        target_hours=np.array(target_hours),
        hourly_stats=hourly_stats,
        a=a,
        b=b,
        c=c,
        energy_quadratic=energy_quadratic,
    )
    
    refined_point_pred_rate = np.array(refined_rate_preds).reshape(-1, 1)
    refined_point_pred_rate = np.ones(mc_arr.shape) * refined_point_pred_rate
    p_ref_E_uncorr, p_ref_E_corr = compute_energy_arrays(
        mc_arr=refined_point_pred_rate,
        target_hours=np.array(target_hours),
        hourly_stats=hourly_stats,
        a=a,
        b=b,
        c=c,
        energy_quadratic=energy_quadratic,
    )

    # for i in range(point_pred_rate.shape[0]):
    #     print(f"point_pred_rate[{i}] = {point_pred_rate[i,0]}, mc_arr[{i}] mean = {mc_arr[i].mean()}")
    #     input("debug")


    # ----------------------------
    # 5b) Save as pickles (still 2-D arrays)
    # ----------------------------
    with open(args.out_pkl_uncorr, "wb") as f:
        pickle.dump(E_uncorr, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.out_pkl_corr, "wb") as f:
        pickle.dump(E_corr, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote 2-D pickles:")
    print(f"  {args.out_pkl_uncorr} (shape {E_uncorr.shape})")
    print(f"  {args.out_pkl_corr} (shape {E_corr.shape})")


    # ----------------------------
    # 6) Summary stats per hour (corrected) + merge with ground truth
    # ----------------------------

    # Build per-row (per target_hour row) summary first
    summary = pd.DataFrame({
        "hour": target_hours,
        "median": np.median(E_corr, axis=1),
        "mean": E_corr.mean(axis=1),
        "std":  E_corr.std(axis=1, ddof=0),
        "p05":  np.quantile(E_corr, 0.05, axis=1),
        "p50":  np.quantile(E_corr, 0.50, axis=1),
        "p95":  np.quantile(E_corr, 0.95, axis=1),
    })

    # If multiple MC rows map to the same hour, collapse them safely.
    # (If mapping is 1:1, this is a no-op.)
    # summary = (
    #     summary.groupby("hour", as_index=False)
    #     .agg(
    #         mean=("mean", "mean"),
    #         std=("std", "mean"),
    #         p05=("p05", "mean"),
    #         p50=("p50", "mean"),
    #         p95=("p95", "mean"),
    #     )
    # )

    # Ensure ground-truth is unique per hour (defensive)
    gt_unique = (
        gt_hourly_energy.groupby("hour", as_index=False)["true_hourly_energy"]
        .sum()
    )

    # Merge and compute error
    summary = summary.merge(gt_unique, on="hour", how="left")
    summary["mean_minus_truth"] = summary["mean"] - summary["true_hourly_energy"]

    summary.to_csv("energy_summary.csv", index=False)
    print("Save to energy_summary.csv")
    
    corr_score = smape(summary["true_hourly_energy"], summary["median"])
    print(f"Corrected energy SMAPE: {corr_score:.2f}%")
    uncorr_score = smape(summary["true_hourly_energy"], E_uncorr.mean(axis=1))
    print(f"Uncorrected energy SMAPE: {uncorr_score:.2f}%")

    original_point_of_estimation_rates_energy = np.mean(p_ori_E_corr, axis=1)
    rate_score = smape(summary["true_hourly_energy"], original_point_of_estimation_rates_energy)
    print(f"Original point-estimation energy rate SMAPE: {rate_score:.2f}%")
    
    refined_point_pred_rate_energy = np.mean(p_ref_E_corr, axis=1)
    rate_score = smape(summary["true_hourly_energy"], refined_point_pred_rate_energy)
    print(f"Refined point-estimation energy rate SMAPE: {rate_score:.2f}%")
    # input("debug")

    # p_ci = np.array(pickle.load(open("gb_t1_1step_predci_14d.pkl", "rb")))
    
    # ci_samples = pickle.load(open("/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/forecast/gb_t1_1step_samples_14d.pkl", "rb"))
    
    # gb_ci_data: CI_data = pickle.load(open("ci_data_LT_14d.pkl", "rb"))
    gb_ci_data: CI_data = pickle.load(open(f"{DATA_PATH}/ci_data_{region}_window_{window_size}_{days}d_{'vertical' if is_vertical else 'horizontal'}_no_bias_correction.pkl", "rb"))
    print(f"[ci] verification code: {gb_ci_data.verification_code}")
    print(f"[ci] path: {DATA_PATH}/ci_data_{region}_window_{window_size}_{days}d_{'vertical' if is_vertical else 'horizontal'}.pkl")
    print(f"[ci] gb_ci_data[\"predicted_ci_list\"].shape: {np.array(gb_ci_data.predicted_ci_list).shape}")
    # input("[ci] loaded gb_ci_data")
    # residuals = gb_ci_data.actual_ci_list - gb_ci_data.predicted_ci_list
    # mean_residuals = np.mean(residuals)
    # print(f"[*ci*] region: {region}, original mean residuals (actual - predicted): {mean_residuals:.6g}")
    # test_lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    # print(f"[*ci*] Ljung-Box test results for original CI residuals:\n{test_lb}\n")
    
    
    # # 1) Original refined residuals
    # refined_residuals = np.asarray(gb_ci_data.actual_ci_list) - np.asarray(gb_ci_data.predicted_refined_list)

    # refined_mean_residuals = float(np.mean(refined_residuals))
    # print(f"[*ci*] region: {region}, refined mean residuals (actual - predicted): {refined_mean_residuals:.6g}")

    # refined_test_lb = acorr_ljungbox(refined_residuals, lags=[10], return_df=True)
    # print(f"[*ci*] Ljung-Box test results for refined CI residuals:\n{refined_test_lb}\n")

    # # 2) Fit AR(p) on residuals
    # y_true = gb_ci_data.actual_ci_list
    # y_pred = gb_ci_data.predicted_refined_list
    
    # e = y_true - y_pred
    
    # print(acorr_ljungbox(e, lags=[1, 2, 24], return_df=True).to_string())
    # # input("debug")
    
    # hour = np.arange(len(e)) % 24
    
    # hour_bias = (
    #     pd.Series(e)
    #     .groupby(hour)
    #     .mean()
    #     .reindex(range(24), fill_value=0.0)
    # )
    
    # y_pred_hourly_adjusted = y_pred + hour_bias.to_numpy()[hour]
    
    # e = np.asarray(y_true) - np.asarray(y_pred_hourly_adjusted)
    
    # print(f"after hourly adjustment: ")
    # print(acorr_ljungbox(e, lags=[1, 2, 24], return_df=True).to_string())

    # hour = np.arange(len(e)) % 24

    # hourly_stats = (
    #     pd.DataFrame({
    #         "residual": e,
    #         "hour": hour,
    #         "predicted": y_pred_hourly_adjusted,
    #     })
    #     .groupby("hour")
    #     .agg(
    #         count=("residual", "count"),
    #         mean=("residual", "mean"),
    #         var=("residual", "var"),      # sample variance
    #         std=("residual", "std"),      # sample std
    #         y_mean=("predicted", "mean"),
    #     )
    # )

    # print(hourly_stats)
    # input("debug")
    # # out = residual_seasonal_ar_correction(
    # #     y_true=y_true,
    # #     y_pred=y_pred,
    # #     lags=(1, 2, 24),       # key change: include 24
    # #     min_train=200,         # increase if you have lots of data
    # #     lb_lags=(1, 2, 24),
    # # )

    # model = SARIMAX(
    #     endog=y_true,
    #     exog=y_pred_hourly_adjusted,
    #     order=(1, 0, 1),              # ARMA errors
    #     seasonal_order=(1, 0, 1, 24), # DAILY seasonal ARMA
    #     enforce_stationarity=False,
    #     enforce_invertibility=False,
    # ).fit(disp=False)

    # resid = model.resid
    
    # # fixed_24_y_pred = 
    

    # print(acorr_ljungbox(resid, lags=[1, 2, 24], return_df=True).to_string())
    
    
    # # r24 = acf(out["e_corrected"], nlags=24, fft=True)[24]
    # # print("[*ci*] ACF at lag 24:", r24)

    # # print("[*ci*] Corrected mean residual:", out["mean_corrected_residual"])
    # # print("[*ci*] Ljung-Box (corrected):")
    # # print(out["lb_table"].to_string())

    # print("[*ci*]" + "=" * 80)
    
    gb_ci_data_length = len(gb_ci_data.predicted_ci_list)
    print(f"[ci] loaded ci data length: {gb_ci_data_length}")
    min_length = min(gb_ci_data_length, rate_data_length)
    gb_ci_data = change_CI_length(gb_ci_data, length=min_length)
    print(f"[ci] ground truth original predicted smape: {gb_ci_data.smape_before:.2f}%")
    print(f"[ci] ground truth refined predicted smape: {gb_ci_data.smape_after:.2f}%")
    # print(f"[ci] mc_100: {gb_ci_data.samples_list[100]}")
    # input("debug")
    
    # print(ci_samples[0])
    # input("[ci] loaded ci samples")

    # gt_ci = pickle.load(open("/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/GB_direct_rolling_t1_eval_actual_ci_14d.pkl", "rb"))
    ci_df = pd.DataFrame(gb_ci_data.actual_ci_list, columns=["true_hourly_ci"])
    # ci_df["ci_samples_mean"] = .mean(axis=1)
    # ci_df["ci_samples_median"] = np.median(gb_ci_data.predicted_refined_list)
    ci_original_predicted_smape = smape(gb_ci_data.actual_ci_list, gb_ci_data.predicted_ci_list)
    ci_refined_predicted_smape = smape(gb_ci_data.actual_ci_list, gb_ci_data.predicted_refined_list)
    print(f"[ci] ci original predicted smape: {ci_original_predicted_smape:.2f}%")
    print(f"[ci] ci refined predicted smape: {ci_refined_predicted_smape:.2f}%")
    # ci_df["predicted_ci"] = gb_ci_data.predicted
    # ci_df.to_csv("ci_stats.csv", index=False)
    # input("debug")

    min_len = min(gb_ci_data.predicted_ci_list.shape[0], E_corr.shape[0])
    
    
    E_corr = E_corr[:min_len, :]
    # point_of_estimation_rates_energy = point_of_estimation_rates_energy[:min_len]

    original_point_carbon_prediction = gb_ci_data.predicted_ci_list * p_ori_E_corr.mean(axis=1)
    refined_point_carbon_prediction = gb_ci_data.predicted_refined_list * p_ref_E_corr.mean(axis=1)

    m = min_len
    n = E_corr.shape[1]
    k = gb_ci_data.samples_list.shape[1]
    # final_carbon_mc_samples = gb_ci_data.samples_list * E_corr
    final_carbon_mc_samples = (gb_ci_data.samples_list[:, :, None] * E_corr[:, None, :]).reshape(m, n * k)
    # final_carbon_samples = ci_samples * E_corr
    # final_carbon_samples = point_of_estimation_rates_energy.reshape(-1, 1) * ci_samples

    final_carbon_stats = pd.DataFrame({
        "mc_mean": np.mean(final_carbon_mc_samples, axis=1),
        "mc_median": np.median(final_carbon_mc_samples, axis=1),
        "std": np.std(final_carbon_mc_samples, axis=1, ddof=0),
        "p05": np.quantile(final_carbon_mc_samples, 0.05, axis=1),
        "p50": np.quantile(final_carbon_mc_samples, 0.50, axis=1),
        "p95": np.quantile(final_carbon_mc_samples, 0.95, axis=1),
    })

    final_carbon_gt = gt_hourly_energy["true_hourly_energy"].to_numpy() * gb_ci_data.actual_ci_list
    original_point_carbon_prediction_smape = smape(final_carbon_gt, original_point_carbon_prediction)
    print(f"[final] point-estimation carbon SMAPE: {original_point_carbon_prediction_smape:.2f}%")
    refined_point_carbon_prediction_smape = smape(final_carbon_gt, refined_point_carbon_prediction)
    print(f"[final] refined point-estimation carbon SMAPE: {refined_point_carbon_prediction_smape:.2f}%")

    final_carbon_stats["true_hourly_carbon"] = final_carbon_gt
    final_carbon_stats["mc_mean_minus_truth"] = final_carbon_stats["mc_mean"] - final_carbon_stats["true_hourly_carbon"]

    final_carbon_stats.to_csv("final_carbon_summary.csv", index=False)
    # print(f"[final] smape: {100.0 * np.mean(np.abs(final_carbon_stats['mc_mean'] - final_carbon_stats['true_hourly_carbon']) / ((np.abs(final_carbon_stats['mc_mean']) + np.abs(final_carbon_stats['true_hourly_carbon'])) / 2.0)):.2f}%")
    mc_score = smape(final_carbon_stats["true_hourly_carbon"], np.median(final_carbon_mc_samples, axis=1))
    print(f"[final] mc smape: {mc_score:.2f}%")
    
    region_result = {
        "hour": target_hours[:min_length],
        "name": region,
        "actual_carbon": final_carbon_gt,
        # "ground_truth_carbon": final_carbon_gt,
        # "original_point_carbon_prediction": original_point_carbon_prediction,
        # "refined_point_carbon_prediction": refined_point_carbon_prediction,
        # "point_estimation_carbon": np.median(final_carbon_mc_samples, axis=1),
        # "point_estimation_carbon": original_point_carbon_prediction,
        "point_estimation_carbon": original_point_carbon_prediction,
        "refined_point_estimation_carbon": refined_point_carbon_prediction,
        "mc_carbon_samples": final_carbon_mc_samples,
    }
    
    return region_result

    # diff = final_energy_stats[final_energy_stats["mc_mean"] != final_energy_stats["point_estimation_carbon"]]
    # print(f"[final] diff count: {len(diff)}")
    # print(diff.head())

def realized_var_cvar_upper_tail(x: np.ndarray, alpha: float = 0.95):
    """
    Realized VaR and CVaR (upper-tail mean) for sample x at confidence level alpha.

    VaR_alpha = empirical alpha-quantile of x
    CVaR_alpha = average of values in x that are >= VaR_alpha
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("Input array is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    var_a = np.quantile(x, alpha, method="linear")  # use interpolation/estimation
    tail_mask = x >= var_a
    if not np.any(tail_mask):
        # Shouldn't happen for typical quantile definitions, but safe fallback
        cvar_a = var_a
        tail_idx = np.array([], dtype=int)
    else:
        cvar_a = x[tail_mask].mean()
        tail_idx = np.flatnonzero(tail_mask)

    return var_a, cvar_a, tail_idx


def cvar_percent_improvement(
    baseline: np.ndarray,
    policy: np.ndarray,
    alpha: float = 0.95,
):
    """
    %ΔCVaR_alpha = 100 * (CVaR_alpha(baseline) - CVaR_alpha(policy)) / CVaR_alpha(baseline)
    Positive means policy improves (reduces tail risk).
    """
    _, cvar_b, _ = realized_var_cvar_upper_tail(baseline, alpha)
    _, cvar_p, _ = realized_var_cvar_upper_tail(policy, alpha)

    if cvar_b == 0:
        raise ZeroDivisionError("Baseline CVaR is 0; percent improvement undefined.")

    pct_improve = 100.0 * (cvar_b - cvar_p) / cvar_b
    return cvar_b, cvar_p, pct_improve



from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


RegionName = str


@dataclass
class AreaState:
    """
    Online Bayesian calibration state for one region.

    Bias model (relative to a chosen central forecast):
        Y = center_forecast + b + eps
        b ~ Normal(mu, s2)
        eps ~ Normal(0, sigma2)

    We update using residual r = Y - center_forecast.
    """
    mu: float = 0.0           # posterior mean of bias
    s2: float = 100.0         # posterior variance of bias (prior uncertainty)
    sigma2: float = 25.0      # observation noise variance (EWMA of residual^2)


def default_switch_cost(prev: Optional[RegionName], new: RegionName, cost_if_switch: float) -> float:
    if prev is None or prev == new:
        return 0.0
    return float(cost_if_switch)


def cvar_upper_tail(samples: np.ndarray, q: float) -> float:
    """
    CVaR_q for the upper tail (worst high-carbon outcomes), q in (0, 1).
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a non-empty 1D array")

    var_q = np.quantile(x, q)
    tail = x[x >= var_q]
    return float(np.mean(tail)) if tail.size else float(np.max(x))


class BiasCalibratedThompsonCVaRRouter:
    """
    Combines:
      (1) Online Bayesian bias calibration per region
      (2) Thompson sampling using MC predictive samples + bias posterior + switching cost
      (3) Tail-risk control via CVaR penalty

    Score minimized each timestep:
      score_i = x*_i + b*_i + switch_cost(prev->i) + risk_aversion * CVaR_q(calibrated_dist_i)

    Where:
      x*_i is a random draw from MC samples (Thompson)
      b*_i is a random draw from bias posterior (Thompson)
      calibrated_dist_i approximates MC samples + bias uncertainty
    """

    def __init__(
        self,
        regions: Iterable[RegionName],
        *,
        prior_mu: float = 0.0,
        prior_s2: float = 100.0,
        init_sigma2: float = 25.0,
        ewma_alpha: float = 0.05,
        discount: float = 1.0,
        switch_cost_fn: Optional[Callable[[Optional[RegionName], RegionName], float]] = None,
        # Risk control:
        cvar_q: float = 0.90,
        risk_aversion: float = 0.0,
        # CVaR sampling controls:
        cvar_num_bias_draws: int = 30,
        cvar_max_mc_used: int = 400,
        # MC sampling control:
        paired_mc_index: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.regions: List[RegionName] = list(regions)
        if len(self.regions) < 2:
            raise ValueError("Provide at least two regions.")

        self.state: Dict[RegionName, AreaState] = {
            r: AreaState(mu=prior_mu, s2=prior_s2, sigma2=init_sigma2) for r in self.regions
        }

        if not (0.0 < ewma_alpha <= 1.0):
            raise ValueError("ewma_alpha must be in (0, 1].")
        self.ewma_alpha = float(ewma_alpha)

        if not (0.0 < discount <= 1.0):
            raise ValueError("discount must be in (0, 1].")
        self.discount = float(discount)

        self.switch_cost_fn = switch_cost_fn or (lambda prev, new: 0.0)

        if not (0.0 < cvar_q < 1.0):
            raise ValueError("cvar_q must be in (0, 1).")
        self.cvar_q = float(cvar_q)
        self.risk_aversion = float(risk_aversion)

        self.cvar_num_bias_draws = int(cvar_num_bias_draws)
        self.cvar_max_mc_used = int(cvar_max_mc_used)

        self.paired_mc_index = bool(paired_mc_index)
        self.rng = rng or np.random.default_rng()

        # Cached per-step for update
        self._last_center_forecast: Dict[RegionName, float] = {}
        self._last_mc_mean: Dict[RegionName, float] = {}

    def decide(
        self,
        mc_samples: Mapping[RegionName, Union[np.ndarray, Sequence[float]]],
        *,
        center_forecast: Optional[Mapping[RegionName, float]] = None,
        prev_choice: Optional[RegionName] = None,
        is_tail_aware: bool = True,
    ) -> Tuple[RegionName, Dict[RegionName, Dict[str, float]]]:
        """
        mc_samples[region] is 1D array-like of MC samples for the next hour.
        center_forecast[region] is the scalar center used for bias calibration residuals.
          - If None, defaults to mean(mc_samples[region]).
          - In your case you should pass point_estimation_carbon[t] for each region.
        """
        self._validate_mc_samples(mc_samples)

        # Cache central forecast used for bias updates
        if center_forecast is None:
            center = {r: float(np.median(np.asarray(mc_samples[r], dtype=float))) for r in self.regions}
        else:
            center = {r: float(center_forecast[r]) for r in self.regions}

        mc_mean = {r: float(np.median(np.asarray(mc_samples[r], dtype=float))) for r in self.regions}
        self._last_center_forecast = center
        self._last_mc_mean = mc_mean

        diagnostics: Dict[RegionName, Dict[str, float]] = {}
        best_region: Optional[RegionName] = None
        best_score = float("inf")

        # Optional: paired MC draw index across regions (useful if MC samples are joint)
        shared_index: Optional[int] = None
        if self.paired_mc_index:
            m_sizes = [np.asarray(mc_samples[r], dtype=float).size for r in self.regions]
            m_min = int(min(m_sizes))
            if m_min < 2:
                raise ValueError("paired_mc_index=True requires each region to have at least 2 MC samples.")
            shared_index = int(self.rng.integers(0, m_min))

        for r in self.regions:
            st = self.state[r]
            X = np.asarray(mc_samples[r], dtype=float)

            # Thompson draw from MC
            if shared_index is not None and shared_index < X.size:
                x_star = float(X[shared_index])
            else:
                x_star = float(X[self.rng.integers(0, X.size)])

            # Thompson draw from bias posterior
            b_star = float(self.rng.normal(loc=st.mu, scale=np.sqrt(max(st.s2, 1e-12))))

            sw = float(self.switch_cost_fn(prev_choice, r))

            cvar_val = 0.0
            if self.risk_aversion != 0.0:
                calibrated = self._calibrated_samples_for_cvar(X, st)
                cvar_val = cvar_upper_tail(calibrated, q=self.cvar_q)

            if not is_tail_aware:
                score = x_star + b_star #+ sw + self.risk_aversion * cvar_val
            else:
                score = x_star + b_star + sw + self.risk_aversion * cvar_val

            diagnostics[r] = {
                "score": float(score),
                "thompson_draw": float(x_star),
                "bias_draw": float(b_star),
                "switch_cost": float(sw),
                "cvar": float(cvar_val),
                "center_forecast": float(center[r]),
                "mc_mean": float(mc_mean[r]),
                "bias_mu": float(st.mu),
                "bias_s2": float(st.s2),
                "sigma2": float(st.sigma2),
            }

            if score < best_score:
                best_score = score
                best_region = r

        assert best_region is not None
        return best_region, diagnostics

    def update(
        self,
        observed_actual: Mapping[RegionName, float],
        *,
        center_forecast_override: Optional[Mapping[RegionName, float]] = None,
    ) -> Dict[RegionName, Dict[str, float]]:
        """
        Update the bias posterior using observed actuals.
        observed_actual may include one region or multiple.
        If you can observe actuals for all regions each timestep, pass them all.
        """
        if center_forecast_override is None:
            if not self._last_center_forecast:
                raise RuntimeError("No cached center_forecast. Call decide() first or pass center_forecast_override.")
            center = {r: float(self._last_center_forecast[r]) for r in observed_actual.keys()}
        else:
            center = {r: float(center_forecast_override[r]) for r in observed_actual.keys()}

        updates: Dict[RegionName, Dict[str, float]] = {}

        for r, y in observed_actual.items():
            if r not in self.state:
                raise ValueError(f"Unknown region '{r}'. Known regions: {self.regions}")

            st = self.state[r]

            # Discounting (forgetting) for non-stationarity: inflate uncertainty before update
            if self.discount < 1.0:
                st.s2 = st.s2 / self.discount

            mu0, s20, sigma2 = st.mu, st.s2, max(st.sigma2, 1e-12)
            rres = float(y - center[r])

            # Conjugate Normal-Normal update
            s2_new = 1.0 / (1.0 / max(s20, 1e-12) + 1.0 / sigma2)
            mu_new = s2_new * (mu0 / max(s20, 1e-12) + rres / sigma2)

            # EWMA update for sigma2
            alpha = self.ewma_alpha
            sigma2_new = (1.0 - alpha) * sigma2 + alpha * (rres * rres)

            st.mu = float(mu_new)
            st.s2 = float(max(s2_new, 1e-12))
            st.sigma2 = float(max(sigma2_new, 1e-12))

            updates[r] = {
                "actual": float(y),
                "center_forecast": float(center[r]),
                "residual": float(rres),
                "mu_old": float(mu0),
                "s2_old": float(s20),
                "sigma2_old": float(sigma2),
                "mu_new": float(st.mu),
                "s2_new": float(st.s2),
                "sigma2_new": float(st.sigma2),
            }

        return updates

    def _validate_mc_samples(self, mc_samples: Mapping[RegionName, Union[np.ndarray, Sequence[float]]]) -> None:
        missing = [r for r in self.regions if r not in mc_samples]
        if missing:
            raise ValueError(f"mc_samples missing regions: {missing}")

        for r in self.regions:
            x = np.asarray(mc_samples[r], dtype=float)
            if x.ndim != 1 or x.size < 2:
                raise ValueError(f"mc_samples['{r}'] must be 1D with length >= 2.")
            if not np.all(np.isfinite(x)):
                raise ValueError(f"mc_samples['{r}'] contains non-finite values.")

    def _calibrated_samples_for_cvar(self, X: np.ndarray, st: AreaState) -> np.ndarray:
        # Subsample MC for efficiency
        X_use = X
        if X.size > self.cvar_max_mc_used:
            idx = self.rng.choice(X.size, size=self.cvar_max_mc_used, replace=False)
            X_use = X[idx]

        K = max(1, int(self.cvar_num_bias_draws))
        # b_draws = np.array([0])
        b_draws = self.rng.normal(loc=st.mu, scale=np.sqrt(max(st.s2, 1e-12)), size=K)
        calibrated = b_draws.reshape(-1, 1) + X_use.reshape(1, -1)
        return calibrated.ravel()


# -------------------------------
# Your data-structure runner
# -------------------------------

def run_routing_on_region_data(
    region_data: List[Dict[str, Any]],
    *,
    switch_cost_if_switch: float = 0.0,
    risk_aversion: float = 0.0,
    cvar_q: float = 0.90,
    paired_mc_index: bool = False,
    prior_s2: float = 100.0,
    init_sigma2: float = 25.0,
    ewma_alpha: float = 0.05,
    discount: float = 1.0,
    seed: Optional[int] = 123,
    verbose: bool = True,
    is_tail_aware: bool = True,
    window_size: int = 1,
    # unbiased_residual_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    region_data: list of region_result dicts, each like:
      {
        "hour": target_hours,                         # length T
        "name": region,                               # str
        "actual_carbon": final_carbon_gt,             # length T
        "point_estimation_carbon": median(samples),   # length T
        "mc_carbon_samples": final_carbon_mc_samples  # shape (T, M)
      }

    For each timestep:
      - Choose a region using Bias-Calibrated Thompson + (optional) CVaR + switching cost
      - Baseline chooses region with lowest point_estimation_carbon at that timestep
      - Report decision and percent increase/decrease of *actual_carbon* vs baseline
    Finally:
      - Report total percent saving/increase vs baseline
    """
    if not region_data:
        raise ValueError("region_data must be a non-empty list.")

    # Parse and validate
    names: List[str] = []
    hours_ref = None
    T_ref: Optional[int] = None

    actual: Dict[str, np.ndarray] = {}
    point: Dict[str, np.ndarray] = {}
    original_point: Dict[str, np.ndarray] = {}
    mc: Dict[str, np.ndarray] = {}

    for rr in region_data:
        if "name" not in rr:
            raise ValueError("Each region_result must include 'name'.")
        rname = str(rr["name"])
        if rname in actual:
            raise ValueError(f"Duplicate region name found: {rname}")
        names.append(rname)

        hours = np.asarray(rr["hour"])
        if hours_ref is None:
            hours_ref = hours
            T_ref = int(hours_ref.shape[0])
        else:
            if hours.shape[0] != T_ref:
                raise ValueError("All regions must have the same number of timesteps (same length of 'hour').")
            # If values differ, you likely need alignment by timestamps; raise for safety.
            if not np.array_equal(hours, hours_ref):
                raise ValueError("All regions must have identical 'hour' arrays. Align timestamps before running.")

        actual_arr = np.asarray(rr["actual_carbon"], dtype=float)
        original_point_arr = np.asarray(rr["point_estimation_carbon"], dtype=float)
        point_arr = np.asarray(rr["refined_point_estimation_carbon"], dtype=float)
        mc_arr = np.asarray(rr["mc_carbon_samples"], dtype=float)

        if actual_arr.shape[0] != T_ref or point_arr.shape[0] != T_ref:
            raise ValueError(f"Region {rname}: 'actual_carbon' and 'point_estimation_carbon' must be length T.")
        if mc_arr.ndim != 2 or mc_arr.shape[0] != T_ref or mc_arr.shape[1] < 2:
            raise ValueError(f"Region {rname}: 'mc_carbon_samples' must have shape (T, M) with M>=2.")
        if not (np.all(np.isfinite(actual_arr)) and np.all(np.isfinite(point_arr)) and np.all(np.isfinite(mc_arr))):
            raise ValueError(f"Region {rname}: arrays contain non-finite values.")

        actual[rname] = actual_arr
        point[rname] = point_arr
        original_point[rname] = original_point_arr
        mc[rname] = mc_arr

    regions = names
    T = int(T_ref)
    hours = hours_ref

    # Router instance
    rng = np.random.default_rng()

    def sw_cost(prev: Optional[str], new: str) -> float:
        return default_switch_cost(prev, new, cost_if_switch=switch_cost_if_switch)

    router = BiasCalibratedThompsonCVaRRouter(
        regions=regions,
        prior_mu=0.0,
        prior_s2=prior_s2,
        init_sigma2=init_sigma2,
        ewma_alpha=ewma_alpha,
        discount=discount,
        switch_cost_fn=sw_cost,
        cvar_q=cvar_q,
        risk_aversion=risk_aversion,
        paired_mc_index=paired_mc_index,
        rng=rng,
    )

    # Run
    prev_choice: Optional[str] = None

    per_step: List[Dict[str, Any]] = []
    algo_total = 0.0
    base_total = 0.0
    algo_choice_emissions = np.array([])
    base_choice_emissions = np.array([])
    
    wrong_decisions = []
    better_decisions = []
    """
    region_result = {
        "hour": target_hours[:min_length],
        "name": region,
        "actual_carbon": final_carbon_gt,
        # "ground_truth_carbon": final_carbon_gt,
        # "original_point_carbon_prediction": original_point_carbon_prediction,
        # "refined_point_carbon_prediction": refined_point_carbon_prediction,
        # "point_estimation_carbon": np.median(final_carbon_mc_samples, axis=1),
        "point_estimation_carbon": original_point_carbon_prediction,
        "mc_carbon_samples": final_carbon_mc_samples
    }
    """
    for t in range(T):
        # Build per-step inputs
        mc_step = {r: mc[r][t, :].astype(float) for r in regions}
        center_step = {r: float(point[r][t]) for r in regions}  # use your provided point estimate as center
        original_center_step = {r: float(original_point[r][t]) for r in regions}

        # Baseline decision: choose region with lowest point estimate
        baseline_choice = min(regions, key=lambda r: original_center_step[r])

        # Algorithm decision
        choice, diag = router.decide(mc_step, center_forecast=center_step, prev_choice=prev_choice, is_tail_aware=is_tail_aware)
        
        # for r in region_data:
        #     unbiased_residual_data[r['name']].append(diag[r['name']]['thompson_draw'] + diag[r['name']]['bias_draw'] - r["actual_carbon"][t])
        
        algo_actual = float(actual[choice][t])
        base_actual = float(actual[baseline_choice][t])

        algo_total += algo_actual
        base_total += base_actual
        
        algo_choice_emissions = np.append(algo_choice_emissions, algo_actual)
        base_choice_emissions = np.append(base_choice_emissions, base_actual)
        

        if base_actual > 0:
            pct_change = (algo_actual - base_actual) / base_actual * 100.0
        else:
            pct_change = float("nan")

        per_step.append(
            {
                "t": t,
                "hour": hours[t],
                "algorithm_choice": choice,
                "baseline_choice": baseline_choice,
                "algorithm_actual_carbon": algo_actual,
                "baseline_actual_carbon": base_actual,
                "pct_change_vs_baseline": pct_change,
                "diagnostics": diag,  # contains per-region scores etc.
            }
        )

        # Update using observed actuals for all regions (fastest learning, since you have ground truth series)
        observed_all = {r: float(actual[r][t]) for r in regions}
        router.update(observed_all, center_forecast_override=center_step)

        prev_choice = choice

        if verbose:
            direction = "decrease" if pct_change < 0 else "increase"
            s = f"[t={t}] hour={hours[t]} " + \
                f"decision={choice} baseline={baseline_choice} optimal={min(regions, key=lambda r: actual[r][t])} " + \
                f"actual(algo)={algo_actual:.6g} actual(base)={base_actual:.6g} " + \
                f"{direction}={pct_change:.3f}%"
                
            # print(
            #     f"[t={t}] hour={hours[t]} "
            #     f"decision={choice} baseline={baseline_choice} optimal={min(regions, key=lambda r: actual[r][t])} "
            #     f"actual(algo)={algo_actual:.6g} actual(base)={base_actual:.6g} "
            #     f"{direction}={pct_change:.3f}%"
            # )
            
            # if round(pct_change, 3) < 0:
            #     better_decisions.append(s)
            # elif round(pct_change, 3) > 0:
            #     wrong_decisions.append(s)
                
    # Final totals
    print(f"=" * 80)        
    print("\nWrong decisions made:")
    for wd in wrong_decisions:
        print(wd)
    print(f"=" * 80)
    print("\nBetter decisions made:")
    for bd in better_decisions:
        print(bd)
    print(f"=" * 80)
    
    if base_total > 0:
        total_pct_change = (algo_total - base_total) / base_total * 100.0
    else:
        total_pct_change = float("nan")

    if verbose:
        direction = "saving" if total_pct_change < 0 else "increase"
        print(f"is_tail_aware={is_tail_aware}, window_size: {window_size}")
        print(
            f"\nTOTAL actual carbon: algo={algo_total:.6g}, baseline={base_total:.6g} "
            f"=> total {direction}={total_pct_change:.3f}%"
        )
        
    alpha = cvar_q
    var_b, cvar_b, idx_b = realized_var_cvar_upper_tail(base_choice_emissions, alpha)
    var_p, cvar_p, idx_p = realized_var_cvar_upper_tail(algo_choice_emissions, alpha)
    cvar_b, cvar_p, pct = cvar_percent_improvement(base_choice_emissions, algo_choice_emissions, alpha)
    print(f"At alpha={alpha:.2f}:")
    print(f"Baseline VaR: {var_b:.6g}")
    print(f"Algorithm VaR: {var_p:.6g}")
    print(f"Baseline CVaR: {cvar_b:.6g}")
    print(f"Algorithm CVaR: {cvar_p:.6g}")
    print(f"Percent improvement: {pct:.3f}%")

    return {
        "per_timestep_report": per_step,
        "total_algorithm_actual_carbon": algo_total,
        "total_baseline_actual_carbon": base_total,
        "total_pct_change_vs_baseline": total_pct_change,
        "final_states": {r: router.state[r] for r in regions},
        "total_savings_pct": total_pct_change,
        "p90_improvement_pct": pct,
    }


def print_actual_carbon_std_var(region_result: Dict[str, Any]) -> None:
    actual = region_result["actual_carbon"]
    std_dev = np.std(actual, ddof=1)
    variance = np.var(actual, ddof=1)
    print(f"Region {region_result['name']} actual carbon std dev: {std_dev:.6g}, variance: {variance:.6g}")
    
def print_predicted_carbon_residual_std_var(region_result: Dict[str, Any]) -> None:
    predicted_residual = region_result["actual_carbon"] - region_result["point_estimation_carbon"]
    std_dev = np.std(predicted_residual, ddof=1)
    variance = np.var(predicted_residual, ddof=1)
    print(f"Region {region_result['name']} predicted_residual carbon std dev: {std_dev:.6g}, variance: {variance:.6g}")
    
def print_unbaised_predicted_carbon_residual_std_var(region_name: str, region_result: list) -> None:
        residuals_array = np.array(region_result)
        std_dev = np.std(residuals_array, ddof=1)
        variance = np.var(residuals_array, ddof=1)
        print(f"Region {region_name} unbiased predicted_residual carbon std dev: {std_dev:.6g}, variance: {variance:.6g}")


if __name__ == "__main__":
    days = 30
    window_size = 7
    
    window_sizes = [1, 3, 7, 10, 24, 72, 96]
    # is_vertical = True
    is_vertical = False
    is_refined = True
    # is_refined = False
    
    total_savings = {}
    total_tail_savings = {}
    
    for window_size in window_sizes:
        total_savings["window_size_" + str(window_size)] = []
        total_tail_savings["window_size_" + str(window_size)] = []
        
        region1_result = main("GB", days, window_size, ".", is_vertical, is_refined)
        region2_result = main("LT", days, window_size, ".", is_vertical, is_refined)
        region3_result = main("CISO", days, window_size, ".", is_vertical, is_refined)
        region4_result = main("NWMT", days, window_size, ".", is_vertical, is_refined)
        # exit()

        region_data = [region1_result, region2_result, region3_result, region4_result]

        residuals = []
        for rr in region_data:
            r = rr["actual_carbon"][:5] - rr["point_estimation_carbon"][:5]
            residuals.append(r)

        all_residuals = np.concatenate(residuals)
        sigma2_init = np.var(all_residuals, ddof=1)
        s2 = sigma2_init / 0.3

        cvar_q = 0.9
        risk_aversion = 0.5
        discount = 0.7
        
        print(f"=" * 80)
        
        # unbiased_residual_data = dict()
        # for rr in region_data:
        #     unbiased_residual_data[rr['name']] = []
        
        
        for i in range(30):
            out = run_routing_on_region_data(
                region_data,
                # switch_cost_if_switch=0.0,
                risk_aversion=risk_aversion,
                cvar_q=cvar_q,
                paired_mc_index=False,
                verbose=True,
                init_sigma2=sigma2_init,
                prior_s2=s2,
                ewma_alpha = 0.03,
                switch_cost_if_switch=0,
                is_tail_aware=True,
                discount=discount,
                window_size=window_size,
                # unbiased_residual_data=unbiased_residual_data
            )
            
            total_savings["window_size_" + str(window_size)].append(out["total_savings_pct"])
            total_tail_savings["window_size_" + str(window_size)].append(out["p90_improvement_pct"])
            
            # print(f"=" * 80)
            # for rr in region_data:
            #     print_actual_carbon_std_var(rr)
            # print(f"=" * 80)
            # for rr in region_data:
            #     print_predicted_carbon_residual_std_var(rr)
            # print(f"=" * 80)
            # for rr in region_data:
            #     print_unbaised_predicted_carbon_residual_std_var(rr['name'], unbiased_residual_data[rr['name']])
            # print(f"=" * 80)
            

            # residuals: 1D array-like

            # for rr in region_data:
            #     # residuals_array = np.array(unbiased_residual_data[rr['name']])
            #     # mean_residual = np.mean(residuals_array)
            #     # print(f"Region {rr['name']} unbiased residuals mean: {mean_residual :.6g}")
            #     # lb_test = acorr_ljungbox(residuals_array, lags=[10], return_df=True)
            #     # print(f"Region {rr['name']} Ljung-Box test results:\n{lb_test}\n")
            #     residuals = rr[""] - rr["point_estimation_carbon"]
            # exit()
            # input("input")
            
            # out = run_routing_on_region_data(
            #     region_data,
            #     # switch_cost_if_switch=0.0,
            #     risk_aversion=risk_aversion,
            #     cvar_q=cvar_q,
            #     paired_mc_index=False,
            #     verbose=True,
            #     init_sigma2=sigma2_init,
            #     prior_s2=s2,
            #     ewma_alpha = 0.03,
            #     switch_cost_if_switch=0,
            #     is_tail_aware=False,
            #     discount=discount,
            #     window_size=window_size
            # )
        
    for window_size in window_sizes:
        print("=" * 80)
        savings_array = np.array(total_savings["window_size_" + str(window_size)])
        tail_savings_array = np.array(total_tail_savings["window_size_" + str(window_size)])
        print(f"Window size: {window_size}, Average Total Savings: {np.mean(savings_array):.3f}%, Std: {np.std(savings_array, ddof=1):.3f}%")
        print(f"Window size: {window_size}, Average Tail Savings: {np.mean(tail_savings_array):.3f}%, Std: {np.std(tail_savings_array, ddof=1):.3f}%")
    