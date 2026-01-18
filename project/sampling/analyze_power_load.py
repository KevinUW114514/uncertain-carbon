#!/usr/bin/env python3
"""
Analyze a power profiling CSV and produce:
1) Summary stats (mean, median, std, percentiles, 90% CI) -> log file
2) A binned trend plot (10 groups) with error bars -> PNG image

Usage:
  python3 analyze_power_profile.py --csv ./power_logs/power_profile_YYYYmmdd_HHMMSS.csv --out-dir ./power_logs
"""

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_dual_plot(df: pd.DataFrame, load_name: str, out_path: Path) -> None:
    """
    Creates a figure with two subplots:
      (1) Power trend over time (10 bins) with 90% CI error bars
      (2) Power vs CPU usage_pct (10 bins) with 90% CI error bars

    Saves to out_path as PNG.
    """
    required = {"power_total_w"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

    # -----------------------------
    # Subplot 1: Time -> Power
    # -----------------------------
    if "elapsed_s" in df.columns:
        x_time = pd.to_numeric(df["elapsed_s"], errors="coerce")
        x_time_label = "elapsed_s"
    elif "sample_index" in df.columns:
        x_time = pd.to_numeric(df["sample_index"], errors="coerce")
        x_time_label = "sample_index"
    else:
        x_time = pd.Series(np.arange(len(df), dtype=float))
        x_time_label = "index"

    y_power = pd.to_numeric(df["power_total_w"], errors="coerce")

    tdf = pd.DataFrame({"x": x_time, "y": y_power}).dropna()
    if tdf.empty:
        raise RuntimeError("No valid numeric samples found for plotting power_total_w over time.")

    tdf = tdf.sort_values("x").reset_index(drop=True)
    tdf["bin"] = pd.qcut(tdf.index, q=10, labels=False, duplicates="drop")

    t_xs, t_means, t_lo, t_hi = [], [], [], []
    for _, g in tdf.groupby("bin", as_index=False):
        x_center = float(g["x"].mean())
        vals = g["y"].to_numpy(dtype=float)
        m, lo, hi = mean_ci_90(vals)
        if np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi):
            t_xs.append(x_center); t_means.append(m); t_lo.append(lo); t_hi.append(hi)

    if len(t_xs) < 2:
        raise RuntimeError("Not enough valid time bins to plot (need at least 2).")

    t_xs = np.array(t_xs, dtype=float)
    t_means = np.array(t_means, dtype=float)
    t_lo = np.array(t_lo, dtype=float)
    t_hi = np.array(t_hi, dtype=float)

    order = np.argsort(t_xs)
    t_xs, t_means, t_lo, t_hi = t_xs[order], t_means[order], t_lo[order], t_hi[order]
    t_yerr = np.vstack([t_means - t_lo, t_hi - t_means])

    # -----------------------------
    # Subplot 2: CPU% -> Power
    # -----------------------------
    if "cpu_usage_pct" not in df.columns:
        raise RuntimeError("CSV missing required column for CPU->Power plot: cpu_usage_pct")

    x_cpu = pd.to_numeric(df["cpu_usage_pct"], errors="coerce")
    cdf = pd.DataFrame({"x": x_cpu, "y": y_power}).dropna()

    # Filter invalid CPU ranges if any
    cdf = cdf[(cdf["x"] >= 0) & (cdf["x"] <= 100)]
    if cdf.empty:
        raise RuntimeError("No valid numeric samples found for CPU->Power plot.")

    # Bin by CPU value (equal-count bins across the observed CPU range)
    # Note: if CPU values are constant (e.g., idle), qcut can collapse bins; duplicates="drop" handles it.
    cdf["bin"] = pd.qcut(cdf["x"], q=10, labels=False, duplicates="drop")

    c_xs, c_means, c_lo, c_hi = [], [], [], []
    for _, g in cdf.groupby("bin", as_index=False):
        x_center = float(g["x"].mean())
        vals = g["y"].to_numpy(dtype=float)
        m, lo, hi = mean_ci_90(vals)
        if np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi):
            c_xs.append(x_center); c_means.append(m); c_lo.append(lo); c_hi.append(hi)

    if len(c_xs) < 2:
        # This can happen for very steady CPU (e.g., perfectly idle).
        # We still plot a scatter so you get something useful.
        c_xs = cdf["x"].to_numpy(dtype=float)
        c_means = cdf["y"].to_numpy(dtype=float)
        c_lo = None
        c_hi = None

    # If we have binned stats, order by CPU
    if c_lo is not None:
        c_xs = np.array(c_xs, dtype=float)
        c_means = np.array(c_means, dtype=float)
        c_lo = np.array(c_lo, dtype=float)
        c_hi = np.array(c_hi, dtype=float)
        order = np.argsort(c_xs)
        c_xs, c_means, c_lo, c_hi = c_xs[order], c_means[order], c_lo[order], c_hi[order]
        c_yerr = np.vstack([c_means - c_lo, c_hi - c_means])

    # -----------------------------
    # Render figure (two subplots)
    # -----------------------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)

    # Left: time trend with capped error bars (cleaner than default)
    axes[0].errorbar(t_xs, t_means, yerr=t_yerr, fmt="-o", capsize=4, elinewidth=1, alpha=0.85)
    axes[0].set_title(f"Power Trend (10 bins) - {load_name}")
    axes[0].set_xlabel(x_time_label)
    axes[0].set_ylabel("power_total_w (W)")
    axes[0].grid(True, alpha=0.3)

    # Right: CPU% -> Power with error bars (if available) and trend line
    if c_lo is not None:
        axes[1].errorbar(c_xs, c_means, yerr=c_yerr, fmt="-o", capsize=4, elinewidth=1, alpha=0.85)
    else:
        # fallback scatter only (e.g., CPU nearly constant)
        axes[1].scatter(c_xs, c_means, s=15, alpha=0.6)

    axes[1].set_title(f"Power vs CPU Usage (10 bins) - {load_name}")
    axes[1].set_xlabel("cpu_usage_pct")
    axes[1].set_ylabel("power_total_w (W)")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze RAPL power profile CSV.")
    p.add_argument(
        "--csv",
        required=False,
        help="Path to CSV file. If omitted, the latest CSV in --out-dir is used."
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory containing CSV files and where outputs will be written."
    )
    return p.parse_args()


def find_latest_csv(directory: Path) -> Path:
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _t_critical_90(df: int) -> float:
    """
    Returns two-sided 90% t critical value for given degrees of freedom.
    Falls back to normal approx if SciPy isn't available.
    """
    # Two-sided 90% => alpha=0.10 => quantile = 1 - alpha/2 = 0.95
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.95, df))
    except Exception:
        # Normal approximation (reasonable for moderate/large n)
        # z(0.95) ≈ 1.64485
        return 1.6448536269514722


def mean_ci_90(x: np.ndarray) -> Tuple[float, float, float]:
    """
    90% CI for the mean: mean ± t_crit * (s / sqrt(n))
    Returns (mean, lo, hi). If insufficient data, returns NaNs.
    """
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        m = float(np.nan) if n == 0 else float(x.mean())
        return m, float("nan"), float("nan")

    m = float(x.mean())
    s = float(x.std(ddof=1))
    tcrit = _t_critical_90(n - 1)
    half = tcrit * (s / math.sqrt(n))
    return m, m - half, m + half


def compute_stats(series: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    out: Dict[str, float] = {}

    out["n"] = float(x.size)
    out["mean"] = float(np.mean(x)) if x.size else float("nan")
    out["median"] = float(np.median(x)) if x.size else float("nan")
    out["std"] = float(np.std(x, ddof=1)) if x.size >= 2 else float("nan")

    # Percentiles (customize as needed)
    for p in [5, 25, 50, 75, 95]:
        out[f"p{p}"] = float(np.percentile(x, p)) if x.size else float("nan")

    m, lo, hi = mean_ci_90(x)
    out["ci90_mean_lo"] = lo
    out["ci90_mean_hi"] = hi

    return out


def write_stats_log(
    out_path: Path,
    load_name: str,
    stats_main: Dict[str, float],
    per_source_stats: Dict[str, Dict[str, float]],
) -> None:
    lines: List[str] = []
    lines.append(f"Power Profile Analysis Log")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Load name: {load_name}")
    lines.append("")

    def fmt(k: str, v: float) -> str:
        if k == "n":
            return f"{int(v)}"
        if v != v:  # NaN
            return "NaN"
        return f"{v:.6f}"

    lines.append("[power_total_w]")
    for k in ["n", "mean", "median", "std", "p5", "p25", "p50", "p75", "p95", "ci90_mean_lo", "ci90_mean_hi"]:
        lines.append(f"{k}: {fmt(k, stats_main.get(k, float('nan')))}")
    lines.append("")

    if per_source_stats:
        lines.append("[per_source_rapl_stats]")
        # Keep deterministic ordering
        for col in sorted(per_source_stats.keys()):
            lines.append(f"\n[{col}]")
            s = per_source_stats[col]
            for k in ["n", "mean", "median", "std", "p5", "p25", "p50", "p75", "p95", "ci90_mean_lo", "ci90_mean_hi"]:
                lines.append(f"{k}: {fmt(k, s.get(k, float('nan')))}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_binned_plot(df: pd.DataFrame, load_name: str, out_path: Path) -> None:
    """
    Group into 10 bins along time order and plot:
      - mean trend line
      - shaded 90% CI band (cleaner than large vertical error bars)
      - optional small capped error bars
    """
    if "power_total_w" not in df.columns:
        raise RuntimeError("CSV does not contain required column: power_total_w")

    # Choose x-axis
    if "elapsed_s" in df.columns:
        x_raw = pd.to_numeric(df["elapsed_s"], errors="coerce")
        x_label = "elapsed_s"
    elif "sample_index" in df.columns:
        x_raw = pd.to_numeric(df["sample_index"], errors="coerce")
        x_label = "sample_index"
    else:
        x_raw = pd.Series(np.arange(len(df), dtype=float))
        x_label = "index"

    y_raw = pd.to_numeric(df["power_total_w"], errors="coerce")

    plot_df = pd.DataFrame({"x": x_raw, "y": y_raw}).dropna()
    if plot_df.empty:
        raise RuntimeError("No valid numeric samples found for plotting power_total_w.")

    # Sort by time and bin into 10 equal-count bins
    plot_df = plot_df.sort_values("x").reset_index(drop=True)
    plot_df["bin"] = pd.qcut(plot_df.index, q=10, labels=False, duplicates="drop")

    xs, means, lo_s, hi_s = [], [], [], []

    for _, g in plot_df.groupby("bin", as_index=False):
        x_center = float(g["x"].mean())
        vals = g["y"].to_numpy(dtype=float)
        m, lo, hi = mean_ci_90(vals)

        # Skip bins that don't have a valid CI (e.g., too few samples)
        if not (np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi)):
            continue

        xs.append(x_center)
        means.append(m)
        lo_s.append(lo)
        hi_s.append(hi)

    if len(xs) < 2:
        raise RuntimeError("Not enough valid bins to plot (need at least 2).")

    xs_arr = np.array(xs, dtype=float)
    means_arr = np.array(means, dtype=float)
    lo_arr = np.array(lo_s, dtype=float)
    hi_arr = np.array(hi_s, dtype=float)

    # Ensure increasing x for a clean line and band
    order = np.argsort(xs_arr)
    xs_arr = xs_arr[order]
    means_arr = means_arr[order]
    lo_arr = lo_arr[order]
    hi_arr = hi_arr[order]

    plt.figure()

    # Clean uncertainty visualization: shaded CI band
    plt.fill_between(xs_arr, lo_arr, hi_arr, alpha=0.2)

    # Trend line + markers
    plt.plot(xs_arr, means_arr, marker="o", linewidth=2)

    # Optional: subtle capped error bars (small, not dominant)
    # Comment this block out if you only want the CI band.
    yerr = np.vstack([means_arr - lo_arr, hi_arr - means_arr])
    plt.errorbar(
        xs_arr, means_arr, yerr=yerr,
        fmt="none", capsize=4, elinewidth=1, alpha=0.6
    )

    plt.title(f"Power Trend (10 bins) - {load_name}")
    plt.xlabel(x_label)
    plt.ylabel("power_total_w (W)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
    else:
        csv_path = find_latest_csv(out_dir)
        print(f"[INFO] --csv not provided, using latest CSV: {csv_path.name}")
    
    ensure_out_dir(out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Load name handling
    load_name = "unknown"
    if "load_name" in df.columns and not df["load_name"].dropna().empty:
        load_name = str(df["load_name"].dropna().iloc[0])

    # Main stats on total power
    if "power_total_w" not in df.columns:
        raise RuntimeError("CSV must contain column: power_total_w")

    stats_total = compute_stats(df["power_total_w"])

    # Optional: compute stats for each RAPL source column as well
    rapl_cols = [c for c in df.columns if c.startswith("rapl_") and c.endswith("_w")]
    per_source = {c: compute_stats(df[c]) for c in rapl_cols}

    stem = csv_path.stem
    log_path = out_dir / f"{stem}_stats.log"
    plot_path = out_dir / f"{stem}_trend.png"

    write_stats_log(log_path, load_name, stats_total, per_source)
    make_binned_plot(df, load_name, plot_path)
    # make_dual_plot(df, load_name, plot_path)

    print(f"Wrote stats log: {log_path}")
    print(f"Wrote plot image: {plot_path}")


if __name__ == "__main__":
    main()

"""
python analyze_power_load.py \
  --csv ./power_logs/intel-manager_idle_20251217_055029.csv \
  --out-dir ./power_logs
  
python analyze_power_load.py \
  --out-dir ./power_logs
"""