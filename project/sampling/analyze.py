import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
NAME = "zinput3"
INPUT_CSV = f"./power_logs/{NAME}-rps-aggregated.csv"
OUTPUT_CSV = f"{NAME}-rps-power-summary.csv"
OUTPUT_PLOT = f"{NAME}_rps_vs_power.png"
OUTPUT_LOG = f"{NAME}_power_stats.log"

# Percentiles to report
PCTS = [5, 10, 25, 50, 75, 90, 95]

# 90% CI z-score (normal approximation)
Z_90 = 1.645


def safe_std(x: pd.Series) -> float:
    """Sample standard deviation; returns 0.0 for n<2."""
    n = x.size
    if n < 2:
        return 0.0
    return float(x.std(ddof=1))


def mean_ci_90(x: pd.Series) -> tuple[float, float]:
    """
    90% confidence interval for the mean using normal approximation:
      mean ± z * (std / sqrt(n))
    Uses sample std (ddof=1). For n<2, returns (mean, mean).
    """
    n = x.size
    mu = float(x.mean()) if n else float("nan")
    if n < 2:
        return mu, mu

    s = float(x.std(ddof=1))
    se = s / math.sqrt(n)
    half = Z_90 * se
    return mu - half, mu + half


def format_group_stats(row: dict) -> str:
    """Format a stats dict into a readable single-line log entry."""
    pct_str = ", ".join([f"p{p}={row[f'p{p}']:.4f}" for p in PCTS])
    return (
        f"rps={row['rps']}, hostname={row['hostname']}, n={row['n']}, "
        f"mean={row['mean']:.4f}, median={row['median']:.4f}, std={row['std']:.4f}, "
        f"{pct_str}, "
        f"ci90_mean=[{row['ci90_low']:.4f}, {row['ci90_high']:.4f}]"
    )


def main() -> None:
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"hostname", "rps", "input_size", "power_total_w"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric
    df["rps"] = pd.to_numeric(df["rps"], errors="coerce")
    df["power_total_w"] = pd.to_numeric(df["power_total_w"], errors="coerce")
    df = df.dropna(subset=["rps", "hostname", "power_total_w"])

    # -----------------------------
    # (A) Per-(rps, hostname) stats for power_total_w -> LOG
    # -----------------------------
    def pct_agg(p: int):
        return lambda s: float(np.percentile(s.to_numpy(), p)) if s.size else float("nan")

    agg_dict = {
        "n": ("power_total_w", "size"),
        "mean": ("power_total_w", "mean"),
        "median": ("power_total_w", "median"),
        "std": ("power_total_w", safe_std),
    }
    for p in PCTS:
        agg_dict[f"p{p}"] = ("power_total_w", pct_agg(p))

    stats = (
        df.groupby(["rps", "hostname"], as_index=False)
          .agg(**agg_dict)
    )

    # CI90 for the mean per group (computed from raw samples to be safe)
    ci_rows = []
    for (rps, hostname), g in df.groupby(["rps", "hostname"]):
        lo, hi = mean_ci_90(g["power_total_w"])
        ci_rows.append({"rps": rps, "hostname": hostname, "ci90_low": lo, "ci90_high": hi})

    ci_df = pd.DataFrame(ci_rows)
    stats = stats.merge(ci_df, on=["rps", "hostname"], how="left")

    # Write log file
    log_path = Path(OUTPUT_LOG)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"power_total_w stats by (rps, hostname)\n")
        f.write(f"source_csv={INPUT_CSV}\n")
        f.write(f"generated_at={datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"percentiles={PCTS}\n")
        f.write(f"ci=90% normal approx (z={Z_90})\n")
        f.write("-" * 80 + "\n")

        # Sorted for readability
        stats_sorted = stats.sort_values(["rps", "hostname"])
        for _, row in stats_sorted.iterrows():
            f.write(format_group_stats(row.to_dict()) + "\n")

    # -----------------------------
    # (B) Host-average per (rps, hostname), then sum across hosts per rps -> CSV
    # -----------------------------
    host_avg = (
        df.groupby(["rps", "hostname"], as_index=False)
          .agg(avg_power_total_w=("power_total_w", "mean"))
    )

    rps_agg = (
        host_avg.groupby("rps")
        .agg(
            sum_avg_power_total_w=("avg_power_total_w", "sum"),
            std_avg_power_total_w=("avg_power_total_w", "std"),
            host_count=("avg_power_total_w", "count"),
        )
        .reset_index()
    )

    input_size_map = (
        df.groupby("rps", as_index=False)
          .agg(input_size=("input_size", "first"))
    )

    result = rps_agg.merge(input_size_map, on="rps", how="left")
    result = result[
        ["rps", "input_size", "sum_avg_power_total_w", "std_avg_power_total_w", "host_count"]
    ].sort_values("rps")

    result.to_csv(OUTPUT_CSV, index=False)

    # -----------------------------
    # (C) Plot with error bars and line of fit
    # -----------------------------
    x = result["rps"].to_numpy()
    y = result["sum_avg_power_total_w"].to_numpy()
    yerr = result["std_avg_power_total_w"].fillna(0.0).to_numpy()

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="Sum of host avg power")

    # Linear regression (line of fit)
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = poly(x_fit)
        plt.plot(x_fit, y_fit, linestyle="--", label="Linear fit")

    plt.xlabel("RPS")
    plt.ylabel("Sum Avg Power Total (W)")
    plt.title("RPS vs Sum of Average Host Power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()

    print(f"Aggregated CSV written to: {OUTPUT_CSV}")
    print(f"Plot saved to: {OUTPUT_PLOT}")
    print(f"Stats log written to: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
