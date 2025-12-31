#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy import stats


def compute_power_stats(csv_file: str) -> None:
    # Load CSV
    df = pd.read_csv(csv_file)

    if "power_total_w" not in df.columns:
        raise ValueError("CSV must contain a 'power_total_w' column")

    power = df["power_total_w"].dropna().to_numpy()

    if len(power) < 2:
        raise ValueError("Not enough data points to compute statistics")

    n = len(power)

    # Basic statistics
    mean_val = np.mean(power)
    median_val = np.median(power)
    std_val = np.std(power, ddof=1)  # sample standard deviation

    # Percentiles
    percentiles = {
        "p5": np.percentile(power, 5),
        "p25": np.percentile(power, 25),
        "p50": np.percentile(power, 50),
        "p75": np.percentile(power, 75),
        "p95": np.percentile(power, 95),
    }

    # 90% confidence interval for the mean
    confidence_level = 0.90
    alpha = 1.0 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * std_val / np.sqrt(n)
    ci_low = mean_val - margin
    ci_high = mean_val + margin

    # Output
    print("Power Statistics (power_total_w)")
    print("=" * 40)
    print(f"Count                : {n}")
    print(f"Mean                 : {mean_val:.4f}")
    print(f"Median               : {median_val:.4f}")
    print((f"Minimum              : {power.min():.4f}"))
    print(f"Maximum              : {power.max():.4f}")
    print(f"Standard Deviation   : {std_val:.4f}")
    print()
    print("Percentiles:")
    for k, v in percentiles.items():
        print(f"  {k:>4}              : {v:.4f}")
    print()
    print("90% Confidence Interval for Mean:")
    print(f"  [{ci_low:.4f}, {ci_high:.4f}]")

def plot_power_distribution_graph(csv_file: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Load CSV
    df = pd.read_csv(csv_file)
    values = df["power_total_w"].dropna()

    # Histogram settings
    bins = 50  # adjust as needed
    counts, bin_edges = np.histogram(values, bins=bins)
    bin_width = bin_edges[1] - bin_edges[0]

    # Create figure and axes
    fig, ax1 = plt.subplots()

    # Plot histogram on ax2 (right Y-axis)
    ax2 = ax1.twinx()
    bars = ax2.hist(values, bins=bin_edges, alpha=0.7, label=f"Bucket size: {bin_width:.2f}")

    # Right Y-axis (count)
    ax2.set_ylabel("Count")

    # Left Y-axis (percentage)
    total = len(values)
    percentages = counts / total * 100
    ax1.set_ylabel("Percentage")
    ax1.set_ylim(0, percentages.max() * 1.1)

    # Convert left Y-axis ticks to percentage format
    yticks = ax1.get_yticks()
    ax1.set_yticklabels([f"{yt:.0f}%" for yt in yticks])

    # X-axis label
    ax1.set_xlabel("Power Total (W)")

    # Title and legend
    ax1.set_title("Distribution of Power Total (Percentage & Count)")
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # plt.show()
    plt.savefig("power_distribution.png", dpi=300)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python power_stats.py <input.csv>")
        sys.exit(1)

    compute_power_stats(sys.argv[1])
    plot_power_distribution_graph(sys.argv[1])