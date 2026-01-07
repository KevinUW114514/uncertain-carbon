import json
import glob
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================
# Configuration
# ============================
INPUT_DIR = "."                   # folder containing your json files
FILE_GLOB = "bo_results_*_*.json" # matches bo_results_energy_...json and bo_results_price_...json
OUTPUT_DIR = "bo_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================
# I/O and utilities
# ============================
def load_records_from_file(path: str) -> List[Dict[str, Any]]:
    """Load one JSON file (top-level array of dicts). Add filename and row index."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected top-level JSON array")

    out = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            row = dict(item)
            row["_filename"] = os.path.basename(path)
            row["_row_in_file"] = i
            out.append(row)
    return out


def percentile_series(s: pd.Series, q: float) -> float:
    x = s.dropna().to_numpy(dtype=float)
    return float(np.percentile(x, q)) if x.size else np.nan


def compute_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Stats per column. ci90 uses normal approx: mean +/- 1.645 * std/sqrt(n).
    Returns rows=metrics, cols=variables in `cols`.
    """
    metrics = [
        "n", "mean", "std",
        "p50", "p75", "p90", "p95", "p99",
        "ci90_low", "ci90_high",
        "min", "max",
    ]
    out = pd.DataFrame(index=metrics, columns=cols, dtype=float)

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        n = int(x.shape[0])
        if n == 0:
            continue

        mean = float(x.mean())
        std = float(x.std(ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else np.nan
        z = 1.645
        out.loc["n", c] = n
        out.loc["mean", c] = mean
        out.loc["std", c] = std
        out.loc["p50", c] = percentile_series(x, 50)
        out.loc["p75", c] = percentile_series(x, 75)
        out.loc["p90", c] = percentile_series(x, 90)
        out.loc["p95", c] = percentile_series(x, 95)
        out.loc["p99", c] = percentile_series(x, 99)
        out.loc["ci90_low", c] = mean - z * se
        out.loc["ci90_high", c] = mean + z * se
        out.loc["min", c] = float(x.min())
        out.loc["max", c] = float(x.max())

    return out


def first_row_per_file(df: pd.DataFrame) -> pd.DataFrame:
    """Return first row (by _row_in_file) for each file."""
    if df.empty:
        return df.copy()
    return (
        df.sort_values(["_filename", "_row_in_file"])
          .groupby("_filename", as_index=False)
          .head(1)
          .reset_index(drop=True)
    )


def scatter_plot(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_path: str) -> None:
    plt.figure()
    plt.scatter(pd.to_numeric(df[xcol], errors="coerce"),
                pd.to_numeric(df[ycol], errors="coerce"),
                alpha=0.6)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================
# Load all files
# ============================
paths = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
if not paths:
    raise FileNotFoundError(f"No files matched: {os.path.join(INPUT_DIR, FILE_GLOB)}")

records: List[Dict[str, Any]] = []
for p in paths:
    records.extend(load_records_from_file(p))

df = pd.DataFrame(records)

required = ["objective_name", "price_cost", "energy_cost"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["objective_name"] = df["objective_name"].astype(str)

df_price = df[df["objective_name"] == "price"].copy()
df_energy = df[df["objective_name"] == "energy"].copy()

cols_for_stats = ["price_cost", "energy_cost"]


# ============================
# Graph 1: energy group (all points)
# ============================
g1_path = os.path.join(OUTPUT_DIR, "scatter_energy_group_all.png")
if not df_energy.empty:
    scatter_plot(
        df_energy, "price_cost", "energy_cost",
        f"Energy objective: price_cost vs energy_cost (all points, n={len(df_energy)})",
        g1_path
    )
else:
    print("Warning: no rows found for objective_name == 'energy'")


# ============================
# Graph 2: price group (all points)
# ============================
g2_path = os.path.join(OUTPUT_DIR, "scatter_price_group_all.png")
if not df_price.empty:
    scatter_plot(
        df_price, "price_cost", "energy_cost",
        f"Price objective: price_cost vs energy_cost (all points, n={len(df_price)})",
        g2_path
    )
else:
    print("Warning: no rows found for objective_name == 'price'")


# ============================
# Graph 3: overlay FIRST row per file (two colors, NOT paired)
# ============================
price_first = first_row_per_file(df_price)
energy_first = first_row_per_file(df_energy)

g3_path = os.path.join(OUTPUT_DIR, "scatter_first_overlay.png")
plt.figure()

if not price_first.empty:
    plt.scatter(
        pd.to_numeric(price_first["price_cost"], errors="coerce"),
        pd.to_numeric(price_first["energy_cost"], errors="coerce"),
        alpha=0.7,
        label=f"price objective (first per file, n={len(price_first)})"
    )
else:
    print("Warning: no first rows found for price objective files")

if not energy_first.empty:
    plt.scatter(
        pd.to_numeric(energy_first["price_cost"], errors="coerce"),
        pd.to_numeric(energy_first["energy_cost"], errors="coerce"),
        alpha=0.7,
        label=f"energy objective (first per file, n={len(energy_first)})"
    )
else:
    print("Warning: no first rows found for energy objective files")

plt.xlabel("price_cost")
plt.ylabel("energy_cost")
plt.title("First row per file: overlay (price objective vs energy objective)")
plt.legend()
plt.tight_layout()
plt.savefig(g3_path, dpi=200)
plt.close()


# ============================
# Stats CSV 1: energy group (all points)
# ============================
s1 = compute_stats(df_energy, cols_for_stats)
s1_path = os.path.join(OUTPUT_DIR, "stats_energy_group_all.csv")
s1.to_csv(s1_path, index=True)


# ============================
# Stats CSV 2: price group (all points)
# ============================
s2 = compute_stats(df_price, cols_for_stats)
s2_path = os.path.join(OUTPUT_DIR, "stats_price_group_all.csv")
s2.to_csv(s2_path, index=True)


# ============================
# Stats CSV 3: overlay FIRST row per file (both objectives in one file)
# Columns are flattened to keep CSV simple.
# ============================
s3_price_first = compute_stats(price_first, cols_for_stats).add_prefix("price_first__")
s3_energy_first = compute_stats(energy_first, cols_for_stats).add_prefix("energy_first__")
s3 = pd.concat([s3_price_first, s3_energy_first], axis=1)

s3_path = os.path.join(OUTPUT_DIR, "stats_first_overlay.csv")
s3.to_csv(s3_path, index=True)


print("Done. Outputs in:", OUTPUT_DIR)
print("Graphs:")
print(" -", g1_path)
print(" -", g2_path)
print(" -", g3_path)
print("Stats CSVs:")
print(" -", s1_path)
print(" -", s2_path)
print(" -", s3_path)
