#!/usr/bin/env python3
"""
Trade-off analysis for BO results (price vs energy).

Inputs:
- JSON files named like:
    bo_results_energy_YYYYMMDD_HHMMSS.json
    bo_results_price_YYYYMMDD_HHMMSS.json

Each JSON file contains a list of result dicts sorted best-first.
We use ONLY the first element of each file (the optimal one).

Outputs (all written to ./bo_outputs/):
- Aggregated CSV: bo_tradeoff_aggregated.csv
- Boxplots:
    1) energy_cost by optimizer objective
    2) price_cost by optimizer objective
    3) regret (normalized) by optimizer objective
    4) duration by optimizer objective (optional)
- Scatter plot: price_cost vs energy_cost
"""

import os
import re
import json
import glob
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

INPUT_PATTERN = "bo_results_*.json"
OUTPUT_DIR = "bo_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILENAME_RE = re.compile(
    r"bo_results_(?P<objective>energy|price)_(?P<date>\d{8})_(?P<time>\d{6})\.json$"
)


# =========================
# Utilities
# =========================

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


# =========================
# Data loading
# =========================

def load_optimal_results(
    path_pattern: str,
    require_feasible: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{path_pattern}'")

    for fp in files:
        base = os.path.basename(fp)
        m = FILENAME_RE.match(base)

        objective_from_name = m.group("objective") if m else None
        run_date = m.group("date") if m else None
        run_time = m.group("time") if m else None

        with open(fp, "r") as f:
            data = json.load(f)

        if not data:
            continue

        best = data[0]
        if not isinstance(best, dict):
            continue

        objective = _safe_get(best, "objective_name", objective_from_name)
        feasible = bool(_safe_get(best, "feasible", True))
        if require_feasible and not feasible:
            continue

        price_cost = _safe_get(best, "price_cost")
        energy_cost = _safe_get(best, "energy_cost")

        # Backward compatibility
        if price_cost is None and objective == "price":
            price_cost = _safe_get(best, "cost")
        if energy_cost is None and objective == "energy":
            energy_cost = _safe_get(best, "cost")

        rows.append(
            {
                "file": fp,
                "optimizer_objective": objective,
                "run_date": run_date,
                "run_time": run_time,
                "price_cost": price_cost,
                "energy_cost": energy_cost,
                "duration": _safe_get(best, "duration"),
                "feasible": feasible,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No usable records loaded")

    df["optimizer_objective"] = (
        df["optimizer_objective"].astype(str).str.lower().str.strip()
    )
    df = df[df["optimizer_objective"].isin(["price", "energy"])]

    return df.reset_index(drop=True)


# =========================
# Regret computation
# =========================

def compute_regrets(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    price_best = df[df["optimizer_objective"] == "price"]["price_cost"].min()
    energy_best = df[df["optimizer_objective"] == "energy"]["energy_cost"].min()

    out = df.copy()
    out["price_regret"] = out["price_cost"] / price_best
    out["energy_regret"] = out["energy_cost"] / energy_best

    return out, price_best, energy_best


# =========================
# Plotting helpers
# =========================

def boxplot_by_objective(df, column, title, ylabel, filename):
    plt.figure()
    df.boxplot(column=column, by="optimizer_objective")
    plt.title(title)
    plt.suptitle("")
    plt.xlabel("Optimizer Objective")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def boxplot_two_cols(df, columns, title, ylabel, filename):
    plt.figure()
    df.boxplot(column=columns, by="optimizer_objective")
    plt.title(title)
    plt.suptitle("")
    plt.xlabel("Optimizer Objective")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def scatter_tradeoff(df, filename):
    plt.figure()
    for obj in ["price", "energy"]:
        sub = df[df["optimizer_objective"] == obj]
        plt.scatter(
            sub["price_cost"],
            sub["energy_cost"],
            label=f"optimizer={obj}",
            alpha=0.8,
        )
    plt.xlabel("Price Cost")
    plt.ylabel("Energy Cost")
    plt.title("Price–Energy Trade-off (Optimal per Run)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


# =========================
# Main
# =========================

def main():
    df = load_optimal_results(INPUT_PATTERN)

    for c in ["price_cost", "energy_cost", "duration"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df, best_price, best_energy = compute_regrets(df)
    print(f"len(df): {len(df)}")

    print(f"[INFO] Records loaded: {len(df)}")
    print(f"[INFO] Best price:  {best_price:.6f}")
    print(f"[INFO] Best energy: {best_energy:.6f}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "bo_tradeoff_aggregated.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")

    # Boxplots
    boxplot_by_objective(
        df,
        "energy_cost",
        "Energy Cost by Optimizer Objective",
        "Energy Cost",
        "box_energy_cost.png",
    )

    boxplot_by_objective(
        df,
        "price_cost",
        "Price Cost by Optimizer Objective",
        "Price Cost",
        "box_price_cost.png",
    )

    boxplot_two_cols(
        df,
        ["price_regret", "energy_regret"],
        "Normalized Trade-off (Regret)",
        "Regret (lower is better)",
        "box_regrets.png",
    )

    if df["duration"].notna().any():
        boxplot_by_objective(
            df,
            "duration",
            "Duration by Optimizer Objective",
            "Duration (s)",
            "box_duration.png",
        )

    # Scatter
    scatter_tradeoff(df, "scatter_price_vs_energy.png")


if __name__ == "__main__":
    main()
