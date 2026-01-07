import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Your existing imports
from bo_utils import (
    update_resource_config,
    WORKFLOW_CONFIG,
    read_worker_energy_snapshot,
    invoke_fission_function_sequence,
    stop_locust,
    calc_total_energy_j,
    calc_cost,
    WORKER_ENERGY_URL,
)

DEFAULT_PATH_PREFIX = "best_resource_config_default_"
ENERGY_PATH_PREFIX = "best_resource_config_energy_"


def sample_cost(resource_config: Dict[str, Any]) -> Tuple[float, float, float]:
    update_resource_config(
        functions=WORKFLOW_CONFIG["functions"], resource_config=resource_config
    )

    start = read_worker_energy_snapshot(WORKER_ENERGY_URL)
    e2e_latency_p99, latencies = invoke_fission_function_sequence(
        functions=WORKFLOW_CONFIG["functions"],
        runs=None,
        duration_s=15,
    )
    end = read_worker_energy_snapshot(WORKER_ENERGY_URL)

    energy_j = calc_total_energy_j(start_snap=start, end_snap=end)

    # NOTE: Your "energy_cost" is actually average power-like (J/ns) given the denominator.
    # Consider renaming to avg_power or normalizing to W if desired.
    energy_cost = energy_j / (end["monotonic_ns"] - start["monotonic_ns"])

    money_cost = calc_cost(
        functions=WORKFLOW_CONFIG["functions"],
        resource_config=resource_config,
        latencies=latencies,
    )
    stop_locust()

    return float(e2e_latency_p99), float(energy_cost), float(money_cost)


@dataclass
class ConfigRecord:
    group: str          # "default" or "energy"
    config_name: str    # file stem
    path: Path
    config: Dict[str, Any]


def discover_configs(directory: Path) -> List[ConfigRecord]:
    records: List[ConfigRecord] = []

    for p in sorted(directory.glob("*.json")):
        name = p.name
        if name.startswith(DEFAULT_PATH_PREFIX):
            group = "default"
        elif name.startswith(ENERGY_PATH_PREFIX):
            group = "energy"
        else:
            continue

        with p.open("r") as f:
            cfg = json.load(f)

        records.append(
            ConfigRecord(
                group=group,
                config_name=p.stem,
                path=p,
                config=cfg,
            )
        )
    print(f"Discovered {len(records)} config(s) in {directory}")

    return records


def nondominated_mask(df: pd.DataFrame, cols_minimize: List[str]) -> pd.Series:
    """
    True for Pareto-optimal points (non-dominated) when minimizing all given columns.
    O(n^2) but fine for typical experiment counts.
    """
    vals = df[cols_minimize].to_numpy()
    n = vals.shape[0]
    mask = [True] * n
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if <= in all and < in at least one
            if (vals[j] <= vals[i]).all() and (vals[j] < vals[i]).any():
                mask[i] = False
                break
    return pd.Series(mask, index=df.index)


def run_experiments(
    records: List[ConfigRecord],
    repeats: int,
    cooldown_s: float,
    results_csv: Path,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Runs sample_cost for each config, possibly multiple repeats.
    Persists incremental results to CSV to avoid re-running.
    """
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
    else:
        existing = pd.DataFrame()

    rows = []
    for rec in records:
        for r in range(repeats):
            run_id = f"{rec.config_name}__rep{r}"

            if skip_existing and not existing.empty:
                if (existing["run_id"] == run_id).any():
                    continue

            print(f"Running {run_id} ({rec.group}) from {rec.path} ...")
            t0 = time.time()

            latency_p99, energy_cost, money_cost = sample_cost(rec.config)

            elapsed_s = time.time() - t0

            row = {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "group": rec.group,
                "config_name": rec.config_name,
                "run_id": run_id,
                "latency_p99": latency_p99,
                "energy_cost": energy_cost,
                "money_cost": money_cost,
                "elapsed_s": elapsed_s,
                "path": str(rec.path),
            }

            rows.append(row)

            # cool-down helps reduce thermal / interference effects
            if cooldown_s > 0:
                time.sleep(cooldown_s)

            # incremental persist
            out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            out.to_csv(results_csv, index=False)

    if rows:
        df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = existing

    return df


import math
import pandas as pd

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    metrics = ["latency_p99", "energy_cost", "money_cost"]
    percentiles = [0.10, 0.50, 0.90, 0.95, 0.99]  # adjust if needed

    # Two-sided 90% CI => alpha=0.10, use t-dist if scipy is available
    def tcrit_90(dof: int) -> float:
        if dof <= 0:
            return float("nan")
        try:
            from scipy.stats import t  # type: ignore
            return float(t.ppf(0.95, dof))  # 1 - alpha/2
        except Exception:
            # Normal approximation if scipy is unavailable
            return 1.6448536269514722

    def ci90_halfwidth(x: pd.Series) -> float:
        x = x.dropna()
        n = int(x.shape[0])
        if n < 2:
            return float("nan")
        s = float(x.std(ddof=1))
        return tcrit_90(n - 1) * s / math.sqrt(n)

    agg_spec = {"n": ("latency_p99", "count")}
    for m in metrics:
        agg_spec.update({
            f"{m}_mean": (m, "mean"),
            f"{m}_std": (m, "std"),
            f"{m}_min": (m, "min"),
            f"{m}_max": (m, "max"),
            f"{m}_ci90_hw": (m, ci90_halfwidth),
        })
        for p in percentiles:
            p_label = int(p * 100)
            agg_spec[f"{m}_p{p_label}"] = (m, lambda s, q=p: float(s.quantile(q)))

    # Whole-group aggregates (pooled across ALL configs + repeats in that group)
    grp = data.groupby("group", as_index=False).agg(**agg_spec)

    # CI bounds
    for m in metrics:
        grp[f"{m}_ci90_lo"] = grp[f"{m}_mean"] - grp[f"{m}_ci90_hw"]
        grp[f"{m}_ci90_hi"] = grp[f"{m}_mean"] + grp[f"{m}_ci90_hw"]

    print("\n=== Group-level summary (pooled across all runs) ===")
    print(grp.to_string(index=False))

    return grp



def plot_comparisons(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ok = df.copy()

    # Use per-config means (reduces noise if repeats)
    by_cfg = ok.groupby(["group", "config_name"], as_index=False).agg(
        latency_p99=("latency_p99", "mean"),
        energy_cost=("energy_cost", "mean"),
        money_cost=("money_cost", "mean"),
    )

    # 1) Scatter: money vs latency
    plt.figure()
    for g, sub in by_cfg.groupby("group"):
        plt.scatter(sub["money_cost"], sub["latency_p99"], label=g)
    plt.xlabel("money_cost (your units)")
    plt.ylabel("latency_p99 (s or ms depending on your function)")
    plt.title("Money vs P99 latency (per-config mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "scatter_money_vs_latency.png", dpi=160)
    plt.close()

    # 2) Scatter: energy vs latency
    plt.figure()
    for g, sub in by_cfg.groupby("group"):
        plt.scatter(sub["energy_cost"], sub["latency_p99"], label=g)
    plt.xlabel("energy_cost (J/ns in current code)")
    plt.ylabel("latency_p99")
    plt.title("Energy vs P99 latency (per-config mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "scatter_energy_vs_latency.png", dpi=160)
    plt.close()

    # 3) Scatter: money vs energy
    plt.figure()
    for g, sub in by_cfg.groupby("group"):
        plt.scatter(sub["money_cost"], sub["energy_cost"], label=g)
    plt.xlabel("money_cost")
    plt.ylabel("energy_cost")
    plt.title("Money vs Energy (per-config mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "scatter_money_vs_energy.png", dpi=160)
    plt.close()

    # 4) Pareto fronts (minimize pairs)
    for x, y, cols, fname, title in [
        ("money_cost", "latency_p99", ["money_cost", "latency_p99"],
         "pareto_money_latency.png", "Pareto: money vs latency (minimize both)"),
        ("energy_cost", "latency_p99", ["energy_cost", "latency_p99"],
         "pareto_energy_latency.png", "Pareto: energy vs latency (minimize both)"),
        ("money_cost", "energy_cost", ["money_cost", "energy_cost"],
         "pareto_money_energy.png", "Pareto: money vs energy (minimize both)"),
    ]:
        plt.figure()
        for g, sub in by_cfg.groupby("group"):
            sub = sub.copy()
            nd = nondominated_mask(sub, cols_minimize=cols)
            plt.scatter(sub[x], sub[y], label=f"{g} (all)")
            plt.scatter(sub.loc[nd, x], sub.loc[nd, y], label=f"{g} (pareto)")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=160)
        plt.close()
        
    # Global Pareto across BOTH groups
    x, y, cols = "money_cost", "energy_cost", ["money_cost", "energy_cost"]
    plt.figure()
    nd_global = nondominated_mask(by_cfg, cols_minimize=cols)

    for g, sub in by_cfg.groupby("group"):
        plt.scatter(sub[x], sub[y], label=f"{g} (all)")

    plt.scatter(by_cfg.loc[nd_global, x], by_cfg.loc[nd_global, y],
                label="global (pareto)")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Pareto: money vs energy (global, minimize both)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pareto_money_energy_global.png", dpi=160)
    plt.close()

    # 5) Boxplots by group (distribution across configs)
    # Note: per-config means shown; switch to ok runs if you want per-run distribution.
    for metric, fname in [
        ("latency_p99", "box_latency.png"),
        ("money_cost", "box_money.png"),
        ("energy_cost", "box_energy.png"),
    ]:
        plt.figure()
        data = [by_cfg[by_cfg["group"] == g][metric].dropna().values for g in ["default", "energy"]]
        plt.boxplot(data, labels=["default", "energy"])
        plt.ylabel(metric)
        plt.title(f"{metric} distribution (per-config mean)")
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=160)
        plt.close()

    print(f"\nSaved plots to: {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing json configs")
    ap.add_argument("--repeats", type=int, default=1, help="Repeats per config")
    ap.add_argument("--cooldown-s", type=float, default=0.0, help="Sleep between runs")
    ap.add_argument("--results-csv", default="results.csv", help="CSV path for raw results")
    ap.add_argument("--plots-dir", default="plots", help="Directory for plots output")
    ap.add_argument("--no-skip-existing", action="store_true", help="Do not skip run_ids already in CSV")
    args = ap.parse_args()
    
    stop_locust()

    directory = Path(args.dir)
    if not directory.exists():
        raise FileNotFoundError(directory)

    records = discover_configs(directory)
    if not records:
        raise RuntimeError(
            f"No json files found with prefixes '{DEFAULT_PATH_PREFIX}' or '{ENERGY_PATH_PREFIX}' in {directory}"
        )

    df = run_experiments(
        records=records,
        repeats=args.repeats,
        cooldown_s=args.cooldown_s,
        results_csv=Path(args.results_csv),
        skip_existing=not args.no_skip_existing,
    )

    agg = summarize(df)

    # Persist config-level aggregates for quick inspection
    agg.to_csv("config_level_summary.csv", index=False)
    print("\nWrote config_level_summary.csv")

    plot_comparisons(df, outdir=Path(args.plots_dir))


if __name__ == "__main__":
    main()

"""
python check_result.py --dir . --no-skip-existing
"""