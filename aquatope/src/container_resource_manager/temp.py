# import json

# resource_config = [{'cpu_m': 355, 'memory_mi': 70}, {'cpu_m': 5410, 'memory_mi': 512}]
# with open("best_resource_config_default_1.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 62}, {'cpu_m': 4708, 'memory_mi': 418}]
# with open("best_resource_config_default_2.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 70}, {'cpu_m': 4935, 'memory_mi': 509}]
# with open("best_resource_config_default_3.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 70}, {'cpu_m': 5481, 'memory_mi': 486}]
# with open("best_resource_config_energy_1.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 50}, {'cpu_m': 6000, 'memory_mi': 400}]
# with open("best_resource_config_energy_2.json", "w") as f:
#     json.dump(resource_config, f, indent=2)    
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 64}, {'cpu_m': 6000, 'memory_mi': 400}]
# with open("best_resource_config_energy_3.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
"PYTHONPATH=/home/cc/uncertain-carbon/aquatope:$PYTHONPATH"
# import locust
# import pandas as pd
# from fissionlib.cli import LOCUST_CSV, start_locust, stop_locust
# import time

# # start_locust()
# # time.sleep(10)  # wait for locust to start
# # stop_locust()

# start = 1767744712
# locust_csv_df = pd.read_csv(f"{LOCUST_CSV}_stats_history.csv")
# start_requests = locust_csv_df.loc[
#     locust_csv_df["Timestamp"] == start,
#     "Total Request Count"
# ].iloc[0]
# print(type(start_requests))   # pandas Series
# print(start_requests)
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

PCTS = [50, 90, 95, 99]


def read_first_cost(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError(f"{path}: expected a non-empty JSON list.")

    first = obj[0]
    if not isinstance(first, dict) or "energy_cost" not in first:
        raise ValueError(f"{path}: first element must be a dict with key 'energy_cost'.")

    return float(first["energy_cost"])


def find_files(directory: str, prefix: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, f"{prefix}*.json")))


def mean_ci_90(x: np.ndarray) -> Dict[str, float]:
    n = int(x.size)
    if n < 2:
        return {"ci90_low": float("nan"), "ci90_high": float("nan")}

    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)

    alpha = 0.10
    df = n - 1

    try:
        from scipy.stats import t  # type: ignore
        tcrit = float(t.ppf(1 - alpha / 2, df))
    except Exception:
        tcrit = 1.6448536269514722  # normal approx

    half = tcrit * se
    return {"ci90_low": mean - half, "ci90_high": mean + half}


@dataclass
class Stats:
    n: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    ci90_low: float
    ci90_high: float


def compute_stats(values: List[float]) -> Stats:
    x = np.array(values, dtype=float)
    n = int(x.size)

    if n == 0:
        return Stats(
            n=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            ci90_low=float("nan"),
            ci90_high=float("nan"),
        )

    pct = np.percentile(x, PCTS, method="linear")
    ci = mean_ci_90(x)

    return Stats(
        n=n,
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if n >= 2 else 0.0,
        min=float(np.min(x)),
        max=float(np.max(x)),
        p50=float(pct[0]),
        p90=float(pct[1]),
        p95=float(pct[2]),
        p99=float(pct[3]),
        ci90_low=float(ci["ci90_low"]),
        ci90_high=float(ci["ci90_high"]),
    )


def summarize_group(name: str, files: List[str]) -> Dict[str, Any]:
    costs: List[float] = []
    errors: List[str] = []

    for fp in files:
        try:
            costs.append(read_first_cost(fp))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")

    return {
        "name": name,
        "files": files,
        "errors": errors,
        "stats": compute_stats(costs),
    }


def print_report(group: Dict[str, Any]) -> None:
    st: Stats = group["stats"]

    print(f"\n=== {group['name']} ===")
    print(f"Files matched: {len(group['files'])}")
    print(f"Costs extracted: n={st.n}")

    if group["errors"]:
        print(f"Files skipped due to errors: {len(group['errors'])}")
        for msg in group["errors"][:10]:
            print(f"  - {msg}")

    print("\nStats (cost):")
    print(f"  mean:  {st.mean:.6g}")
    print(f"  std:   {st.std:.6g}")
    print(f"  min:   {st.min:.6g}")
    print(f"  max:   {st.max:.6g}")
    print(f"  p50:   {st.p50:.6g}")
    print(f"  p90:   {st.p90:.6g}")
    print(f"  p95:   {st.p95:.6g}")
    print(f"  p99:   {st.p99:.6g}")
    print(f"  90% CI (mean): [{st.ci90_low:.6g}, {st.ci90_high:.6g}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory containing the JSON files")
    args = ap.parse_args()

    directory = os.path.abspath(args.dir)

    energy = summarize_group(
        "ENERGY", find_files(directory, "bo_results_energy_")
    )
    price = summarize_group(
        "PRICE", find_files(directory, "bo_results_price_")
    )

    print(f"Scanning directory: {directory}")
    print_report(energy)
    print_report(price)


if __name__ == "__main__":
    main()


    # If you also want the raw arrays printed (commented out by default):
    # print("\nENERGY costs:", energy["costs"])
    # print("PRICE costs:", price["costs"])


if __name__ == "__main__":
    main()
    print(f"{((7.233 - 6.152) / 7.233):.2%}" )
    print(f"{((7.75886 - 6.0043) / 7.75886):.2%}" )