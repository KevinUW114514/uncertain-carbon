import gevent  # isort:skip
from gevent import monkey
import locust# isort:skip

monkey.patch_all()  # isort:skip
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
import time
import json
import threading
import math
import pandas as pd

_LOG_LOCK = threading.Lock()
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

WORKFLOW_CONFIG = json.load(open("ml_workflow.json", "r"))
from config import CONFIG

from fissionlib.cli import update_fission_function_setting, invoke_fission_function_sequence, start_locust, stop_locust, LOCUST_CSV
from utils.config import (
    CPU_MAX,
    CPU_MIN,
    CPU_UNIT_COST,
    MEMORY_MAX,
    MEMORY_MIN,
    MEMORY_UNIT_COST,
    NUM_RESOURCES,
    CPU_UNIT_POWER,
    MEMORY_UNIT_POWER,
    CPU_BASE_POWER,
    MEMORY_BASE_POWER,
)

import os

IS_ENERGY = os.getenv("IS_ENERGY", "0")
if IS_ENERGY == "1":
    IS_ENERGY = True
else:
    IS_ENERGY = False

CACHE = dict()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.double

import time
import requests
from typing import Callable, Dict, List, Tuple, Optional

WORKER_ENERGY_URL = "http://10.52.2.205:9876"

def poll_locust_row(
    csv_path: str,
    name: str,
    ts: int,
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.5,
) -> pd.Series:
    """
    Polls the Locust stats_history CSV until a row exists for (Name==name, Timestamp==ts),
    reloading the CSV every poll.

    Returns:
        pd.Series: the matching row

    Raises:
        TimeoutError: if the row does not appear within timeout_s
    """
    deadline = time.time() + timeout_s
    last_err = None

    while time.time() < deadline:
        try:
            df = pd.read_csv(csv_path)

            # Build mask on the SAME dataframe
            rows = df.loc[(df["Name"] == name) & (df["Timestamp"] == ts)]

            if not rows.empty:
                # If multiple rows exist, return the first (or change policy here)
                return rows.iloc[0]

        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            # File not created or being flushed
            last_err = e
        except Exception as e:
            # CSV mid-write or transient parse error
            last_err = e

        print(f"[info] poll_locust_row: failed to load, try again")
        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for row "
        f"(Name={name!r}, Timestamp={ts}) in {csv_path}. "
        f"Last error: {last_err}"
    )
    
def safe_log_append(log_path: str, text: str) -> None:
    if log_path is None:
        print("[warning] log_path is None; skipping log append")
        return
        
    with _LOG_LOCK:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

def _rapl_delta_uj(end_uj: int, start_uj: int, max_range_uj: Optional[int] = None) -> int:
    """Handle wrap if max_range_uj is provided; otherwise assume no wrap."""
    if end_uj >= start_uj:
        return end_uj - start_uj
    if max_range_uj is not None and max_range_uj > 0:
        return (max_range_uj - start_uj) + end_uj
    # Fallback: treat as no wrap info; return negative -> caller can decide
    return end_uj - start_uj


# def read_worker_energy_snapshot(url: str = WORKER_ENERGY_URL, timeout: float = 2.0) -> dict:
#     """
#     Returns JSON like:
#       { "monotonic_ns": ..., "energy_uj": {domain: value, ...}, ... }
#     """
#     r = requests.get(url, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

def get_worker_energy(timestamp: float, url: str = WORKER_ENERGY_URL, timeout: float = 2.0) -> dict:
    """
    Returns JSON like:
      { "monotonic_ns": ..., "energy_uj": {domain: value, ...}, ... }
    """
    r = requests.get(f"{url}/power?timestamp_utc_ms={math.floor(timestamp * 1000)}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def calc_total_energy_j(
    start_snap: dict,
    end_snap: dict,
    include_dram: bool = True,
    package_keys: Tuple[str, ...] = ("package",),  # substring match
    dram_keys: Tuple[str, ...] = ("dram",),        # substring match
) -> float:
    """
    Compute total energy (J) as sum of deltas of selected domains.

    Notes:
      - This assumes counters are cumulative µJ.
      - Uses substring matching on domain keys.
      - Wrap handling requires max_energy_range_uj; if you want full wrap safety,
        expose it in the /energy endpoint and pass it here.
    """
    s = start_snap["energy_uj"]
    e = end_snap["energy_uj"]

    total_uj = 0
    for k in s.keys():
        kl = k.lower()
        is_pkg = any(tok in kl for tok in package_keys)
        is_dram = any(tok in kl for tok in dram_keys)

        if is_pkg or (include_dram and is_dram):
            if k not in e:
                raise ValueError(f"Energy domain {k} missing from end snapshot")
            # duj = e[k] - s[k]  # wrap-safe variant needs max_range; see note above
            duj = _rapl_delta_uj(end_uj=e[k], start_uj=s[k], max_range_uj=262143328850)
            total_uj += duj

    return total_uj / 1e6  # µJ -> J

def calc_total_energy(
    start_total_energy: dict,
    end_total_energy: dict,
) -> float:
    """
    Compute total energy (J) as sum of deltas of selected domains.

    Notes:
      - This assumes counters are cumulative µJ.
      - Uses substring matching on domain keys.
      - Wrap handling requires max_energy_range_uj; if you want full wrap safety,
        expose it in the /energy endpoint and pass it here.
    """
    s = start_total_energy
    e = end_total_energy

    total_uj = _rapl_delta_uj(end_uj=e, start_uj=s, max_range_uj=262143328850)

    return total_uj / 1e6  # µJ -> J


def from_x_to_resource_config(
    x: torch.Tensor,
    functions: List[str] = WORKFLOW_CONFIG["functions"],
    num_resources: int = NUM_RESOURCES,
    workflow_config: Dict[str, Any] = WORKFLOW_CONFIG
) -> List[Dict[str, int]]:
    """
        Convert scaled tensor x into per-stage actual resource config.

        Returns list of dicts:
        [{"cpu_m": <int>, "memory_mi": <int>}, ...]
    """
    x_list = x.tolist()
    num_stages = int(len(x_list) / num_resources)

    # Default to minimums (actual) for each stage
    resource_config: List[Dict[str, int]] = []
    for i in range(num_stages):
        fn = functions[i]
        cpu_min = workflow_config["min_cpu"][fn]
        mem_min = workflow_config["min_memory"][fn]
        resource_config.append({"cpu_m": cpu_min, "memory_mi": mem_min})

    # Each stage has (scaled_cpu, scaled_memory)
    for i in range(num_stages):
        scaled_cpu = x_list[i * 2]
        scaled_memory = x_list[i * 2 + 1]

        fn = functions[i]
        cpu_max = workflow_config["max_cpu"][fn]
        cpu_min = workflow_config["min_cpu"][fn]
        mem_max = workflow_config["max_memory"][fn]
        mem_min = workflow_config["min_memory"][fn]

        cpu = round(scaled_cpu * (cpu_max - cpu_min) + cpu_min)
        memory = round(scaled_memory * (mem_max - mem_min) + mem_min)

        resource_config[i]["cpu_m"] = cpu
        resource_config[i]["memory_mi"] = memory

    return resource_config


def calc_cost(functions: list, resource_config: list, latencies: dict) -> float:
    cost = 0
    for i, config in enumerate(resource_config):
        cpu, memory = config["cpu_m"], config["memory_mi"]
        fn = functions[i]
        if fn not in latencies:
            raise ValueError(f"Function {fn} not in latencies")
        duration = latencies[fn]
        pods_count = WORKFLOW_CONFIG["function_pods_mapping"][fn]
        cost += cpu * pods_count * duration * CPU_UNIT_COST + memory * pods_count * duration * MEMORY_UNIT_COST
        # print(f"Function {fn}: cpu={cpu}m, memory={memory}Mi, duration={duration:.2f}s")
        # cost += (cpu * CPU_UNIT_POWER + CPU_BASE_POWER) * duration + (memory * MEMORY_UNIT_POWER + MEMORY_BASE_POWER) * duration
    return cost

def update_resource_config(functions: list, resource_config: list, *, concurrency: int = 8):
    """
    Update each function's resource config concurrently.

    Assumes update_fission_function_setting(...) is I/O-bound (API calls).
    """
    if len(resource_config) != len(functions):
        raise ValueError(
            f"resource_config length ({len(resource_config)}) must match functions length ({len(functions)})"
        )

    def _update_one(i: int):
        fn = functions[i]
        container_name = WORKFLOW_CONFIG["container_name_mapping"][fn]
        deployment_name = WORKFLOW_CONFIG["function_deployment_mapping"][fn]

        cpu_m = resource_config[i]["cpu_m"]
        memory_mi = resource_config[i]["memory_mi"]

        cpu_str = f"{cpu_m}m"
        mem_str = f"{memory_mi}Mi"

        print(f"Updating function {fn} to cpu: {cpu_str}, memory: {mem_str}")

        # Keep request policy: min as requests, computed as limits
        CPU_MIN = WORKFLOW_CONFIG["min_cpu"][fn]
        MEMORY_MIN = WORKFLOW_CONFIG["min_memory"][fn]

        requests = {"cpu": f"{CPU_MIN}m", "memory": f"{MEMORY_MIN}Mi"}
        limits = {"cpu": cpu_str, "memory": mem_str}

        update_fission_function_setting(
            deployment_name=deployment_name,
            requests=requests,
            limits=limits,
            container_name=container_name,
        )

        return {"fn": fn, "deployment": deployment_name, "cpu": cpu_str, "memory": mem_str}

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_update_one, i) for i in range(len(functions))]

        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(e)

    if errors:
        # Fail fast with aggregated signal; you can also log per-error details above.
        raise RuntimeError(f"{len(errors)} updates failed. First error: {errors[0]!r}")

    # lines = []
    # for i, r in enumerate(results):
    #     lines.append(
    #         f"stage[{i}] fn={r['fn']} cpu={r['cpu']} memory={r['memory']}"
    #     )
    # lines.append("~" * 80 + "\n")
    # safe_log_append(CONFIG.log_path, "\n".join(lines))

    # start_locust()
    # # return results
    # print("[info]: start sleep for 30s")
    # time.sleep(30)
    # print("[info]: sleep done")

# IS_ENERGY = True

def sample_cost(x: torch.Tensor):
    resource_config = from_x_to_resource_config(x)
    hash_id = hash(str(resource_config))

    # For reproducibility / audit
    start_ts = time.time()
    
    sample_data = CACHE.get(hash_id, None)
    
    if hash_id not in CACHE:
        # Apply configuration
        update_resource_config(
            functions=WORKFLOW_CONFIG["functions"],
            resource_config=resource_config
        )

        # Give system time to stabilize
        start_locust()
        print("[info] start sleep for 20s")
        time.sleep(20)
        print("[info] sleep done")

        # start = read_worker_energy_snapshot(WORKER_ENERGY_URL)
        start_time = time.time()
        e2e_latency, latencies = invoke_fission_function_sequence(
            functions=WORKFLOW_CONFIG["functions"],
            runs=None,
            duration_s=15
        )
        # time.sleep(15)
        end_time = time.time()
        time.sleep(3)
        
        locust_csv_df = pd.read_csv(f"{LOCUST_CSV}_stats_history.csv")
        locust_csv_e2e = locust_csv_df.loc[
            locust_csv_df["Name"] == "ml-image-processing-e2e"
        ]
        start_requests = locust_csv_e2e.loc[
            locust_csv_df["Timestamp"] == math.floor(start_time),
            "Total Request Count"
        ].iloc[0]
        end_row = poll_locust_row(f"{LOCUST_CSV}_stats_history.csv", "ml-image-processing-e2e", math.floor(end_time), timeout_s=10, poll_interval_s=0.5)
        end_requests   = end_row["Total Request Count"]
        total_requests = end_requests - start_requests
        print(f"[info] total_requests: {total_requests}")
        
        e2e_latency = end_row["99%"] / 1000.0  # ms -> s
        print(f"[info] e2e_latency (p99): {e2e_latency:.3f} s")
        
        start_energy_snap = get_worker_energy(timestamp=start_time, url=WORKER_ENERGY_URL)
        # end = read_worker_energy_snapshot(WORKER_ENERGY_URL)
        end_energy_snap = get_worker_energy(timestamp=end_time, url=WORKER_ENERGY_URL)
        energy_j = calc_total_energy_j(start_snap=start_energy_snap, end_snap=end_energy_snap)
        print(f"[info] energy_j: {energy_j:.6f} J")
        
        served_rps = end_row["Requests/s"]
        print(f"[info] served_rps: {served_rps:.2f} req/s")
        
        duration_s = end_time - start_time
        print(f"[info]: start_time: {start_time}, end_time: {end_time}, duration_s: {duration_s:.2f} s")
        
        energy_per_request = energy_j / total_requests if total_requests > 0 else float('inf')
        energy_cost = energy_per_request
        print(f"[info] energy_cost: {energy_cost:.6f} J/request")
        
        power = energy_j / duration_s if duration_s > 0 else float('inf')
        print(f"[info] average power: {power:.6f} W")
        
        price_cost = calc_cost(
            functions=WORKFLOW_CONFIG["functions"],
            resource_config=resource_config,
            latencies=latencies,
        )
        # energy_j = calc_total_energy_j(start_snap=start, end_snap=end)
        # duration_s = (end["monotonic_ns"] - start["monotonic_ns"])
        # energy_cost = energy_j / duration_s
        
        stop_locust()
        
        if IS_ENERGY:
            objective_name = "energy"
            cost = energy_cost
        else:
            objective_name = "price"
            cost = price_cost

        # Cache results
        CACHE[hash_id] = {
            "cost": cost,
            "duration": e2e_latency,
            "latencies": latencies,
            "price_cost": price_cost,
            "energy_cost": energy_cost,
            "objective_name": objective_name,
            "feasible": e2e_latency <= WORKFLOW_CONFIG["qos"],
            "resource_config": resource_config,
            "energy_j": energy_j,
            "total_requests": total_requests,
            "served_rps": served_rps,
            "duration_s": duration_s,
            "power": power,
        }
        
        sample_data = CACHE[hash_id]

        # ---------- LOGGING (single source of truth) ----------
        lines = [
            "=" * 80,
            f"objective      : {objective_name}",
            # f"hash_id        : {hash_id}",
            f"resource_config: {resource_config}, latencies: {latencies}, price_cost: {price_cost:.6f}, energy_cost: {energy_cost:.6f}, latency_p99 : {e2e_latency:.6f}, feasible: {sample_data['feasible']}, total_requests: {total_requests}, served_rps: {served_rps:.2f}, total_energy_j: {energy_j:.6f} J, profiling_duration_s: {duration_s:.6f}s, average_power: {power:.6f} W",
        ]

        # if IS_ENERGY:
        #     lines.extend([
        #         f"energy_j       : {energy_j:.6f}",
        #         f"window_s       : {duration_s:.6f}",
        #     ])

        lines.append(f"wall_time_s    : {time.time() - start_ts:.3f}")
        lines.append("=" * 80 + "\n")

        safe_log_append(CONFIG.log_path, "\n".join(lines))
        print("\n".join(lines))
    else:
        print("!!!! Using cached result !!!!")

    return torch.tensor([cost], dtype=dtype), torch.tensor([e2e_latency], dtype=dtype), sample_data


def sample_duration(x: torch.tensor):
    resource_config = from_x_to_resource_config(x)
    hash_id = hash(str(resource_config))
    cost = 0
    if hash_id not in CACHE:
        print("!!! No cached result found for duration sampling; invoking functions !!!")
        update_resource_config(
            functions=WORKFLOW_CONFIG["functions"], resource_config=resource_config
        )
        e2e_latency, latencies = invoke_fission_function_sequence(
            functions=WORKFLOW_CONFIG["functions"]
        )
        CACHE[hash_id] = dict()
        CACHE[hash_id]["cost"] = cost
        CACHE[hash_id]["duration"] = e2e_latency
    duration = CACHE[hash_id]["duration"]
    # stop_locust()
    return torch.tensor([duration], dtype=dtype)


def sample_cost_parallel(X: torch.tensor):
    # jobs = []
    # for i, x in enumerate(X):
    #     job = gevent.spawn(sample_cost, x=x)
    #     jobs.append(job)
    # gevent.joinall(jobs)
    # res = torch.tensor([job.value for job in jobs], dtype=dtype)
    # return res 
    
    results = []
    for x in X:
        results.append(sample_cost(x=x))
    return torch.tensor(results, dtype=dtype)


def sample_duration_parallel(X: torch.tensor):
    # jobs = []
    # for i, x in enumerate(X):
    #     job = gevent.spawn(sample_duration, x=x)
    #     jobs.append(job)
    # gevent.joinall(jobs)
    # res = torch.tensor([job.value for job in jobs], dtype=dtype)
    # return res
    
    results = []
    for x in X:
        results.append(sample_duration(x=x))
    return torch.tensor(results, dtype=dtype)

def sample_cost_duration(X: torch.tensor):
    obj_list = []
    cons_list = []
    sample_data_list = []
    for x in X:
        obj, cons, sample_data = sample_cost(x=x)
        obj_list.append(obj)
        cons_list.append(cons)
        sample_data_list.append(sample_data)
        
    return torch.stack(obj_list), torch.stack(cons_list), sample_data_list
# n = 3
# rand_x = torch.rand(n, 6)
# res = sample_cost_parallel(rand_x)
# print(res)
