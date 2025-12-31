import subprocess
import json
import time


def run(cmd, deployment_name):
    print("Running:", cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error updating function {deployment_name}")
        print("Return code:", e.returncode)
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        raise

    print(result.stdout)
    result.check_returncode()


def verify_resources(deployment_name, cpu, memory, container_name, interval_seconds=0.5):
    print("\nVerifying applied resource configuration (with while-loop)...")

    # start_time = time.time()

    while True:
        # # Timeout protection
        # if time.time() - start_time > timeout_seconds:
        #     print(f"Timeout exceeded ({timeout_seconds}s). Verification failed.")
        #     sys.exit(1)

        # Get all pods belonging to this deployment
        get_pods_cmd = (
            f"kubectl get pods "
            f"-l app={deployment_name} -o json"
        )
        pods_json = json.loads(run(get_pods_cmd, container_name))
        pods = pods_json.get("items", [])

        if not pods:
            print("No pods found yet, retrying...")
            time.sleep(interval_seconds)
            continue

        all_verified = True

        for pod in pods:
            pod_name = pod["metadata"]["name"]

            # Get resources for target container
            res_cmd = (
                f"kubectl get pod {pod_name} -n "
                f"-o jsonpath='{{.spec.containers[?(@.name==\"{container_name}\")].resources}}'"
            )
            res_output = run(res_cmd)

            try:
                resources = json.loads(res_output)
            except json.JSONDecodeError:
                print(f"Failed to read JSON resources for {pod_name}, retrying...")
                all_verified = False
                continue

            req = resources.get("requests", {})
            lim = resources.get("limits", {})

            print(f"Pod: {pod_name}")
            print(f"  Requests: {req}")
            print(f"  Limits:   {lim}")

            # Check if matches requested settings
            if req.get("cpu") != cpu or req.get("memory") != memory or \
               lim.get("cpu") != cpu or lim.get("memory") != memory:
                all_verified = False

        if all_verified:
            print("\nAll Pods show updated CPU and memory resources.")
            break

        print(f"Not ready yet, retrying in {interval_seconds}s...\n")
        time.sleep(interval_seconds)


import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from utils.config import FISSION_HOST

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.container_resource_manager.manager import WORKFLOW_CONFIG

import subprocess
import redis
from .kube import update_kube_deployment

qos = WORKFLOW_CONFIG["qos"]

result = subprocess.run(
    ["bash", "-c", "ls -l | grep py"],
    capture_output=True,
    text=True,
    check=True
)

print(result.stdout)

r = redis.Redis(
    host="127.0.0.1",   # NOT "http://127.0.0.1"
    port=32204,
    decode_responses=True,
)

def send_request(url, payload):
    response = requests.post(url, json=payload)
    res_payload = response.json()
    # print(f"payload: {payload}")
    print("Status code:", response.status_code)
    print("Response body:", response.json())
    req_id = res_payload.get("req_id", "N/A")
    return req_id


def poll_result(
    req_id: str,
    timeout_s: float = qos * 2,
):
    deadline = time.monotonic() + timeout_s

    while True:
        val = r.get(req_id)
        if val is not None:
            # r.delete(result_key)
            print(f"{req_id}: {val}")
            return json.loads(val)

        if time.monotonic() >= deadline:
            return None

        time.sleep(0.5)


def update_fission_function_setting(
    deployment_name,
    requests,
    limits,
    container_name
):
    print("Updating Deployment resource configuration...\n")
    # cmd = (
    #     f"kubectl set resources deployment/{deployment_name} "
    #     f"-c {container_name} "
    #     f"--requests=cpu={cpu},memory={memory} "
    #     f"--limits=cpu={cpu},memory={memory}"
    # )
    # run(cmd, container_name)
    
    # print("Waiting for Deployment rollout to complete...\n")
    # cmd = f"kubectl rollout status deployment/{deployment_name}"
    # run(cmd, container_name)
    
    # time.sleep(2)
    # verify_resources(deployment_name, cpu, memory, container_name)

    # print(f"container_name: {container_name}, deployment_name: {deployment_name}, requests: {requests}, limits: {limits}")
    # input("debug")
    update_kube_deployment(
        container_name=container_name,
        deployment_name=deployment_name,
        requests=requests,
        limits=limits,
        timeout_s=600,
    )
    

# from typing import List, Tuple, Dict, Any

# def invoke_fission_function_sequence(functions: list, runs: int = 20) -> Tuple[float, Dict[str, float]]:
#     """
#     Run the workflow `runs` times.
#       - e2e_latency returned is the p99 across runs (with 20 runs, this is effectively max).
#       - per-stage latencies returned are the average across runs.
#     """
#     if not functions:
#         raise ValueError("functions must be a non-empty list")

#     e2e_latencies: List[float] = []
#     # stage -> list of observed latencies across runs
#     stage_latencies: Dict[str, List[float]] = {}

#     for _ in range(runs):
#         seq_start_t = time.monotonic()
#         req_id = send_request(
#             f"{FISSION_HOST}/{functions[0]}",
#             WORKFLOW_CONFIG["params"].get(functions[0]),
#         )
#         latencies = poll_result(req_id)  # expected: dict-like {stage_name: seconds, ...}
#         seq_end_t = time.monotonic()

#         e2e_latencies.append(seq_end_t - seq_start_t)

#         if isinstance(latencies, dict):
#             for stage, v in latencies.items():
#                 if v is None:
#                     continue
#                 stage_latencies.setdefault(stage, []).append(float(v))

#     # # p99 with "nearest-rank" method; with n=20 => index ceil(0.99*n)-1 = 19 => max.
#     # e2e_sorted = sorted(e2e_latencies)
#     # idx = max(0, min(len(e2e_sorted) - 1, int((0.99 * len(e2e_sorted)) + 0.999999999) - 1))
#     # e2e_p99 = e2e_sorted[idx]

#     avg_latencies: Dict[str, float] = {
#         stage: (sum(vals) / len(vals)) for stage, vals in stage_latencies.items() if vals
#     }
#     e2e_avg = sum(e2e_latencies) / len(e2e_latencies)

#     print(f"Workflow e2e p99 latency over {runs} runs: {e2e_avg:.3f}s")
#     if avg_latencies:
#         print("Average stage latencies over runs:")
#         for stage, avg in sorted(avg_latencies.items()):
#             print(f"  - {stage}: {avg:.3f}s")

#     return e2e_avg, avg_latencies


import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


def invoke_fission_function_sequence(
    functions: list,
    *,
    workers: int = 8,
    # Mode A: fixed count
    runs: Optional[int] = 80,
    # Mode B: timed sampling
    duration_s: Optional[float] = None,
    submit_interval_s: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Run the workflow in parallel using a fixed worker pool.

    Exactly one of:
      - runs (int)   : fixed number of invocations
      - duration_s (float): submit invocations for this many seconds

    Concurrency:
      - At most `workers` in-flight requests at any time.

    Metrics:
      - Returns e2e p99 latency across completed runs
      - Returns per-stage average latency across completed runs
    """
    if not functions:
        raise ValueError("functions must be a non-empty list")
    if workers <= 0:
        raise ValueError("workers must be > 0")

    mode_runs = runs is not None
    mode_time = duration_s is not None
    if mode_runs == mode_time:
        raise ValueError("Specify exactly one of `runs` or `duration_s`")

    if mode_runs and runs <= 0:
        raise ValueError("runs must be > 0")
    if mode_time and duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if submit_interval_s < 0:
        raise ValueError("submit_interval_s must be >= 0")

    fn0 = functions[0]
    url = f"{FISSION_HOST}/{fn0}"
    payload = WORKFLOW_CONFIG["params"].get(fn0)

    def one_run() -> Tuple[float, Dict[str, float]]:
        t0 = time.monotonic()
        req_id = send_request(url, payload)
        latencies = poll_result(req_id)  # expected dict-like {stage_name: seconds, ...}
        t1 = time.monotonic()

        if latencies is None:
            e2e = qos * 3
        else:
            e2e = t1 - t0

        stages: Dict[str, float] = {}
        if isinstance(latencies, dict):
            for stage, v in latencies.items():
                if v is None:
                    continue
                stages[stage] = float(v)

        return e2e, stages

    e2e_latencies: List[float] = []
    stage_latencies: Dict[str, List[float]] = {}

    submitted = 0
    completed = 0

    start = time.monotonic()
    deadline = (start + float(duration_s)) if mode_time else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        in_flight = set()

        def can_submit_more() -> bool:
            if mode_runs:
                return submitted < int(runs)
            # mode_time
            return time.monotonic() < deadline

        # Prime the pool to full utilization (or until we hit runs / time).
        while len(in_flight) < workers and can_submit_more():
            in_flight.add(ex.submit(one_run))
            submitted += 1
            if submit_interval_s:
                time.sleep(submit_interval_s)

        # Drain/submit loop: whenever one finishes, record it and submit another if allowed.
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

            for fut in done:
                e2e, stages = fut.result()
                e2e_latencies.append(e2e)
                completed += 1

                for stage, v in stages.items():
                    stage_latencies.setdefault(stage, []).append(v)

            while len(in_flight) < workers and can_submit_more():
                in_flight.add(ex.submit(one_run))
                submitted += 1
                if submit_interval_s:
                    time.sleep(submit_interval_s)

    if not e2e_latencies:
        raise RuntimeError("No runs completed; cannot compute metrics.")

    e2e_arr = np.asarray(e2e_latencies, dtype=np.float64)
    # p99 (numpy default interpolation behavior differs across versions; use method="higher" when available)
    try:
        e2e_p99 = float(np.quantile(e2e_arr, 0.99, method="higher"))
    except TypeError:
        # older numpy: use interpolation="higher"
        e2e_p99 = float(np.quantile(e2e_arr, 0.99, interpolation="higher"))

    avg_latencies: Dict[str, float] = {
        stage: float(np.mean(np.asarray(vals, dtype=np.float64)))
        for stage, vals in stage_latencies.items()
        if vals
    }

    elapsed = time.monotonic() - start
    mode_desc = f"{completed} completed" if mode_runs else f"{completed} completed in {elapsed:.1f}s"

    print(f"Workflow e2e p99 latency ({mode_desc}, parallel, {workers} workers): {e2e_p99:.3f}s")
    if avg_latencies:
        print("Average stage latencies:")
        for stage, avg in sorted(avg_latencies.items()):
            print(f"  - {stage}: {avg:.3f}s")

    # def stat(data: List[float]) -> Tuple[float, float, float]:
    #     arr = np.asarray(data, dtype=np.float64)
    #     print(f"mean", np.mean(arr))
    #     print(f"median", np.median(arr))
    #     print(f"std", np.std(arr))
    #     print(f"p90", np.quantile(arr, 0.90))
    #     print(f"p95", np.quantile(arr, 0.95))
    #     print(f"p99", np.quantile(arr, 0.99))
    
    # stat(e2e_latencies)

    return e2e_p99, avg_latencies
