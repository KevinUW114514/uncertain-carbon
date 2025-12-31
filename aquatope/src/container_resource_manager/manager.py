import gevent  # isort:skip
from gevent import monkey  # isort:skip

monkey.patch_all()  # isort:skip
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

import bayesian_optimization

WORKFLOW_CONFIG = json.load(open("ml_workflow.json", "r"))

def explain_resource_config(functions: list, resource_config: list):
    lines = []
    for i, fn in enumerate(functions):
        scaled_cpu, scaled_mem = resource_config[i]

        CPU_MAX = WORKFLOW_CONFIG["max_cpu"][fn]
        CPU_MIN = WORKFLOW_CONFIG["min_cpu"][fn]
        MEM_MAX = WORKFLOW_CONFIG["max_memory"][fn]
        MEM_MIN = WORKFLOW_CONFIG["min_memory"][fn]

        cpu_m = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN)
        mem_mi = round(scaled_mem * (MEM_MAX - MEM_MIN) + MEM_MIN)

        lines.append(
            f"{fn}: scaled(cpu={scaled_cpu:.3f}, mem={scaled_mem:.3f}) -> "
            f"cpu={cpu_m}m (range {CPU_MIN}-{CPU_MAX}m), "
            f"mem={mem_mi}Mi (range {MEM_MIN}-{MEM_MAX}Mi)"
        )
    return "\n".join(lines)


def main():
    global WORKFLOW_CONFIG

    parser = argparse.ArgumentParser(description="Container pool scheduler")
    parser.add_argument("--n_init", action="store", type=int)
    parser.add_argument("--n_batch", action="store", type=int)
    parser.add_argument("--mc_samples", action="store", type=int)
    parser.add_argument("--batch_size", action="store", type=int)
    parser.add_argument("--num_restarts", action="store", type=int)
    parser.add_argument("--raw_samples", action="store", type=int)
    parser.add_argument("--infeasible_cost", action="store", type=float)
    parser.add_argument("--anomaly_detection", action="store_true")
    parser.add_argument("--confidence", action="store", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workflow_config", action="store", type=str, default="ml_workflow.json")

    args = parser.parse_args()
    # n_init = args.n_init
    # n_batch = args.n_batch
    # mc_samples = args.mc_samples
    # batch_size = args.batch_size
    # num_restarts = args.num_restarts
    # raw_samples = args.raw_samples
    # infeasible_cost = args.infeasible_cost
    # anomaly_detection = args.anomaly_detection
    # confidence = args.confidence
    # verbose = args.verbose
    workflow_config_path = args.workflow_config

    with open(workflow_config_path, "r") as f:
        WORKFLOW_CONFIG = json.load(f)
        # print(WORKFLOW_CONFIG)

    start_time = time.time()
    best_cost, resource_config = bayesian_optimization.bo_loop(
        workflow_config=WORKFLOW_CONFIG,
        # n_init=n_init,
        # n_batch=n_batch,
        # mc_samples=mc_samples,
        # batch_size=batch_size,
        # num_restarts=num_restarts,
        # raw_samples=raw_samples,
        # infeasible_cost=infeasible_cost,
        # anomaly_detection=anomaly_detection,
        # confidence=confidence,
        # verbose=verbose,
    )
    end_time = time.time()
    print(f"BO loop time: {end_time - start_time:.2f} seconds")

    # best_cost = 0.03552659365574232
    # resource_config= [[0.8111610417730569, 0.7809624320991828], [0.26167575761264583, 0.14111209835747085]]
    print(f"Best cost: {best_cost}")
    print(f"Best resource configuration: {resource_config}")
    print(explain_resource_config(WORKFLOW_CONFIG["functions"], resource_config))


if __name__ == "__main__":
    main()
    # bayesian_optimization.bo_loop()
