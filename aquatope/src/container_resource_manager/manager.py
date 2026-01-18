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
from config import CONFIG
import os
from bo_utils import to_jsonable

IS_ENERGY = os.getenv("IS_ENERGY", "0")
if IS_ENERGY == "1":
    IS_ENERGY = True
else:
    IS_ENERGY = False

WORKFLOW_CONFIG = json.load(open("ml_workflow.json", "r"))

def main():
    global WORKFLOW_CONFIG
    global IS_ENERGY

    parser = argparse.ArgumentParser(description="Container pool scheduler")
    parser.add_argument("--n_init", action="store", type=int)
    parser.add_argument("--n_batch", action="store", type=int, default=10)
    parser.add_argument("--mc_samples", action="store", type=int)
    parser.add_argument("--batch_size", action="store", type=int)
    parser.add_argument("--num_restarts", action="store", type=int)
    parser.add_argument("--raw_samples", action="store", type=int)
    parser.add_argument("--infeasible_cost", action="store", type=float)
    parser.add_argument("--anomaly_detection", action="store_true")
    parser.add_argument("--confidence", action="store", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workflow_config", action="store", type=str, default="ml_workflow.json")
    parser.add_argument("--is_energy", action="store_true")
    parser.add_argument("--model_path", action="store", type=str, default="")
    parser.add_argument("--sample_path", action="store", type=str, default="")

    args = parser.parse_args()
    # n_init = args.n_init
    n_batch = args.n_batch
    # mc_samples = args.mc_samples
    # batch_size = args.batch_size
    # num_restarts = args.num_restarts
    # raw_samples = args.raw_samples
    # infeasible_cost = args.infeasible_cost
    # anomaly_detection = args.anomaly_detection
    # confidence = args.confidence
    # verbose = args.verbose
    workflow_config_path = args.workflow_config

    print(f"IS_ENERGY: {IS_ENERGY}")

    with open(workflow_config_path, "r") as f:
        WORKFLOW_CONFIG = json.load(f)
        # print(WORKFLOW_CONFIG)

    ts = f"{time.strftime('%Y%m%d_%H%M%S')}"
    suffix = f"{'energy' if IS_ENERGY else 'price'}_{ts}"
    
    if args.model_path == "":
        log_path = f"bo_log_{suffix}.log"
        model_path = f"bo_model_{suffix}.pt"
        CONFIG.set_log_path(log_path)
        CONFIG.set_json_path(f"bo_results_{suffix}.json")
        CONFIG.set_sample_path(f"bo_samples_{suffix}.json")
    else:
        model_id = Path(args.model_path).stem[-15:]
        model_path = args.model_path
        log_path = f"resume_bo_log_{model_id}_{suffix}.log"
        CONFIG.set_log_path(log_path)
        CONFIG.set_json_path(f"resume_bo_results_{model_id}_{suffix}.json")
        CONFIG.set_sample_path(f"resume_bo_samples_{model_id}_{suffix}.json")

    start_time = time.time()
    best_cost, resource_config = bayesian_optimization.bo_loop(
        workflow_config=WORKFLOW_CONFIG,
        suffix=suffix,
        # n_init=n_init,
        n_batch=n_batch,
        # mc_samples=mc_samples,
        # batch_size=batch_size,
        # num_restarts=num_restarts,
        # raw_samples=raw_samples,
        # infeasible_cost=infeasible_cost,
        # anomaly_detection=anomaly_detection,
        # confidence=confidence,
        # verbose=verbose,
        log_path=log_path,
        save_path=model_path,
        sample_path=args.sample_path
    )
    end_time = time.time()
    s = f"BO loop time: {end_time - start_time:.2f} seconds\n" + \
        f"Best cost: {best_cost}\n" + \
        f"Best resource configuration: {resource_config}\n"

    result = dict()
    result["objective"] = "energy" if IS_ENERGY else "price"
    result["best_cost"] = best_cost
    result["resource_config"] = resource_config
    
    print(s)
    with open(log_path, "a") as f:
        f.write(s)

    with open(f"best_resource_config_{suffix}.json", "w") as f:
        json.dump(to_jsonable(result), f, indent=2)


if __name__ == "__main__":
    main()
    # bayesian_optimization.bo_loop()
