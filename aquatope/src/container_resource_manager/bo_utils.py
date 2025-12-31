import gevent  # isort:skip
from gevent import monkey# isort:skip

monkey.patch_all()  # isort:skip
import sys
from pathlib import Path

import torch
import time

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

from manager import WORKFLOW_CONFIG

from fissionlib.cli import update_fission_function_setting, invoke_fission_function_sequence
from utils.config import (
    CPU_MAX,
    CPU_MIN,
    CPU_UNIT_COST,
    MEMORY_MAX,
    MEMORY_MIN,
    MEMORY_UNIT_COST,
    NUM_RESOURCES,
)

CACHE = dict()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.double


def from_x_to_resource_config(x: torch.tensor) -> dict:
    x_list = x.tolist()
    num_stages = int(len(x_list) / NUM_RESOURCES)
    resource_config = [[CPU_MIN, MEMORY_MIN] for _ in range(num_stages)]
    for i in range(int(len(x_list) / 2)):
        scaled_cpu = x_list[i * 2]
        scaled_memory = x_list[i * 2 + 1]
        # resource_config[i][0] = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN)
        # resource_config[i][1] = round(
        #     scaled_memory * (MEMORY_MAX - MEMORY_MIN) + MEMORY_MIN
        # )
        resource_config[i][0] = scaled_cpu
        resource_config[i][1] = scaled_memory
    return resource_config


def calc_cost(functions: list, resource_config: list, latencies: dict) -> float:
    cost = 0
    for i, config in enumerate(resource_config):
        cpu, memory = config
        fn = functions[i]
        if fn not in latencies:
            raise ValueError(f"Function {fn} not in latencies")
        duration = latencies[fn]
        cost += cpu * duration * CPU_UNIT_COST + memory * duration * MEMORY_UNIT_COST
    return cost


def update_resource_config(functions: list, resource_config: list):
    for i, fn in enumerate(functions):
        scaled_cpu, scaled_memory = resource_config[i]
        CPU_MAX = WORKFLOW_CONFIG["max_cpu"][fn]
        CPU_MIN = WORKFLOW_CONFIG["min_cpu"][fn]
        MEMORY_MAX = WORKFLOW_CONFIG["max_memory"][fn]
        MEMORY_MIN = WORKFLOW_CONFIG["min_memory"][fn]
        container_name = WORKFLOW_CONFIG["container_name_mapping"][fn]
        memory = round(scaled_memory * (MEMORY_MAX - MEMORY_MIN) + MEMORY_MIN)
        cpu = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN)
        cpu = str(cpu) + "m"
        memory = str(memory) + "Mi"
        print(f"Updating function {fn} to cpu: {cpu}, memory: {memory}")
        deployment_name = WORKFLOW_CONFIG["function_deployment_mapping"][fn]
        requests = {
            "cpu": str(CPU_MIN) + "m",
            "memory": str(MEMORY_MIN) + "Mi"
        }
        limits = {
            "cpu": cpu,
            "memory": memory
        }
        
        update_fission_function_setting(deployment_name=deployment_name, requests=requests, limits=limits, container_name=container_name)
        
    time.sleep(1)

def sample_cost(x: torch.tensor):
    resource_config = from_x_to_resource_config(x)
    hash_id = hash(str(resource_config))
    if hash_id not in CACHE:
        update_resource_config(
            functions=WORKFLOW_CONFIG["functions"], resource_config=resource_config
        )
        e2e_latency, latencies = invoke_fission_function_sequence(
            functions=WORKFLOW_CONFIG["functions"]
        )
        cost = calc_cost(
            functions=WORKFLOW_CONFIG["functions"],
            resource_config=resource_config,
            latencies=latencies,
        )
        CACHE[hash_id] = dict()
        CACHE[hash_id]["cost"] = cost
        CACHE[hash_id]["duration"] = e2e_latency
    cost = CACHE[hash_id]["cost"]
    return torch.tensor([cost], dtype=dtype)


def sample_duration(x: torch.tensor):
    resource_config = from_x_to_resource_config(x)
    hash_id = hash(str(resource_config))
    cost = 0
    if hash_id not in CACHE:
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


# n = 3
# rand_x = torch.rand(n, 6)
# res = sample_cost_parallel(rand_x)
# print(res)
