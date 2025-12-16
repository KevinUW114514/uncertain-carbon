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

result = subprocess.run(
    ["bash", "-c", "ls -l | grep py"],
    capture_output=True,
    text=True,
    check=True
)

print(result.stdout)


def update_fission_function_setting(
    function_name,
    cpu,
    memory,
):
    try:
        result = subprocess.run(
            ["bash", "-c", f"fission fn update --name {function_name} --mincpu {cpu} --maxcpu {cpu} --minmemory {memory} --maxmemory {memory}"],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error updating function {function_name}")
        print("Return code:", e.returncode)
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        raise
    
    
    print(result.stdout)
    return result.returncode
    

def invoke_fission_function_sequence(functions: list):
    seq_start_t = time.monotonic()
    latencies = dict()
    for name in functions:
        response = requests.post(
            url=f"{FISSION_HOST}/{name}",
            json=WORKFLOW_CONFIG["params"].get(name)
        )
        # print(f"Invoked function {name}")
        # print(f"url: {FISSION_HOST}/{name}")
        # ct = response.headers.get("Content-Type")
        # print("status:", response.status_code)
        # print("content-type:", ct)
        # print("raw body (first 500 chars):", repr(response.text[:500]))
        # input("debug")
        response = response.json()
        print(f"Function {name} response: {response}")
        latencies[name] = response.get("duration")
    seq_end_t = time.monotonic()
    e2e_latency = seq_end_t - seq_start_t
    

    return  e2e_latency, latencies
