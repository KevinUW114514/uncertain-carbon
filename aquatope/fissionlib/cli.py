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
    timeout_s: float = 5,
):
    deadline = time.monotonic() + timeout_s

    while True:
        val = r.get(req_id)
        if val is not None:
            # r.delete(result_key)
            return json.loads(val)

        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for {req_id}")

        time.sleep(0.5)


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
    req_id = send_request(f"{FISSION_HOST}/{functions[0]}", WORKFLOW_CONFIG["params"].get(functions[0]))
    latencies = poll_result(req_id)
    seq_end_t = time.monotonic()
    e2e_latency = seq_end_t - seq_start_t

    return  e2e_latency, latencies
