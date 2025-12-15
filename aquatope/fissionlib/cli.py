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
    result = subprocess.run(
        ["bash", "-c", f"fission fn update-container --name {function_name} --mincpu {cpu} --maxcpu {cpu} --minmemory {memory} --maxmemory {memory}"],
        capture_output=True,
        text=True,
        check=True
    )
    
    print(result.stdout)
    return result.returncode
    

def invoke_fission_function_sequence(sequence_name: str, params: dict):
    start_t = time.monotonic()
    response = requests.post(
        url=f"{FISSION_HOST}/{sequence_name}",
        json=params
    )
    end_t = time.monotonic()
    e2e_latency = end_t - start_t
    

    return  e2e_latency 
