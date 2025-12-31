"""
Write a python server program by using a simple using FastAPI. The program should be also a sender and a receiver, open port 11451 at 0.0.0.0. Let's define the following workflow. The send function sends a future time (5 seconds later from now) to multiple receivers. Both sender and all receivers start a bash program at the exact future time. After, all receivers send a pandas dataframe object back to the sender. The sender collects them all and aggregate them and save to a aggreated csv, named as hostname-load_name-time. For simplicity, you can assume the user won't initiate a new workflow before the old one finished, so there won't be multiple workflow happened simultaneously
"""
import os
import time
import json
import socket
import asyncio
import subprocess
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import pandas as pd
import httpx
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_HOST = "0.0.0.0"
APP_PORT = 11451

app = FastAPI(title="Sender/Receiver Workflow Server")

# ----------------------------
# Models
# ----------------------------

class StartWorkflowRequest(BaseModel):
    receivers: List[str] = Field(
        ...,
        description="List of receiver base URLs, e.g. ['http://10.0.0.2:11451', 'http://10.0.0.3:11451']"
    )
    load_name: str = Field(..., description="A label for this run (used in output filenames).")
    bash_cmd: str = Field(..., description="The bash command to run at the scheduled time.")


class ScheduleRequest(BaseModel):
    run_id: str
    start_time_epoch: float
    sender_url: str
    load_name: str
    bash_cmd: str


class ReceiverResult(BaseModel):
    run_id: str
    receiver_host: str
    load_name: str
    start_time_epoch: float
    dataframe_json: str  # df.to_json(orient="split")
    bash_returncode: int
    bash_stdout: str
    bash_stderr: str


# ----------------------------
# Simple in-memory state (single workflow at a time)
# ----------------------------

class SenderRunState:
    def __init__(self):
        self.in_progress: bool = False
        self.run_id: Optional[str] = None
        self.load_name: Optional[str] = None
        self.start_time_epoch: Optional[float] = None
        self.expected_receivers: List[str] = []
        self.received: Dict[str, ReceiverResult] = {}

state = SenderRunState()

# Receiver-side state (so receiver won't accept multiple schedules concurrently)
receiver_in_progress = False
receiver_current_run_id: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------

def now_epoch() -> float:
    return time.time()

def iso_utc_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def hostname() -> str:
    return socket.gethostname()

async def wait_until_epoch(ts: float) -> None:
    # Sleep in short chunks for better precision near the boundary
    while True:
        remaining = ts - now_epoch()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.25))

def run_bash(cmd: str) -> subprocess.CompletedProcess:
    # Use bash -lc so things like PATH / shell features behave like a normal bash session
    return subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True
    )

def sample_dataframe(load_name: str, start_time_epoch: float) -> pd.DataFrame:
    # Example "load" dataframe; customize for your real measurements
    cpu = psutil.cpu_percent(interval=0.2)
    vm = psutil.virtual_memory()
    return pd.DataFrame([{
        "receiver_host": hostname(),
        "load_name": load_name,
        "start_time_epoch": start_time_epoch,
        "timestamp_epoch": now_epoch(),
        "cpu_percent": cpu,
        "mem_used": vm.used,
        "mem_total": vm.total
    }])

def aggregate_and_save(run_id: str, load_name: str, start_time_epoch: float, results: List[ReceiverResult]) -> Dict[str, Any]:
    # Parse each receiver df and concat
    dfs = []
    for r in results:
        df = pd.read_json(r.dataframe_json, orient="split")
        # Add bash metadata
        df["bash_returncode"] = r.bash_returncode
        df["bash_stdout"] = r.bash_stdout
        df["bash_stderr"] = r.bash_stderr
        dfs.append(df)

    if not dfs:
        raise ValueError("No receiver dataframes to aggregate.")

    agg = pd.concat(dfs, ignore_index=True)

    # Output filename format requested: `hostname-load_name-time`
    # Interpret "hostname" as the sender's hostname.
    sender_host = hostname()
    tstamp = iso_utc_from_epoch(start_time_epoch)
    out_name = f"{sender_host}-{load_name}-{tstamp}.csv"
    agg.to_csv(out_name, index=False)

    # Optional: also save a "combined" JSON for traceability
    meta_name = f"{sender_host}-{load_name}-{tstamp}.meta.json"
    meta = {
        "run_id": run_id,
        "sender_host": sender_host,
        "load_name": load_name,
        "start_time_epoch": start_time_epoch,
        "start_time_utc": datetime.fromtimestamp(start_time_epoch, tz=timezone.utc).isoformat(),
        "receiver_count": len(results),
        "receivers": [r.receiver_host for r in results],
    }
    with open(meta_name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"csv": out_name, "meta": meta_name, "rows": len(agg)}


# ----------------------------
# Sender endpoints
# ----------------------------

@app.post("/sender/start")
async def sender_start(req: StartWorkflowRequest):
    """
    Sender initiates a single workflow:
      1) schedule time = now + 5s
      2) send schedule to all receivers
      3) sender also waits for schedule time and runs bash
      4) receivers send back dataframe
      5) sender aggregates + saves CSV
    """
    if state.in_progress:
        raise HTTPException(status_code=409, detail="A workflow is already in progress.")

    run_id = f"{hostname()}-{int(now_epoch()*1000)}"
    start_time_epoch = now_epoch() + 5.0

    # Determine sender_url from env or default guess
    sender_url = os.environ.get("SENDER_URL", f"http://{hostname()}:{APP_PORT}")

    state.in_progress = True
    state.run_id = run_id
    state.load_name = req.load_name
    state.start_time_epoch = start_time_epoch
    state.expected_receivers = req.receivers
    state.received = {}

    schedule = ScheduleRequest(
        run_id=run_id,
        start_time_epoch=start_time_epoch,
        sender_url=sender_url,
        load_name=req.load_name,
        bash_cmd=req.bash_cmd
    )

    # Fan-out schedule to receivers
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for r in req.receivers:
            # Expect receiver endpoint at /receiver/schedule
            tasks.append(client.post(f"{r.rstrip('/')}/receiver/schedule", json=schedule.model_dump()))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    failed = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception) or getattr(res, "status_code", 500) >= 300:
            failed.append(req.receivers[idx])

    if failed:
        # Reset state to allow retry (since no concurrency assumed, this is simplest)
        state.in_progress = False
        raise HTTPException(status_code=502, detail={"message": "Failed to schedule some receivers", "failed": failed})

    # Sender itself waits and runs bash at scheduled time
    await wait_until_epoch(start_time_epoch)
    sender_proc = run_bash(req.bash_cmd)

    return {
        "run_id": run_id,
        "start_time_epoch": start_time_epoch,
        "start_time_utc": datetime.fromtimestamp(start_time_epoch, tz=timezone.utc).isoformat(),
        "scheduled_receivers": req.receivers,
        "sender_bash": {
            "returncode": sender_proc.returncode,
            "stdout": sender_proc.stdout,
            "stderr": sender_proc.stderr
        }
    }


@app.post("/sender/submit")
async def sender_submit(result: ReceiverResult):
    """
    Receivers POST their result here.
    Once all expected receivers have submitted, aggregate and write CSV.
    """
    if not state.in_progress or result.run_id != state.run_id:
        raise HTTPException(status_code=409, detail="No matching workflow in progress.")

    state.received[result.receiver_host] = result

    done = (len(state.received) >= len(state.expected_receivers))
    payload = {
        "received_count": len(state.received),
        "expected_count": len(state.expected_receivers),
        "done": done
    }

    if done:
        out = aggregate_and_save(
            run_id=state.run_id,
            load_name=state.load_name or "load",
            start_time_epoch=state.start_time_epoch or now_epoch(),
            results=list(state.received.values())
        )
        # Clear state
        state.in_progress = False
        payload["output"] = out

    return payload


@app.get("/sender/status")
async def sender_status():
    return {
        "in_progress": state.in_progress,
        "run_id": state.run_id,
        "load_name": state.load_name,
        "start_time_epoch": state.start_time_epoch,
        "expected_receivers": state.expected_receivers,
        "received_hosts": list(state.received.keys())
    }


# ----------------------------
# Receiver endpoint
# ----------------------------

@app.post("/receiver/schedule")
async def receiver_schedule(req: ScheduleRequest):
    """
    Receiver:
      1) wait until start_time_epoch
      2) run bash_cmd
      3) build dataframe
      4) send back to sender /sender/submit
    """
    global receiver_in_progress, receiver_current_run_id

    if receiver_in_progress:
        raise HTTPException(status_code=409, detail="Receiver already running a workflow.")
    receiver_in_progress = True
    receiver_current_run_id = req.run_id

    try:
        await wait_until_epoch(req.start_time_epoch)
        proc = run_bash(req.bash_cmd)

        df = sample_dataframe(req.load_name, req.start_time_epoch)
        df_json = df.to_json(orient="split")

        result = ReceiverResult(
            run_id=req.run_id,
            receiver_host=hostname(),
            load_name=req.load_name,
            start_time_epoch=req.start_time_epoch,
            dataframe_json=df_json,
            bash_returncode=proc.returncode,
            bash_stdout=proc.stdout,
            bash_stderr=proc.stderr
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{req.sender_url.rstrip('/')}/sender/submit", json=result.model_dump())
            if resp.status_code >= 300:
                raise RuntimeError(f"Sender submit failed: {resp.status_code} {resp.text}")

        return {"ok": True, "sent_to": req.sender_url, "receiver_host": hostname()}
    finally:
        receiver_in_progress = False
        receiver_current_run_id = None


# ----------------------------
# Entrypoint hint
# ----------------------------
# Run with:
#   uvicorn server:app --host 0.0.0.0 --port 11451
