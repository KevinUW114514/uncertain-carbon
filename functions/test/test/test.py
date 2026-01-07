import json
import time
import random

import redis
from locust import FastHttpUser, task, events
import locust.stats
locust.stats.CONSOLE_STATS_INTERVAL_SEC = 5

# ---------------------------------------------------------------------------
# Shared Redis client (one per Locust worker process)
# ---------------------------------------------------------------------------
r = redis.Redis(
    host="127.0.0.1",  # Same as your mock-client example
    port=32204,
    decode_responses=True,
)

SLO_MS = 3000           # QoS target: > 3s counts as failure
POLL_TIMEOUT_S = 3.1   # "Did not complete" threshold (should be > SLO)
POLL_INTERVAL_S = 0.5

def poll_result(req_id: str, timeout_s: float, poll_interval_s: float):
    """
    Poll Redis for the given req_id until a value is found or timeout occurs.
    Returns parsed JSON on success; raises TimeoutError on timeout.
    """
    deadline = time.monotonic() + timeout_s

    while True:
        val = r.get(req_id)
        if val is not None:
            return json.loads(val)

        if time.monotonic() >= deadline:
            raise TimeoutError()

        time.sleep(poll_interval_s)


def fire_e2e(name: str, e2e_start: float, exception: Exception | None):
    """
    Record an E2E request result into Locust stats.
    """
    e2e_rt_ms = (time.monotonic() - e2e_start) * 1000.0
    events.request.fire(
        request_type="E2E",
        name=name,
        response_time=e2e_rt_ms,
        response_length=0,
        exception=exception,
    )


class ServerlessUser(FastHttpUser):
    """
    Locust user that:
    1. POSTs to /ml-image-processing to start async workflow.
    2. Polls Redis for req_id completion.
    3. Records an E2E metric from POST start to Redis completion.
    """

    # You can also set host on CLI; set here if you prefer:
    # host = "http://127.0.0.1:8000"

    # Same idea as your wait_time
    def wait_time(self):
        return 1 # random.expovariate(1)  # mean 1s

    @task
    def ml_image_processing_e2e(self):
        # image_name = "000b7b74-0a22-4d0c-b717-e240fdc5d555_processed.png" # s1 object-detection
        image_name = "000b7b74-0a22-4d0c-b717-e240fdc5d555.png" # s1 image-processing
        # image_name = "0d74cfde-b4d2-48dc-bf92-2234717025a8.png"   # s2
        # image_name = "2f36e9dd-b8c2-407d-bac9-f64fa23fd1a6.png"  # test-s3
        # image_name = "e937afcb-aad7-4478-a3e8-59ff4e97262a.png"  # test-s4
        payload = {"image_name": image_name}

        metric_name = "ml-image-processing-e2e"
        e2e_start = time.monotonic()

        # -----------------------------
        # Step 1: create async request
        # -----------------------------
        with self.client.post(
            "/ml-image-processing",
            # "/ml-object-detection",
            json=payload,
            name="ml-image-processing-create",
            catch_response=True,
            timeout=1,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}")
                fire_e2e(metric_name, e2e_start, Exception(f"Create failed: HTTP {resp.status_code}"))
                return

            try:
                res_payload = resp.json()
            except ValueError:
                resp.failure("Invalid JSON in create response")
                fire_e2e(metric_name, e2e_start, Exception("Create failed: invalid JSON"))
                return

            req_id = res_payload.get("req_id")
            if not req_id:
                resp.failure("Missing req_id in create response")
                fire_e2e(metric_name, e2e_start, Exception("Create failed: missing req_id"))
                return

            # Mark the HTTP create call as successful
            resp.success()

        # -------------------------------------------------------------------
        # Step 2: poll Redis for completion (equivalent to your poll_result)
        # -------------------------------------------------------------------
        try:
            _ = poll_result(req_id, timeout_s=POLL_TIMEOUT_S, poll_interval_s=POLL_INTERVAL_S)
        except TimeoutError as exc:
            # Did not complete => failure
            fire_e2e(metric_name, e2e_start, exc)
            return
        except Exception as exc:
            # Any other polling/Redis error => failure
            fire_e2e(metric_name, e2e_start, Exception())
            return

        # -----------------------------
        # Step 3: enforce SLO and record
        # -----------------------------
        e2e_rt_ms = (time.monotonic() - e2e_start) * 1000.0
        if e2e_rt_ms > SLO_MS:
            # Completed but violated QoS => failure
            fire_e2e(metric_name, e2e_start, Exception())
        else:
            # Completed within QoS => success
            fire_e2e(metric_name, e2e_start, None)



# import base64
# import json
# import logging
# import os
# import pickle
# import random
# import string
# import sys
# import time
# import uuid
# from pathlib import Path
# import pickle


# import urllib3

# import threading
# import locust.stats
# from locust import HttpUser, LoadTestShape, TaskSet, between, constant, tag, task, events
# from locust.contrib.fasthttp import FastHttpUser
# from locust.runners import MasterRunner

# import statistics


# # PROJECT_DIR = Path(__file__).resolve().parents[1]
# # sys.path.append(str(PROJECT_DIR))

# locust.stats.CSV_STATS_INTERVAL_SEC = 1  # second
# # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# random.seed(114514)
# logging.basicConfig(level=logging.INFO)

# ACTION_NAME = "my-function"

# results = []
# lock = threading.Lock()


# class ServerlessUser(FastHttpUser):
    

#     def wait_time(self):
#         return random.expovariate(1.4)  # mean 0.7s

#     @tag("ml-image-processing")
#     @task
#     def test(self):

#         action_params = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"}
#         # url_params = {"blocking": "true", "result": "false"}
#         response = self.client.post(
#             url="/ml-image-processing",
#             # params=url_params,
#             json=action_params,
#             # auth=(USER_PASS[0], USER_PASS[1]),
#             name=ACTION_NAME,
#         )
#         data = response
#         print(data.content)
        
#     # @tag("ml-object-detection")
#     # @task
#     # def test(self):

#     #     action_params = {"image_name": "000b7b74-0a22-4d0c-b717-e240fdc5d555_processed.png"}
#     #     # url_params = {"blocking": "true", "result": "false"}
#     #     response = self.client.post(
#     #         url="/ml-object-detection",
#     #         # params=url_params,
#     #         json=action_params,
#     #         # auth=(USER_PASS[0], USER_PASS[1]),
#     #         name=ACTION_NAME,
#     #     )
#     #     data = response
#     #     print(data.content)


# # class StagesShape(LoadTestShape):
# #     # stages = [
# #     #     {"duration": 30, "users": 500, "spawn_rate": 100},
# #     #     {"duration": 60, "users": 600, "spawn_rate": 100},
# #     #     {"duration": 90, "users": 700, "spawn_rate": 100},
# #     #     {"duration": 120, "users": 800, "spawn_rate": 100},
# #     #     {"duration": 150, "users": 900, "spawn_rate": 100},
# #     #     {"duration": 180, "users": 1000, "spawn_rate": 100},
# #     # ]
# #     with open("/mnt/locust/trace.pickle", "rb") as fp:  # Unpickling
# #         stages = pickle.load(fp)

# #     def tick(self):
# #         run_time = self.get_run_time()

# #         self.stages = self.stages
# #         for stage in self.stages:
# #             if run_time < stage["duration"]:
# #                 tick_data = (stage["users"], stage["spawn_rate"])
# #                 return tick_data

# #         return None
