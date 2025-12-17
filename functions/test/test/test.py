import json
import time
import random

import redis
from locust import FastHttpUser, task, events

# ---------------------------------------------------------------------------
# Shared Redis client (one per Locust worker process)
# ---------------------------------------------------------------------------
r = redis.Redis(
    host="127.0.0.1",  # Same as your mock-client example
    port=32204,
    decode_responses=True,
)


def poll_result(req_id: str, timeout_s: float = 5.0, poll_interval_s: float = 0.5):
    """
    Poll Redis for the given req_id until a value is found or timeout occurs.
    Returns the parsed JSON value from Redis on success.
    Raises TimeoutError on timeout.
    """
    deadline = time.monotonic() + timeout_s

    while True:
        val = r.get(req_id)
        if val is not None:
            # Optional: r.delete(req_id) if you want to clean up
            return json.loads(val)

        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for {req_id}")

        time.sleep(poll_interval_s)


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
        return random.expovariate(1.4)  # mean 0.7s

    @task
    def ml_image_processing_e2e(self):
        image_name = "000b7b74-0a22-4d0c-b717-e240fdc5d555.png"
        payload = {"image_name": image_name}

        # Start E2E timer before sending the request
        e2e_start = time.monotonic()

        # -------------------------------------------------------------------
        # Step 1: send async request (equivalent to your send_request)
        # -------------------------------------------------------------------
        with self.client.post(
            "/ml-image-processing",
            json=payload,
            name="ml-image-processing-create",
            catch_response=True,
        ) as resp:
            # Non-200 => mark HTTP request failed AND record a failed E2E event
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}")

                e2e_rt_ms = (time.monotonic() - e2e_start) * 1000
                events.request.fire(
                    request_type="E2E",
                    name="ml-image-processing-e2e",
                    response_time=e2e_rt_ms,
                    response_length=0,
                    exception=Exception(
                        f"Create failed with status {resp.status_code}"
                    ),
                )
                return

            # Parse req_id from response JSON
            try:
                res_payload = resp.json()
            except ValueError:
                resp.failure("Invalid JSON in create response")

                e2e_rt_ms = (time.monotonic() - e2e_start) * 1000
                events.request.fire(
                    request_type="E2E",
                    name="ml-image-processing-e2e",
                    response_time=e2e_rt_ms,
                    response_length=0,
                    exception=Exception("Invalid JSON in create response"),
                )
                return

            req_id = res_payload.get("req_id")
            if not req_id:
                resp.failure("Missing req_id in create response")

                e2e_rt_ms = (time.monotonic() - e2e_start) * 1000
                events.request.fire(
                    request_type="E2E",
                    name="ml-image-processing-e2e",
                    response_time=e2e_rt_ms,
                    response_length=0,
                    exception=Exception("Missing req_id in create response"),
                )
                return

            # If we reach here, the create call is considered successful
            resp.success()

        # -------------------------------------------------------------------
        # Step 2: poll Redis for completion (equivalent to your poll_result)
        # -------------------------------------------------------------------
        try:
            result = poll_result(req_id, timeout_s=5.1)
            # Optionally log/inspect:
            # print(result)
        except TimeoutError as exc:
            # Poll loop timeout => record failed E2E event
            e2e_rt_ms = (time.monotonic() - e2e_start) * 1000
            events.request.fire(
                request_type="E2E",
                name="ml-image-processing-e2e",
                response_time=e2e_rt_ms,
                response_length=0,
                exception=exc,
            )
            return

        # -------------------------------------------------------------------
        # Step 3: record successful E2E metric
        # -------------------------------------------------------------------
        e2e_rt_ms = (time.monotonic() - e2e_start) * 1000
        # If you want to treat Redis completion as a “success” request:
        events.request.fire(
            request_type="E2E",
            name="ml-image-processing-e2e",
            response_time=e2e_rt_ms,
            response_length=0,
            exception=None,
        )
        # If you also want to see the result per request, you could attach it
        # to logs, but avoid printing too much during large tests.
        # print(f"E2E result for {req_id}: {result}")



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
