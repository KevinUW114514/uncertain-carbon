import base64
import json
import logging
import os
import pickle
import random
import string
import sys
import time
import uuid
from pathlib import Path
import pickle


import urllib3

import threading
import locust.stats
from locust import HttpUser, LoadTestShape, TaskSet, between, constant, tag, task, events
from locust.contrib.fasthttp import FastHttpUser
from locust.runners import MasterRunner

import statistics


# PROJECT_DIR = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_DIR))

locust.stats.CSV_STATS_INTERVAL_SEC = 1  # second
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
random.seed(114514)
logging.basicConfig(level=logging.INFO)

ACTION_NAME = "my-function"

results = []
lock = threading.Lock()


class ServerlessUser(FastHttpUser):
    

    def wait_time(self):
        return random.expovariate(1.4)  # mean 0.7s

    @tag(ACTION_NAME)
    @task
    def test(self):

        action_params = {}
        # url_params = {"blocking": "true", "result": "false"}
        response = self.client.post(
            url="/function/my-function",
            # params=url_params,
            json=action_params,
            # auth=(USER_PASS[0], USER_PASS[1]),
            name=ACTION_NAME,
        )
        data = response
        print(data.content)


# class StagesShape(LoadTestShape):
#     # stages = [
#     #     {"duration": 30, "users": 500, "spawn_rate": 100},
#     #     {"duration": 60, "users": 600, "spawn_rate": 100},
#     #     {"duration": 90, "users": 700, "spawn_rate": 100},
#     #     {"duration": 120, "users": 800, "spawn_rate": 100},
#     #     {"duration": 150, "users": 900, "spawn_rate": 100},
#     #     {"duration": 180, "users": 1000, "spawn_rate": 100},
#     # ]
#     with open("/mnt/locust/trace.pickle", "rb") as fp:  # Unpickling
#         stages = pickle.load(fp)

#     def tick(self):
#         run_time = self.get_run_time()

#         self.stages = self.stages
#         for stage in self.stages:
#             if run_time < stage["duration"]:
#                 tick_data = (stage["users"], stage["spawn_rate"])
#                 return tick_data

#         return None
