import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd

# INPUT = "/home/kevin/carbon/huawei-dataset/data/output.csv"
INPUT = "/home/kevin/carbon/aquatope/src/container_pool_scheduler/train_samples.csv"
COLUMNS = ["invocation_rate", "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos"]
GROUP_SIZE = 60

# Load only the needed columns
df = pd.read_csv(INPUT, usecols=COLUMNS)

# Convert each row to a tuple in the specified order
input_data = list(df.itertuples(index=False, name=None))

# Group every 60 records (drop any remainder)
# groups = [rows_as_tuples[i:i+GROUP_SIZE]
#           for i in range(0, len(rows_as_tuples) - len(rows_as_tuples) % GROUP_SIZE, GROUP_SIZE)]

# 'groups' is now: List[List[Tuple(invocation_rate, hour_sin, hour_cos, day_sin, day_cos)]]
# print(f"Created {len(groups)} groups of {GROUP_SIZE} records each.")
# print("First tuple of first group:", groups[0][0] if groups and groups[0] else None)


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

import models.variational_dropout as vd
from models.predict import *

import data
import utils

cpu = lambda x: x.cpu().detach().numpy()

MODEL = None
MODEL_ARTIFACTS_DIR = SCHED_DIR / "model_artifacts"


def load_trained_model(model_artifacts_dir: str, device: str):
    predict_loc = os.path.join(model_artifacts_dir, "lstm_encoder_decoder.pt")
    predict = torch.load(predict_loc, map_location=device, weights_only=False).eval()
    return predict.to(device)


def dropout_on(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.train()


def dropout_off(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.eval()


def inference(x: list, external: list, mc_dropout: bool = False, batch_size: int = 1):
    global MODEL

    device = utils.get_device()
    if MODEL is None:
        MODEL = load_trained_model(
            model_artifacts_dir=MODEL_ARTIFACTS_DIR, device=device
        )
    if mc_dropout:
        MODEL = MODEL.apply(dropout_on)
    else:
        MODEL = MODEL.apply(dropout_off)

    x = np.expand_dims(x, axis=0)
    external = np.expand_dims(external, axis=0)
    x = torch.tensor(np.array(x, dtype=np.float32), device=device)
    external = torch.tensor(np.array(external, dtype=np.float32), device=device)
    # print(x.shape, external.shape)
    # input("Nao")

    res = []
    print(f"batch_size: {batch_size}")
    for _ in range(batch_size):
        # res.append(MODEL((x, external)).to(device).item())
        res.append(MODEL(x).cpu().detach())
    print(f"res len: {len(res)}, res[0] shape: {res[0].shape}")
    print(res[0])
    mean = np.mean(res)
    var = np.var(res)
    return mean, var


def main():
    # --------------------------------------------------------------------------
    # Parse args
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train prediction network")
    parser.add_argument(dest="filenames", metavar="filename", nargs="*")
    parser.add_argument("--n_input_steps", action="store", type=int)
    parser.add_argument("--n_output_steps", action="store", type=int)

    args = parser.parse_args()
    n_input_steps = args.n_input_steps
    n_output_steps = args.n_output_steps
    model_artifacts_dir = SCHED_DIR / "model_artifacts"

    device = utils.get_device()
    predict = load_trained_model(model_artifacts_dir=model_artifacts_dir, device=device)

    x = []
    for i in range(n_input_steps):
        x.append(input_data[i])
    x = np.array(x, dtype=np.float32)
    print(x[:, 0])
    external = [0, 0, 0, 0]
    start = time.time()
    mean, var = inference(x=x, external=external, mc_dropout=True, batch_size=1)
    end = time.time()
    print("time:", end - start)
    print(mean, var)


if __name__ == "__main__":
    main()
