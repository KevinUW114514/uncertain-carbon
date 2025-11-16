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

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

import models.variational_dropout as vd
from models.predict import *

import huawei_data as data
import utils

cpu = lambda x: x.cpu().detach().numpy()

MODEL_ARTIFACTS_DIR = SCHED_DIR / "model_artifacts"


def load_trained_model(model_artifacts_dir: str, device: str):
    predict_loc = os.path.join(model_artifacts_dir, "predict.pt")
    predict = torch.load(predict_loc, map_location=device, weights_only=False).eval()
    return predict.to(device)

def main():
    # --------------------------------------------------------------------------
    # Parse args
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train prediction network")
    parser.add_argument("--n_input_steps", action="store", type=int)
    parser.add_argument("--n_output_steps", action="store", type=int)
    parser.add_argument("--dataset_dir", action="store", type=str)

    args = parser.parse_args()
    n_input_steps = args.n_input_steps
    n_output_steps = args.n_output_steps
    dataset_path = args.dataset_dir
    model_artifacts_dir = SCHED_DIR / "model_artifacts"

    device = utils.get_device()
    predict = load_trained_model(model_artifacts_dir=model_artifacts_dir, device=device)
    
    samples = data.pipeline(
        n_input_steps=n_input_steps,
        n_pred_steps=n_output_steps,
        dataset_path=dataset_path,
        is_inference=True,
    )

    datasets = data.get_datasets(
        samples=samples, n_input_steps=n_input_steps, pretraining=False
    )

    start = time.time()
    mean, var = utils.inference(datasets=datasets, model=predict, mc_dropout=True, batch_size=128)
    end = time.time()
    print("time:", end - start)
    print(mean, var)


if __name__ == "__main__":
    main()
