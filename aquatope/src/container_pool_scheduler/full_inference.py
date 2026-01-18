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

from personal.projects.server.qa.python_models.add_sub import model

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
    utils.inference(datasets=datasets, model=predict, not_used1=False, not_used2=128)
    # df = utils.inference_conformal(datasets=datasets, model=predict, mc_dropout=False)
    df = utils.inference_conformal(
        datasets=datasets,
        model=predict,
        k_neighbors: int = 2,
        alpha: float = 0.1,                # miscoverage; 90% CI => alpha=0.1
        mc_samples: int = 300,
        agg: str = "median",               # "median", "mean", "trimmed_mean"
        lam_recency: float = 0.0,          # recency decay weights inside conditioning window
        noise: str = "gaussian",           # "studentt" or "gaussian"
        studentt_df: float = 4.0,
        bandwidth_scale: float = 0.3,      # typical 0.3–0.8
        clamp_to_ci: bool = False,
        # NEW: decouple bias tracking from uncertainty estimation
        bias_window: int = 1,              # keep 1 if that gives best point accuracy
        scale_window: int = 12,            # longer history for uncertainty (e.g., 72 hours)
        min_history: int = 5,              # minimum history before generating non-degenerate samples
        use_knn: bool = True,              # actually use regime conditioning (was disabled in your code)
        min_bw: float = 1e-6,
        # Optional: if your model output is a delta relative to last x, set use_delta=True
        use_delta: bool = False,
    )
    
#     # 1) Build profile on a held-out window
#     utils.build_and_save_signed_evt_profile(
#         datasets=datasets,
#         model=predict,
#         loader_key="calibration",
#         u_quantile=0.6,       # start here; go 0.975/0.99 when you have more data
#         min_tail_each=20,      # raise when you have more data
#         out_json="evt_signed.json",
#         out_bulk="evt_bulk_signed.npy",
#     )

#     # 2) Inference with intervals and MC mean forecast
#     utils.inference_with_signed_EVT_MC(
#         datasets=datasets,
#         model=predict,
#         evt_json_path="evt_signed.json",
#         evt_bulk_path="evt_bulk_signed.npy",
#         K=10000,               # important for stable p01/p99 and mean under heavy tails
# )


    end = time.time()
    print("time:", end - start)
    
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt

    # CSV_PATH = "inference_results.csv"


    # df = pd.read_csv(CSV_PATH)

    # required = ["predicted", "target", "p05", "p95", "p01", "p99"]
    # missing = [c for c in required if c not in df.columns]
    # if missing:
    #     raise ValueError(f"Missing columns in {CSV_PATH}: {missing}")

    # # Ensure numeric
    # for c in required:
    #     df[c] = pd.to_numeric(df[c], errors="coerce")
    # df = df.dropna(subset=required).reset_index(drop=True)

    # # Create an x-axis index in file order (proxy for time)
    # t = np.arange(len(df))

    # # ----- Plot 1: Target/Predicted with uncertainty bands -----
    # plt.figure()
    # plt.fill_between(t, df["p01"].values, df["p99"].values, alpha=0.2, label="p01–p99")
    # plt.fill_between(t, df["p05"].values, df["p95"].values, alpha=0.35, label="p05–p95")
    # plt.plot(t, df["target"].values, label="target")
    # plt.plot(t, df["predicted"].values, label="predicted")
    # plt.title("Traffic forecast with EVT-MC uncertainty bands (row order)")
    # plt.xlabel("index (row order)")
    # plt.ylabel("requests/min")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("plot_timeseries_bands.png", dpi=160)

    # # ----- Plot 2: Parity plot predicted vs target -----
    # # Color by absolute percentage error (avoid division by zero)
    # denom = np.maximum(df["target"].values, 1e-12)
    # ape = np.abs(df["predicted"].values - df["target"].values) / denom

    # plt.figure()
    # plt.scatter(df["target"].values, df["predicted"].values, s=8, alpha=0.5)
    # # 45-degree line
    # mn = min(df["target"].min(), df["predicted"].min())
    # mx = max(df["target"].max(), df["predicted"].max())
    # plt.plot([mn, mx], [mn, mx])
    # plt.title("Parity plot: predicted vs target")
    # plt.xlabel("target")
    # plt.ylabel("predicted")
    # plt.tight_layout()
    # plt.savefig("plot_parity.png", dpi=160)

    # # ----- Plot 3: Interval diagnostics -----
    # width_90 = (df["p95"] - df["p05"]).values
    # width_98 = (df["p99"] - df["p01"]).values

    # inside_90 = ((df["target"] >= df["p05"]) & (df["target"] <= df["p95"])).mean()
    # inside_98 = ((df["target"] >= df["p01"]) & (df["target"] <= df["p99"])).mean()

    # plt.figure()
    # plt.hist(width_90, bins=50, alpha=0.6, label="width p05–p95")
    # plt.hist(width_98, bins=50, alpha=0.6, label="width p01–p99")
    # plt.title(f"Interval width distributions | coverage90={inside_90:.3f}, coverage98={inside_98:.3f}")
    # plt.xlabel("interval width")
    # plt.ylabel("count")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("plot_interval_widths.png", dpi=160)

    # # ----- Print concise summary -----
    # print("Saved plots:")
    # print("  plot_timeseries_bands.png")
    # print("  plot_parity.png")
    # print("  plot_interval_widths.png")
    # print()
    # print("Coverage (empirical):")
    # print(f"  inside p05–p95 (nominal 0.90): {inside_90:.4f}")
    # print(f"  inside p01–p99 (nominal 0.98): {inside_98:.4f}")
    # print()
    # print("Interval widths (summary):")
    # print(f"  p05–p95 width median={np.median(width_90):.4f}, mean={np.mean(width_90):.4f}")
    # print(f"  p01–p99 width median={np.median(width_98):.4f}, mean={np.mean(width_98):.4f}")

if __name__ == "__main__":
    main()
