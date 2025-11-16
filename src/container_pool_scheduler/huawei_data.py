import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import json

class AzureFunctionDataset(Dataset):
    """
    PyTorch Dataset class for Metro Traffic dataset
    """

    def __init__(self, samples: dict, n_input_steps: int,
                 key: str = 'train', pretraining: bool = True):
        # calculate normalisation parameters for columns `invocation_rate`
        # from training data
        # self.X_train = samples['train'][:, :n_input_steps, :].copy()

        cols_to_normalise = [1]
        json_data = json.load(open("train_invocation_rate_normalization.json", "r"))
        self.train_mu, self.train_sigma = [json_data["mu"]], [json_data["sigma"]]
        # normalise dataset
        self.original_X = samples[key][:, 0, 1].copy()
        self.X = samples[key][:, :n_input_steps, :].copy()
        self.y = samples[key][:, n_input_steps:, :].copy()
        for c, col in enumerate(cols_to_normalise):
            self.X[:, :, col] = (self.X[:, :, col] -
                                 self.train_mu[c]) / (self.train_sigma[c])
            self.y[:, :, col] = (self.y[:, :, col] -
                                 self.train_mu[c]) / (self.train_sigma[c])

        # provide external features for prediction network
        self.pretraining = pretraining
        self.prediction_cols = [2, 3]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        invocation_idx = 1
        # print(f"[hu] __getitem__ idx: {idx}")
        # input("debug")
        # x = torch.Tensor(self.X[idx, :, :]).float()
        # print(f"[hu] x shape: {x.shape}")
        # input("debug")
        x = torch.Tensor(self.X[idx, :, invocation_idx]).float().unsqueeze(-1)
        # print(f"[hu] x shape: {x.shape}")
        # input("debug")
        if self.pretraining:
            y = torch.Tensor(self.y[idx, :, invocation_idx] -
                             self.X[idx, 0, invocation_idx]).float()
            # y = torch.Tensor(self.y[idx, :, invocation_idx]).float()
            
            # print((f"x: {x}"))
            # print(f"1st: {self.y[idx, :, invocation_idx]}")
            # print(f"2nd: {self.X[idx, 0, invocation_idx]}")
            # print(f"y: {y}")
            # input("debug")
        else:
            y = self.y[idx, :, :].copy()
            y[:, invocation_idx] -= self.X[idx, 0, invocation_idx]
            y = torch.Tensor(
                y[:, [invocation_idx] + self.prediction_cols]).float()

            # print(f"[hu] y shape: {y.shape}")
            # input("debug")
            
        return x, y, self.X[idx, 0, 0], self.y[idx, 0, 0], self.original_X[idx]  


def build_features(dataset_path: str, units_per_hour: int, data_type: str) -> pd.DataFrame:
    if data_type == "train":
        file_name = "train.csv"
    elif data_type == "valid":
        file_name = "valid.csv"
    elif data_type == "inference":
        file_name = "test.csv"
        
    df = pd.read_csv(dataset_path + file_name)
    data = df.copy().sort_values("time").reset_index(drop=True)

    hours_since_0 = (data["time"] // units_per_hour).astype(int)
    data["hour_of_day"] = (hours_since_0 % 24).astype(int)
    data["day_of_week"] = ((hours_since_0 // 24) % 7).astype(int)
    data["hour"] = (data["time"] // units_per_hour).astype(int)

    hourly_df = (
        data.groupby("hour", as_index=False)
            .agg(
                hour_invocation=("invocation_rate", "mean"),
                hour_of_day=("hour_of_day", "first"),
                day_of_week=("day_of_week", "first"),
            )
    )

    # (Optional) ensure nice dtypes
    hourly_df["hour"] = hourly_df["hour"].astype(int)
    # hourly_df["hour_of_day_sin"] = np.sin(2 * np.pi * (hourly_df["hour_of_day"] / 24)).astype(float)
    # hourly_df["hour_of_day_cos"] = np.cos(2 * np.pi * (hourly_df["hour_of_day"] / 24)).astype(float)
    # hourly_df["day_of_week_sin"] = np.sin(2 * np.pi * (hourly_df["day_of_week"] / 7)).astype(float)
    # hourly_df["day_of_week_cos"] = np.cos(2 * np.pi * (hourly_df["day_of_week"] / 7)).astype(float)
    # hourly_df.drop(columns=["hour_of_day", "day_of_week"], inplace=True)

    # hourly_df.sort_values(["day_of_week", "hour_of_day"], ascending=True, inplace=True)
    hourly_df.sort_values(["hour"], ascending=True, inplace=True)
    hourly_df.to_csv(f"debug_24h_{data_type}.csv", index=False)
    
    print(f"[hu] save debug_24h_{data_type}.csv")
    # input("debug")
    
    return hourly_df


def get_datasets(samples: dict, n_input_steps: int, pretraining=True) -> dict:
    datasets = {}
    for key, sample in samples.items():
        datasets[key] = AzureFunctionDataset(
            samples, n_input_steps, key, pretraining)

    return datasets


def get_dataloaders(datasets: dict, train_batch_size: int = 0) -> dict:
    dataloaders = {}
    for key, dataset in datasets.items():
        if key == 'train':
            dataloaders[key] = DataLoader(dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True)
        else:
            dataloaders[key] = DataLoader(dataset,
                                          batch_size=len(dataset),
                                          shuffle=False)

    return dataloaders


def pipeline(n_input_steps: int, n_pred_steps: int,
             dataset_path: str,
             num_days: int = -1,
             hash_function: str = "",
             is_inference: bool = False
            ) -> Tuple[pd.DataFrame, dict, dict]:
    datasets = dict()
    
    if not is_inference:
        train_df = build_features(dataset_path=dataset_path, units_per_hour=3600, data_type="train")
        valid_df = build_features(dataset_path=dataset_path, units_per_hour=3600, data_type="valid")
        datasets['train'] = train_df
        datasets['valid'] = valid_df
    else:
        infer_df = build_features(dataset_path=dataset_path, units_per_hour=3600, data_type="inference")
        datasets['inference'] = infer_df
        
    samples = create_samples(datasets, n_input_steps, n_pred_steps)

    return samples


def full_pipeline(params):
    # run the data preprocessing pipeline to create dataset
    df, split_dfs, samples = pipeline(
        n_input_steps=params['data']['n_input_steps'],
        n_pred_steps=params['models']['prediction']['n_output_steps'],
        dataset_dir='../data')

    # we modify the get_datasets function to return external features in the y labels
    datasets = get_datasets(
        samples, params['data']['n_input_steps'], pretraining=False)

    dataloaders = get_dataloaders(datasets, train_batch_size=256)

    return df, dataloaders


def create_samples(datasets: dict, n_input_steps: int, n_pred_steps: int) -> dict:
    data = {}
    for key, dataset in datasets.items():
        dataset = datasets[key]
        
        n_timesteps = n_input_steps + n_pred_steps

        # 1) Drop rows with NaN or 0 in invocation_rate
        df_clean = dataset.copy()
        df_clean = df_clean.dropna(subset=["hour_invocation"])
        df_clean = df_clean[df_clean["hour_invocation"] != 0]

        # If all rows were dropped, return an empty tensor
        if df_clean.empty:
           raise ValueError(f"Not enough data after cleaning for {key} dataset.")

        # We preserve the original order of rows; no sorting.
        hours = df_clean["hour"].to_numpy()

        if key == "train":
            train_mu = df_clean["hour_invocation"].mean()
            train_sigma = df_clean["hour_invocation"].std()
            json.dump({"mu": train_mu, "sigma": train_sigma},
                      open("train_invocation_rate_normalization.json", "w"))
        
        # Use only numeric columns for the tensor (torch requires numeric types)
        values = df_clean.to_numpy()

        windows = []

        # 2 & 3) Sliding window with consecutive hour constraint
        max_start = len(df_clean) - n_timesteps
        for start in range(max_start + 1):
            end = start + n_timesteps

            window_hours = hours[start:end]

            # Check consecutiveness: hour[i+1] - hour[i] == 1 for all i
            if np.all(np.diff(window_hours) == 1):
                # Append the corresponding records (all numeric features)
                windows.append(values[start:end])

        if not windows:
            raise ValueError(f"No valid samples found for {key} dataset.")

        windows_arr = np.stack(windows, axis=0)
        data[key] = windows_arr

        # print(data[key].shape)
        print(data[key].shape[0],
              f'samples of {n_input_steps} input steps and {n_pred_steps} output steps in', key)
        # print(windows_arr[0].astype(np.int32))
        # input("debug")
        
    return data


if __name__ == "__main__":
    pipeline()
