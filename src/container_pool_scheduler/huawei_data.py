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

    def __init__(
        self,
        samples: dict,
        n_input_steps: int,
        key: str = "train",
        pretraining: bool = True,
    ):
        # calculate normalisation parameters for columns `invocation_rate`
        # from training data
        # self.X_train = samples['train'][:, :n_input_steps, :].copy()

        cols_to_normalise = [2]
        json_data = json.load(open("train_invocation_rate_normalization.json", "r"))
        self.train_mu, self.train_sigma = [json_data["mu"]], [json_data["sigma"]]
        # normalise dataset
        self.original_X = samples[key][:, 0, 1].copy()
        self.X = samples[key][:, :n_input_steps, :].copy()
        self.y = samples[key][:, n_input_steps:, :].copy()
        # print(self.X[0][0])
        # input("debug")
        for c, col in enumerate(cols_to_normalise):
            self.X[:, :, col] = (self.X[:, :, col] - self.train_mu[c]) / (
                self.train_sigma[c]
            )
            self.y[:, :, col] = (self.y[:, :, col] - self.train_mu[c]) / (
                self.train_sigma[c]
            )

        # provide external features for prediction network
        self.pretraining = pretraining
        self.prediction_cols = [0, 3]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        invocation_idx = 2
        # print(f"[hu] __getitem__ idx: {idx}")
        # input("debug")
        # x = torch.Tensor(self.X[idx, :, :]).float()
        # print(f"[hu] x shape: {x.shape}")
        # input("debug")
        
        # print(f"[hu] x shape: {x.shape}")
        # input("debug")
        if self.pretraining:
            x = torch.Tensor(self.X[idx, :, invocation_idx]).float().unsqueeze(-1)
            # y = torch.Tensor(self.y[idx, :, invocation_idx]).float()

            # print((f"x: {x}"))
            # print(f"1st: {self.y[idx, :, invocation_idx]}")
            # print(f"2nd: {self.X[idx, 0, invocation_idx]}")
            # print(f"y: {y}")
            # print(torch.Tensor(self.X).float().shape)
            # print(torch.Tensor(self.X[idx, :, [invocation_idx]]).float().shape)
            # input("debug")
        else:
            x = torch.Tensor(self.X).float()[idx, :, [invocation_idx] + self.prediction_cols]
            
        y = torch.Tensor(
                self.y[idx, :, invocation_idx] - self.X[idx, 0, invocation_idx]
            ).float()
        
        return x, y, self.X[idx, 0, 0], self.y[idx, 0, 0], self.original_X[idx]


def build_features(
    dataset_path: str, units_per_hour: int, data_type: str
) -> pd.DataFrame:
    if data_type == "train":
        file_name = "train.csv"
    elif data_type == "valid":
        file_name = "valid.csv"
    elif data_type == "inference":
        file_name = "test.csv"

    df = pd.read_csv(dataset_path + file_name)
    data = df.copy().sort_values("time").reset_index(drop=True)

    hours_since_0 = (data["time"] // units_per_hour).astype(int)
    data["hour_of_day"] = ((hours_since_0) % 24).astype(int)
    data["day_of_week"] = ((hours_since_0 // 24) % 7).astype(int)
    data["hour"] = (data["time"] // units_per_hour).astype(int)

    hourly_df = data.groupby("hour", as_index=False).agg(
        hour_invocation=("invocation_rate", "mean"),
        hour_of_day=("hour_of_day", "first"),
        day_of_week=("day_of_week", "first"),
    )

    # (Optional) ensure nice dtypes
    hourly_df["hour"] = hourly_df["hour"].astype(int)
    hourly_df = hourly_df.dropna(subset=["hour_invocation"])
    hourly_df = hourly_df[hourly_df["hour_invocation"] >= 10]
    # hourly_df["hour_of_day_sin"] = np.sin(2 * np.pi * (hourly_df["hour_of_day"] / 24)).astype(float)
    # hourly_df["hour_of_day_cos"] = np.cos(2 * np.pi * (hourly_df["hour_of_day"] / 24)).astype(float)
    # hourly_df["day_of_week_sin"] = np.sin(2 * np.pi * (hourly_df["day_of_week"] / 7)).astype(float)
    # hourly_df["day_of_week_cos"] = np.cos(2 * np.pi * (hourly_df["day_of_week"] / 7)).astype(float)
    # hourly_df.drop(columns=["hour_of_day", "day_of_week"], inplace=True)

    # hourly_df.sort_values(["day_of_week", "hour_of_day"], ascending=True, inplace=True)
    hourly_df.sort_values(["hour"], ascending=True, inplace=True)
    hourly_df.to_csv(f"debug_24h_{data_type}.csv", index=False)
    
    minutes_15_df = (
        df.groupby(df.index // 15, as_index=False)
        .agg(
            {
                "time": "first",  # take the first timestamp in each 4-record group
                "invocation_rate": "mean",  # take the mean of the 4 rates
            }
        )
        .rename(columns={"time": "hour", "invocation_rate": "hour_invocation"})  # rename the field
    )
    # print(minutes_15_df.columns)
    # input("debug")

    minutes_15_df["hour"] = (minutes_15_df["hour"] / units_per_hour).astype(float)
    minutes_15_df = minutes_15_df.dropna(subset=["hour_invocation"])
    minutes_15_df = minutes_15_df[minutes_15_df["hour_invocation"] >= 10]
    minutes_15_df["day_of_week"] = minutes_15_df["hour"].astype(int) // 24 % 7
    minutes_15_df["hour_of_day"] = (minutes_15_df["hour"].astype(int) % 24)
    minutes_15_df.to_csv(f"debug_15min_{data_type}.csv", index=False)

    print(f"[hu] save debug_24h_{data_type}.csv")
    # input("debug")

    return hourly_df, minutes_15_df


def get_datasets(samples: dict, n_input_steps: int, pretraining=True) -> dict:
    datasets = {}
    for key, sample in samples.items():
        datasets[key] = AzureFunctionDataset(samples, n_input_steps, key, pretraining)

    return datasets


def get_dataloaders(datasets: dict, train_batch_size: int = 0) -> dict:
    dataloaders = {}
    for key, dataset in datasets.items():
        if key == "train":
            dataloaders[key] = DataLoader(
                dataset, batch_size=train_batch_size, shuffle=True
            )
        else:
            dataloaders[key] = DataLoader(
                dataset, batch_size=len(dataset), shuffle=False
            )

    return dataloaders


def pipeline(
    n_input_steps: int,
    n_pred_steps: int,
    dataset_path: str,
    num_days: int = -1,
    hash_function: str = "",
    is_inference: bool = False,
) -> Tuple[pd.DataFrame, dict, dict]:
    datasets = dict()
    units_per_hour = 3600

    if not is_inference:
        hour_train_df, minutes_train_df = build_features(
            dataset_path=dataset_path, units_per_hour=units_per_hour, data_type="train"
        )
        hour_valid_df, minutes_valid_df = build_features(
            dataset_path=dataset_path, units_per_hour=units_per_hour, data_type="valid"
        )
        datasets["train"] = (hour_train_df, minutes_train_df)
        datasets["valid"] = (hour_valid_df, minutes_valid_df)
    else:
        hour_infer_df, minutes_infer_df = build_features(
            dataset_path=dataset_path,
            units_per_hour=units_per_hour,
            data_type="inference",
        )
        datasets["inference"] = (hour_infer_df, minutes_infer_df)

    samples = create_samples(datasets, n_input_steps, n_pred_steps)

    return samples


def full_pipeline(params):
    # run the data preprocessing pipeline to create dataset
    df, split_dfs, samples = pipeline(
        n_input_steps=params["data"]["n_input_steps"],
        n_pred_steps=params["models"]["prediction"]["n_output_steps"],
        dataset_dir="../data",
    )

    # we modify the get_datasets function to return external features in the y labels
    datasets = get_datasets(samples, params["data"]["n_input_steps"], pretraining=False)

    dataloaders = get_dataloaders(datasets, train_batch_size=256)

    return df, dataloaders


# def create_samples(datasets: dict, n_input_steps: int, n_pred_steps: int) -> dict:
#     data = {}
#     for key, dataset in datasets.items():
#         dataset = datasets[key]

#         n_timesteps = n_input_steps + n_pred_steps

#         # 1) Drop rows with NaN or 0 in invocation_rate
#         df_clean = dataset.copy()
#         df_clean = df_clean.dropna(subset=["hour_invocation"])
#         df_clean = df_clean[df_clean["hour_invocation"] >= 10]

#         # If all rows were dropped, return an empty tensor
#         if df_clean.empty:
#             raise ValueError(f"Not enough data after cleaning for {key} dataset.")

#         # We preserve the original order of rows; no sorting.
#         hours = df_clean["hour"].to_numpy()

#         if key == "train":
#             train_mu = df_clean["hour_invocation"].mean()
#             train_sigma = df_clean["hour_invocation"].std()
#             json.dump(
#                 {"mu": train_mu, "sigma": train_sigma},
#                 open("train_invocation_rate_normalization.json", "w"),
#             )

#         # Use only numeric columns for the tensor (torch requires numeric types)
#         values = df_clean.to_numpy()

#         windows = []

#         # 2 & 3) Sliding window with consecutive hour constraint
#         max_start = len(df_clean) - n_timesteps
#         for start in range(max_start + 1):
#             end = start + n_timesteps

#             window_hours = hours[start:end]

#             # Check consecutiveness: hour[i+1] - hour[i] == 1 for all i
#             if np.all(np.diff(window_hours) == 1):
#                 # Append the corresponding records (all numeric features)
#                 windows.append(values[start:end])

#         if not windows:
#             raise ValueError(f"No valid samples found for {key} dataset.")

#         windows_arr = np.stack(windows, axis=0)
#         data[key] = windows_arr

#         # print(data[key].shape)
#         print(
#             data[key].shape[0],
#             f"samples of {n_input_steps} input steps and {n_pred_steps} output steps in",
#             key,
#         )
#         # print(windows_arr[0].astype(np.int32))
#         # input("debug")

#     return data

def create_samples(datasets: dict, n_input_steps: int, n_pred_steps: int) -> dict:
    data = {}
    for key, dataset in datasets.items():
        hourly_df, minutes_15_df = datasets[key]

        n_timesteps = n_input_steps + n_pred_steps

        hourly_df = hourly_df.copy().sort_values("hour")
        minutes_15_df = minutes_15_df.copy().sort_values("hour")
        
        # Use an index on 'hour' for fast .loc selection
        hourly_df = hourly_df.set_index("hour", drop=False)
        minutes_15_df = minutes_15_df.set_index("hour", drop=False)

        # Pre-calc the maximum starting hour we can consider
        max_hour = hourly_df["hour"].max()
        max_start = max_hour - n_timesteps # because we need x..x+23

        # Candidate starting hours: all hourly hours that can support a full lag span
        candidate_starts = hourly_df.loc[hourly_df["hour"] <= max_start, "hour"].values
        # print(f"candidate_starts: {candidate_starts}")
        # input("debug")

        all_feature_cols = sorted(set(hourly_df.columns) | set(minutes_15_df.columns))
        # print(all_feature_cols)
        # input("debug")
        windows = []

        for x in candidate_starts:
            # Required hourly hours: x .. x+23 (24 hourly points)
            hourly_hours_needed = [x + i for i in range(n_input_steps - 4)]

            # Required minutes_15 hours: x+22 + {0, 0.25, 0.5, 0.75}
            base_minute_hour = x + (n_timesteps - 5)  # x+22 when n_timesteps=24
            minutes_hours_needed = [base_minute_hour + 0.25 * k for k in range(4)]
            
            # print(f"hourly_hours_needed: {hourly_hours_needed}")
            # print(f"minutes_hours_needed: {minutes_hours_needed}")
            # input("debug")

            # Check availability of all hourly and quarter-hour timestamps
            if not set(hourly_hours_needed).issubset(hourly_df.index):
                print(f"missing hourly for start {x}")
                # input("debug")
                continue
            if not set(minutes_hours_needed).issubset(minutes_15_df.index):
                # print(minutes_hours_needed[0])
                # print(minutes_15_df.index)
                print(f"missing minutes for start {x}")
                # input("debug")
                continue

            # Retrieve blocks
            # First 23 hourly rows: x..x+22
            hourly_first = hourly_df.loc[hourly_hours_needed[:], all_feature_cols]
            # print(f"hourly_first shape: {hourly_first.shape}")
            # print(hourly_first)
            # input("debug")
            # Last hourly row: x+23
            hourly_last = hourly_df.loc[list(range(x + n_timesteps - n_pred_steps - 3, x + n_timesteps - 3)), all_feature_cols]
            # print(f"hourly_last shape: {hourly_last.shape}")
            # print(hourly_last)
            # input("debug")
            # 4 quarter-hour rows: x+22.00, x+22.25, x+22.50, x+22.75
            minutes_block = minutes_15_df.loc[minutes_hours_needed, all_feature_cols]
            # print(f"minutes_block shape: {minutes_block.shape}")
            # input("debug")

            # Concatenate to match required order:
            # 23 hourly, then 4 minutes, then 1 hourly = 28 rows total
            window_df = pd.concat([hourly_first, minutes_block, hourly_last], axis=0)
            # print(window_df.columns)
            # print(f"window_df shape: {window_df.shape}")
            # print(window_df)
            # input("debug")

            if len(window_df) != n_timesteps:
                # Guard in case of unexpected data issues
                print(f"unexpected window length for start {x}, got {len(window_df)}")
                input("debug")
                continue

            # Convert to NumPy and append
            window_arr = window_df.to_numpy()  # shape (28, n_features)
            windows.append(window_arr)

        if not windows:
            raise ValueError(f"No valid samples found for {key} dataset.")
        else:
            # Stack into (n_windows, 28, n_features)
            final_array = np.stack(windows, axis=0)

        # Optional: write a flattened debug CSV (windows × (time × features))
        if final_array.size > 0:
            n_windows, n_steps, n_feats = final_array.shape
            flat = final_array.reshape(n_windows, n_steps * n_feats)

            # Build column names like "<feature>_t00", "<feature>_t01", ...
            cols = []
            for t in range(n_steps):
                for feat in all_feature_cols:
                    cols.append(f"{feat}_t{t:02d}")

            flat_df = pd.DataFrame(flat, columns=cols)
            flat_df.to_csv(f"debug_{key}_samples.csv", index=False)
            print(f"[hu] save debug_{key}_samples")

        if key == "train":
            train_mu = hourly_df["hour_invocation"].mean()
            train_sigma = hourly_df["hour_invocation"].std()
            json.dump(
                {"mu": train_mu, "sigma": train_sigma},
                open("train_invocation_rate_normalization.json", "w"),
            )

        data[key] = final_array
        # print(data[key].shape)
        print(
            data[key].shape[0],
            f"samples of {n_input_steps} input steps and {n_pred_steps} output steps in",
            key,
        )
        # print(windows_arr[0].astype(np.int32))
        # input("debug")

    return data


if __name__ == "__main__":
    pipeline()
