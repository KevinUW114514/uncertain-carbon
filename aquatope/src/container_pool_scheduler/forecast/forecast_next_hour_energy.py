import pandas as pd

# Load data
df = pd.read_csv("/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/valid.csv")

# Define quadratic energy function coefficients
a, b, c = -0.157702, 79.9895, 3050.61

# Compute hour bucket
df["hour"] = df["time"] // 3600

# Compute per-minute energy
df["energy_minute"] = a * df["invocation_rate"]**2 \
                     + b * df["invocation_rate"] \
                     + c

# Aggregate to hourly ground truth energy
hourly_energy = (
    df.groupby("hour", as_index=False)["energy_minute"]
      .sum()
      .rename(columns={"energy_minute": "true_hourly_energy"})
)

print(hourly_energy.head())

import pickle

with open("/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/GB_direct_rolling_t1_eval_actual_ci_14d.pkl", "rb") as f:
    ci = pickle.load(f)
    
ci_df = pd.DataFrame({"ci": ci})
ci_df.to_csv("ground_truth_ci.csv", index=False)

# Ensure same length
min_len = min(len(hourly_energy), len(ci_df))

hourly_energy = hourly_energy.iloc[:min_len].reset_index(drop=True)
ci_df = ci_df.iloc[:min_len].reset_index(drop=True)
# Multiply hourly energy by carbon intensity
hourly_energy["ground_truth_carbon"] = (
    hourly_energy["true_hourly_energy"] * ci_df["ci"]
)
ground_truth = pd.concat(
    [hourly_energy[["hour", "true_hourly_energy"]], ci_df],
    axis=1
)

ground_truth["ground_truth_carbon"] = (
    ground_truth["true_hourly_energy"] * ground_truth["ci"]
)

print(ground_truth.head())
ground_truth.to_csv("ground_truth_hourly_carbon.csv", index=False)