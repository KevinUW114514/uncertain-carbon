#!/usr/bin/env python3
"""
Train a regression model f(rps, input_size) -> power_total_w from a CSV with columns:
hostname,timestamp,rps,sample_index,elapsed_s,dt_s,input_size,rapl_package_0_0_w,rapl_package_1_1_w,power_total_w

What this script does:
1) Loads CSV
2) (Optional) filters / cleans
3) Aggregates repeated samples into one row per (rps, input_size) using mean(power_total_w)
   and keeps std + count for diagnostics
4) Trains a regression model:
   - default: linear with interaction term (rps * input_size)
   - optional: polynomial (degree 2)
   - optional: random forest
5) Evaluates with train/test split and reports MAE, RMSE, R^2
6) Saves:
   - aggregated dataset CSV
   - metrics JSON
   - trained model artifact (joblib)
7) Provides a callable f(rps, input_size) via a small wrapper class

Example:
  python train_power_model.py --csv rps-aggregated.csv --out-dir ./out --model linear

Then, to do a quick prediction after training:
  python train_power_model.py --csv rps-aggregated.csv --out-dir ./out --model linear --predict 120 --predict-input-size 256
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


REQUIRED_COLUMNS = [
    "rps",
    "input_size",
    "power_total_w",
]


@dataclass
class PowerModel:
    """
    Thin wrapper so you can treat the result as f(rps, input_size).
    """
    sklearn_pipeline: Pipeline
    feature_names: Tuple[str, str] = ("rps", "input_size")

    def predict_one(self, rps: float, input_size: float) -> float:
        X = pd.DataFrame([{self.feature_names[0]: rps, self.feature_names[1]: input_size}])
        y = self.sklearn_pipeline.predict(X)
        return float(y[0])

    def predict_many(self, rps: np.ndarray, input_size: np.ndarray) -> np.ndarray:
        if rps.shape != input_size.shape:
            raise ValueError("rps and input_size must have the same shape")
        X = pd.DataFrame({self.feature_names[0]: rps, self.feature_names[1]: input_size})
        return self.sklearn_pipeline.predict(X)


def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nFound columns: {list(df.columns)}")
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only relevant columns for this modeling task.
    # (You can extend features later.)
    df = df.copy()

    # Convert to numeric safely
    for col in ["rps", "input_size", "power_total_w"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["rps", "input_size", "power_total_w"])

    # Remove obviously invalid values (adjust to your environment)
    df = df[(df["rps"] >= 0) & (df["input_size"] >= 0) & (df["power_total_w"] >= 0)]

    return df


def aggregate_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse time-series samples into one row per (rps, input_size).
    """
    agg = (
        df.groupby(["rps", "input_size"], as_index=False)
          .agg(
              power_mean_w=("power_total_w", "mean"),
              power_std_w=("power_total_w", "std"),
              n_samples=("power_total_w", "size"),
          )
    )

    # std is NaN if n_samples == 1; replace with 0 for convenience
    agg["power_std_w"] = agg["power_std_w"].fillna(0.0)
    return agg


def build_pipeline(model_kind: str, random_state: int = 42) -> Pipeline:
    """
    Build an sklearn pipeline that:
    - imputes missing values (just in case)
    - transforms features as needed
    - fits a regression model
    """
    numeric_features = ["rps", "input_size"]

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_kind == "linear":
        # Linear + interaction term: include polynomial features degree 2 but only interaction and squares
        # If you want strictly interaction w/out squares, set interaction_only=True.
        reg = LinearRegression()
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ("reg", reg),
            ]
        )
        return pipe

    if model_kind == "ridge":
        # More stable than pure OLS if you have multicollinearity; still interpretable.
        reg = Ridge(alpha=1.0, random_state=random_state)
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ("reg", reg),
            ]
        )
        return pipe

    if model_kind == "poly2":
        # Polynomial degree 2 (same as above essentially), but keep as separate option
        reg = LinearRegression()
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("reg", reg),
            ]
        )
        return pipe

    if model_kind == "rf":
        reg = RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("reg", reg),
            ]
        )
        return pipe

    raise ValueError(f"Unknown --model '{model_kind}'. Choose from: linear, ridge, poly2, rf")


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return {"mae_w": mae, "rmse_w": rmse, "r2": r2}


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--out-dir", default="./power_model_out", help="Output directory")
    p.add_argument("--model", default="linear", choices=["linear", "ridge", "poly2", "rf"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--min-samples-per-point", type=int, default=1,
                   help="After aggregation, drop (rps,input_size) points with fewer samples than this")
    p.add_argument("--no-aggregate", action="store_true",
                   help="If set, train directly on raw rows instead of aggregating by (rps,input_size)")
    p.add_argument("--model-file", default="power_model.joblib", help="Saved model filename inside out-dir")

    # Optional prediction after training (or you can load the model separately)
    p.add_argument("--predict", type=float, default=None, help="If set, print f(rps,input_size) for this rps")
    p.add_argument("--predict-input-size", type=float, default=None, help="Input size for --predict")

    # Optional: load an existing model and just predict
    p.add_argument("--load-only", action="store_true",
                   help="If set, skip training and only load model from out-dir/model-file for prediction")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir)
    model_path = os.path.join(args.out_dir, args.model_file)

    if args.load_only:
        if args.predict is None or args.predict_input_size is None:
            raise ValueError("--load-only requires --predict and --predict-input-size")
        pipe = load(model_path)
        pm = PowerModel(pipe)
        pred = pm.predict_one(args.predict, args.predict_input_size)
        print(f"f(rps={args.predict}, input_size={args.predict_input_size}) = {pred:.6f} W")
        return

    df = load_and_validate(args.csv)
    df = clean_df(df)

    if args.no_aggregate:
        train_df = df[["rps", "input_size", "power_total_w"]].copy()
        train_df = train_df.rename(columns={"power_total_w": "y"})
        meta = {
            "training_rows": int(len(train_df)),
            "mode": "raw",
        }
    else:
        agg = aggregate_samples(df)
        agg = agg[agg["n_samples"] >= args.min_samples_per_point].copy()

        # target is mean power at each point
        train_df = agg[["rps", "input_size", "power_mean_w", "power_std_w", "n_samples"]].copy()
        train_df = train_df.rename(columns={"power_mean_w": "y"})
        meta = {
            "training_rows": int(len(train_df)),
            "mode": "aggregated",
            "min_samples_per_point": int(args.min_samples_per_point),
        }

        # Save the aggregated dataset for transparency/reproducibility
        agg_out = os.path.join(args.out_dir, "aggregated_power_by_rps_input_size.csv")
        train_df.to_csv(agg_out, index=False)

    X = train_df[["rps", "input_size"]]
    y = train_df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipe = build_pipeline(args.model, random_state=args.random_state)
    pipe.fit(X_train, y_train)

    metrics = evaluate(pipe, X_test, y_test)
    metrics.update(meta)
    metrics.update(
        {
            "model_kind": args.model,
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
        }
    )

    # Save model + metrics
    dump(pipe, model_path)
    save_json(metrics, os.path.join(args.out_dir, "metrics.json"))

    print("Training complete.")
    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {os.path.join(args.out_dir, 'metrics.json')}")
    if not args.no_aggregate:
        print(f"Saved data:    {os.path.join(args.out_dir, 'aggregated_power_by_rps_input_size.csv')}")
    print("Metrics:")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Optional: show a single prediction using the trained model
    if args.predict is not None and args.predict_input_size is not None:
        pm = PowerModel(pipe)
        pred = pm.predict_one(args.predict, args.predict_input_size)
        print(f"\nPrediction:")
        print(f"f(rps={args.predict}, input_size={args.predict_input_size}) = {pred:.6f} W")
    elif args.predict is not None or args.predict_input_size is not None:
        print("\nNote: to predict, provide both --predict and --predict-input-size.")


if __name__ == "__main__":
    main()


"""
# 1) Train (recommended default: aggregate + linear with degree-2 terms, including interaction)
python train_power_model.py --csv rps-aggregated.csv --out-dir ./out --model linear

# 2) Train and predict one point
python train_power_model.py --csv rps-aggregated.csv --out-dir ./out --model linear --predict 200 --predict-input-size 256

# 3) Load the saved model later and predict
python train_power_model.py --csv hu.csv --out-dir ./out --model-file power_model.joblib \
  --load-only --predict 32 --predict-input-size 822355

# 4) If you want to train on raw per-sample rows (usually noisier)
python train_power_model.py --csv rps-aggregated.csv --out-dir ./out --model ridge --no-aggregate
"""