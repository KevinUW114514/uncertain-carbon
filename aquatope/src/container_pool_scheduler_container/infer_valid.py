#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from forecast_next_hour import build_features

# Use joblib so it works for most scikit-learn models/pipelines
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

def load_model(path):
    if joblib_load is None:
        print("joblib is required to load the model. Install with `pip install joblib`.", file=sys.stderr)
        sys.exit(2)
    return joblib_load(path)

def main():
    p = argparse.ArgumentParser(
        description="Run inference on validation CSV; save predictions, real average, and error percentage (MAPE)."
    )
    p.add_argument("--valid", default="valid.cs", help="Validation file (CSV). Default: valid.cs")
    p.add_argument("--model", default="model.joblib", help="Trained model/pipeline file (joblib).")
    p.add_argument("--target", required=True, help="Name of the ground-truth target column in the CSV.")
    p.add_argument("--output", default="valid_with_preds.csv", help="Output CSV with predictions appended.")
    p.add_argument("--metrics", default="metrics.json", help="Where to write metrics JSON.")
    p.add_argument("--features", nargs="*", default=None,
                   help="Optional explicit list of feature columns (space-separated). If omitted, uses all non-target columns.")
    args = p.parse_args()

    # Read validation data (treat .cs as .csv)
    df = pd.read_csv("valid.csv")
    
    X, y, feature_cols = build_features(df, 3600)

    # Load model and predict
    model = load_model(args.model)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        print(f"Model prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Attach predictions
    out_df = df.copy()
    pred_col = "prediction"
    # Avoid name collision
    if pred_col in out_df.columns:
        i = 1
        while f"{pred_col}_{i}" in out_df.columns:
            i += 1
        pred_col = f"{pred_col}_{i}"
    out_df[pred_col] = y_pred

    # Metrics: real average and error percentage (MAPE)
    y_true = y
    y_pred = pd.Series(y_pred, index=df.index).astype(float)

    real_average = float(y_true.mean())

    # MAPE: mean(|(y_true - y_pred)/y_true|)*100, skipping y_true == 0
    nonzero_mask = y_true != 0
    skipped = int((~nonzero_mask).sum())
    if nonzero_mask.any():
        mape = float((np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])).mean() * 100.0)
    else:
        mape = float("nan")

    metrics = {
        "target": 'y',
        "feature_columns": feature_cols,
        "n_rows": int(len(df)),
        "n_skipped_for_mape_due_to_zero_truth": skipped,
        "real_average": real_average,
        "error_percentage_mape": mape
    }

    # Write outputs
    out_df.to_csv(args.output, index=False)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote predictions to: {args.output}")
    print(f"Wrote metrics to: {args.metrics}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
