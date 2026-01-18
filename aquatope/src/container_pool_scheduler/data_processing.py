#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys
import joblib

model = joblib.load("polynomial_regression_model.joblib")


def search_consecutive_intervals(numbers, window_size):
    # Ensure numbers are sorted and unique
    left, right = -1, -1
    intervals = []
    
    for i in range(0, len(numbers), window_size):
        window = numbers[i:i + window_size]

        if len(window) < window_size:
            break
        if window[-1] - window[0] == window_size - 1:
            intervals.append((window[0], window[-1]))
    
    print(f"intervals: {intervals}")
    return intervals

def print_percentiles(data, column_name):
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.05, 99.9, 99.99, 99.999]
    stats = data[column_name].quantile([p / 100 for p in percentiles])
    print("\nInvocation rate percentiles:")
    for p in percentiles:
        print(f"{p:>3}th percentile: {stats[p / 100]:.6f}, num samples below: {(data[column_name] <= stats[p / 100]).sum()}")
        
def drop_above_percentile(
    df: pd.DataFrame,
    column: str,
    percentile: float = 0.99,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop rows where `column` exceeds the given percentile.

    Args:
        df: Input DataFrame
        column: Column to filter
        percentile: Percentile threshold (0 < p < 1)
        verbose: Whether to print drop statistics

    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    threshold = df[column].quantile(percentile)

    before = len(df)
    filtered = df[df[column] <= threshold].copy()
    after = len(filtered)

    if verbose:
        dropped = before - after
        print(
            f"Dropped {dropped} rows above "
            f"{percentile}th percentile "
            f"(threshold={threshold})"
        )

    return filtered


def main():
    p = argparse.ArgumentParser(description="Keep day, time, and a user-specified column; fill NaN in that column with 0; concat and save.")
    p.add_argument("--column", help="Name of the input column to keep (in addition to 'day' and 'time').")
    p.add_argument("-p", "--path", type=str, required=True, help="Path to the directory containing the CSV files.")
    p.add_argument("-o", "--output", default="combined.csv", help="Output CSV filename (default: combined.csv)")
    p.add_argument("-t", "--train", action="store_true", help="If set, use training days (0-13); else use test days (14-27).")
    p.add_argument("-v", "--valid", action="store_true", help="If set, use validation days (28-41); else use test days (14-27).")
    p.add_argument("-c", "--calibration", action="store_true", help="If set, use calibration days (21-27); else use test days (14-20).")
    p.add_argument("--test", action="store_true", help="If set, use test days (14-27); else use training days (0-13).")
    args = p.parse_args()
    
    train_range = range(0, 30)
    valid_range = range(30, 60)
    calibration_range = range(28, 62)
    
    if args.valid:
        print("Processing validation data...")
        print("=" * 80)
        required = ["day", "time", args.column]

        files = [Path(f"{args.path}/day_{i:03d}.csv") for i in valid_range]
        args.output = "valid.csv"

        dfs = []
        for f in files:
            if not f.exists():
                print(f"[skip] {f} not found", file=sys.stderr)
                continue
            df = pd.read_csv(f)
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[skip] {f} missing columns: {', '.join(missing)}", file=sys.stderr)
                continue
            # Select and fill NaN in the user column
            subset = df[["day", "time"]].copy()
            subset["invocation_rate"] = df[args.column].fillna(0)
            dfs.append(subset)

        if not dfs:
            print("No valid files to concatenate.", file=sys.stderr)
            sys.exit(1)

        out = pd.concat(dfs, ignore_index=True)
        print_percentiles(out, "invocation_rate")
        out = drop_above_percentile(out, "invocation_rate", percentile=0.9905)
        
        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output} with {len(out)} rows from {len(dfs)} files.")

    if args.train:
        print("Processing training data...")
        print("=" * 80)
        
        required = ["day", "time", args.column]

        files = [Path(f"{args.path}/day_{i:03d}.csv") for i in train_range]
        args.output = "train.csv"

        dfs = []
        for f in files:
            if not f.exists():
                print(f"[skip] {f} not found", file=sys.stderr)
                continue
            df = pd.read_csv(f)
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[skip] {f} missing columns: {', '.join(missing)}", file=sys.stderr)
                continue
            # Select and fill NaN in the user column
            subset = df[["day", "time"]].copy()
            subset["invocation_rate"] = df[args.column].fillna(0)
            dfs.append(subset)

        if not dfs:
            print("No valid files to concatenate.", file=sys.stderr)
            sys.exit(1)
            
        out = pd.concat(dfs, ignore_index=True)
        print_percentiles(out, "invocation_rate")
        out = drop_above_percentile(out, "invocation_rate", percentile=0.9905)

        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output} with {len(out)} rows from {len(dfs)} files.")
        
        
    if args.calibration:
        print("Processing calibration data...")
        print("=" * 80)
        
        required = ["day", "time", args.column]

        files = [Path(f"{args.path}/day_{i:03d}.csv") for i in calibration_range]
        args.output = "calibration.csv"

        dfs = []
        for f in files:
            if not f.exists():
                print(f"[skip] {f} not found", file=sys.stderr)
                continue
            df = pd.read_csv(f)
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[skip] {f} missing columns: {', '.join(missing)}", file=sys.stderr)
                continue
            # Select and fill NaN in the user column
            subset = df[["day", "time"]].copy()
            subset["invocation_rate"] = df[args.column].fillna(0)
            dfs.append(subset)

        if not dfs:
            print("No valid files to concatenate.", file=sys.stderr)
            sys.exit(1)
            
        out = pd.concat(dfs, ignore_index=True)
        print_percentiles(out, "invocation_rate")
        out = drop_above_percentile(out, "invocation_rate", percentile=0.9905)

        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output} with {len(out)} rows from {len(dfs)} files.")
    
    if args.test:
        print("Processing test data...")
        print("=" * 80)
        
        required = ["day", "time", args.column]

        files = [Path(f"{args.path}/day_{i:03d}.csv") for i in range(43, 365)]
        filtered_files = []
        for f in files:
            if not f.exists():
                print(f"[skip] {f} not found", file=sys.stderr)
                continue
            filtered_files.append(f)
        nao = sorted([int(str(s)[-7:-4]) for s in filtered_files])
        
        print(nao)
        for i in range(30, -1, -1):
            result = search_consecutive_intervals(nao, i)
            if len(result) > 0:
                print(f"window size: {i}")
                break
        print(f"{result}")
        input("debug")
        
        files = [Path(f"requests_minute/day_{i:03d}.csv") for i in range(147, 165)]
        args.output = "test.csv"

        dfs = []
        for f in files:
            if not f.exists():
                print(f"[skip] {f} not found", file=sys.stderr)
                continue
            df = pd.read_csv(f)
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[skip] {f} missing columns: {', '.join(missing)}", file=sys.stderr)
                continue
            # Select and fill NaN in the user column
            subset = df[["day", "time"]].copy()
            subset["invocation_rate"] = df[args.column].fillna(0)
            dfs.append(subset)

        if not dfs:
            print("No valid files to concatenate.", file=sys.stderr)
            sys.exit(1)

        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output} with {len(out)} rows from {len(dfs)} files.")
        
if __name__ == "__main__":
    main()
