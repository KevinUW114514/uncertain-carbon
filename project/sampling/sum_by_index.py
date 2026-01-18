#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import pandas as pd


REQUIRED_COLS = ["hostname", "timestamp", "rps", "input_size", "power_total_w"]


def main() -> int:
    p = argparse.ArgumentParser(
        description="For each rps, sort two hostnames by timestamp, align by index, sum power_total_w, output rps,input_size,power."
    )
    p.add_argument("--in-csv", required=True, help="Input CSV path")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    p.add_argument("--timestamp-format", default=None, help="Optional pandas timestamp format string")
    p.add_argument(
        "--truncate-to-min",
        action="store_true",
        help="If the two hostnames have different row counts for an rps, truncate to the shorter instead of erroring.",
    )
    p.add_argument(
        "--skip-input-size-check",
        action="store_true",
        help="Do not validate that input_size aligns between the two hostnames after sorting.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format=args.timestamp_format, errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        raise ValueError(
            f"Found {bad_ts} rows with unparseable timestamps. "
            f"Provide --timestamp-format if needed."
        )

    # Group robustly (avoids float grouping surprises)
    df["_rps_key"] = df["rps"].astype(str)

    out_rows = []
    for rps_key, g in df.groupby("_rps_key", sort=True):
        rps_val = g["rps"].dropna().iloc[0] if g["rps"].notna().any() else rps_key

        hosts = sorted(g["hostname"].dropna().unique().tolist())
        if len(hosts) != 2:
            raise ValueError(f"Expected exactly 2 hostnames for rps={rps_val}, found {len(hosts)}: {hosts}")

        h1, h2 = hosts
        a = g[g["hostname"] == h1].sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        b = g[g["hostname"] == h2].sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        if len(a) != len(b):
            if args.truncate_to_min:
                n = min(len(a), len(b))
                a = a.iloc[:n].reset_index(drop=True)
                b = b.iloc[:n].reset_index(drop=True)
            else:
                raise ValueError(
                    f"Row count mismatch for rps={rps_val}: {h1} has {len(a)} rows, {h2} has {len(b)} rows. "
                    f"Use --truncate-to-min to truncate."
                )

        if not args.skip_input_size_check:
            # After sorting and aligning by index, input_size should match row-by-row.
            mismatch = (a["input_size"].astype(str).values != b["input_size"].astype(str).values).sum()
            if mismatch:
                raise ValueError(
                    f"input_size mismatch after alignment for rps={rps_val}: {mismatch} rows differ. "
                    f"Use --skip-input-size-check to bypass."
                )

        power = pd.to_numeric(a["power_total_w"], errors="coerce") + pd.to_numeric(b["power_total_w"], errors="coerce")

        # Output one row per aligned sample
        out = pd.DataFrame(
            {
                "rps": [rps_val] * len(power),
                "input_size": a["input_size"].values,  # take from host A (validated to match host B)
                "power_total_w": power.values,
            }
        )
        out_rows.append(out)

    result = pd.concat(out_rows, ignore_index=True)
    result.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(result):,} rows to {args.out_csv} with columns: rps,input_size,power")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
