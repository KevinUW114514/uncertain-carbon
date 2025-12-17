#!/usr/bin/env python3
"""
Profile Intel RAPL power on an Ubuntu host for a single load.

Samples energy counters once per interval, converts to power (W),
shows a progress bar, and writes a single CSV named with datetime.

Requires:
  - Intel RAPL exposed under /sys/class/powercap
  - pandas, tqdm
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import os
import socket

import pandas as pd
from tqdm import tqdm


RAPL_ROOT = Path("/sys/class/powercap")


def discover_rapl_domains():
    domains = []
    for p in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        name_file = p / "name"
        energy_file = p / "energy_uj"
        if name_file.exists() and energy_file.exists():
            name = name_file.read_text().strip()
            is_package = p.name.count(":") == 1
            key = (
                name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                + "_"
                + p.name.split("intel-rapl:")[1].replace(":", "_")
            )
            domains.append({
                "key": key,
                "name": name,
                "energy": energy_file,
                "is_package": is_package,
            })
    return domains


def read_energy(domains):
    return {d["key"]: int(d["energy"].read_text()) for d in domains}


def main():
    parser = argparse.ArgumentParser(description="Intel RAPL power profiler")
    parser.add_argument("--load-name", required=True, help="Name of the workload")
    parser.add_argument("--duration", type=int, required=True, help="Profiling duration (seconds)")
    parser.add_argument("--interval", type=float, required=True, help="Sampling interval (seconds)")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = discover_rapl_domains()
    if not domains:
        raise RuntimeError("No Intel RAPL domains found under /sys/class/powercap")

    rows = []

    prev_energy = read_energy(domains)
    prev_time = time.time()
    start_time = prev_time

    samples = int(args.duration / args.interval)

    for i in tqdm(range(samples), desc="Profiling power", unit="sample"):
        time.sleep(args.interval)

        now = time.time()
        cur_energy = read_energy(domains)
        dt = now - prev_time

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "load_name": args.load_name,
            "sample_index": i,
            "elapsed_s": round(now - start_time, 6),
            "dt_s": round(dt, 6),
        }

        total_power = 0.0

        for d in domains:
            delta_uj = cur_energy[d["key"]] - prev_energy[d["key"]]
            power_w = (delta_uj / 1e6) / dt if delta_uj >= 0 else float("nan")
            col = f"rapl_{d['key']}_w"
            row[col] = power_w

            if d["is_package"] and power_w == power_w:
                total_power += power_w

        row["power_total_w"] = total_power
        rows.append(row)

        prev_energy = cur_energy
        prev_time = now

    df = pd.DataFrame(rows)

    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{hostname}_{args.load_name}_{timestamp}.csv"
    df.to_csv(out_file, index=False)
    
    # Ensure non-root users can read the CSV
    os.chmod(out_file, 0o644)

    # Ensure output directory is traversable
    os.chmod(out_dir, 0o755)

    print(f"Power profile written to: {out_file}")


if __name__ == "__main__":
    main()


"""
sudo -E $(which python3) power_profile_rapl.py \
  --load-name idle \
  --duration 60 \
  --interval 1 \
  --out-dir ./power_logs
"""