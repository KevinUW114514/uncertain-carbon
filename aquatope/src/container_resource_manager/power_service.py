#!/usr/bin/env python3
"""
Intel RAPL energy profiler with:
  - duration=-1 infinite loop
  - streaming CSV append (row-per-sample, flushed each interval)
  - time-sync-safe measurement via monotonic timestamps
  - optional HTTP endpoint:
      GET /health -> ok
      GET /energy -> current raw RAPL counters snapshot
      GET /power?timestamp_utc_ms=... -> nearest 0.1s-bucket sample:
            - power_total_w
            - energy_uj (raw per-domain counters captured at that sample)
            - returned_timestamp_utc_ms (actual sample timestamp)
            - bucket_utc_ms (nearest 100ms bucket for the query)
            - found/source

Key change requested:
  - Query does NOT require perfect timestamp match.
  - It rounds the query timestamp to the nearest 100ms bucket and returns the
    nearest-bucket sample (with small neighbor fallback).

Time base:
  - timestamp_utc_ms is epoch milliseconds (UTC), integer.

Requires:
  - Intel RAPL under /sys/class/powercap (intel-rapl:*)
  - psutil, pandas (kept for compatibility)
"""

import argparse
import csv
import json
import os
import socket
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, parse_qs

import psutil
import pwd
import grp

import pandas as pd  # kept in case you still want DataFrame-based work later
from http.server import BaseHTTPRequestHandler, HTTPServer

RAPL_ROOT = Path("/sys/class/powercap")

# Ownership settings
USER_NAME = "cc"
GROUP_NAME = "cc"
UID = pwd.getpwnam(USER_NAME).pw_uid
GID = grp.getgrnam(GROUP_NAME).gr_gid


def utc_iso_now_seconds() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def epoch_ms_now() -> int:
    return int(time.time() * 1000)


def bucket_100ms(ts_ms: int) -> int:
    """
    Nearest 100ms bucket (0.1s granularity).
    Example: 1767758176270 -> 1767758176300 (nearest 100ms)
    """
    # round to nearest 100ms
    return int(((ts_ms + 50) // 100) * 100)


def discover_rapl_domains() -> List[dict]:
    """
    Discovers intel-rapl domains and captures:
      - key
      - name
      - energy_uj path
      - max_energy_range_uj (for wrap handling)
      - is_package
    """
    domains = []
    for p in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        name_file = p / "name"
        energy_file = p / "energy_uj"
        max_range_file = p / "max_energy_range_uj"

        if not (name_file.exists() and energy_file.exists()):
            continue

        name = name_file.read_text().strip()
        is_package = p.name.count(":") == 1

        key = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_"
            + p.name.split("intel-rapl:")[1].replace(":", "_")
        )

        max_range_uj = None
        if max_range_file.exists():
            try:
                max_range_uj = int(max_range_file.read_text().strip())
            except Exception:
                max_range_uj = None

        domains.append(
            {
                "key": key,
                "name": name,
                "energy": energy_file,
                "max_range_uj": max_range_uj,
                "is_package": is_package,
            }
        )

    return domains


def read_energy_uj(domains: List[dict]) -> Dict[str, int]:
    return {d["key"]: int(d["energy"].read_text()) for d in domains}


def delta_with_wrap(cur: int, prev: int, max_range_uj: Optional[int]) -> Optional[int]:
    if cur >= prev:
        return cur - prev
    if max_range_uj is not None and max_range_uj > 0:
        return (max_range_uj - prev) + cur
    return None


def ensure_access(path: Path):
    try:
        os.chown(path, UID, GID)
    except PermissionError:
        pass
    try:
        os.chmod(path, 0o644)
    except PermissionError:
        pass


def ensure_dir_access(path: Path):
    try:
        os.chmod(path, 0o755)
    except PermissionError:
        pass


def _parse_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _parse_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v == v:
            return v
    except Exception:
        return None
    return None


def lookup_nearest_bucket_from_csv(
    csv_path: Path,
    query_ts_ms: int,
    desired_bucket_ms: int,
    energy_cols_by_key: Dict[str, str],
    neighbor_buckets: List[int],
) -> Tuple[Optional[dict], Optional[int]]:
    """
    CSV fallback: scan CSV and return the best sample among neighbor buckets.

    Returns:
      (sample_dict, chosen_bucket_ms)
      sample_dict := {"power_total_w": float, "energy_uj": {...}, "timestamp_utc_ms": int}
    """
    if not csv_path.exists():
        return None, None

    best: Optional[dict] = None
    best_bucket: Optional[int] = None
    best_score: Optional[Tuple[int, int]] = None  # (abs(bucket-desired_bucket), abs(sample_ts-query_ts))

    neighbor_set = set(neighbor_buckets)

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_ms = _parse_int(row.get("timestamp_utc_ms"))
                if ts_ms is None:
                    continue

                b = bucket_100ms(ts_ms)
                if b not in neighbor_set:
                    continue

                p = _parse_float(row.get("power_total_w"))
                if p is None:
                    continue

                e: Dict[str, int] = {}
                ok = True
                for key, col in energy_cols_by_key.items():
                    v = _parse_int(row.get(col))
                    if v is None:
                        ok = False
                        break
                    e[key] = v
                if not ok:
                    continue

                score = (abs(b - desired_bucket_ms), abs(ts_ms - query_ts_ms))
                # Prefer closer bucket; tie-break by closer actual timestamp; tie-break by "latest" (keep last seen)
                if best_score is None or score < best_score or (score == best_score and ts_ms > best["timestamp_utc_ms"]):
                    best_score = score
                    best_bucket = b
                    best = {"power_total_w": p, "energy_uj": e, "timestamp_utc_ms": ts_ms}
    except Exception:
        return None, None

    return best, best_bucket


class BucketSampleCache:
    """
    Rolling cache keyed by 100ms bucket (epoch ms rounded to nearest 100ms).
    Each bucket stores the latest sample recorded for that bucket:
      {
        "timestamp_utc_ms": int,   # actual sample ts
        "power_total_w": float,
        "energy_uj": {domain_key: int, ...}
      }
    """

    def __init__(self, window_ms: int = 60_000):
        self.window_ms = window_ms
        self._by_bucket: Dict[int, dict] = {}
        self._order: deque[int] = deque()  # insertion order of buckets (may include duplicates)
        self._lock = Lock()

    def put(self, bucket_ms: int, sample: dict):
        with self._lock:
            self._by_bucket[bucket_ms] = sample
            self._order.append(bucket_ms)

            cutoff = bucket_ms - self.window_ms
            while self._order:
                oldest = self._order[0]
                if oldest >= cutoff:
                    break
                self._order.popleft()
                if oldest not in self._order:
                    self._by_bucket.pop(oldest, None)

            # Secondary cap
            max_len = max(2_000, int(self.window_ms / 10))
            while len(self._order) > max_len:
                oldest = self._order.popleft()
                if oldest not in self._order:
                    self._by_bucket.pop(oldest, None)

    def get(self, bucket_ms: int) -> Optional[dict]:
        with self._lock:
            return self._by_bucket.get(bucket_ms)

    def get_best_among(self, desired_bucket_ms: int, query_ts_ms: int, buckets: List[int]) -> Tuple[Optional[dict], Optional[int]]:
        """
        Choose best sample among candidate buckets using:
          1) abs(bucket - desired_bucket)
          2) abs(sample_ts - query_ts)
        """
        best_sample = None
        best_bucket = None
        best_score = None

        with self._lock:
            for b in buckets:
                s = self._by_bucket.get(b)
                if s is None:
                    continue
                score = (abs(b - desired_bucket_ms), abs(int(s["timestamp_utc_ms"]) - query_ts_ms))
                if best_score is None or score < best_score or (score == best_score and int(s["timestamp_utc_ms"]) > int(best_sample["timestamp_utc_ms"])):
                    best_score = score
                    best_sample = s
                    best_bucket = b

        return best_sample, best_bucket


class EnergyState:
    """
    Shared state for HTTP handler.
    """

    def __init__(
        self,
        domains: List[dict],
        csv_path: Path,
        energy_cols_by_key: Dict[str, str],
        cache_window_ms: int = 60_000,
        bucket_ms: int = 100,
    ):
        self.domains = domains
        self.csv_path = csv_path
        self.energy_cols_by_key = energy_cols_by_key
        self.cache = BucketSampleCache(window_ms=cache_window_ms)
        self.bucket_ms = bucket_ms

    def snapshot_energy(self) -> dict:
        mono_s = time.monotonic()
        wall_utc_s = time.time()
        energy = read_energy_uj(self.domains)
        return {
            "hostname": socket.gethostname(),
            "wall_utc_s": wall_utc_s,
            "wall_utc_ms": epoch_ms_now(),
            "monotonic_ns": int(mono_s * 1e9),
            "energy_uj": energy,
        }

    def record_sample(self, timestamp_utc_ms: int, power_total_w: float, energy_uj: Dict[str, int]):
        if power_total_w != power_total_w or not energy_uj:
            return
        b = bucket_100ms(timestamp_utc_ms)
        self.cache.put(
            b,
            {
                "timestamp_utc_ms": timestamp_utc_ms,
                "power_total_w": power_total_w,
                "energy_uj": energy_uj,
            },
        )

    def query_sample_nearest_100ms(self, query_ts_ms: int) -> Tuple[Optional[dict], str, int, Optional[int]]:
        """
        Returns (sample, source, desired_bucket_ms, chosen_bucket_ms)
          - desired_bucket_ms: nearest 100ms bucket for the query
          - chosen_bucket_ms: bucket we actually returned (may be neighbor if missing)
        """
        desired_bucket = bucket_100ms(query_ts_ms)

        # Neighbor fallback: try desired, then +/- 100ms, then +/- 200ms (adjust as desired)
        neighbor_buckets = [
            desired_bucket,
            # desired_bucket - 100,
            # desired_bucket + 100,
            # desired_bucket - 200,
            # desired_bucket + 200,
        ]

        # 1) cache
        sample, chosen_bucket = self.cache.get_best_among(desired_bucket, query_ts_ms, neighbor_buckets)
        if sample is not None:
            return sample, "cache", desired_bucket, chosen_bucket

        # 2) csv fallback
        sample, chosen_bucket = lookup_nearest_bucket_from_csv(
            self.csv_path,
            query_ts_ms=query_ts_ms,
            desired_bucket_ms=desired_bucket,
            energy_cols_by_key=self.energy_cols_by_key,
            neighbor_buckets=neighbor_buckets,
        )
        if sample is not None and chosen_bucket is not None:
            # populate cache for future queries
            self.cache.put(chosen_bucket, sample)
            return sample, "csv", desired_bucket, chosen_bucket

        return None, "miss", desired_bucket, None


class EnergyHandler(BaseHTTPRequestHandler):
    """
    GET /health -> ok
    GET /energy -> JSON snapshot from worker clock/counters
    GET /power?timestamp_utc_ms=... -> nearest 0.1s bucket sample:
        {
          "query_timestamp_utc_ms": ...,
          "bucket_utc_ms": ...,                 # nearest 100ms bucket for the query
          "returned_bucket_utc_ms": ...,        # bucket actually returned (could be neighbor)
          "returned_timestamp_utc_ms": ...,     # actual sample timestamp
          "power_total_w": ...,
          "energy_uj": {...},
          "found": true/false,
          "source": "cache|csv|miss"
        }
    """

    state: EnergyState = None

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"ok": True})
            return

        if path == "/energy":
            try:
                snap = self.state.snapshot_energy()
                self._send_json(snap)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/power":
            try:
                ts_vals = qs.get("timestamp_utc_ms", [])
                if not ts_vals:
                    self._send_json({"error": "missing query param: timestamp_utc_ms"}, status=400)
                    return

                query_ts_ms = _parse_int(ts_vals[0])
                if query_ts_ms is None:
                    self._send_json({"error": "timestamp_utc_ms must be an integer epoch-milliseconds value"}, status=400)
                    return

                sample, source, desired_bucket, chosen_bucket = self.state.query_sample_nearest_100ms(query_ts_ms)
                if sample is None:
                    self._send_json(
                        {
                            "query_timestamp_utc_ms": query_ts_ms,
                            "bucket_utc_ms": desired_bucket,
                            "returned_bucket_utc_ms": None,
                            "returned_timestamp_utc_ms": None,
                            "power_total_w": None,
                            "energy_uj": None,
                            "found": False,
                            "source": source,
                        },
                        status=404,
                    )
                    return

                self._send_json(
                    {
                        "query_timestamp_utc_ms": query_ts_ms,
                        "bucket_utc_ms": desired_bucket,
                        "returned_bucket_utc_ms": chosen_bucket,
                        "returned_timestamp_utc_ms": sample.get("timestamp_utc_ms"),
                        "power_total_w": sample.get("power_total_w"),
                        "energy_uj": sample.get("energy_uj"),
                        "found": True,
                        "source": source,
                    }
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "not found"}, status=404)

    def log_message(self, fmt, *args):
        return


def start_http_server(state: EnergyState, host: str, port: int) -> Thread:
    EnergyHandler.state = state
    httpd = HTTPServer((host, port), EnergyHandler)
    t = Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser(description="Intel RAPL power profiler (streaming + time-sync safe)")
    parser.add_argument("--duration", type=int, required=True, help="Profiling duration (seconds). Use -1 for infinite.")
    parser.add_argument("--interval", type=float, required=True, help="Sampling interval (seconds). e.g., 0.1 for 10Hz.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV")
    parser.add_argument("--rps", type=int, required=True, help="Load RPS for labeling")
    parser.add_argument("--input-size", type=int, required=True, help="Input size for labeling")
    parser.add_argument("--series", type=int, required=True, help="Series number for labeling")

    parser.add_argument("--serve", action="store_true", help="Expose HTTP endpoint (/energy, /power)")
    parser.add_argument("--serve-host", default="0.0.0.0", help="HTTP bind host (default 0.0.0.0)")
    parser.add_argument("--serve-port", type=int, default=9876, help="HTTP port (default 9876)")

    parser.add_argument("--fsync", action="store_true", help="Call os.fsync() each row (stronger durability, more overhead)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dir_access(out_dir)

    domains = discover_rapl_domains()
    if not domains:
        raise RuntimeError("No Intel RAPL domains found under /sys/class/powercap")

    hostname = socket.gethostname()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{hostname}_s{args.series}_rps{args.rps}_{ts}_server.csv"

    rapl_power_cols = [f"rapl_{d['key']}_w" for d in domains]
    rapl_energy_cols = [f"rapl_{d['key']}_energy_uj" for d in domains]
    energy_cols_by_key = {d["key"]: f"rapl_{d['key']}_energy_uj" for d in domains}

    fieldnames = [
        "hostname",
        "timestamp_utc_ms",
        "monotonic_ns",
        "rps",
        "input_size",
        "series",
        "sample_index",
        "elapsed_s",
        "dt_s",
        "cpu_power_total_w",
        "memory_power_total_w",
        "power_total_w",
    ] + rapl_power_cols + rapl_energy_cols

    prev_energy = read_energy_uj(domains)
    start_mono = time.monotonic()
    prev_mono = start_mono

    psutil.cpu_percent(interval=None)

    infinite = (args.duration == -1)
    end_time_wall = None if infinite else (time.time() + args.duration)

    state = EnergyState(domains=domains, csv_path=out_file, energy_cols_by_key=energy_cols_by_key, cache_window_ms=60_000)

    if args.serve:
        start_http_server(state, args.serve_host, args.serve_port)
        print(f"HTTP endpoints running at http://{args.serve_host}:{args.serve_port}/")
        print("  - /energy")
        print("  - /power?timestamp_utc_ms=<epoch_milliseconds>  (returns nearest 0.1s sample: power_total_w + energy_uj)")

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        if args.fsync:
            os.fsync(f.fileno())

        print(f"Streaming power profile to: {out_file}")
        print(f"Sampling interval: {args.interval:.6f}s")

        sample_index = 0
        while True:
            if not infinite and time.time() >= end_time_wall:
                break

            time.sleep(args.interval)

            now_mono = time.monotonic()
            now_wall_ms = epoch_ms_now()

            cur_energy = read_energy_uj(domains)

            dt_s = now_mono - prev_mono
            elapsed_s = now_mono - start_mono

            row = {
                "hostname": hostname,
                "timestamp_utc_ms": now_wall_ms,
                "monotonic_ns": int(now_mono * 1e9),
                "rps": args.rps,
                "input_size": args.input_size,
                "series": args.series,
                "sample_index": sample_index,
                "elapsed_s": round(elapsed_s, 6),
                "dt_s": round(dt_s, 6),
            }

            cpu_power_total = 0.0
            memory_power_total = 0.0

            # Raw energy counters for this sample
            for d in domains:
                key = d["key"]
                row[f"rapl_{key}_energy_uj"] = cur_energy[key]

            # Per-domain power + totals
            for d in domains:
                key = d["key"]
                duj = delta_with_wrap(cur_energy[key], prev_energy[key], d.get("max_range_uj"))

                if duj is None or dt_s <= 0:
                    power_w = float("nan")
                else:
                    power_w = (duj / 1e6) / dt_s

                row[f"rapl_{key}_w"] = power_w

                if power_w == power_w:
                    if d.get("is_package", False):
                        cpu_power_total += power_w
                    if "dram" in key.lower():
                        memory_power_total += power_w

            total_power = cpu_power_total + memory_power_total
            row["cpu_power_total_w"] = cpu_power_total
            row["memory_power_total_w"] = memory_power_total
            row["power_total_w"] = total_power

            writer.writerow(row)
            f.flush()
            if args.fsync:
                os.fsync(f.fileno())

            # Cache for nearest-0.1s /power queries
            state.record_sample(timestamp_utc_ms=now_wall_ms, power_total_w=total_power, energy_uj=cur_energy)

            prev_energy = cur_energy
            prev_mono = now_mono
            sample_index += 1

    ensure_access(out_file)
    print(f"Done. Wrote: {out_file}")


if __name__ == "__main__":
    main()



"""
sudo -E $(which python3) ./power_service.py --duration -1 --interval 1 --out-dir ./ --rps 109 --input-size 563135 --series 0 --serve


stat /home/cc/uncertain-carbon/functions/source-images/images/0d74cfde-b4d2-48dc-bf92-2234717025a8.png

sudo -E $(which python3) power_profile_rapl.py   \
    --rps 94 \
    --duration 60 \
    --interval 1 \
    --out-dir ./power_logs \
    --input-size 563135 \
    --series 11
    
python sum_by_index.py --in-csv ./power_logs/intel-manager_s7_rps150_20251229_065853.csv --out-csv summed_hu.csv

python avg.py
"""