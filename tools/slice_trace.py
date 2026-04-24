"""Slice request_trace.csv and system_metrics.csv from a log dir to a time window.

Useful for zoomed paper figures — run full simulation once, then extract
just the interesting window (e.g. a traffic burst) for focused plotting.

Usage:
    python tools/slice_trace.py logs/our_sac/infer/seed_42 \
        --start 40h --end 60h --output logs/zoom_42/

    python tools/slice_trace.py logs/our_sac/infer/seed_42 \
        --start 144000 --end 216000 --output logs/zoom_42/

Time arguments accept plain seconds or suffixes: 's', 'm', 'h', 'd'.
"""

from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd


_SUFFIX_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_time(value: str) -> float:
    """Parse '40h' / '2400m' / '86400' into seconds."""
    value = value.strip().lower()
    if value and value[-1] in _SUFFIX_SECONDS:
        return float(value[:-1]) * _SUFFIX_SECONDS[value[-1]]
    return float(value)


def slice_csv(src: str, dst: str, time_col: str, start: float, end: float) -> int:
    """Read src, filter rows where `time_col` in [start, end], write dst.
    Returns number of rows kept."""
    df = pd.read_csv(src)
    if time_col not in df.columns:
        raise KeyError(f"{src}: missing column '{time_col}' (have: {list(df.columns)})")
    kept = df[(df[time_col] >= start) & (df[time_col] <= end)]
    kept.to_csv(dst, index=False)
    return len(kept)


def main():
    parser = argparse.ArgumentParser(description="Slice trace/metrics CSVs to a time window")
    parser.add_argument("log_dir", help="Log dir (contains request_trace.csv + system_metrics.csv)")
    parser.add_argument("--start", required=True, help="Window start (seconds, or e.g. '40h')")
    parser.add_argument("--end", required=True, help="Window end (seconds, or e.g. '60h')")
    parser.add_argument("--output", required=True, help="Output dir")
    parser.add_argument("--copy-summary", action="store_true",
                        help="Also copy summary.json (global stats, not re-sliced)")
    args = parser.parse_args()

    start = parse_time(args.start)
    end = parse_time(args.end)
    if end <= start:
        parser.error(f"--end ({end}) must be > --start ({start})")

    os.makedirs(args.output, exist_ok=True)

    trace_src = os.path.join(args.log_dir, "request_trace.csv")
    metrics_src = os.path.join(args.log_dir, "system_metrics.csv")
    summary_src = os.path.join(args.log_dir, "summary.json")

    print(f"Slicing [{start:.0f}s, {end:.0f}s] (= [{start/3600:.2f}h, {end/3600:.2f}h])")

    if os.path.exists(trace_src):
        n = slice_csv(trace_src, os.path.join(args.output, "request_trace.csv"),
                      "arrival_time", start, end)
        print(f"  request_trace.csv: {n} rows kept")
    else:
        print(f"  request_trace.csv: not found in {args.log_dir}")

    if os.path.exists(metrics_src):
        n = slice_csv(metrics_src, os.path.join(args.output, "system_metrics.csv"),
                      "time", start, end)
        print(f"  system_metrics.csv: {n} rows kept")
    else:
        print(f"  system_metrics.csv: not found in {args.log_dir}")

    if args.copy_summary and os.path.exists(summary_src):
        shutil.copy(summary_src, os.path.join(args.output, "summary.json"))
        print(f"  summary.json: copied (note: stats cover full run, not window)")

    print(f"\nOutput: {args.output}/")


if __name__ == "__main__":
    main()
