#!/usr/bin/env python3
"""Fit gamma windows from a trace CSV's inter-arrival times."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_GAMMA_WINDOW = "20m"
_DURATION_PATTERN = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]*)\s*$")


def _normalize_column_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _resolve_timestamp_column(columns: List[str]) -> str:
    normalized_to_raw = {_normalize_column_name(col): col for col in columns if col}
    ts_col = normalized_to_raw.get("timestamp")
    if ts_col is None:
        raise ValueError("Input CSV is missing required 'Timestamp' column.")
    return ts_col


def _fit_gamma_moments(samples: List[float]) -> tuple[float, float]:
    # Gamma(shape=alpha, scale=beta): mean=alpha*beta, var=alpha*beta^2
    if len(samples) < 2:
        return float("nan"), float("nan")
    mean = sum(samples) / float(len(samples))
    if mean <= 0:
        return float("nan"), float("nan")
    var = sum((value - mean) ** 2 for value in samples) / float(len(samples))
    if var <= 0:
        return float("nan"), float("nan")
    alpha = (mean * mean) / var
    beta = var / mean
    return alpha, beta


def _parse_duration_seconds(raw: str, arg_name: str) -> float:
    text = str(raw).strip().lower()
    match = _DURATION_PATTERN.match(text)
    if not match:
        raise ValueError(f"Invalid {arg_name} value: {raw}")

    value = float(match.group("value"))
    unit = match.group("unit")
    if unit in {"", "s", "sec", "second", "seconds", "giay"}:
        multiplier = 1.0
    elif unit in {"m", "min", "minute", "minutes", "phut"}:
        multiplier = 60.0
    elif unit in {"h", "hr", "hour", "hours", "gio"}:
        multiplier = 3600.0
    elif unit in {"d", "day", "days", "ngay"}:
        multiplier = 86400.0
    elif unit in {"mo", "mon", "month", "months", "thang"}:
        multiplier = 30.0 * 86400.0
    else:
        raise ValueError(
            f"Unsupported unit in {arg_name}: {raw}. "
            "Use s/m/h/d/mo (or sec/min/hour/day/month variants)."
        )

    seconds = value * multiplier
    if seconds <= 0:
        raise ValueError(f"{arg_name} must be > 0.")
    return seconds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split trace into configurable windows, fit gamma(alpha,beta) from "
            "inter-arrival times per window."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input trace CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV file for gamma windows "
            "(default: <input_dir>/<input_stem>_gamma_windows.csv)."
        ),
    )
    parser.add_argument(
        "--gamma-window",
        type=str,
        default=DEFAULT_GAMMA_WINDOW,
        help=(
            "Window size for gamma fitting, supports units "
            "(e.g. 20m, 10m, 1h, 900s). Default: 20m."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = args.input
    gamma_output_path = (
        args.output
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_gamma_windows.csv")
    )

    if not input_path.exists() or not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    try:
        window_seconds = _parse_duration_seconds(args.gamma_window, "--gamma-window")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    try:
        header_df = pd.read_csv(input_path, nrows=0)
    except Exception as error:
        raise SystemExit(f"Failed to read CSV header: {error}") from error

    columns = [str(col) for col in header_df.columns]
    if not columns:
        raise SystemExit(f"Input CSV has no header: {input_path}")

    try:
        ts_col = _resolve_timestamp_column(columns)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    df = pd.read_csv(
        input_path,
        usecols=[ts_col],
        dtype=str,
        keep_default_na=False,
    )

    ts_series = pd.to_numeric(df[ts_col].astype(str).str.strip(), errors="coerce")
    ts_series = ts_series[ts_series.notna()]
    if ts_series.empty:
        raise SystemExit("No valid timestamp rows found in input CSV.")

    ts_series = ts_series.astype(float)
    ts_series = ts_series[ts_series >= 0.0]
    if ts_series.empty:
        raise SystemExit("No non-negative timestamps found in input CSV.")

    timestamps = sorted(ts_series.tolist())
    base_ts = timestamps[0]
    last_ts = timestamps[-1]
    window_count = int(math.floor((last_ts - base_ts) / window_seconds)) + 1

    per_window_timestamps: Dict[int, List[float]] = {idx: [] for idx in range(window_count)}
    for timestamp in timestamps:
        window_idx = int((timestamp - base_ts) // window_seconds)
        per_window_timestamps[window_idx].append(timestamp)

    rows: List[dict] = []
    for window_idx in range(window_count):
        window_start = base_ts + window_idx * window_seconds
        window_end = window_start + window_seconds
        window_timestamps = per_window_timestamps.get(window_idx, [])

        inter_arrivals = [
            window_timestamps[idx] - window_timestamps[idx - 1]
            for idx in range(1, len(window_timestamps))
        ]
        positive_inter_arrivals = [value for value in inter_arrivals if value > 0]
        alpha, beta = _fit_gamma_moments(positive_inter_arrivals)

        rows.append(
            {
                "window_index": window_idx,
                "window_seconds": window_seconds,
                "window_start_timestamp": window_start,
                "window_end_timestamp": window_end,
                "request_count": len(window_timestamps),
                "inter_arrival_count": len(positive_inter_arrivals),
                "alpha": alpha,
                "beta": beta,
            }
        )

    gamma_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(gamma_output_path, index=False)

    print(f"Saved {len(rows)} gamma windows to {gamma_output_path}")


if __name__ == "__main__":
    main()
