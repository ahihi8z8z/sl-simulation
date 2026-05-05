"""Generate synthetic bursty traffic traces in two formats:
  - aggregate trace : minute,count  (compatible with AggregateTraceGenerator)
  - trace replay    : timestamp     (compatible with TraceReplayGenerator, seconds)

Traffic model
-------------
Traffic alternates between *peak* and *valley* minutes on a fixed period:

  [valley ... valley | peak | valley ... valley | peak | ...]

Within each minute the arrival process is Poisson with rate = count/60 req/s.
The inter-peak gap (valley minutes) is filled at `valley_rate` req/min.

Usage examples
--------------
  # defaults — writes to datasets/traffic_pattern/synthetic/
  python tools/gen_bursty_trace.py

  # custom params
  python tools/gen_bursty_trace.py \\
      --peak-rate 600 \\
      --valley-rate 20 \\
      --period 60 \\
      --duration 1440 \\
      --seed 42 \\
      --name my_burst \\
      --out datasets/traffic_pattern/synthetic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate(
    peak_rate: float,
    valley_rate: float,
    period: int,
    peak_duration: int,
    duration: int,
    seed: int,
    function_id: str = "synthetic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    agg_df   : DataFrame with columns [minute, function_id, count]
    replay_df: DataFrame with columns [timestamp]  (seconds from t=0)
    """
    rng = np.random.default_rng(seed)

    agg_rows: list[dict] = []
    replay_timestamps: list[float] = []

    for minute in range(duration):
        # Peak spans [0, peak_duration) minutes within each period
        is_peak = (minute % period) < peak_duration
        rate = peak_rate if is_peak else valley_rate

        # Number of arrivals this minute ~ Poisson(rate)
        count = int(rng.poisson(rate))
        if count == 0:
            continue

        agg_rows.append({"minute": minute, "function_id": function_id, "count": float(count)})

        # Scatter arrivals uniformly within the minute (Poisson process)
        offsets = np.sort(rng.uniform(0.0, 60.0, count))
        base = minute * 60.0
        replay_timestamps.extend((base + offsets).tolist())

    agg_df = pd.DataFrame(agg_rows, columns=["minute", "function_id", "count"])
    replay_df = pd.DataFrame({"timestamp": replay_timestamps})
    return agg_df, replay_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--peak-rate",   type=float, default=500,
                        help="Requests/min at peak minutes (default 500)")
    parser.add_argument("--valley-rate", type=float, default=20,
                        help="Requests/min between peaks (default 20)")
    parser.add_argument("--period",        type=int,   default=60,
                        help="Minutes between consecutive peak starts (default 60)")
    parser.add_argument("--peak-duration", type=int,   default=1,
                        help="Duration of each peak in minutes (default 1)")
    parser.add_argument("--duration",    type=int,   default=1440,
                        help="Total trace length in minutes (default 1440 = 1 day)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--function-id", default="synthetic",
                        help="function_id label in aggregate trace (default 'synthetic')")
    parser.add_argument("--name",        default=None,
                        help="Output file stem (default: auto-generated from params)")
    parser.add_argument("--out",         default="datasets/traffic_pattern/synthetic",
                        help="Output directory (default: datasets/traffic_pattern/synthetic)")
    args = parser.parse_args()

    if args.period < 1:
        sys.exit("--period must be >= 1")
    if args.peak_duration < 1:
        sys.exit("--peak-duration must be >= 1")
    if args.peak_duration >= args.period:
        sys.exit("--peak-duration must be < --period")
    if args.peak_rate < args.valley_rate:
        print(f"Warning: peak_rate ({args.peak_rate}) < valley_rate ({args.valley_rate})")

    agg_df, replay_df = generate(
        peak_rate=args.peak_rate,
        valley_rate=args.valley_rate,
        period=args.period,
        peak_duration=args.peak_duration,
        duration=args.duration,
        seed=args.seed,
        function_id=args.function_id,
    )

    stem = args.name or (
        f"bursty_p{args.peak_rate:.0f}_v{args.valley_rate:.0f}"
        f"_T{args.period}_pd{args.peak_duration}_d{args.duration}_s{args.seed}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path    = out_dir / f"{stem}_aggregate.csv"
    replay_path = out_dir / f"{stem}_replay.csv"

    agg_df.to_csv(agg_path, index=False)
    replay_df.to_csv(replay_path, index=False)

    n_peaks   = args.duration // args.period
    total_req = len(replay_df)

    print(f"Generated bursty trace: {stem}")
    print(f"  duration    : {args.duration} min")
    print(f"  peaks       : {n_peaks}  (every {args.period} min, {args.peak_duration} min wide)")
    print(f"  peak rate   : {args.peak_rate} req/min")
    print(f"  valley rate : {args.valley_rate} req/min")
    print(f"  total reqs  : {total_req:,}")
    print(f"  seed        : {args.seed}")
    print(f"  aggregate   : {agg_path}")
    print(f"  replay      : {replay_path}")


if __name__ == "__main__":
    main()
