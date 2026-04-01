"""Predictive pre-warming policy driven by an external forecast CSV.

Reads a predicted invocation count file (e.g. LSTM output) and sets
pool_target for the first pool state based on the predicted count at
the current simulation time.

CSV format (aggregate trace style)::

    minute,function_id,count,duration,predicted_count,phase
    60,Java_APIG-S,12.0,0.03,2.5,train
    120,Java_APIG-S,0.0,0.00,1.8,test

Config example::

    "controller": {
        "enabled": true,
        "policy": "predictive",
        "interval": 60.0,
        "predict_path": "experimental/lstm_baseline/predicted/Java_APIG-S_lstm_pred.csv",
        "predict_column": "predicted_count",
        "predict_scale": 1.0
    }
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING

from serverless_sim.controller.policies.base_policy import BaseControlPolicy

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class PredictivePolicy(BaseControlPolicy):
    """Set pool_target from a pre-computed forecast file.

    Parameters
    ----------
    predict_path : str
        Path to CSV with predictions.
    predict_column : str
        Column name for predicted values (default: "predicted_count").
    minute_column : str
        Column name for minute timestamps (default: "minute").
    function_id_column : str
        Column name for function/service id (default: "function_id").
    predict_scale : float
        Multiplier applied to predicted values (default: 1.0).
    """

    def __init__(
        self,
        predict_path: str,
        predict_column: str = "predicted_count",
        minute_column: str = "minute",
        function_id_column: str = "function_id",
        predict_scale: float = 1.0,
        avg_duration: float = 0.0,
        interval: float = 3600.0,
    ):
        self._predict_scale = predict_scale
        self._avg_duration = avg_duration
        self._interval = interval
        # Load predictions: {(minute, function_id): predicted_count}
        self._predictions: dict[tuple[int, str], float] = {}
        # Also build sorted minute list per function_id for nearest lookup
        self._minutes_by_func: dict[str, list[int]] = {}

        with open(predict_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(predict_column, "")
                if not val or val == "":
                    continue
                minute = int(float(row[minute_column]))
                func_id = row[function_id_column]
                self._predictions[(minute, func_id)] = float(val)
                self._minutes_by_func.setdefault(func_id, []).append(minute)

        # Sort minutes for binary search
        for func_id in self._minutes_by_func:
            self._minutes_by_func[func_id].sort()

    def _lookup(self, sim_time: float, func_id: str) -> float | None:
        """Find predicted count for the current simulation time.

        Converts sim_time (seconds) to minutes and finds the nearest
        minute with a prediction.
        """
        current_minute = int(sim_time / 60.0)
        minutes = self._minutes_by_func.get(func_id)
        if not minutes:
            return None

        # Exact match
        val = self._predictions.get((current_minute, func_id))
        if val is not None:
            return val

        # Find nearest minute (binary search)
        import bisect
        idx = bisect.bisect_left(minutes, current_minute)

        # Check closest neighbors
        candidates = []
        if idx < len(minutes):
            candidates.append(minutes[idx])
        if idx > 0:
            candidates.append(minutes[idx - 1])

        if not candidates:
            return None

        nearest = min(candidates, key=lambda m: abs(m - current_minute))
        return self._predictions.get((nearest, func_id))

    def decide(self, snapshot: dict, ctx: SimContext) -> list[dict]:
        actions = []
        if ctx.autoscaling_manager is None:
            return actions

        sim_time = ctx.env.now

        for svc_id in ctx.workload_manager.services:
            predicted = self._lookup(sim_time, svc_id)
            if predicted is None:
                continue

            # predicted = invocations per hour (unscaled)
            # pool_target = concurrent containers needed
            # = predicted_requests * avg_duration / interval
            scaled_requests = predicted * self._predict_scale
            if self._avg_duration > 0:
                pool_count = max(0, math.ceil(scaled_requests * self._avg_duration / self._interval))
            else:
                pool_count = max(0, math.ceil(scaled_requests))

            pool_states = ctx.autoscaling_manager._get_pool_states(svc_id)
            first_state = pool_states[0] if pool_states else "prewarm"
            current = ctx.autoscaling_manager.get_pool_target(svc_id, first_state)

            if pool_count != current:
                actions.append({
                    "action": "set_pool_target",
                    "service_id": svc_id,
                    "state": first_state,
                    "value": pool_count,
                })

        return actions
