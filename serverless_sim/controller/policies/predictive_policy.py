"""Predictive pre-warming policy driven by external forecast CSVs.

Reads a predicted invocation count file per service (e.g. LSTM output)
and sets pool_target for the first pool state based on the predicted
count at the current simulation time.

CSV format (per-service, single forecast file)::

    minute,count,predicted_count,phase
    60,12.0,2.5,train
    120,0.0,1.8,test

Each service points at its own forecast file via
``services[i].predict_path``.  Controller config holds only global
knobs (column name, scale, interval).

Config example::

    "services": [
        {
            "service_id": "Java_APIG-S",
            "predict_path": "experimental/lstm_baseline/predicted/Java_APIG-S_lstm_pred.csv",
            ...
        }
    ],
    "controller": {
        "enabled": true,
        "policy": "predictive",
        "interval": 60.0,
        "predict_column": "predicted_count",
        "predict_scale": 1.0
    }
"""

from __future__ import annotations

import bisect
import csv
import math
from typing import TYPE_CHECKING

from serverless_sim.controller.policies.base_policy import BaseControlPolicy

if TYPE_CHECKING:
    from serverless_sim.core.simulation.sim_context import SimContext


class PredictivePolicy(BaseControlPolicy):
    """Set pool_target from per-service forecast files.

    Parameters
    ----------
    predict_paths : dict[str, str]
        Mapping ``service_id -> CSV path``.
    predict_column : str
        Column name for predicted values (default: "predicted_count").
    minute_column : str
        Column name for minute timestamps (default: "minute").
    predict_scale : float
        Multiplier applied to predicted values (default: 1.0).
    """

    def __init__(
        self,
        predict_paths: dict[str, str],
        predict_column: str = "predicted_count",
        minute_column: str = "minute",
        predict_scale: float = 1.0,
        avg_duration: float = 0.0,
        interval: float = 3600.0,
    ):
        self._predict_scale = predict_scale
        self._avg_duration = avg_duration
        self._interval = interval
        self._predictions: dict[str, dict[int, float]] = {}
        self._minutes_by_func: dict[str, list[int]] = {}

        for func_id, path in predict_paths.items():
            preds: dict[int, float] = {}
            minutes: list[int] = []
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val = row.get(predict_column, "")
                    if not val:
                        continue
                    minute = int(float(row[minute_column]))
                    preds[minute] = float(val)
                    minutes.append(minute)
            minutes.sort()
            self._predictions[func_id] = preds
            self._minutes_by_func[func_id] = minutes

    def _lookup(self, sim_time: float, func_id: str) -> float | None:
        """Find predicted count for the current simulation time."""
        current_minute = int(sim_time / 60.0)
        preds = self._predictions.get(func_id)
        minutes = self._minutes_by_func.get(func_id)
        if not preds or not minutes:
            return None

        val = preds.get(current_minute)
        if val is not None:
            return val

        idx = bisect.bisect_left(minutes, current_minute)
        candidates = []
        if idx < len(minutes):
            candidates.append(minutes[idx])
        if idx > 0:
            candidates.append(minutes[idx - 1])
        if not candidates:
            return None

        nearest = min(candidates, key=lambda m: abs(m - current_minute))
        return preds.get(nearest)

    def decide(self, snapshot: dict, ctx: SimContext) -> list[dict]:
        actions = []
        if ctx.autoscaling_manager is None:
            return actions

        sim_time = ctx.env.now

        for svc_id in ctx.workload_manager.services:
            predicted = self._lookup(sim_time, svc_id)
            if predicted is None:
                continue

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
