from serverless_sim.controller.policies.base_policy import BaseControlPolicy


class ThresholdPolicy(BaseControlPolicy):
    """Rule-based policy: adjust prewarm/idle_timeout based on CPU and latency thresholds."""
    pass
