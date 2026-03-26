from __future__ import annotations

import csv
import os

from serverless_sim.lifecycle.state_definition import StateDefinition
from serverless_sim.lifecycle.transition_definition import TransitionDefinition
from serverless_sim.lifecycle.transition_model import (
    BaseTransitionModel,
    BaseStateResourceModel,
    DeterministicTransitionModel,
    FixedStateResourceModel,
)


class OpenWhiskExtendedStateMachine:
    """Holds state/transition definitions with a linear cold-start chain.

    The cold-start chain is an ordered list of states from ``null`` to
    ``warm``.  Transitions only exist between consecutive chain members,
    plus ``warm ↔ running`` and ``* → evicted``.

    Default chain::

        null → prewarm → warm

    Extended example::

        null → prewarm → code_loaded → warm
    """

    def __init__(self):
        self.states: dict[str, StateDefinition] = {}
        self.transitions: dict[tuple[str, str], TransitionDefinition] = {}
        self._cold_start_chain: list[str] = []
        self.transition_model: BaseTransitionModel = DeterministicTransitionModel()
        self.state_resource_model: BaseStateResourceModel | None = None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def add_state(self, sd: StateDefinition) -> None:
        self.states[sd.state_name] = sd

    def add_transition(self, td: TransitionDefinition) -> None:
        key = (td.from_state, td.to_state)
        self.transitions[key] = td

    def set_cold_start_chain(self, chain: list[str]) -> None:
        """Set the linear cold-start chain and validate it.

        The chain must start with ``"null"`` and end with ``"warm"``.
        Every state in the chain must be registered, and transitions
        must exist between each consecutive pair.
        """
        if len(chain) < 2:
            raise ValueError("Cold-start chain must have at least 2 states (null → warm)")
        if chain[0] != "null":
            raise ValueError(f"Cold-start chain must start with 'null', got '{chain[0]}'")
        if chain[-1] != "warm":
            raise ValueError(f"Cold-start chain must end with 'warm', got '{chain[-1]}'")

        for state_name in chain:
            if state_name not in self.states:
                raise ValueError(f"Chain state '{state_name}' not found in registered states")

        for i in range(len(chain) - 1):
            key = (chain[i], chain[i + 1])
            if key not in self.transitions:
                raise ValueError(
                    f"Missing transition {chain[i]} → {chain[i+1]} "
                    f"(required by cold_start_chain)"
                )

        self._cold_start_chain = list(chain)

    def get_transition(self, from_state: str, to_state: str) -> TransitionDefinition | None:
        return self.transitions.get((from_state, to_state))

    def get_state(self, name: str) -> StateDefinition | None:
        return self.states.get(name)

    # ------------------------------------------------------------------
    # Cold-start path
    # ------------------------------------------------------------------

    def get_evictable_states(self) -> set[str]:
        """Return state names that are stable, idle-evictable (not null/evicted/running)."""
        return {
            name for name, sd in self.states.items()
            if sd.is_stable and name not in ("null", "evicted")
        }

    def get_cold_start_path(self) -> list[str]:
        """Return the linear cold-start chain (null → ... → warm)."""
        return list(self._cold_start_chain)

    def find_path(self, from_state: str, to_state: str) -> list[str] | None:
        """Return the sub-chain between *from_state* and *to_state*.

        Only supports states within the cold-start chain.  Returns
        ``None`` if either state is not in the chain or *from_state*
        comes after *to_state*.
        """
        if from_state == to_state:
            return [from_state]
        if from_state not in self._cold_start_chain or to_state not in self._cold_start_chain:
            return None
        start = self._cold_start_chain.index(from_state)
        end = self._cold_start_chain.index(to_state)
        if start > end:
            return None
        return self._cold_start_chain[start:end + 1]

    # ------------------------------------------------------------------
    # Default minimal chain
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> OpenWhiskExtendedStateMachine:
        """Create the minimal OpenWhisk-style state machine.

        Chain: null → prewarm → warm
        """
        sm = cls()

        sm.add_state(StateDefinition("null", "stable", cpu=0.0, memory=0.0))
        sm.add_state(StateDefinition("prewarm", "stable", cpu=0.0, memory=128.0))
        sm.add_state(StateDefinition("warm", "stable", cpu=0.1, memory=256.0, service_bound=True, reusable=True))
        sm.add_state(StateDefinition("running", "transient", cpu=1.0, memory=256.0, service_bound=True, reusable=False))
        sm.add_state(StateDefinition("evicted", "stable", cpu=0.0, memory=0.0, reusable=False))

        # Cold-start chain transitions
        sm.add_transition(TransitionDefinition("null", "prewarm", transition_time=0.5))
        sm.add_transition(TransitionDefinition("prewarm", "warm", transition_time=0.3))

        # Execution cycle
        sm.add_transition(TransitionDefinition("warm", "running", transition_time=0.0))
        sm.add_transition(TransitionDefinition("running", "warm", transition_time=0.0))

        # Eviction
        sm.add_transition(TransitionDefinition("warm", "evicted", transition_time=0.0))
        sm.add_transition(TransitionDefinition("prewarm", "evicted", transition_time=0.0))

        sm.set_cold_start_chain(["null", "prewarm", "warm"])

        # Build deterministic transition model from TransitionDefinitions
        sm._build_deterministic_model()
        return sm

    def _build_deterministic_model(self) -> None:
        """Populate transition_model from TransitionDefinition values."""
        model = DeterministicTransitionModel()
        for (f, t), td in self.transitions.items():
            model.set(f, t, time=td.transition_time,
                      cpu=td.transition_cpu, memory=td.transition_memory)
        self.transition_model = model

    @classmethod
    def from_lifecycle_config(cls, lifecycle_cfg: dict) -> OpenWhiskExtendedStateMachine:
        """Build a state machine from a lifecycle config dict.

        Config format (deterministic transitions)::

            {
              "cold_start_chain": ["null", "prewarm", "code_loaded", "warm"],
              "states": [...],
              "transitions": [{"from": "null", "to": "prewarm", "time": 0.5}, ...]
            }

        Config format (CSV trace, no transitions needed)::

            {
              "cold_start_chain": ["null", "prewarm", "code_loaded", "warm"],
              "states": [...],
              "transition_profile": "path/to/transitions.csv"
            }

        When ``transition_profile`` is set and ``transitions`` is absent,
        the graph structure is auto-generated from ``cold_start_chain``
        plus ``warm ↔ running`` and ``* → evicted`` for all stable
        intermediate states.  Time/cpu/memory values come from the CSV.
        """
        sm = cls()

        for s_cfg in lifecycle_cfg.get("states", []):
            sd = StateDefinition(
                state_name=s_cfg["name"],
                category=s_cfg.get("category", "stable"),
                cpu=s_cfg.get("cpu", 0.0),
                memory=s_cfg.get("memory", 0.0),
                service_bound=s_cfg.get("service_bound", False),
                reusable=s_cfg.get("reusable", True),
            )
            sm.add_state(sd)

        # Auto-add internal states if not declared
        if "evicted" not in sm.states:
            sm.add_state(StateDefinition("evicted", "stable", cpu=0.0, memory=0.0, reusable=False))
        if "running" not in sm.states:
            sm.add_state(StateDefinition("running", "transient", service_bound=True, reusable=False))

        # State resource model from CSV if provided
        state_profile = lifecycle_cfg.get("state_profile")
        if state_profile:
            from serverless_sim.lifecycle.transition_model import CsvSampleStateResourceModel
            sm.state_resource_model = CsvSampleStateResourceModel.from_csv(state_profile)
            # Also set fixed values from mean (for peak_memory calculation etc.)
            sm._apply_state_profile_means(state_profile)

        # Validate required states
        required = {"null", "warm"}
        missing = required - set(sm.states.keys())
        if missing:
            raise ValueError(f"Lifecycle config missing required states: {missing}")

        transitions_cfg = lifecycle_cfg.get("transitions")
        chain_cfg = lifecycle_cfg.get("cold_start_chain")

        if transitions_cfg:
            # Explicit transitions
            for t_cfg in transitions_cfg:
                td = TransitionDefinition(
                    from_state=t_cfg["from"],
                    to_state=t_cfg["to"],
                    transition_time=t_cfg.get("time", 0.0),
                    transition_cpu=t_cfg.get("cpu", 0.0),
                    transition_memory=t_cfg.get("memory", 0.0),
                )
                sm.add_transition(td)
        elif chain_cfg:
            # Auto-generate transitions from chain
            sm._generate_transitions_from_chain(chain_cfg)
        else:
            raise ValueError(
                "Lifecycle config requires either 'transitions' or "
                "'cold_start_chain' (to auto-generate transitions)"
            )

        # Set cold-start chain
        if chain_cfg:
            sm.set_cold_start_chain(chain_cfg)
        else:
            chain = sm._infer_chain()
            sm.set_cold_start_chain(chain)

        # Build transition model
        csv_path = lifecycle_cfg.get("transition_profile")
        if csv_path:
            from serverless_sim.lifecycle.transition_model import CsvSampleTransitionModel
            sm.transition_model = CsvSampleTransitionModel.from_csv(csv_path)
        else:
            sm._build_deterministic_model()

        return sm

    @classmethod
    def from_config(cls, config: dict) -> OpenWhiskExtendedStateMachine:
        """Build from top-level config (reads config["lifecycle"]).

        Falls back to default() if no lifecycle section.
        """
        lifecycle_cfg = config.get("lifecycle")
        if not lifecycle_cfg:
            return cls.default()
        return cls.from_lifecycle_config(lifecycle_cfg)

    def _apply_state_profile_means(self, csv_path: str) -> None:
        """Set state cpu/memory to mean values from CSV (for peak calculations).

        The actual runtime values are sampled by state_resource_model.
        This sets the StateDefinition fields to means so that
        peak_memory/peak_cpu calculations work correctly.
        """
        by_state: dict[str, list[tuple[float, float]]] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                state_name = row["state"]
                cpu = float(row.get("cpu", 0))
                mem = float(row.get("memory", 0))
                by_state.setdefault(state_name, []).append((cpu, mem))

        for state_name, values in by_state.items():
            if state_name in self.states:
                mean_cpu = sum(v[0] for v in values) / len(values)
                mean_mem = sum(v[1] for v in values) / len(values)
                self.states[state_name].cpu = mean_cpu
                self.states[state_name].memory = mean_mem

    def _generate_transitions_from_chain(self, chain: list[str]) -> None:
        """Auto-generate transitions from cold_start_chain.

        Creates:
        - Chain transitions: each consecutive pair in the chain
        - Execution cycle: warm → running, running → warm
        - Eviction: every stable intermediate state → evicted, plus warm → evicted
        """
        # Chain transitions (null→prewarm, prewarm→warm, etc.)
        for i in range(len(chain) - 1):
            self.add_transition(TransitionDefinition(chain[i], chain[i + 1]))

        # Execution cycle
        self.add_transition(TransitionDefinition("warm", "running"))
        self.add_transition(TransitionDefinition("running", "warm"))

        # Eviction: all stable states except null and evicted → evicted
        for name, sd in self.states.items():
            if sd.is_stable and name not in ("null", "evicted"):
                key = (name, "evicted")
                if key not in self.transitions:
                    self.add_transition(TransitionDefinition(name, "evicted"))

    def _infer_chain(self) -> list[str]:
        """Infer linear chain from null to warm by following transitions.

        Each state (except warm) must have exactly one forward transition
        to another chain candidate (excluding running and evicted).
        """
        exclude = {"running", "evicted"}
        chain = ["null"]
        visited = {"null"}

        while chain[-1] != "warm":
            current = chain[-1]
            # Find all forward transitions from current that aren't to running/evicted
            candidates = [
                to_state for (from_s, to_state) in self.transitions
                if from_s == current and to_state not in exclude and to_state not in visited
            ]
            if len(candidates) == 0:
                raise ValueError(
                    f"Cannot infer cold_start_chain: no forward transition from "
                    f"'{current}' (chain so far: {chain}). "
                    f"Add 'cold_start_chain' to lifecycle config explicitly."
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Cannot infer cold_start_chain: '{current}' has multiple "
                    f"forward transitions {candidates}. "
                    f"Add 'cold_start_chain' to lifecycle config explicitly."
                )
            chain.append(candidates[0])
            visited.add(candidates[0])

        return chain
