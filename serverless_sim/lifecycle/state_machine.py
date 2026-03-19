from __future__ import annotations

from serverless_sim.lifecycle.state_definition import StateDefinition
from serverless_sim.lifecycle.transition_definition import TransitionDefinition


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

        sm.add_state(StateDefinition("null", "stable"))
        sm.add_state(StateDefinition("prewarm", "stable", steady_memory=0.0))
        sm.add_state(StateDefinition("warm", "stable", service_bound=True, reusable=True))
        sm.add_state(StateDefinition("running", "transient", service_bound=True, reusable=False))
        sm.add_state(StateDefinition("evicted", "stable", reusable=False))

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
        return sm

    @classmethod
    def from_config(cls, config: dict) -> OpenWhiskExtendedStateMachine:
        """Build a state machine from config, falling back to default if absent.

        Config format::

            {
              "lifecycle": {
                "cold_start_chain": ["null", "prewarm", "code_loaded", "warm"],
                "states": [...],
                "transitions": [...]
              }
            }

        If ``cold_start_chain`` is not provided, it is inferred as
        ``["null", ..., "warm"]`` by following forward transitions.
        """
        lifecycle_cfg = config.get("lifecycle")
        if not lifecycle_cfg:
            return cls.default()

        sm = cls()

        for s_cfg in lifecycle_cfg.get("states", []):
            sd = StateDefinition(
                state_name=s_cfg["name"],
                category=s_cfg.get("category", "stable"),
                steady_cpu=s_cfg.get("steady_cpu", 0.0),
                steady_memory=s_cfg.get("steady_memory", 0.0),
                service_bound=s_cfg.get("service_bound", False),
                reusable=s_cfg.get("reusable", True),
            )
            sm.add_state(sd)

        for t_cfg in lifecycle_cfg.get("transitions", []):
            td = TransitionDefinition(
                from_state=t_cfg["from"],
                to_state=t_cfg["to"],
                transition_time=t_cfg.get("time", 0.0),
                transition_cpu=t_cfg.get("cpu", 0.0),
                transition_memory=t_cfg.get("memory", 0.0),
            )
            sm.add_transition(td)

        # Validate required states
        required = {"null", "warm", "running", "evicted"}
        missing = required - set(sm.states.keys())
        if missing:
            raise ValueError(f"Lifecycle config missing required states: {missing}")

        # Set cold-start chain
        chain_cfg = lifecycle_cfg.get("cold_start_chain")
        if chain_cfg:
            sm.set_cold_start_chain(chain_cfg)
        else:
            # Infer chain by following forward transitions from null → warm
            chain = sm._infer_chain()
            sm.set_cold_start_chain(chain)

        return sm

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
