from __future__ import annotations

from serverless_sim.lifecycle.state_definition import StateDefinition
from serverless_sim.lifecycle.transition_definition import TransitionDefinition


class OpenWhiskExtendedStateMachine:
    """Holds state/transition definitions, computes transition paths.

    Minimal default graph:
        null → prewarm → warm → running → warm → evicted
    """

    def __init__(self):
        self.states: dict[str, StateDefinition] = {}
        self.transitions: dict[tuple[str, str], TransitionDefinition] = {}
        self._adjacency: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def add_state(self, sd: StateDefinition) -> None:
        self.states[sd.state_name] = sd
        self._adjacency.setdefault(sd.state_name, [])

    def add_transition(self, td: TransitionDefinition) -> None:
        key = (td.from_state, td.to_state)
        self.transitions[key] = td
        self._adjacency.setdefault(td.from_state, []).append(td.to_state)

    def get_transition(self, from_state: str, to_state: str) -> TransitionDefinition | None:
        return self.transitions.get((from_state, to_state))

    def get_state(self, name: str) -> StateDefinition | None:
        return self.states.get(name)

    # ------------------------------------------------------------------
    # Path finding (BFS)
    # ------------------------------------------------------------------

    def find_path(self, from_state: str, to_state: str) -> list[str] | None:
        """Return the shortest state path (including endpoints), or None."""
        if from_state == to_state:
            return [from_state]
        visited = {from_state}
        queue = [[from_state]]
        while queue:
            path = queue.pop(0)
            current = path[-1]
            for neighbor in self._adjacency.get(current, []):
                if neighbor == to_state:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    # ------------------------------------------------------------------
    # Default minimal graph
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> OpenWhiskExtendedStateMachine:
        """Create the minimal OpenWhisk-style state machine."""
        sm = cls()

        sm.add_state(StateDefinition("null", "stable"))
        sm.add_state(StateDefinition("prewarm", "stable", steady_memory=0.0))
        sm.add_state(StateDefinition("warm", "stable", service_bound=True, reusable=True))
        sm.add_state(StateDefinition("running", "transient", service_bound=True, reusable=False))
        sm.add_state(StateDefinition("evicted", "stable", reusable=False))

        # null → prewarm (container creation)
        sm.add_transition(TransitionDefinition("null", "prewarm", transition_time=0.5))
        # prewarm → warm (service binding / code loading)
        sm.add_transition(TransitionDefinition("prewarm", "warm", transition_time=0.3))
        # warm → running (start execution)
        sm.add_transition(TransitionDefinition("warm", "running", transition_time=0.0))
        # running → warm (finish execution)
        sm.add_transition(TransitionDefinition("running", "warm", transition_time=0.0))
        # warm → evicted
        sm.add_transition(TransitionDefinition("warm", "evicted", transition_time=0.0))
        # prewarm → evicted
        sm.add_transition(TransitionDefinition("prewarm", "evicted", transition_time=0.0))

        return sm
