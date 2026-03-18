# Design Specification: SimPy-based Serverless System Simulator with Gymnasium and Stable-Baselines3

## 1. Document purpose

This document specifies the system design for a modular, extensible serverless simulator inspired by OpenWhisk. The primary purpose is to serve as implementation input for a coding agent. It defines:

* scope and goals
* architectural decomposition
* module responsibilities
* class and interface contracts
* event semantics in SimPy
* configuration schema expectations
* logging and output behavior
* Gymnasium wrapper behavior
* PPO training and inference integration with Stable-Baselines3
* runtime flow and implementation order

This is a design document, not source code. The coding agent should use it as the authoritative blueprint for file layout, module boundaries, API contracts, and expected behavior.

---

## 2. High-level goals

The system must support three usage modes:

1. **Standalone SimPy simulation**
2. **Gymnasium-wrapped simulation environment**
3. **RL training/inference using PPO from Stable-Baselines3**

The simulated system models a serverless platform with the following functional parts:

1. workload / request arrival
2. scheduling / dispatching
3. worker nodes / compute classes / serving model
4. function instance lifecycle
5. autoscaling / memory-bounded pools / idle timeout / LRU eviction
6. monitoring / metric collection

Additional required supporting parts:

* controller module that reads metrics and updates autoscaling settings
* configuration subsystem using JSON files
* logging subsystem for all modules
* CSV export subsystem with selectable detail level
* run directory management under `logs/`
* sample configs and documentation

The implementation must emphasize:

* clean modular separation
* extensibility for new policies/mechanisms
* deterministic and reproducible runs via seeding
* compatibility with vectorized Gymnasium environments
* no shared mutable global state across environments

---

## 3. Non-goals

The simulator does **not** need to reproduce all OpenWhisk internals exactly. The following are explicitly abstracted unless later extended:

* Kafka/controller implementation details (the per-node `simpy.Store` acts as an abstract Kafka-like queue)
* persistent activation database
* authentication/API gateway
* detailed network stack
* exact container runtime internals
* exact OpenWhisk deployment topology

The simulator should preserve the important behavioral semantics needed for research and control:

* request arrival and queueing
* resource-aware dispatching
* warm/cold/extended lifecycle transitions
* memory-bounded autoscaling with LRU eviction and idle timeout handling
* QoS/resource metrics

---

## 4. System architecture overview

### 4.1 Main architectural layers

The system is organized into the following layers:

1. **Core runtime layer**

   * SimPy environment
   * configuration loading
   * logging setup
   * simulation builder / simulation context

2. **Domain simulation layer**

   * workload generation
   * scheduling/dispatching (ShardingContainerPoolBalancer)
   * cluster/node model (with per-node simpy.Store queues)
   * lifecycle state machine (stable/transient states, simpy.Resource concurrency)
   * autoscaling logic (memory-bounded, LRU eviction, global idle timeout)
   * monitoring
   * controller

3. **Integration layer**

   * Gymnasium wrapper
   * Stable-Baselines3 training/inference integration
   * exporters
   * CLI entrypoints

### 4.2 Runtime data flow

Normal simulation flow:

1. load simulation config
2. build simulation context and all modules
3. initialize cluster (create nodes with `simpy.Store` queues), lifecycle definitions, and prewarm containers
4. start monitor process (runs independently on its own collection_interval)
5. start autoscaling reconcile process
6. start per-node request pull loops
7. start workload generators
8. optionally start controller process
9. run the SimPy environment
10. collect logs, metrics, and export (mode 0/1/2)

RL-wrapped flow:

1. create a fresh simulation for each Gym env instance
2. on `reset()`, rebuild simulator state
3. on `step(action)`, apply control action to autoscaling/controller-facing APIs
4. run the simulator forward by a fixed decision interval
5. collect metrics from monitor
6. build observation, reward, done flags, info

---

## 5. Required repository layout

The implementation should use the following repository structure.

```text
serverless_sim/
├── README.md
├── docs/
│   ├── config_reference.md
│   ├── simpy_standalone.md
│   ├── gym_wrapper.md
│   └── rl_training.md
├── configs/
│   ├── simulation/
│   │   ├── sample_minimal.json
│   │   ├── sample_multi_service.json
│   │   └── sample_extended_states.json
│   ├── gym/
│   │   └── sample_gym_discrete.json
│   └── rl/
│       ├── sample_ppo_train.json
│       └── sample_ppo_infer.json
├── logs/
├── outputs/
├── runtime/
│   ├── cli.py
│   └── app.py
├── core/
│   ├── simulation/
│   ├── events/
│   ├── config/
│   ├── logging/
│   └── utils/
├── workload/
├── scheduling/
├── cluster/
├── lifecycle/
├── autoscaling/
├── monitoring/
├── controller/
│   ├── base_controller.py
│   └── policies/
│       ├── base_policy.py
│       └── threshold_policy.py
├── export/
├── gym_env/
│   ├── serverless_gym_env.py
│   ├── observation_builder.py
│   ├── action_mapper.py
│   └── reward_calculator.py
└── rl_agent/
    └── train.py
```

This structure is mandatory unless the coding agent has a compelling implementation reason to preserve the same module boundaries using a close equivalent.

---

## 6. Core simulation concepts

### 6.1 SimContext

`SimContext` is the central runtime dependency container. It must hold references to all major modules and runtime state needed during simulation.

Required fields:

* `env`: SimPy environment
* `config`: loaded simulation config object
* `rng`: root RNG object or seed manager
* `logger`: root/system logger
* `workload_manager`
* `dispatcher`
* `cluster_manager`
* `lifecycle_manager`
* `autoscaling_manager`
* `monitor_manager`
* `controller` (optional)
* `request_table`: central store for all invocations
* `export_manager` (optional)
* `run_dir`: output path for this run

`SimContext` must not contain policy logic itself. It only wires references.

### 6.2 SimulationBuilder

`SimulationBuilder` constructs the entire simulator from config.

Responsibilities:

* parse JSON config
* create run directory
* initialize logging
* create SimPy environment
* construct all managers and policy objects
* register APIs between modules
* return a configured `SimulationEngine`

### 6.3 SimulationEngine

`SimulationEngine` is the top-level runtime object for standalone simulation.

Required methods:

* `setup()`
* `run(until: float)`
* `shutdown()`
* `get_snapshot()`

Responsibilities:

* start long-running SimPy processes
* coordinate run lifecycle
* flush exporters and final logs at the end

---

## 7. Part 1: Workload / request arrival

### 7.1 Scope

Current required mode:

* Poisson arrivals
* fixed job size per service
* all parameters loaded from the shared simulation JSON config

### 7.2 Required behavior

For each service class:

* inter-arrival times follow exponential distribution parameterized by rate `lambda`
* each request has a fixed `job_size`
* each request also carries timeout and basic metadata
* each service defines `max_concurrency`, the maximum number of requests a single bound container instance may execute simultaneously
* each service has independent generation process

The system must support multiple services at once.

### 7.3 Main classes

#### ServiceClass

Represents a function/service type.

Required fields:

* `service_id: str`
* `display_name: str`
* `arrival_mode: str`
* `arrival_rate: float`
* `job_size: float`
* `timeout: float`
* `max_concurrency: int`
* `resource_hint_cpu: float`
* `resource_hint_memory: float`
* `per_request_resource: ResourceProfile` (additional CPU consumed per active request on the node; memory=0)

#### Invocation

Represents a single request.

Required fields:

* `request_id: str`
* `service_id: str`
* `arrival_time: float`
* `job_size: float`
* `timeout: float`
* `dispatch_time: Optional[float]`
* `queue_enter_time: Optional[float]`
* `execution_start_time: Optional[float]`
* `execution_end_time: Optional[float]`
* `completion_time: Optional[float]`
* `assigned_node_id: Optional[str]`
* `assigned_instance_id: Optional[str]`
* `cold_start: bool`
* `dropped: bool`
* `timed_out: bool`
* `drop_reason: Optional[str]` (one of `queue_full`, `timeout`, `no_capacity`, or None)
* `status: str`

Derived metrics must be computable from this record:

* end-to-end latency
* queue delay
* execution duration
* dispatch delay

#### BaseGenerator

Interface for workload generators.

Required methods:

* `attach(sim_context)`
* `start_for_service(service: ServiceClass)`

#### PoissonFixedSizeGenerator

Concrete generator implementing exponential inter-arrival with fixed request size.

Implementation expectations:

* one SimPy process per service
* uses per-service RNG stream or deterministic seeded substream

#### WorkloadManager

Responsibilities:

* store all services
* instantiate generator objects
* start workload processes
* create invocations and forward them to the dispatcher

Required methods:

* `start()`
* `emit_invocation(service_id: str)`
* `register_service(service: ServiceClass)`

### 7.4 SimPy event/process semantics

Each service must run as an independent generator loop:

1. sample next inter-arrival time
2. `yield env.timeout(delta)`
3. create `Invocation`
4. register invocation in central request table
5. pass invocation to dispatcher

No separate event bus is required for SimPy internals, but event records should be written into logs/trace where helpful.

### 7.5 Configuration requirements

Simulation config must define services in a section similar to:

* service id
* display name
* arrival mode = `poisson`
* arrival rate
* fixed job size
* timeout
* max concurrency per container
* resource hints (cpu, memory) for container allocation
* per_request_resource (cpu cost per active request; memory=0)

The coding agent should validate unsupported modes and fail clearly.

### 7.6 Extensibility contract

Future workload modes should fit under the same generator interface, such as:

* trace-driven arrivals
* burst phase model
* time-varying Poisson
* DAG/workflow arrivals

Therefore generator selection must be registry-based.

---

## 8. Part 2: Control plane / scheduling / invoker selection

### 8.1 Scope

The scheduling layer uses a single load-balancing strategy: **ShardingContainerPoolBalancer**, which uses consistent hashing on `service_id` to select a primary node. There are no alternative scheduling policies; the previous `hash_affinity`, `least_connection`, and `weighted_round_robin` policies have been removed.

### 8.2 Scheduling responsibilities

When a request arrives:

1. hash the `service_id` onto a consistent hash ring of enabled nodes to find the primary node
2. check whether the primary node has enough available memory to host a container for this service (read `node.available_memory` in real time, not from a periodic snapshot)
3. if the primary node lacks memory, walk the hash ring to the next node and repeat the check
4. if a suitable node is found, put the invocation into that node's `simpy.Store` queue (see Section 9.4)
5. if no node on the ring has enough memory, drop the request immediately (`drop_reason: no_capacity`)
6. record the scheduling decision in the request record and logs

Dispatch is **asynchronous**: the load balancer returns immediately after putting the request into the node's store. It does not wait for execution to complete.

Scheduling only chooses a node and enqueues. It must not directly allocate containers or run lifecycle transitions.

### 8.3 Main classes

#### ShardingContainerPoolBalancer

Implements consistent hashing on `service_id` with fallback walk.

Required methods:

* `select_node(invocation, cluster_manager) -> Optional[str]` -- returns node_id or None if no capacity
* `_build_hash_ring(enabled_nodes)` -- rebuild ring when node set changes

Behavior:

* hash `service_id` to a position on a consistent hash ring of enabled node IDs
* the resulting node is the primary; check `node.available_memory >= service.resource_hint_memory` by reading the node's current state directly (real-time, not snapshot)
* if primary lacks memory, walk clockwise to next node on ring and repeat
* if full ring traversal finds no suitable node, return None (request will be dropped)

#### Dispatcher

Responsibilities:

* hold the ShardingContainerPoolBalancer
* receive new invocations
* call balancer to select node
* put invocation into the selected node's `simpy.Store` queue
* if balancer returns None, mark invocation as dropped (`drop_reason: no_capacity`)

Required methods:

* `on_request_arrival(invocation)`
* `dispatch(invocation)`

### 8.4 Data dependencies

The balancer needs real-time read access to:

* node enabled/disabled status
* `node.available_memory` (total - baseline - used_dynamic memory)

These are read directly from node objects via cluster manager. No periodic snapshot is used for scheduling decisions.

---

## 9. Part 3: Worker nodes / compute classes / serving model

### 9.1 Scope

Current required capabilities:

* only serving model is `fixed_rate`
* processing time is proportional to job size
* processing factor loaded from config
* resources: CPU and memory
* configurable per-service container concurrency
* multiple compute classes supported
* each compute class config includes:

  * number of nodes
  * serving model
  * resources
* nodes can be enabled or disabled
* enabled nodes consume baseline CPU and memory
* all values loaded from config

### 9.2 Main entities

#### ResourceProfile

Represents a CPU/memory pair.

Required fields:

* `cpu: float`
* `memory: float`

Required operations:

* add
* subtract
* compare capacity feasibility
* clone/serialize

#### ComputeClass

Represents a family of nodes.

Required fields:

* `class_id: str`
* `node_count: int`
* `serving_model_type: str`
* `processing_factor: float`
* `total_resource: ResourceProfile`
* `baseline_when_enabled: ResourceProfile`
* optional weight / labels for scheduling

#### Node

Represents a single worker node.

Required fields:

* `node_id: str`
* `class_id: str`
* `enabled: bool`
* `total_resource: ResourceProfile`
* `baseline_when_enabled: ResourceProfile`
* `used_dynamic_resource: ResourceProfile` (includes both container steady-state resources and per-request CPU)
* `request_store: simpy.Store` (per-node queue; acts as a Kafka-like topic that the node pulls from)
* `active_requests: int`
* `instances: dict[instance_id, ContainerInstance]`
* `serving_model`

Derived resource values:

* `baseline_resource = baseline_when_enabled if enabled else zero`
* `available = total - baseline - used_dynamic`
* `available_memory = total.memory - baseline.memory - used_dynamic.memory`

Required methods:

* `enable()`
* `disable()`
* `can_allocate(resource_profile) -> bool`
* `allocate_resource(resource_profile)`
* `release_resource(resource_profile)`
* `get_state_snapshot()`

#### BaseServingModel

Required method:

* `estimate_service_time(invocation, instance, node) -> float`

#### FixedRateModel

Behavior:

* service time = `job_size * processing_factor`

The design must support replacing this with future models.

#### ClusterManager

Responsibilities:

* create nodes from compute class config
* keep node registry
* return enabled nodes
* expose cluster snapshot to scheduler/monitor
* support node enable/disable operations

Required methods:

* `get_enabled_nodes()`
* `get_node(node_id)`
* `enable_node(node_id)`
* `disable_node(node_id)`
* `get_cluster_snapshot()`

### 9.3 Node behavior

A node does not directly choose scaling policy. It is an execution and hosting resource that **pulls** requests from its local store.

Each node runs a SimPy process loop that continuously pulls from its `request_store`:

```
while True:
    invocation = yield request_store.get()
    env.process(handle_request(invocation))
```

For each pulled request:

1. lifecycle manager is consulted to obtain a usable instance with available concurrency capacity
2. per-request CPU admission is checked: node must have enough CPU for `service.per_request_resource`
3. if both a concurrency slot and per-request CPU are available, execution begins
4. on execution start: allocate `per_request_resource` to `node.used_dynamic_resource`
5. execution duration is computed via serving model
6. on execution finish: release `per_request_resource` from `node.used_dynamic_resource`
7. resource/accounting and request state are updated

Admission has **two gates** for each request:
1. a concurrency slot is available on a suitable container instance
2. the node has enough CPU for `service.per_request_resource`

Both must be satisfied before execution can begin.

### 9.4 Queue semantics (pull model with simpy.Store)

Each node owns a `simpy.Store(env, capacity=queue_capacity)` that acts as its request queue.

* The load balancer (dispatcher) puts requests into a node's store: `store.put(invocation)`
* The node's process loop pulls from the store: `yield store.get()`
* Queue capacity is configurable in simulation config (default: unlimited / very large)
* When the store is full (at capacity), the put fails and the request is dropped (`drop_reason: queue_full`)
* Dispatch is asynchronous: the load balancer returns immediately after put

**Timeout handling in the queue:** Each request races its queue wait and execution against a timeout event measured from `arrival_time` (client perspective). If the timeout fires while the request is still in the queue, the request is removed from the queue and marked `timed_out=True` with `drop_reason: timeout`. If the timeout fires during execution, the execution is aborted immediately and resources (per-request CPU + concurrency slot) are released. Implementation uses a SimPy event race between execution completion and a timeout event per request.

### 9.5 Extensibility contract

Future additions may include:

* CPU sharing / slowdown under contention
* per-service processing factor
* node power-on delay
* heterogeneous interference
* adaptive concurrency tuning
* per-state concurrency overrides

Therefore node internals should keep serving model separate from queueing and lifecycle.

---

## 10. Part 4: Function instance lifecycle

### 10.1 Scope

The simulator must support an OpenWhisk-inspired lifecycle with required base states and configurable extended states.

States are classified into two categories:

* **Stable states** -- the system (autoscaler/controller) can trigger transitions from these states. Stable states are: `null`, `prewarm`, any extended states, `warm`, `evicted`.
* **Transient states** -- the system must wait for these to resolve before acting. Transient states are: `running`, and any ongoing transition between stable states (e.g., prewarm-to-warm).

`running` is a **transient** state, not a stable state. A container enters `running` when its `active_request_count` goes from 0 to 1 (warm to running transition). It returns to `warm` when `active_request_count` goes from 1 to 0. Additional concurrent requests simply increment/decrement the count without triggering state transitions.

The autoscaler and controller can only downgrade or evict containers that are in a **stable** state. They cannot act on containers that are `running` (transient) or mid-transition (transient). They must wait for the container to return to a stable state.

Additionally, the config may define arbitrary extended states between prewarm and warm, representing partial loading stages.

A bound container instance may execute more than one request concurrently, up to the configured `max_concurrency` for its service. Concurrency is managed via `simpy.Resource(env, capacity=max_concurrency)` on each ContainerInstance (see Section 10.3). Concurrency is therefore a first-class admission constraint in addition to lifecycle state readiness.

Resource profiles must be configurable both for:

* steady-state occupancy of a state
* transient resource usage during transitions

Per-request resource cost: each active request consumes additional CPU on the node (not memory). This is defined by `ServiceClass.per_request_resource`. On request start, the per-request CPU is allocated to `node.used_dynamic_resource`. On request finish, it is released. See Section 9.3 for the two-gate admission model.

Transition times must also be configurable.

### 10.2 State model requirements

Each state definition must include:

* state name
* state category/type
* steady resource profile
* whether bound to a specific service
* whether reusable

Each transition definition must include:

* from-state
* to-state
* transition time
* transition resource profile
* optional service constraints

### 10.3 Main classes

#### StateDefinition

Required fields:

* `state_name: str`
* `category: str`
* `steady_resource: ResourceProfile`
* `service_bound: bool`
* `reusable: bool`

#### TransitionDefinition

Required fields:

* `from_state: str`
* `to_state: str`
* `transition_time: float`
* `transition_resource: ResourceProfile`
* `allowed_services: Optional[list[str]]`

#### ContainerInstance

Represents an instance/container hosted on a node.

Required fields:

* `instance_id: str`
* `node_id: str`
* `bound_service_id: Optional[str]`
* `current_state: str`
* `target_state: Optional[str]`
* `state_enter_time: float`
* `last_used_time: Optional[float]`
* `max_concurrency: int`
* `active_request_count: int`
* `slots: simpy.Resource(env, capacity=max_concurrency)` -- concurrency slot resource

Derived state:

* `busy = active_request_count > 0`
* `is_stable = current_state in STABLE_STATES and target_state is None`
* `is_transient = not is_stable` (running, or mid-transition)

Required methods:

* `can_serve(service_id) -> bool`
* `has_capacity() -> bool` -- checks `slots.count < slots.capacity`
* `bind_service(service_id)`
* `acquire_slot()` -- wraps `slots.request()`; on acquisition triggers warm-to-running if `active_request_count` goes 0->1
* `release_slot()` -- wraps `slots.release()`; on release triggers running-to-warm if `active_request_count` goes 1->0
* `transition_to(state_name)`
* `get_snapshot()`

The LifecycleManager checks `instance.slots.count < instance.slots.capacity` to determine if an instance has available concurrency. The `acquire_slot()` / `release_slot()` methods wrap SimPy Resource request/release and trigger the warm-to-running and running-to-warm state transitions based on `active_request_count` crossing the 0/1 boundary.

#### OpenWhiskExtendedStateMachine

Responsibilities:

* hold all state definitions and transition definitions
* validate config consistency
* compute possible transition paths
* identify best reusable instance candidate for a request

Required methods:

* `get_transition(from_state, to_state)`
* `find_upgrade_path(current_state, target_state, service_id=None)`
* `resolve_target_ready_state(service_id)`
* `score_instance_reuse(instance, service_id)`

#### LifecycleManager

Responsibilities:

* locate a reusable instance on a node, including an already-running instance with available concurrency capacity
* create a new instance when needed
* perform transitions
* bind instance to service if needed
* start and finish execution
* handle idle and eviction handoff to autoscaler

Required methods:

* `find_reusable_instance(node, service_id)`
* `prepare_instance_for_service(node, service_id)`
* `start_execution(instance, invocation)`
* `finish_execution(instance, invocation)`
* `evict_instance(instance)`
* `downgrade_instance(instance, target_state)`

### 10.4 Instance selection semantics

When a request needs service on a node, instance search priority must be:

1. a same-service instance already bound to the service and with `slots.count < slots.capacity` (checked via `simpy.Resource`), preferring `warm` or `running` instances that require no transition
2. a service-bound extended state closest to ready
3. a reusable `prewarm` instance
4. create a new instance from `null` (may trigger LRU eviction if node lacks memory; see Section 11.5)

This priority should be implemented by lifecycle manager using the state machine scoring/path logic.

### 10.5 Resource semantics

For each instance, there are two separate resource concepts:

1. **steady-state resource** while staying in current state
2. **transition resource** consumed temporarily during state changes

The implementation must account for both.

Concurrency admission must also be enforced separately from resource accounting. A request may only start on an instance when `active_request_count < max_concurrency`, even if the node still has free CPU and memory capacity.

A transition should:

1. verify node has enough resource for transition
2. allocate transition resource
3. wait transition time
4. release transition resource
5. update steady-state resource occupancy from old state to new state

The coding agent must not collapse transition cost and steady-state cost into one value.

### 10.6 Execution semantics

Execution follows a well-defined state transition model:

* When `active_request_count` transitions from 0 to 1, the instance state changes from `warm` to `running` (transient).
* When `active_request_count` transitions from 1 to 0, the instance state changes from `running` back to `warm` (stable).
* Additional requests arriving while the instance is already `running` simply increment `active_request_count` (no state change).
* Requests completing while `active_request_count > 1` simply decrement the count (no state change).

Under this model, a single container instance may host multiple overlapping executions for the same bound service, but never more than `max_concurrency` (enforced by `simpy.Resource`). The implementation must update request counts atomically on slot acquire/release so monitoring and autoscaling observe the correct in-flight level.

**Per-request resource handling during execution:**

1. Before execution starts: allocate `service.per_request_resource` (CPU only) to `node.used_dynamic_resource`
2. During execution: the per-request CPU remains allocated
3. On execution finish (or timeout abort): release `service.per_request_resource` from `node.used_dynamic_resource`

**Timeout during execution:** Each request runs as a SimPy process that races execution completion against a timeout event (measured from `arrival_time`). If the timeout fires first, execution is aborted immediately, the concurrency slot is released, per-request CPU is released, and the invocation is marked `timed_out=True` with `drop_reason: timeout`.

Because `running` is a transient state, the autoscaler/controller cannot downgrade or evict a running instance. It must wait for all requests to complete and the instance to return to `warm`.

### 10.7 Extensibility contract

Future lifecycle extensions may include:

* multiple partial loading states
* runtime image caches
* service-specific load stages
* downgrade chains rather than immediate eviction

Therefore all states and transitions must be data-driven from config.

---

## 11. Part 5: Autoscaling / memory-bounded pools / idle timeout / LRU eviction

### 11.1 Scope

The autoscaling system uses a **memory-bounded pool model** (OpenWhisk style). There are no count-based pool targets. Instead, containers are kept alive as long as memory is available and they have not exceeded the idle timeout.

Key concepts:

* each node's memory budget is its `total_resource.memory`
* prewarm containers: configurable count per runtime, maintained independently per node
* warm containers: kept alive until memory pressure or idle timeout expiry; no count target
* when a new container is needed but no memory is available, evict the LRU idle container (by `last_used_time`)
* a single global idle timeout controls how long idle containers survive
* runtime API for external modules to update the idle timeout

External control (controller or RL agent) can change:

* the global idle timeout value

### 11.2 Conceptual model

Memory-bounded pool management:

* each node independently manages its containers within its memory budget
* prewarm pool: a configurable number of unbound containers per runtime kept in `prewarm` state on each node
* warm containers remain in `warm` state after their last request completes, consuming their steady-state memory
* when memory is needed for a new container and the node is full, the autoscaler evicts the container with the oldest `last_used_time` (LRU eviction)
* if an idle container's time since `last_used_time` exceeds the global idle timeout, it is evicted regardless of memory pressure

There is a single global idle timeout (not per-service, not per-state). It is configurable in the simulation config and can be modified at runtime by the RL agent via the autoscaling API.

### 11.3 Main classes

The `PoolEntry`, `PoolManager` classes and `set_pool_target` / `get_pool_target` APIs are **removed**. Pool management is implicit via memory bounds and LRU eviction.

#### OpenWhiskPoolAutoscaler

Responsibilities:

* maintain periodic reconcile loop
* ensure prewarm counts are maintained per runtime per node
* apply idle timeout: evict containers idle longer than the global timeout
* on memory pressure: evict LRU idle container (lowest `last_used_time`) to free memory
* can only act on containers in **stable** states (cannot evict/downgrade running or mid-transition containers)

Required methods:

* `periodic_reconcile()`
* `handle_idle_instance(instance)`
* `evict_lru(node)` -- find and evict the idle container with the oldest `last_used_time` on the given node
* `set_idle_timeout(value)` -- set the global idle timeout
* `get_idle_timeout() -> float` -- get the current global idle timeout

#### AutoscalingAPI

Public-facing control API used by controller or RL integration.

Required methods:

* `get_idle_timeout() -> float`
* `set_idle_timeout(value)`
* `trigger_reconcile()`

#### AutoscalingManager

Responsibilities:

* hold autoscaler and API facade
* start autoscaler processes

### 11.4 Reconcile behavior

At each autoscaling interval:

1. for each node, check prewarm counts per runtime; create new prewarm containers if below target
2. scan all idle containers (in stable states only): if `env.now - instance.last_used_time > idle_timeout`, evict the instance
3. the autoscaler must skip any container in a transient state (running or mid-transition)

### 11.5 Memory pressure eviction

When a new container needs to be created on a node but the node lacks sufficient memory:

1. find all idle containers on the node in stable states (warm, prewarm, extended states)
2. sort by `last_used_time` ascending (oldest first = LRU)
3. evict containers one by one until enough memory is freed
4. if no idle containers remain and still not enough memory, the request cannot be served (upstream handles the drop)

### 11.6 API mutability requirements

The API must support runtime changes without restarting simulation.

Controller or Gym action application must be able to invoke updates safely during a run.

### 11.7 Extensibility contract

Future autoscaling mechanisms may include:

* predictive scaling
* latency-SLO-based scaling
* RL-based scaling
* per-node pool targeting
* multi-tenant quota-aware scaling

Therefore current autoscaling logic must remain policy-modular.

---

## 12. Part 6: Monitoring module

### 12.1 Scope

A dedicated monitoring module acts like Prometheus inside the simulator.

Requirements:

* periodic metric collection
* configurable collection interval from simulation config
* query API for other modules
* support system snapshots and time series

### 12.2 Metrics to collect

At minimum, the monitor must support the following metric families.

#### Request metrics

* total requests seen
* completed requests
* dropped requests
* timed out requests
* cold start count
* warm hit count
* latency statistics per service
* queue delay per service
* service time per service
* p50 / p95 / p99 latency per service
* concurrency saturation per service

#### Cluster metrics

* nodes enabled/disabled
* CPU used / available / total
* memory used / available / total
* utilization ratios
* active requests per node
* queue length per node
* active requests per instance
* concurrency slot utilization per instance

#### Lifecycle metrics

* instance counts per state
* instance counts per service and per state
* transition counts by `(from_state, to_state)`
* evictions
* downgrades

#### Autoscaling metrics

* prewarm target counts per runtime
* prewarm actual counts per runtime per node
* global idle timeout value
* LRU eviction count
* reconcile count

### 12.3 Main classes

#### MetricStore

Responsibilities:

* hold latest values
* hold time series in a **ring buffer** with configurable `max_history_length` (default: 1000)
* serve queries
* when the ring buffer is full, the oldest entry is overwritten

Required fields:

* `max_history_length: int` (configurable in simulation config, default 1000)

Required methods:

* `put(metric_name, timestamp, value)`
* `get_latest(metric_name)`
* `query_range(metric_name, start, end)`
* `get_snapshot(metric_names=None)`

#### BaseCollector

Required method:

* `collect(env_time, sim_context) -> dict[str, object]`

#### Concrete collectors

Required collector modules:

* `RequestCollector`
* `ClusterCollector`
* `LifecycleCollector`
* `AutoscalingCollector`

Each collector should only read from simulator state and write metrics; it must not mutate system behavior.

#### MonitorManager

Responsibilities:

* own collectors and metric store
* start periodic collection process that runs independently with its own `collection_interval`
* provide API object

The monitor runs **independently** on its own collection interval. The Gym wrapper does **not** trigger `collect_once()`; instead it calls `monitor.get_latest()` to read whatever data is currently available (which may be slightly stale depending on when the last collection occurred).

Required methods:

* `start()`
* `collect_once()` (called internally by the monitor's own periodic process, not by external callers)
* `get_latest()` -- returns the most recent snapshot (called by Gym wrapper)

#### MonitorAPI

Required methods:

* `get_latest(metric_name)`
* `query_range(metric_name, start, end)`
* `get_snapshot(metric_names=None)`

### 12.4 Snapshot semantics

A snapshot is a flat or structured dict of most recent metric values.

The Gym wrapper calls `monitor.get_latest()` which returns whatever was last collected. The data may be slightly stale (up to one `collection_interval` old). This is by design: the monitor is an independent process, not demand-driven.

### 12.5 Extensibility contract

Future features may include:

* histogram buckets
* Prometheus-style export format
* streaming metric sink
* alert conditions

Therefore monitor storage and collectors should remain decoupled.

---

## 13. Controller module

### 13.1 Scope

The controller is an **independent module** that periodically observes the system via the monitor API and makes autoscaling decisions. It is completely decoupled from Gymnasium and Stable-Baselines3.

The controller supports **pluggable policies**. Each policy implements the same interface but uses different decision logic. Examples:

* rule-based threshold policy (sample implementation required)
* future: other heuristic policies

RL inference does **not** go through the controller. Instead, RL inference uses the Gymnasium wrapper directly with `model.predict()` (see Section 18.4). This keeps observation/action format guaranteed identical to training.

### 13.2 Main classes

#### BaseController

Orchestrates the periodic control loop. Delegates decisions to the active policy.

Required fields:

* `monitor_api`
* `autoscaling_api`
* `control_interval: float`
* `policy: BaseControlPolicy`

Required method:

* `run()` — SimPy process: every `control_interval`, call `policy.decide()` and apply returned actions

#### BaseControlPolicy

Interface for pluggable control policies.

Required method:

* `decide(monitor_snapshot) -> list[ControlAction]`

`ControlAction` is a simple structure describing one autoscaling change (e.g., set idle timeout, set prewarm count).

#### ThresholdPolicy

A sample rule-based policy. Required as baseline implementation.

Example behavior:

* if average CPU utilization > 80%: increase prewarm count
* if average CPU utilization < 20%: decrease prewarm count
* if p95 latency > threshold: decrease idle timeout (keep containers warm longer)

Policy parameters (thresholds, step sizes) are loaded from simulation config.

### 13.3 Architecture overview

```
controller/
  base_controller.py        # BaseController with periodic SimPy loop
  policies/
    base_policy.py           # BaseControlPolicy interface
    threshold_policy.py      # sample rule-based policy
```

### 13.4 Runtime semantics

If controller is enabled in standalone simulation:

* it must run as its own SimPy process
* every `control_interval` it reads monitor snapshot, calls active policy, applies returned actions via autoscaling API

Controller logic must not directly manipulate internal module state bypassing APIs.

### 13.5 Relationship with RL

The controller and RL inference are **separate code paths**:

| Mode | How it runs |
|---|---|
| Rule-based | Standalone sim + controller + ThresholdPolicy |
| RL training | Gym env + SB3 `PPO.learn()` (no controller) |
| RL inference | Gym env + SB3 `model.predict()` loop (no controller) |

ObservationBuilder, ActionMapper, and RewardCalculator live inside `gym_env/` and are only used by the Gym wrapper and RL training/inference. The controller does **not** use these components.

---

## 14. Export subsystem

### 14.1 Scope

Export is a **simulation-level** concern, not part of the Gymnasium wrapper. After a run, outputs are written according to the configured export mode.

Three export modes are supported, configured in the simulation config via an integer `export_mode`:

* **Mode 0**: `summary.txt` only -- total requests, average latency, drop rate, cold start rate, etc.
* **Mode 1**: Mode 0 + `system_metrics.csv` -- time series data from the monitor (one row per collection interval)
* **Mode 2**: Mode 1 + `request_trace.csv` -- all individual request details

### 14.2 Main classes

#### BatchCSVWriter

A utility for buffered CSV writing.

Behavior:

* buffer rows in memory until `batch_size` is reached (configurable, default 500)
* on reaching batch_size, flush buffered rows to disk in **append** mode
* CSV header is written on the **first flush** only; subsequent flushes append data rows
* a final flush is called at simulation end to write any remaining buffered rows

Required fields:

* `file_path: str`
* `batch_size: int` (default 500)
* `header: list[str]`
* `buffer: list[list]`

Required methods:

* `add_row(row: list)`
* `flush()`
* `close()` -- flush remaining rows

#### SummaryWriter

Responsibilities:

* at simulation end, compute and write `summary.txt` with aggregate statistics
* total requests, completed, dropped (by reason), timed out, cold starts, warm hits
* average / p50 / p95 / p99 latency, average queue delay
* cluster utilization averages

Required methods:

* `write_summary(path, sim_context)`

#### SystemMetricsExporter

Responsibilities:

* uses `BatchCSVWriter` to write `system_metrics.csv`
* records one row per monitor collection interval with metric snapshot values

Required methods:

* `record_snapshot(env_time, metric_snapshot)`
* `close()`

#### RequestTraceExporter

Responsibilities:

* uses `BatchCSVWriter` to write `request_trace.csv`
* records one row per finalized request

Required methods:

* `record_request(invocation)`
* `close()`

#### ExportManager

Responsibilities:

* manage the selected export mode (0, 1, or 2)
* coordinate writers based on mode
* create files in run directory
* call `close()` on all writers at simulation end

### 14.3 Export semantics

**Mode 0 outputs:**

* `summary.txt` -- written once at simulation end

**Mode 1 outputs (in addition to Mode 0):**

* `system_metrics.csv` -- one row per monitor collection interval, columns are the monitored metric fields

**Mode 2 outputs (in addition to Mode 1):**

* `request_trace.csv` -- one row per request, with at minimum:
  * request id
  * service id
  * arrival time
  * dispatch time
  * queue delay
  * execution start/end
  * completion time
  * latency
  * cold start
  * assigned node
  * assigned instance
  * dropped / timed out flags
  * drop_reason

---

## 15. Logging subsystem

### 15.1 Scope

Every module must have its own logger.

Logs must be written under a unique run directory inside `logs/`.

Logging mode is provided at CLI runtime.

### 15.2 Required logging modes

* `console`
* `file`
* `both`

### 15.3 Required behavior

For every run, create a timestamped run directory such as:

```text
logs/2026-03-17_14-30-00_train/
```

Expected contents may include:

* `runtime.log`
* copied config files
* CSV outputs
* optional training logs / checkpoints metadata

### 15.4 LoggerFactory

The logging subsystem should expose a factory such as:

* `create_logger(module_name, run_dir, mode, level)`

All modules must use this shared setup and avoid creating ad-hoc logger behavior.

---

## 16. Configuration subsystem

### 16.1 Required config files

Three different JSON config types are required.

1. **simulation config**
2. **gym config**
3. **RL config**

### 16.2 Simulation config responsibilities

Must define at least:

* simulation horizon and seed
* services, including per-service `max_concurrency` and `per_request_resource`
* cluster compute classes and node settings
* lifecycle states and transitions
* autoscaling settings: prewarm counts per runtime, global idle timeout
* monitoring: `collection_interval`, enabled metric groups, `max_history_length` for ring buffer
* export settings: `export_mode` (0, 1, or 2), `batch_size` for BatchCSVWriter
* per-node queue capacity (default unlimited)
* optional controller settings

### 16.3 Gym config responsibilities

Must define at least:

* decision interval
* episode time limit
* observation definition with `aggregate` and `per_service` field lists (see Section 17.5)
* reward weights (see Section 17.7)
* discrete action mapping (see Section 17.6)
* optional vectorization compatibility settings

### 16.4 RL config responsibilities

Must define at least:

* mode: train or inference
* algorithm: PPO
* PPO hyperparameters
* number of environments
* output/model paths
* inference episode settings

### 16.5 Validation requirements

The coding agent should implement config validation with clear error messages for:

* missing required keys
* malformed resource profiles
* invalid `max_concurrency` values
* invalid `per_request_resource` values
* invalid transition graph references
* invalid metric names in observation/export config
* invalid `export_mode` values (must be 0, 1, or 2)
* invalid idle timeout values

---

## 17. Gymnasium wrapper design

### 17.1 Scope

The simulator must be wrapped as a Gymnasium environment using **fixed-time decision intervals**.

This corresponds to:

* the agent acts every fixed simulated time interval
* between decisions, the underlying SimPy simulation runs freely

### 17.2 Main class

#### ServerlessGymEnv

Required methods:

* `__init__(sim_config_path, gym_config_path, seed=None)`
* `reset(seed=None, options=None)`
* `step(action)`
* `close()`

### 17.3 Reset behavior

`reset()` must:

1. create a fresh simulator instance from configs
2. reset internal episode counters
3. run any minimal initialization if needed
4. return initial observation and info

Each environment instance must be fully isolated.

### 17.4 Step behavior

A single `step(action)` must:

1. apply the action through approved control API surface
2. advance the SimPy simulation by exactly `decision_interval`
3. call `monitor.get_latest()` to obtain whatever snapshot data is available (the Gym wrapper does NOT trigger `collect_once()`; the monitor runs independently)
4. construct observation
5. compute reward (weighted sum of penalties over the decision interval window)
6. determine `terminated` and `truncated`
7. return `(obs, reward, terminated, truncated, info)`

Export is handled at the simulation level (see Section 14), not by the Gym wrapper.

The Gym wrapper must not bypass APIs to mutate core autoscaling state directly.

### 17.5 ObservationBuilder

A separate observation builder module must convert monitor snapshot to a fixed-size numeric vector.

The gym config defines observation fields in two categories:

```json
{
  "observation": {
    "aggregate": ["cluster_cpu_util", "cluster_memory_util"],
    "per_service": ["p95_latency", "warm_count", "running_count"]
  }
}
```

* `aggregate`: cluster-wide metrics, listed once
* `per_service`: metrics that are automatically expanded for ALL services defined in the simulation config (the gym config only lists field names, not service IDs)

Observation vector size = `len(aggregate) + len(services) * len(per_service)`

Changing the number of services in the simulation config changes the observation size, which requires retraining the RL model. This is acceptable since the gym/RL integration is a toy/research tool.

Requirements:

* observation field list comes from gym config in the format above
* per-service metrics auto-apply to all services; no per-service enumeration in gym config
* missing metrics must be handled explicitly, not silently ignored
* output must be compatible with numpy arrays and SB3

### 17.6 Action mapping

The Gym action space uses a **Discrete** action space (not continuous).

Available actions:

* no-op
* increase pool target (prewarm count)
* decrease pool target (prewarm count)
* increase idle timeout
* decrease idle timeout

The concrete action mapping must be defined in gym config. Each discrete action index maps to one of the above operations with a configured step size.

Since count-based pool targets are removed, "increase/decrease pool target" refers to adjusting the prewarm count per runtime. The idle timeout is the single global value.

### 17.7 Reward calculation

Reward is a **weighted sum of penalties**, computed as a single function. There is no separate builder class and no complex normalization.

Penalty terms (examples):

* p95 latency penalty
* cold start penalty
* resource cost penalty
* timeout/drop penalty

Weights for each penalty term are defined in the gym config:

```json
{
  "reward": {
    "weights": {
      "p95_latency": -0.5,
      "cold_start_rate": -0.3,
      "drop_rate": -1.0,
      "cpu_utilization": -0.2
    }
  }
}
```

The reward is calculated over a **single decision interval window** (the metrics accumulated since the last step). The reward computation should live in a dedicated utility function, not inline in the Gym env.

### 17.8 Vectorization compatibility requirements

This is critical.

The implementation must support use with `DummyVecEnv` and `SubprocVecEnv`.

To ensure this:

* no global mutable simulator state
* each env has separate configs/state/run directory
* each env in a VecEnv uses `seed = base_seed + env_index` for deterministic reproducibility
* all objects used by subprocess env creation must be import-safe and picklable where needed
* CLI-only constructs must not leak into env constructor

---

## 18. RL agent integration with Stable-Baselines3

### 18.1 Scope

The RL integration is a **toy/research tool**. Training is a separate module. Inference uses the Gym env directly with `model.predict()`.

The controller module is **not involved** in RL training or inference (see Section 13.5).

### 18.2 Required files/modules

```
rl_agent/
  train.py                   # training entry point
gym_env/
  serverless_gym_env.py      # Gym env (used by both training and inference)
  observation_builder.py     # monitor snapshot → numpy vector
  action_mapper.py           # action index → autoscaling API calls
  reward_calculator.py       # monitor snapshot → reward float
```

ObservationBuilder, ActionMapper, and RewardCalculator live inside `gym_env/` because they are only used by the Gym wrapper. This guarantees observation/action format is identical between training and inference.

### 18.3 Training behavior

Training flow:

1. load simulation config, gym config, RL config
2. create vectorized Gym envs according to `n_envs` (each env uses `seed = base_seed + env_index`)
3. initialize PPO model
4. train for configured timesteps
5. save model/checkpoints/logs

### 18.4 Inference behavior

Inference uses the **same Gym env** as training, guaranteeing observation/action compatibility:

1. load configs
2. load saved PPO model
3. create Gym env (single env, not vectorized)
4. loop: `obs = env.reset()` → `action = model.predict(obs)` → `obs, reward, done, info = env.step(action)`
5. save metrics/logs/exports

Inference is run via the `infer` CLI command, which loads the Gym env and model directly. No controller is involved.

### 18.5 PPO configuration expectations

The RL config should support standard PPO parameters such as:

* learning rate
* total timesteps
* n_steps
* batch_size
* gamma
* gae_lambda
* ent_coef
* vf_coef
* clip_range
* device
* n_envs

### 18.6 Reproducibility and seed propagation

Each environment in a `VecEnv` uses `seed = base_seed + env_index`. This is simple, deterministic, and reproducible.

Training and inference runners must propagate seeds carefully to:

* SB3 (set seed on model creation)
* Gym env construction (each env gets `base_seed + env_index`)
* simulation RNGs inside each env (seeded from the env's seed)

---

## 19. CLI and runtime entrypoints

### 19.1 Required command modes

The CLI must support at least three commands:

* `simulate`
* `train`
* `infer`

### 19.2 Expected command behavior

#### simulate

* requires simulation config
* may optionally enable controller
* runs standalone SimPy simulation
* writes logs and exports

#### train

* requires simulation config, gym config, RL config
* trains PPO
* writes logs, models, and exports

#### infer

* requires simulation config, gym config, RL config
* loads trained model, creates Gym env, runs `model.predict()` loop
* does NOT use the controller module
* writes logs and exports

### 19.3 CLI arguments

Required arguments should include:

* `--sim-config`
* `--gym-config`
* `--rl-config`
* `--log-mode`
* `--log-level`
* `--run-name` (optional)
* optional override for CSV export mode

---

## 20. Required runtime flow details

### 20.1 Standalone simulation startup order

The implementation should follow this order:

1. load config and validate
2. create run directory and loggers
3. build cluster (nodes with `simpy.Store` queues) and lifecycle definitions
4. initialize prewarm containers per runtime per node
5. initialize monitor (with ring buffer MetricStore) and APIs
6. initialize dispatcher (with ShardingContainerPoolBalancer)
7. initialize autoscaling (memory-bounded, LRU, global idle timeout)
8. initialize workload generators
9. optionally initialize controller
10. start background processes (monitor collection, autoscale reconcile, per-node pull loops)
11. run until simulation horizon
12. finalize and export (mode 0/1/2)

### 20.2 Gym step runtime order

For every Gym step:

1. receive action (discrete)
2. translate action into autoscaling API updates (idle timeout or prewarm count adjustment)
3. run simulator forward by fixed interval
4. call `monitor.get_latest()` to read available snapshot (do not trigger collection)
5. build observation (aggregate + per_service fields)
6. compute reward (weighted sum of penalties over decision interval)
7. determine done flags
8. return outputs

### 20.3 Request completion flow

A request typically follows:

1. creation by workload manager
2. dispatcher calls ShardingContainerPoolBalancer to select node via consistent hashing
3. dispatcher puts invocation into the selected node's `simpy.Store` (async, returns immediately)
4. node's pull loop gets invocation from store: `yield store.get()`
5. per-request SimPy process is spawned, racing execution vs timeout (from `arrival_time`)
6. lifecycle manager selects or prepares instance (acquire concurrency slot via `simpy.Resource`)
7. per-request CPU is allocated to `node.used_dynamic_resource`
8. state transitions happen if needed (warm-to-running on first request)
9. execution starts
10. execution ends (or timeout fires, aborting execution)
11. per-request CPU is released; concurrency slot is released (running-to-warm if last request)
12. request record finalized
13. monitor/exporters observe finalized state

Failure paths:
* if no node has enough memory: dropped at step 2 (`drop_reason: no_capacity`)
* if node's queue is full: dropped at step 3 (`drop_reason: queue_full`)
* if timeout fires during queue wait: removed from queue (`drop_reason: timeout`)
* if timeout fires during execution: aborted, resources released (`drop_reason: timeout`)

---

## 21. Data ownership and boundaries

To avoid architecture drift, ownership must be respected.

### 21.1 Workload owns

* service definitions
* request generation timing
* invocation creation

### 21.2 Scheduling owns

* node selection only

### 21.3 Cluster owns

* node registry
* resource accounting on nodes
* node enable/disable state

### 21.4 Lifecycle owns

* instance creation
* instance transition paths
* instance readiness for execution

### 21.5 Autoscaling owns

* prewarm counts per runtime
* global idle timeout
* LRU eviction under memory pressure
* reconcile logic
* downgrade/eviction decisions through lifecycle integration (only for containers in stable states)

### 21.6 Monitoring owns

* metric collection and query only
* no state mutation

### 21.7 Controller owns

* periodic observation via monitor API
* pluggable policy selection
* autoscaling decisions via APIs only
* NOT responsible for RL inference (that goes through Gym env)

### 21.8 Gym wrapper owns

* observation/action/reward translation (ObservationBuilder, ActionMapper, RewardCalculator)
* episode management
* used for both RL training AND RL inference
* no direct simulator policy logic

---

## 22. Error handling expectations

The coding agent should implement explicit error handling for:

* invalid config values
* unsupported policy names
* missing state transitions
* resource over-allocation attempts
* invalid action values from Gym wrapper
* failure to create run directories or write exports

Errors should be logged with enough context to diagnose the simulation state and config path involved.

---

## 23. Documentation requirements

The repository must include:

### README.md

Must cover:

* project purpose
* installation/dependencies
* quick start for standalone simulation
* quick start for Gym wrapper use
* quick start for PPO train/inference
* run outputs and logs

### docs/config_reference.md

Must explain all JSON fields across:

* simulation config
* gym config
* RL config

### docs/simpy_standalone.md

Must explain:

* how to run standalone mode
* what gets logged/exported
* how to enable controller

### docs/gym_wrapper.md

Must explain:

* reset/step semantics
* action mapping
* observation mapping
* reward definition
* vectorization notes

### docs/rl_training.md

Must explain:

* train/inference commands
* model save/load paths
* callback/checkpoint behavior
* seed and reproducibility notes

---

## 24. Implementation priorities

The coding agent should implement in the following order.

### Phase 1

* config loader and validation
* logging and run directory management
* cluster manager with per-node `simpy.Store` queue and fixed-rate node execution with configurable per-instance concurrency via `simpy.Resource`
* workload manager with Poisson arrivals
* dispatcher with ShardingContainerPoolBalancer (consistent hashing)
* minimal lifecycle: prewarm, warm, running (transient), evicted; stable/transient state classification
* per-request CPU resource accounting
* timeout handling (SimPy event race)
* basic monitor with ring buffer MetricStore
* standalone simulation CLI
* export Mode 0 (summary.txt)

### Phase 2

* extended lifecycle states and transition resources
* memory-bounded autoscaling with LRU eviction and global idle timeout
* controller with pluggable policies (BaseController + ThresholdPolicy sample)
* export Mode 1 (system_metrics.csv) and Mode 2 (request_trace.csv) with BatchCSVWriter
* richer monitoring

### Phase 3

* Gymnasium wrapper with discrete action space
* simplified observation builder (aggregate + per_service)
* simplified reward (weighted penalty sum)
* vector env compatibility testing with seed propagation (`base_seed + env_index`)
* PPO training module (`rl_agent/train.py`)
* RL inference via Gym env + model.predict()
* sample configs and docs completion

---

## 25. Final architectural constraints

These constraints are mandatory.

1. **Simulation core must run independently of Gym and RL.**
2. **All modules must be separately testable.**
3. **All mutable runtime behavior must be instance-local, not global.**
4. **Lifecycle definitions must be selected from config, not hard-coded. Stable vs transient state classification must be respected by all modules.**
5. **Controller and Gym wrapper may only control autoscaling through exposed APIs (idle timeout, prewarm count).**
6. **Monitoring must remain read-only and run independently on its own interval.**
7. **Export mode (0/1/2) must be selectable at config level. Export is a simulation-level concern, not a Gym concern.**
8. **Every run must write into a unique directory under `logs/`.**
9. **The design must remain open for new state types without major refactor.**
10. **Autoscaler/controller can only act on containers in stable states; transient containers must not be evicted or downgraded.**
11. **Each VecEnv environment uses seed = base_seed + env_index for deterministic reproducibility.**

---

## 26. Summary for coding agent

Implement a modular serverless simulator using SimPy with the following hard requirements:

* multi-service Poisson workload with fixed job size and per-request CPU resource cost
* ShardingContainerPoolBalancer using consistent hashing on service_id with memory-aware fallback
* pull model: per-node `simpy.Store` queues, nodes pull requests asynchronously
* multi-class nodes with fixed-rate serving model, CPU/memory accounting, and configurable per-service container concurrency via `simpy.Resource`
* OpenWhisk-like lifecycle with stable states (null, prewarm, extended, warm, evicted) and transient states (running, mid-transition); autoscaler/controller can only act on stable states
* memory-bounded autoscaling with LRU eviction, global idle timeout, configurable prewarm counts per runtime
* timeout handling: SimPy event race per request, counted from arrival_time
* three drop reasons: queue_full, timeout, no_capacity
* periodic monitor running independently with ring buffer MetricStore
* controller module with pluggable policies (ThresholdPolicy sample), independent from Gym/RL
* Gymnasium wrapper with discrete action space, simplified observation (aggregate + per_service), weighted penalty reward; used for both RL training and inference
* PPO training as separate module (`rl_agent/train.py`); RL inference via Gym env + model.predict()
* seed propagation: base_seed + env_index
* full logging, run directories, 3-mode export (summary / system_metrics.csv / request_trace.csv) with BatchCSVWriter, docs, and sample JSON configs

The implementation must follow the module boundaries and responsibilities defined in this document as closely as possible.
