import numpy as np

from hermes.core import (
    Backend,
    BackendState,
    Candidate,
    CompletedObservation,
    EventSimulator,
    ExperimentConfig,
    InMemoryReplayOracle,
    OracleRecord,
    PendingJob,
    ProposedJob,
    ResourceBudget,
    SchedulerState,
    Source,
    SourceMeanRuntimeModel,
)
from hermes.schedulers.base import BaseScheduler


def test_core_types_and_runtime_model() -> None:
    candidate = Candidate(candidate_id=1, smiles="C", features=np.array([1.0, 2.0]))
    source = Source(source_id="def2tzvp", name="def2-TZVP", is_target=True, nominal_cost_rank=5)
    backend = Backend(
        backend_id="gpu_fast",
        name="Fast GPU",
        capacity_slots=2,
        dollar_per_second=0.5,
        startup_latency_seconds=10.0,
        runtime_multiplier_by_source={"def2tzvp": 0.25},
    )
    runtime_model = SourceMeanRuntimeModel(
        source_mean_runtime_seconds={"def2tzvp": 100.0},
        backends={"gpu_fast": backend},
    )

    assert candidate.features.tolist() == [1.0, 2.0]
    assert source.is_target
    assert runtime_model.predict(1, "def2tzvp", "gpu_fast") == 35.0
    assert runtime_model.predict_cost(1, "def2tzvp", "gpu_fast") == 17.5


def test_scheduler_visible_state_contains_no_oracle() -> None:
    backend = Backend(
        backend_id="cpu_queue",
        name="CPU queue",
        capacity_slots=1,
        dollar_per_second=0.1,
        startup_latency_seconds=0.0,
        runtime_multiplier_by_source={"sto3g": 1.0},
    )
    job = ProposedJob(candidate_id=7, source_id="sto3g", backend_id="cpu_queue")
    pending = PendingJob(
        job=job,
        start_time=0.0,
        predicted_finish_time=5.0,
        predicted_runtime_seconds=5.0,
        predicted_cost=0.5,
    )
    completed = CompletedObservation(
        job=ProposedJob(candidate_id=3, source_id="sto3g", backend_id="cpu_queue"),
        start_time=0.0,
        finish_time=2.0,
        y=1.2,
        runtime_seconds=2.0,
        cost=0.2,
    )
    state = SchedulerState(
        current_time=2.0,
        completed=[completed],
        pending=[pending],
        backend_states={"cpu_queue": BackendState(backend=backend, busy_jobs=[pending])},
        remaining_budget=ResourceBudget(
            remaining_wallclock_seconds=100.0,
            remaining_dollars=10.0,
            remaining_gpu_seconds=None,
            remaining_target_queries=5,
        ),
        already_target_validated={3},
        already_queried_pairs={(3, "sto3g"), (7, "sto3g")},
    )

    assert state.pending_pairs == {(7, "sto3g")}
    assert state.completed_pairs == {(3, "sto3g")}
    assert state.occupied_slots_by_backend == {"cpu_queue": 1}


def test_oracle_and_simulator_boundaries() -> None:
    source = Source(source_id="sto3g", name="STO-3G", is_target=False, nominal_cost_rank=1)
    backend = Backend(
        backend_id="cpu_queue",
        name="CPU queue",
        capacity_slots=2,
        dollar_per_second=0.1,
        startup_latency_seconds=0.0,
        runtime_multiplier_by_source={"sto3g": 1.0},
    )
    oracle = InMemoryReplayOracle({(1, "sto3g"): OracleRecord(y=0.7, true_runtime_seconds=4.0)})
    runtime_model = SourceMeanRuntimeModel(
        source_mean_runtime_seconds={"sto3g": 5.0},
        backends={"cpu_queue": backend},
    )
    simulator = EventSimulator(
        replay_oracle=oracle,
        runtime_model=runtime_model,
        sources={"sto3g": source},
        backends={"cpu_queue": backend},
    )
    state = SchedulerState(
        current_time=0.0,
        completed=[],
        pending=[],
        backend_states={"cpu_queue": BackendState(backend=backend)},
        remaining_budget=ResourceBudget(
            remaining_wallclock_seconds=10.0,
            remaining_dollars=None,
            remaining_gpu_seconds=None,
            remaining_target_queries=None,
        ),
    )

    assert oracle.query_hidden(1, "sto3g").y == 0.7
    assert simulator.get_available_backend_slots(state) == {"cpu_queue": 2}
    assert ExperimentConfig(horizon_seconds=10.0).horizon_seconds == 10.0


def test_scheduler_protocol_is_structural() -> None:
    class DummyScheduler:
        name = "dummy"

        def reset(self, seed, initial_observations):
            del seed, initial_observations

        def update(self, completed_jobs):
            del completed_jobs

        def propose(self, state, available_slots):
            del state, available_slots
            return []

    assert isinstance(DummyScheduler(), BaseScheduler)
