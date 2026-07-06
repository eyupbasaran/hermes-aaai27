from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from hermes.core.event_simulator import EventSimulator, ExperimentConfig
from hermes.core.replay_oracle import InMemoryReplayOracle, OracleRecord
from hermes.core.runtime_model import SourceMeanRuntimeModel
from hermes.core.types import (
    Backend,
    BackendState,
    CompletedObservation,
    ProposedJob,
    ResourceBudget,
    SchedulerState,
    Source,
)


TARGET = "target"
LOW = "low"
CPU = "cpu"
GPU = "gpu"


@pytest.fixture
def sources() -> dict[str, Source]:
    return {
        LOW: Source(source_id=LOW, name="Low fidelity", is_target=False, nominal_cost_rank=1),
        TARGET: Source(source_id=TARGET, name="Target fidelity", is_target=True, nominal_cost_rank=2),
    }


@pytest.fixture
def backends() -> dict[str, Backend]:
    return {
        CPU: Backend(
            backend_id=CPU,
            name="CPU queue",
            capacity_slots=3,
            dollar_per_second=0.01,
            startup_latency_seconds=0.0,
            runtime_multiplier_by_source={LOW: 1.0, TARGET: 1.0},
        ),
        GPU: Backend(
            backend_id=GPU,
            name="GPU queue",
            capacity_slots=1,
            dollar_per_second=0.05,
            startup_latency_seconds=0.0,
            runtime_multiplier_by_source={LOW: 0.5, TARGET: 0.5},
        ),
    }


@pytest.fixture
def cap2_backends(backends: dict[str, Backend]) -> dict[str, Backend]:
    backends = dict(backends)
    backends[CPU] = Backend(
        backend_id=CPU,
        name="CPU queue",
        capacity_slots=2,
        dollar_per_second=0.01,
        startup_latency_seconds=0.0,
        runtime_multiplier_by_source={LOW: 1.0, TARGET: 1.0},
    )
    return backends


@pytest.fixture
def oracle_records() -> dict[tuple[int, str], OracleRecord]:
    # True runtime intentionally disagrees with the runtime model to test
    # asynchronous event ordering and runtime leakage.
    return {
        (0, TARGET): OracleRecord(y=0.0, true_runtime_seconds=30.0),
        (1, TARGET): OracleRecord(y=1.0, true_runtime_seconds=10.0),
        (2, TARGET): OracleRecord(y=2.0, true_runtime_seconds=20.0),
        (3, TARGET): OracleRecord(y=3.0, true_runtime_seconds=5.0),
        (4, TARGET): OracleRecord(y=4.0, true_runtime_seconds=60.0),
        (0, LOW): OracleRecord(y=0.1, true_runtime_seconds=3.0),
        (1, LOW): OracleRecord(y=1.1, true_runtime_seconds=1.0),
        (2, LOW): OracleRecord(y=2.1, true_runtime_seconds=2.0),
        (3, LOW): OracleRecord(y=3.1, true_runtime_seconds=1.0),
        (4, LOW): OracleRecord(y=4.1, true_runtime_seconds=6.0),
    }


@pytest.fixture
def oracle(oracle_records: dict[tuple[int, str], OracleRecord]) -> InMemoryReplayOracle:
    return InMemoryReplayOracle(oracle_records)


@pytest.fixture
def runtime_model(backends: dict[str, Backend]) -> SourceMeanRuntimeModel:
    # Scheduler-visible predicted runtimes. These are not the true replay runtimes.
    return SourceMeanRuntimeModel(
        source_mean_runtime_seconds={LOW: 10.0, TARGET: 100.0},
        backends=backends,
    )


@pytest.fixture
def simulator(
    oracle: InMemoryReplayOracle,
    runtime_model: SourceMeanRuntimeModel,
    sources: dict[str, Source],
    backends: dict[str, Backend],
) -> EventSimulator:
    return EventSimulator(
        replay_oracle=oracle,
        runtime_model=runtime_model,
        sources=sources,
        backends=backends,
    )


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        horizon_seconds=120.0,
        max_target_queries=10,
        metadata={"seed": 7},
    )


@pytest.fixture
def initial_observations() -> list[CompletedObservation]:
    return [
        CompletedObservation(
            job=ProposedJob(candidate_id=99, source_id=TARGET, backend_id=CPU),
            start_time=0.0,
            finish_time=1.0,
            y=-1.0,
            runtime_seconds=1.0,
            cost=0.01,
        )
    ]


def make_state(
    *,
    backends: dict[str, Backend],
    current_time: float = 0.0,
    completed: list[CompletedObservation] | None = None,
    pending: list[Any] | None = None,
    remaining_budget: ResourceBudget | None = None,
) -> SchedulerState:
    pending = list(pending or [])
    backend_states = {
        backend_id: BackendState(backend=backend, busy_jobs=[])
        for backend_id, backend in backends.items()
    }
    for pending_job in pending:
        backend_states[pending_job.job.backend_id].busy_jobs.append(pending_job)

    return SchedulerState(
        current_time=current_time,
        completed=list(completed or []),
        pending=pending,
        backend_states=backend_states,
        remaining_budget=remaining_budget
        or ResourceBudget(
            remaining_wallclock_seconds=120.0,
            remaining_dollars=10.0,
            remaining_gpu_seconds=None,
            remaining_target_queries=10,
        ),
        already_target_validated={
            obs.job.candidate_id
            for obs in completed or []
            if obs.job.source_id == TARGET
        },
        already_queried_pairs={
            (obs.job.candidate_id, obs.job.source_id)
            for obs in completed or []
        }
        | {
            (job.job.candidate_id, job.job.source_id)
            for job in pending
        },
    )


class ScriptedScheduler:
    """A deterministic scheduler for simulator contract tests."""

    name = "scripted"

    def __init__(self, batches: list[list[ProposedJob]]) -> None:
        self._batches = [list(batch) for batch in batches]
        self.reset_calls: list[tuple[int, list[CompletedObservation]]] = []
        self.update_calls: list[list[CompletedObservation]] = []
        self.propose_states: list[SchedulerState] = []
        self.available_slots_seen: list[dict[str, int]] = []

    def reset(self, seed: int, initial_observations: list[CompletedObservation]) -> None:
        self.reset_calls.append((seed, initial_observations))

    def update(self, completed_jobs: list[CompletedObservation]) -> None:
        self.update_calls.append(completed_jobs)

    def propose(
        self,
        state: SchedulerState,
        available_slots: dict[str, int],
    ) -> list[ProposedJob]:
        self.propose_states.append(state)
        self.available_slots_seen.append(dict(available_slots))
        if not self._batches:
            return []
        return self._batches.pop(0)


def assert_no_scheduler_secrets(obj: Any) -> None:
    """Recursively assert that a scheduler-visible object does not expose secrets.

    Allowed:
    - completed observations may expose observed y/runtime/cost after completion.

    Forbidden anywhere in scheduler-visible state:
    - ReplayOracle-like object;
    - OracleRecord-like object;
    - query_hidden method;
    - true_* fields;
    - hidden labels;
    - final top-K membership/ranks;
    - future completion order.
    """

    seen: set[int] = set()
    forbidden_exact = {
        "replay_oracle",
        "oracle",
        "oracle_records",
        "hidden_records",
        "hidden_labels",
        "target_labels",
        "target_y",
        "final_topk",
        "final_top_k",
        "topk_membership",
        "top_k_membership",
        "target_rank",
        "final_rank",
        "future_completion_order",
    }

    def visit(value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, int, float, bool)):
            return

        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(value, "query_hidden"):
            raise AssertionError(f"Scheduler-visible secret oracle at {path}: {type(value)!r}")

        type_name = type(value).__name__.lower()
        if "oracle" in type_name and "runtime" not in type_name:
            raise AssertionError(f"Scheduler-visible oracle-like object at {path}: {type(value)!r}")

        if isinstance(value, dict):
            for key, item in value.items():
                key_str = str(key).lower()
                if key_str in forbidden_exact or key_str.startswith("true_"):
                    raise AssertionError(f"Forbidden scheduler-visible key at {path}: {key!r}")
                visit(item, f"{path}[{key!r}]")
            return

        if isinstance(value, (list, tuple, set, frozenset)):
            for idx, item in enumerate(value):
                visit(item, f"{path}[{idx}]")
            return

        if is_dataclass(value):
            for field in fields(value):
                name = field.name.lower()
                if name in forbidden_exact or name.startswith("true_"):
                    raise AssertionError(
                        f"Forbidden scheduler-visible dataclass field at {path}.{field.name}"
                    )
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return

    visit(obj, "root")
