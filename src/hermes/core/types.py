"""Typed domain objects for HERMES.

These objects define the public boundary between schedulers and the replay
simulator. Keep hidden oracle records and true future outcomes out of
``SchedulerState`` unless they belong to already completed observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Metadata = dict[str, Any]


def _copy_metadata(metadata: Metadata | None) -> Metadata:
    return dict(metadata or {})


@dataclass(frozen=True, slots=True)
class Candidate:
    """A finite-pool molecular candidate."""

    candidate_id: int
    smiles: str | None
    features: np.ndarray
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        if features.ndim != 1:
            raise ValueError("Candidate features must be a one-dimensional array.")
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Source:
    """A scientific information source or fidelity."""

    source_id: str
    name: str
    is_target: bool
    nominal_cost_rank: int

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("Source source_id must be non-empty.")
        if not self.name:
            raise ValueError("Source name must be non-empty.")
        if self.nominal_cost_rank < 0:
            raise ValueError("Source nominal_cost_rank must be non-negative.")


@dataclass(slots=True)
class Backend:
    """An execution backend with runtime, cost, and capacity behavior."""

    backend_id: str
    name: str
    capacity_slots: int
    dollar_per_second: float
    startup_latency_seconds: float
    runtime_multiplier_by_source: dict[str, float]
    max_batch_size: int = 1

    def __post_init__(self) -> None:
        if not self.backend_id:
            raise ValueError("Backend backend_id must be non-empty.")
        if not self.name:
            raise ValueError("Backend name must be non-empty.")
        if self.capacity_slots < 0:
            raise ValueError("Backend capacity_slots must be non-negative.")
        if self.dollar_per_second < 0:
            raise ValueError("Backend dollar_per_second must be non-negative.")
        if self.startup_latency_seconds < 0:
            raise ValueError("Backend startup_latency_seconds must be non-negative.")
        if self.max_batch_size < 1:
            raise ValueError("Backend max_batch_size must be at least 1.")
        if any(multiplier < 0 for multiplier in self.runtime_multiplier_by_source.values()):
            raise ValueError("Backend runtime multipliers must be non-negative.")


@dataclass(frozen=True, slots=True)
class ProposedJob:
    """A scheduler-selected candidate/source/backend launch request."""

    candidate_id: int
    source_id: str
    backend_id: str


@dataclass(slots=True)
class PendingJob:
    """A launched job whose observation is not yet visible to the scheduler."""

    job: ProposedJob
    start_time: float
    predicted_finish_time: float
    true_finish_time: float
    predicted_runtime_seconds: float
    true_runtime_seconds: float
    predicted_cost: float
    true_cost: float

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("PendingJob start_time must be non-negative.")
        if self.predicted_runtime_seconds < 0:
            raise ValueError("PendingJob predicted_runtime_seconds must be non-negative.")
        if self.true_runtime_seconds < 0:
            raise ValueError("PendingJob true_runtime_seconds must be non-negative.")
        if self.predicted_cost < 0:
            raise ValueError("PendingJob predicted_cost must be non-negative.")
        if self.true_cost < 0:
            raise ValueError("PendingJob true_cost must be non-negative.")
        if self.predicted_finish_time < self.start_time:
            raise ValueError("PendingJob predicted_finish_time cannot precede start_time.")
        if self.true_finish_time < self.start_time:
            raise ValueError("PendingJob true_finish_time cannot precede start_time.")


@dataclass(slots=True)
class CompletedObservation:
    """An observation revealed to schedulers after a job completes."""

    job: ProposedJob
    start_time: float
    finish_time: float
    y: float
    runtime_seconds: float
    cost: float

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("CompletedObservation start_time must be non-negative.")
        if self.finish_time < self.start_time:
            raise ValueError("CompletedObservation finish_time cannot precede start_time.")
        if self.runtime_seconds < 0:
            raise ValueError("CompletedObservation runtime_seconds must be non-negative.")
        if self.cost < 0:
            raise ValueError("CompletedObservation cost must be non-negative.")


@dataclass(slots=True)
class ResourceBudget:
    """Remaining resources visible to schedulers."""

    remaining_wallclock_seconds: float
    remaining_dollars: float | None
    remaining_gpu_seconds: float | None
    remaining_target_queries: int | None

    def __post_init__(self) -> None:
        if self.remaining_wallclock_seconds < 0:
            raise ValueError("remaining_wallclock_seconds must be non-negative.")
        if self.remaining_dollars is not None and self.remaining_dollars < 0:
            raise ValueError("remaining_dollars must be non-negative when provided.")
        if self.remaining_gpu_seconds is not None and self.remaining_gpu_seconds < 0:
            raise ValueError("remaining_gpu_seconds must be non-negative when provided.")
        if self.remaining_target_queries is not None and self.remaining_target_queries < 0:
            raise ValueError("remaining_target_queries must be non-negative when provided.")

    def can_afford(
        self,
        *,
        predicted_runtime_seconds: float,
        predicted_cost: float,
        is_target_query: bool,
        predicted_gpu_seconds: float = 0.0,
    ) -> bool:
        """Check budget feasibility using only predicted pre-launch quantities."""

        if predicted_runtime_seconds > self.remaining_wallclock_seconds:
            return False
        if self.remaining_dollars is not None and predicted_cost > self.remaining_dollars:
            return False
        if self.remaining_gpu_seconds is not None and predicted_gpu_seconds > self.remaining_gpu_seconds:
            return False
        if is_target_query and self.remaining_target_queries == 0:
            return False
        return True


@dataclass(slots=True)
class BackendState:
    """Occupancy state for one backend."""

    backend: Backend
    busy_jobs: list[PendingJob] = field(default_factory=list)

    @property
    def idle_slots(self) -> int:
        return self.backend.capacity_slots - len(self.busy_jobs)

    def has_idle_slot(self) -> bool:
        return self.idle_slots > 0


@dataclass(slots=True)
class SchedulerState:
    """The scheduler-visible state at an event-driven decision epoch."""

    current_time: float
    completed: list[CompletedObservation]
    pending: list[PendingJob]
    backend_states: dict[str, BackendState]
    remaining_budget: ResourceBudget
    already_target_validated: set[int] = field(default_factory=set)
    already_queried_pairs: set[tuple[int, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.current_time < 0:
            raise ValueError("SchedulerState current_time must be non-negative.")

    @property
    def pending_pairs(self) -> set[tuple[int, str]]:
        return {(job.job.candidate_id, job.job.source_id) for job in self.pending}

    @property
    def completed_pairs(self) -> set[tuple[int, str]]:
        return {(obs.job.candidate_id, obs.job.source_id) for obs in self.completed}

    @property
    def occupied_slots_by_backend(self) -> dict[str, int]:
        return {backend_id: len(state.busy_jobs) for backend_id, state in self.backend_states.items()}


__all__ = [
    "Backend",
    "BackendState",
    "Candidate",
    "CompletedObservation",
    "Metadata",
    "PendingJob",
    "ProposedJob",
    "ResourceBudget",
    "SchedulerState",
    "Source",
]
