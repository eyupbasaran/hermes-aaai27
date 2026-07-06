"""Asynchronous event-driven replay simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hermes.core.replay_oracle import ReplayOracle
from hermes.core.runtime_model import RuntimeModel
from hermes.core.types import (
    Backend,
    CompletedObservation,
    PendingJob,
    ProposedJob,
    SchedulerState,
    Source,
)
from hermes.schedulers.base import BaseScheduler


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Minimal simulator configuration surface.

    YAML-specific configuration loading will be added in the experiment runner.
    """

    horizon_seconds: float
    output_dir: Path | None = None
    max_target_queries: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon_seconds < 0:
            raise ValueError("ExperimentConfig horizon_seconds must be non-negative.")
        if self.max_target_queries is not None and self.max_target_queries < 0:
            raise ValueError("ExperimentConfig max_target_queries must be non-negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Return object for a simulator run."""

    scheduler_name: str
    final_state: SchedulerState
    events: list[Mapping[str, Any]]
    completions: list[CompletedObservation]
    metrics: dict[str, Any] = field(default_factory=dict)


class EventSimulator:
    """Asynchronous replay simulator API.

    The behavior is intentionally not implemented in this scaffold step. The
    next step should add tests for event ordering, capacity, duplicates,
    leakage, budgets, and baseline fairness before filling these methods in.
    """

    def __init__(
        self,
        *,
        replay_oracle: ReplayOracle,
        runtime_model: RuntimeModel,
        sources: Mapping[str, Source],
        backends: Mapping[str, Backend],
    ) -> None:
        self.replay_oracle = replay_oracle
        self.runtime_model = runtime_model
        self.sources = dict(sources)
        self.backends = dict(backends)

    def run(
        self,
        scheduler: BaseScheduler,
        initial_observations: list[CompletedObservation],
        config: ExperimentConfig,
    ) -> SimulationResult:
        del scheduler, initial_observations, config
        raise NotImplementedError("EventSimulator.run will be implemented after simulator tests.")

    def collect_completed_jobs(self, state: SchedulerState) -> list[CompletedObservation]:
        del state
        raise NotImplementedError("collect_completed_jobs will be implemented after event tests.")

    def get_available_backend_slots(self, state: SchedulerState) -> dict[str, int]:
        return {
            backend_id: backend_state.idle_slots
            for backend_id, backend_state in state.backend_states.items()
            if backend_state.idle_slots > 0
        }

    def validate_proposed_jobs(
        self,
        jobs: list[ProposedJob],
        state: SchedulerState,
    ) -> None:
        del jobs, state
        raise NotImplementedError("validate_proposed_jobs will be implemented after API tests.")

    def launch(self, jobs: list[ProposedJob], state: SchedulerState) -> list[PendingJob]:
        del jobs, state
        raise NotImplementedError("launch will be implemented after leakage and budget tests.")

    def advance_to_next_event(self, state: SchedulerState) -> float:
        del state
        raise NotImplementedError("advance_to_next_event will be implemented after event tests.")


__all__ = ["EventSimulator", "ExperimentConfig", "SimulationResult"]
