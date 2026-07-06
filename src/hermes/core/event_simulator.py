"""Asynchronous event-driven replay simulator."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hermes.core.replay_oracle import ReplayOracle
from hermes.core.runtime_model import RuntimeModel
from hermes.core.types import (
    Backend,
    BackendState,
    CompletedObservation,
    PendingJob,
    ProposedJob,
    ResourceBudget,
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


@dataclass(slots=True)
class _InternalPendingJob:
    """Simulator-private pending job with hidden replay details."""

    visible: PendingJob
    true_finish_time: float
    true_runtime_seconds: float
    true_cost: float


class EventSimulator:
    """Asynchronous replay simulator with a sanitized scheduler boundary."""

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
        self._current_time = 0.0
        self._completed: list[CompletedObservation] = []
        self._pending: list[_InternalPendingJob] = []
        self._horizon_seconds = 0.0
        self._max_target_queries: int | None = None
        self._events: list[Mapping[str, Any]] = []

    def run(
        self,
        scheduler: BaseScheduler,
        initial_observations: list[CompletedObservation],
        config: ExperimentConfig,
    ) -> SimulationResult:
        self._reset_run_state(initial_observations, config)

        seed = int(config.metadata.get("seed", 0))
        scheduler.reset(seed, copy.deepcopy(self._completed))

        while self._current_time <= config.horizon_seconds:
            completed_jobs = self.collect_completed_jobs(self._snapshot_state())
            scheduler.update(copy.deepcopy(completed_jobs))

            state_for_scheduler = self._snapshot_state()
            available_slots = self.get_available_backend_slots(state_for_scheduler)

            if available_slots:
                proposed = scheduler.propose(
                    copy.deepcopy(state_for_scheduler),
                    dict(available_slots),
                )
                self._assert_proposed_jobs(proposed)

                if proposed:
                    validation_state = self._snapshot_state()
                    self.validate_proposed_jobs(proposed, validation_state)
                    self.launch(proposed, validation_state)

            if self._pending:
                self._current_time = self.advance_to_next_event(self._snapshot_state())
                continue

            break

        final_state = self._snapshot_state()
        return SimulationResult(
            scheduler_name=scheduler.name,
            final_state=final_state,
            events=list(self._events),
            completions=copy.deepcopy(self._completed),
            metrics={},
        )

    def collect_completed_jobs(self, state: SchedulerState) -> list[CompletedObservation]:
        del state
        completed_now: list[CompletedObservation] = []
        still_pending: list[_InternalPendingJob] = []

        for pending in self._pending:
            if pending.true_finish_time <= self._current_time:
                record = self.replay_oracle.query_hidden(
                    pending.visible.job.candidate_id,
                    pending.visible.job.source_id,
                )
                completed_now.append(
                    CompletedObservation(
                        job=pending.visible.job,
                        start_time=pending.visible.start_time,
                        finish_time=pending.true_finish_time,
                        y=record.y,
                        runtime_seconds=pending.true_runtime_seconds,
                        cost=pending.true_cost,
                    )
                )
            else:
                still_pending.append(pending)

        completed_now.sort(key=lambda obs: (obs.finish_time, obs.job.candidate_id, obs.job.source_id))
        self._pending = still_pending
        self._completed.extend(copy.deepcopy(completed_now))
        return completed_now

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
        self._assert_proposed_jobs(jobs)

        proposed_pairs: set[tuple[int, str]] = set()
        proposed_by_backend: dict[str, int] = {}

        for job in jobs:
            if job.source_id not in self.sources:
                raise ValueError(f"unknown source_id: {job.source_id!r}")
            if job.backend_id not in self.backends or job.backend_id not in state.backend_states:
                raise ValueError(f"unknown backend_id: {job.backend_id!r}")

            pair = (job.candidate_id, job.source_id)
            if pair in proposed_pairs:
                raise ValueError(f"duplicate candidate/source pair already proposed: {pair!r}")
            if pair in state.completed_pairs:
                raise ValueError(f"duplicate candidate/source pair already completed: {pair!r}")
            if pair in state.pending_pairs:
                raise ValueError(f"duplicate candidate/source pair already pending: {pair!r}")
            if pair in state.already_queried_pairs:
                raise ValueError(f"duplicate candidate/source pair already queried: {pair!r}")
            proposed_pairs.add(pair)

            proposed_by_backend[job.backend_id] = proposed_by_backend.get(job.backend_id, 0) + 1

            predicted_runtime = self.runtime_model.predict(
                job.candidate_id,
                job.source_id,
                job.backend_id,
            )
            predicted_cost = self.runtime_model.predict_cost(
                job.candidate_id,
                job.source_id,
                job.backend_id,
            )
            is_target_query = self.sources[job.source_id].is_target
            predicted_gpu_seconds = predicted_runtime if self._is_gpu_backend(job.backend_id) else 0.0

            if not state.remaining_budget.can_afford(
                predicted_runtime_seconds=predicted_runtime,
                predicted_cost=predicted_cost,
                is_target_query=is_target_query,
                predicted_gpu_seconds=predicted_gpu_seconds,
            ):
                raise ValueError(
                    "proposed job is not affordable under predicted budget "
                    f"(wallclock/cost/target): {job!r}"
                )

        for backend_id, proposed_count in proposed_by_backend.items():
            backend_state = state.backend_states[backend_id]
            if proposed_count > backend_state.idle_slots:
                raise ValueError(
                    f"backend capacity/slot violation for {backend_id!r}: "
                    f"requested {proposed_count}, idle {backend_state.idle_slots}"
                )

        if state.remaining_budget.remaining_target_queries is not None:
            target_count = sum(1 for job in jobs if self.sources[job.source_id].is_target)
            if target_count > state.remaining_budget.remaining_target_queries:
                raise ValueError("target query budget exceeded by proposed batch")

        if state.remaining_budget.remaining_dollars is not None:
            total_predicted_cost = sum(
                self.runtime_model.predict_cost(job.candidate_id, job.source_id, job.backend_id)
                for job in jobs
            )
            if total_predicted_cost > state.remaining_budget.remaining_dollars:
                raise ValueError("dollar cost budget exceeded by proposed batch")

        if state.remaining_budget.remaining_gpu_seconds is not None:
            total_gpu_seconds = sum(
                self.runtime_model.predict(job.candidate_id, job.source_id, job.backend_id)
                for job in jobs
                if self._is_gpu_backend(job.backend_id)
            )
            if total_gpu_seconds > state.remaining_budget.remaining_gpu_seconds:
                raise ValueError("gpu-second budget exceeded by proposed batch")

    def launch(self, jobs: list[ProposedJob], state: SchedulerState) -> list[PendingJob]:
        self.validate_proposed_jobs(jobs, state)

        visible_jobs: list[PendingJob] = []
        for job in jobs:
            predicted_runtime = self.runtime_model.predict(
                job.candidate_id,
                job.source_id,
                job.backend_id,
            )
            predicted_cost = self.runtime_model.predict_cost(
                job.candidate_id,
                job.source_id,
                job.backend_id,
            )

            record = self.replay_oracle.query_hidden(job.candidate_id, job.source_id)
            true_runtime = self._true_runtime_seconds(job, record.true_runtime_seconds)
            true_cost = true_runtime * self.backends[job.backend_id].dollar_per_second
            visible = PendingJob(
                job=job,
                start_time=self._current_time,
                predicted_finish_time=self._current_time + predicted_runtime,
                predicted_runtime_seconds=predicted_runtime,
                predicted_cost=predicted_cost,
            )
            self._pending.append(
                _InternalPendingJob(
                    visible=visible,
                    true_finish_time=self._current_time + true_runtime,
                    true_runtime_seconds=true_runtime,
                    true_cost=true_cost,
                )
            )
            visible_jobs.append(visible)
            self._events.append(
                {
                    "time": self._current_time,
                    "candidate_id": job.candidate_id,
                    "source_id": job.source_id,
                    "backend_id": job.backend_id,
                    "predicted_runtime_seconds": predicted_runtime,
                    "predicted_cost": predicted_cost,
                }
            )

        return copy.deepcopy(visible_jobs)

    def advance_to_next_event(self, state: SchedulerState) -> float:
        del state
        if not self._pending:
            return self._current_time
        return min(pending.true_finish_time for pending in self._pending)

    def _reset_run_state(
        self,
        initial_observations: list[CompletedObservation],
        config: ExperimentConfig,
    ) -> None:
        self._current_time = 0.0
        self._completed = copy.deepcopy(initial_observations)
        self._pending = []
        self._horizon_seconds = config.horizon_seconds
        self._max_target_queries = config.max_target_queries
        self._events = []

    def _snapshot_state(self) -> SchedulerState:
        visible_pending = [copy.deepcopy(pending.visible) for pending in self._pending]
        visible_completed = copy.deepcopy(self._completed)
        backend_states = {
            backend_id: BackendState(
                backend=copy.deepcopy(backend),
                busy_jobs=[
                    copy.deepcopy(pending.visible)
                    for pending in self._pending
                    if pending.visible.job.backend_id == backend_id
                ],
            )
            for backend_id, backend in self.backends.items()
        }
        completed_pairs = {
            (obs.job.candidate_id, obs.job.source_id)
            for obs in visible_completed
        }
        pending_pairs = {
            (pending.job.candidate_id, pending.job.source_id)
            for pending in visible_pending
        }
        target_validated = {
            obs.job.candidate_id
            for obs in visible_completed
            if self.sources.get(obs.job.source_id) is not None
            and self.sources[obs.job.source_id].is_target
        }
        remaining_target_queries = self._remaining_target_queries()

        return SchedulerState(
            current_time=self._current_time,
            completed=visible_completed,
            pending=visible_pending,
            backend_states=backend_states,
            remaining_budget=ResourceBudget(
                remaining_wallclock_seconds=max(self._horizon_seconds - self._current_time, 0.0),
                remaining_dollars=None,
                remaining_gpu_seconds=None,
                remaining_target_queries=remaining_target_queries,
            ),
            already_target_validated=target_validated,
            already_queried_pairs=completed_pairs | pending_pairs,
        )

    def _remaining_target_queries(self) -> int | None:
        if self._max_target_queries is None:
            return None
        launched_targets = sum(
            1
            for obs in self._completed
            if obs.job.source_id in self.sources and self.sources[obs.job.source_id].is_target
        ) + sum(
            1
            for pending in self._pending
            if pending.visible.job.source_id in self.sources
            and self.sources[pending.visible.job.source_id].is_target
        )
        return max(self._max_target_queries - launched_targets, 0)

    def _assert_proposed_jobs(self, jobs: object) -> None:
        if not isinstance(jobs, list):
            raise TypeError("scheduler return value must be a list[ProposedJob]")
        if not all(isinstance(job, ProposedJob) for job in jobs):
            raise TypeError("scheduler return value must contain only ProposedJob instances")

    def _is_gpu_backend(self, backend_id: str) -> bool:
        lowered = backend_id.lower()
        return "gpu" in lowered or "cuda" in lowered

    def _true_runtime_seconds(self, job: ProposedJob, source_runtime_seconds: float) -> float:
        backend = self.backends[job.backend_id]
        multiplier = backend.runtime_multiplier_by_source.get(job.source_id, 1.0)
        return source_runtime_seconds * multiplier + backend.startup_latency_seconds


__all__ = ["EventSimulator", "ExperimentConfig", "SimulationResult"]
