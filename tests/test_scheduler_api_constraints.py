from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from hermes.core.types import CompletedObservation, ProposedJob
from hermes.schedulers.base import BaseScheduler

from .conftest import CPU, TARGET, make_state


def test_proposed_job_is_exactly_candidate_source_backend_and_frozen() -> None:
    field_names = [field.name for field in fields(ProposedJob)]
    assert field_names == ["candidate_id", "source_id", "backend_id"]

    job = ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)
    with pytest.raises(FrozenInstanceError):
        job.candidate_id = 123  # type: ignore[misc]


def test_validate_rejects_unknown_source(
    simulator,
    backends,
) -> None:
    state = make_state(backends=backends)
    job = ProposedJob(candidate_id=0, source_id="unknown_source", backend_id=CPU)

    with pytest.raises(ValueError, match="source|unknown"):
        simulator.validate_proposed_jobs([job], state)


def test_validate_rejects_unknown_backend(
    simulator,
    backends,
) -> None:
    state = make_state(backends=backends)
    job = ProposedJob(candidate_id=0, source_id=TARGET, backend_id="unknown_backend")

    with pytest.raises(ValueError, match="backend|unknown"):
        simulator.validate_proposed_jobs([job], state)


class BadReturnScheduler:
    name = "bad_return"

    def reset(self, seed, initial_observations):
        pass

    def update(self, completed_jobs):
        pass

    def propose(self, state, available_slots):
        return [
            CompletedObservation(
                job=ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU),
                start_time=0.0,
                finish_time=1.0,
                y=1.0,
                runtime_seconds=1.0,
                cost=0.0,
            )
        ]


def test_scheduler_must_return_only_proposed_jobs(
    simulator,
    initial_observations,
    config,
) -> None:
    with pytest.raises(TypeError, match="ProposedJob|scheduler.*return"):
        simulator.run(BadReturnScheduler(), initial_observations, config)


def test_base_scheduler_protocol_runtime_checkable_shape() -> None:
    class GoodScheduler:
        name = "good"

        def reset(self, seed, initial_observations):
            pass

        def update(self, completed_jobs):
            pass

        def propose(self, state, available_slots):
            return []

    assert isinstance(GoodScheduler(), BaseScheduler)
