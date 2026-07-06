from __future__ import annotations

import pytest

from hermes.core.types import CompletedObservation, ProposedJob

from .conftest import CPU, TARGET, ScriptedScheduler


def test_rejects_candidate_source_pair_already_completed(
    simulator,
    config,
) -> None:
    initial = [
        CompletedObservation(
            job=ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU),
            start_time=0.0,
            finish_time=1.0,
            y=123.0,
            runtime_seconds=1.0,
            cost=0.01,
        )
    ]
    scheduler = ScriptedScheduler(
        batches=[
            [ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)]
        ]
    )

    with pytest.raises(ValueError, match="duplicate|already.*queried|completed"):
        simulator.run(scheduler, initial, config)


def test_rejects_duplicate_candidate_source_pair_inside_same_batch(
    simulator,
    initial_observations,
    config,
) -> None:
    job = ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)
    scheduler = ScriptedScheduler(batches=[[job, job]])

    with pytest.raises(ValueError, match="duplicate|already.*proposed"):
        simulator.run(scheduler, initial_observations, config)


def test_rejects_candidate_source_pair_already_pending(
    simulator,
    initial_observations,
    config,
) -> None:
    """A long pending job cannot be relaunched while a short job triggers
    another decision epoch.
    """

    scheduler = ScriptedScheduler(
        batches=[
            [
                ProposedJob(candidate_id=4, source_id=TARGET, backend_id=CPU),  # true runtime 60
                ProposedJob(candidate_id=3, source_id=TARGET, backend_id=CPU),  # true runtime 5
            ],
            [
                ProposedJob(candidate_id=4, source_id=TARGET, backend_id=CPU),
            ],
        ]
    )

    with pytest.raises(ValueError, match="duplicate|pending|already.*queried"):
        simulator.run(scheduler, initial_observations, config)
