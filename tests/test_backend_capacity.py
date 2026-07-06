from __future__ import annotations

import pytest

from hermes.core.event_simulator import EventSimulator
from hermes.core.runtime_model import SourceMeanRuntimeModel
from hermes.core.types import ProposedJob

from .conftest import CPU, LOW, TARGET, ScriptedScheduler


def test_scheduler_cannot_overfill_backend_capacity(
    oracle,
    sources,
    cap2_backends,
    initial_observations,
    config,
) -> None:
    runtime_model = SourceMeanRuntimeModel(
        source_mean_runtime_seconds={LOW: 10.0, TARGET: 100.0},
        backends=cap2_backends,
    )
    simulator = EventSimulator(
        replay_oracle=oracle,
        runtime_model=runtime_model,
        sources=sources,
        backends=cap2_backends,
    )
    scheduler = ScriptedScheduler(
        batches=[
            [
                ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU),
                ProposedJob(candidate_id=1, source_id=TARGET, backend_id=CPU),
                ProposedJob(candidate_id=2, source_id=TARGET, backend_id=CPU),
            ]
        ]
    )

    with pytest.raises(ValueError, match="capacity|slot|backend"):
        simulator.run(scheduler, initial_observations, config)


def test_scheduler_may_fill_backend_to_exact_capacity(
    oracle,
    sources,
    cap2_backends,
    initial_observations,
    config,
) -> None:
    runtime_model = SourceMeanRuntimeModel(
        source_mean_runtime_seconds={LOW: 10.0, TARGET: 100.0},
        backends=cap2_backends,
    )
    simulator = EventSimulator(
        replay_oracle=oracle,
        runtime_model=runtime_model,
        sources=sources,
        backends=cap2_backends,
    )
    scheduler = ScriptedScheduler(
        batches=[
            [
                ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU),
                ProposedJob(candidate_id=1, source_id=TARGET, backend_id=CPU),
            ],
            [],
        ]
    )

    result = simulator.run(scheduler, initial_observations, config)

    completed_ids = {obs.job.candidate_id for obs in result.completions}
    assert {0, 1}.issubset(completed_ids)
