from __future__ import annotations

from hermes.core.types import ProposedJob

from .conftest import CPU, TARGET, ScriptedScheduler


def test_simulator_completes_jobs_in_true_chronological_order(
    simulator,
    initial_observations,
    config,
) -> None:
    """The internal simulator uses true replay runtimes to complete jobs.

    The scheduler may receive only sanitized pending-job predictions, but the
    simulator itself must complete jobs in actual chronological order.
    """

    scheduler = ScriptedScheduler(
        batches=[
            [
                ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU),  # true finish 30
                ProposedJob(candidate_id=1, source_id=TARGET, backend_id=CPU),  # true finish 10
                ProposedJob(candidate_id=2, source_id=TARGET, backend_id=CPU),  # true finish 20
            ],
            [],
            [],
            [],
        ]
    )

    result = simulator.run(scheduler, initial_observations, config)

    launched = [
        obs for obs in result.completions
        if obs.job.candidate_id in {0, 1, 2} and obs.job.source_id == TARGET
    ]

    assert [obs.job.candidate_id for obs in launched] == [1, 2, 0]
    assert [obs.finish_time for obs in launched] == [10.0, 20.0, 30.0]


def test_advance_to_next_event_uses_next_internal_completion_not_prediction(
    simulator,
    initial_observations,
    config,
) -> None:
    """Predicted finish times are scheduler-visible guesses; true finish times
    drive simulator event order internally.
    """

    scheduler = ScriptedScheduler(
        batches=[
            [
                ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU),  # true 30
                ProposedJob(candidate_id=1, source_id=TARGET, backend_id=CPU),  # true 10
            ],
            [],
        ]
    )

    result = simulator.run(scheduler, initial_observations, config)

    assert result.final_state.current_time >= 30.0
    assert [obs.job.candidate_id for obs in result.completions if obs.job.candidate_id in {0, 1}] == [
        1,
        0,
    ]
