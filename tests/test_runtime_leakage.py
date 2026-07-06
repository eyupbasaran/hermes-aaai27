from __future__ import annotations

from dataclasses import fields

import pytest

from hermes.core.types import PendingJob, ProposedJob
from hermes.schedulers.base import BaseScheduler

from .conftest import CPU, LOW, TARGET, ScriptedScheduler, assert_no_scheduler_secrets


def test_scheduler_visible_pending_job_type_has_no_true_runtime_fields() -> None:
    """Public pending jobs must not reveal true completion data.

    If this test fails, split the current pending type into:
    - public PendingJob with predicted fields only;
    - private _InternalPendingJob with true runtime/finish/cost.
    """

    field_names = {field.name for field in fields(PendingJob)}

    assert "true_finish_time" not in field_names
    assert "true_runtime_seconds" not in field_names
    assert "true_cost" not in field_names


class LeakageProbeScheduler(ScriptedScheduler):
    name = "leakage_probe"

    def reset(self, seed, initial_observations):
        assert_no_scheduler_secrets(initial_observations)
        super().reset(seed, initial_observations)

    def update(self, completed_jobs):
        # Completed observations may reveal observed y/runtime/cost. They should
        # still not contain oracle handles, hidden target labels, or future info.
        assert_no_scheduler_secrets(completed_jobs)
        super().update(completed_jobs)

    def propose(self, state, available_slots):
        assert_no_scheduler_secrets(state)
        assert_no_scheduler_secrets(available_slots)
        return super().propose(state, available_slots)


def test_scheduler_snapshots_do_not_expose_oracle_hidden_labels_or_future_completion_data(
    simulator,
    initial_observations,
    config,
) -> None:
    """At the second decision epoch, one job is still pending. The scheduler
    snapshot must still not expose true runtime/finish/order for that pending job.
    """

    scheduler = LeakageProbeScheduler(
        batches=[
            [
                ProposedJob(candidate_id=4, source_id=TARGET, backend_id=CPU),  # long true runtime
                ProposedJob(candidate_id=3, source_id=TARGET, backend_id=CPU),  # short true runtime
            ],
            [],
        ]
    )

    simulator.run(scheduler, initial_observations, config)


def test_base_scheduler_api_does_not_accept_oracle_or_hidden_dataset() -> None:
    """The public scheduler protocol should remain narrow."""

    import inspect

    reset_params = list(inspect.signature(BaseScheduler.reset).parameters)
    update_params = list(inspect.signature(BaseScheduler.update).parameters)
    propose_params = list(inspect.signature(BaseScheduler.propose).parameters)

    assert reset_params == ["self", "seed", "initial_observations"]
    assert update_params == ["self", "completed_jobs"]
    assert propose_params == ["self", "state", "available_slots"]

    for params in (reset_params, update_params, propose_params):
        joined = " ".join(params).lower()
        assert "oracle" not in joined
        assert "hidden" not in joined
        assert "topk" not in joined
        assert "label" not in joined


class CountingOracle:
    """ReplayOracle test double that counts hidden lookups."""

    def __init__(self, records):
        self.records = dict(records)
        self.num_queries = 0

    def query_hidden(self, candidate_id: int, source_id: str):
        self.num_queries += 1
        return self.records[(candidate_id, source_id)]


def test_rejected_budget_infeasible_jobs_do_not_query_hidden_oracle(
    oracle_records,
    runtime_model,
    sources,
    backends,
    initial_observations,
) -> None:
    """Budget validation should happen before hidden oracle lookup."""

    from hermes.core.event_simulator import EventSimulator, ExperimentConfig

    counting_oracle = CountingOracle(oracle_records)
    simulator = EventSimulator(
        replay_oracle=counting_oracle,
        runtime_model=runtime_model,
        sources=sources,
        backends=backends,
    )
    config = ExperimentConfig(
        horizon_seconds=1.0,
        max_target_queries=10,
        metadata={"seed": 7},
    )
    scheduler = ScriptedScheduler(
        batches=[
            [ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)]
        ]
    )

    with pytest.raises(ValueError, match="budget|wallclock|afford"):
        simulator.run(scheduler, initial_observations, config)

    assert counting_oracle.num_queries == 0
