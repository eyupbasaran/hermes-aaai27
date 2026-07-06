from __future__ import annotations

import pytest

from hermes.core.types import ProposedJob, ResourceBudget

from .conftest import CPU, TARGET, make_state


def test_validate_rejects_job_exceeding_remaining_wallclock(
    simulator,
    backends,
) -> None:
    state = make_state(
        backends=backends,
        remaining_budget=ResourceBudget(
            remaining_wallclock_seconds=1.0,
            remaining_dollars=100.0,
            remaining_gpu_seconds=None,
            remaining_target_queries=10,
        ),
    )
    job = ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)

    with pytest.raises(ValueError, match="wallclock|budget|afford"):
        simulator.validate_proposed_jobs([job], state)


def test_validate_rejects_job_exceeding_remaining_dollars(
    simulator,
    backends,
) -> None:
    state = make_state(
        backends=backends,
        remaining_budget=ResourceBudget(
            remaining_wallclock_seconds=10_000.0,
            remaining_dollars=0.001,
            remaining_gpu_seconds=None,
            remaining_target_queries=10,
        ),
    )
    job = ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)

    with pytest.raises(ValueError, match="dollar|cost|budget|afford"):
        simulator.validate_proposed_jobs([job], state)


def test_validate_rejects_target_query_when_target_budget_is_zero(
    simulator,
    backends,
) -> None:
    state = make_state(
        backends=backends,
        remaining_budget=ResourceBudget(
            remaining_wallclock_seconds=10_000.0,
            remaining_dollars=100.0,
            remaining_gpu_seconds=None,
            remaining_target_queries=0,
        ),
    )
    job = ProposedJob(candidate_id=0, source_id=TARGET, backend_id=CPU)

    with pytest.raises(ValueError, match="target|budget|afford"):
        simulator.validate_proposed_jobs([job], state)


def test_resource_budget_can_afford_uses_only_predicted_quantities() -> None:
    budget = ResourceBudget(
        remaining_wallclock_seconds=10.0,
        remaining_dollars=1.0,
        remaining_gpu_seconds=20.0,
        remaining_target_queries=1,
    )

    assert budget.can_afford(
        predicted_runtime_seconds=9.0,
        predicted_cost=0.9,
        is_target_query=True,
        predicted_gpu_seconds=19.0,
    )
    assert not budget.can_afford(
        predicted_runtime_seconds=11.0,
        predicted_cost=0.9,
        is_target_query=True,
        predicted_gpu_seconds=19.0,
    )
    assert not budget.can_afford(
        predicted_runtime_seconds=9.0,
        predicted_cost=1.1,
        is_target_query=True,
        predicted_gpu_seconds=19.0,
    )
    assert not budget.can_afford(
        predicted_runtime_seconds=9.0,
        predicted_cost=0.9,
        is_target_query=True,
        predicted_gpu_seconds=21.0,
    )
