from __future__ import annotations

from hermes.core.types import ProposedJob

from .conftest import CPU, TARGET


class MutatingScheduler:
    name = "mutating_scheduler"

    def reset(self, seed, initial_observations):
        self.seed = seed
        # Try to corrupt the caller's initial list and observations.
        if initial_observations:
            initial_observations[0].y = 999_999.0
        initial_observations.append(initial_observations[0])

    def update(self, completed_jobs):
        if completed_jobs:
            completed_jobs[0].y = 888_888.0

    def propose(self, state, available_slots):
        # Try to corrupt simulator state through scheduler-visible objects.
        state.current_time = 999_999.0
        if state.completed:
            state.completed[0].y = 777_777.0
        for backend_state in state.backend_states.values():
            backend_state.backend.capacity_slots = 999_999
        for key in list(available_slots):
            available_slots[key] = 999_999
        return []


class RecordingScheduler:
    name = "recording_scheduler"

    def __init__(self):
        self.seed = None
        self.initial_snapshot = None
        self.first_state_snapshot = None
        self.available_slots_snapshot = None

    def reset(self, seed, initial_observations):
        self.seed = seed
        self.initial_snapshot = [
            (obs.job.candidate_id, obs.job.source_id, obs.y)
            for obs in initial_observations
        ]

    def update(self, completed_jobs):
        pass

    def propose(self, state, available_slots):
        if self.first_state_snapshot is None:
            self.first_state_snapshot = {
                "current_time": state.current_time,
                "completed": [
                    (obs.job.candidate_id, obs.job.source_id, obs.y)
                    for obs in state.completed
                ],
                "capacity": {
                    backend_id: backend_state.backend.capacity_slots
                    for backend_id, backend_state in state.backend_states.items()
                },
            }
            self.available_slots_snapshot = dict(available_slots)
        return []


def test_scheduler_mutation_cannot_pollute_next_baseline_run(
    simulator,
    initial_observations,
    config,
) -> None:
    """Baseline fairness requires defensive copies around scheduler calls."""

    simulator.run(MutatingScheduler(), initial_observations, config)

    recorder = RecordingScheduler()
    simulator.run(recorder, initial_observations, config)

    assert recorder.seed == 7
    assert recorder.initial_snapshot == [(99, TARGET, -1.0)]
    assert recorder.first_state_snapshot["current_time"] == 0.0
    assert recorder.first_state_snapshot["completed"] == [(99, TARGET, -1.0)]
    assert recorder.first_state_snapshot["capacity"][CPU] == 3
    assert recorder.available_slots_snapshot[CPU] == 3


def test_same_seed_and_initial_observations_are_passed_to_each_scheduler(
    simulator,
    initial_observations,
    config,
) -> None:
    first = RecordingScheduler()
    second = RecordingScheduler()

    simulator.run(first, initial_observations, config)
    simulator.run(second, initial_observations, config)

    assert first.seed == second.seed == 7
    assert first.initial_snapshot == second.initial_snapshot == [(99, TARGET, -1.0)]
