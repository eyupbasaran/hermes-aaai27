"""Common scheduler protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hermes.core.types import CompletedObservation, ProposedJob, SchedulerState


@runtime_checkable
class BaseScheduler(Protocol):
    """Public scheduler API used by the event simulator."""

    name: str

    def reset(self, seed: int, initial_observations: list[CompletedObservation]) -> None:
        """Reset scheduler state before a replay run."""
        ...

    def update(self, completed_jobs: list[CompletedObservation]) -> None:
        """Incorporate observations completed since the previous decision event."""
        ...

    def propose(
        self,
        state: SchedulerState,
        available_slots: dict[str, int],
    ) -> list[ProposedJob]:
        """Return launch requests for currently available backend slots."""
        ...


__all__ = ["BaseScheduler"]
