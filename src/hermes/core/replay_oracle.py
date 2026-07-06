"""Hidden replay oracle for labels and true runtimes.

Schedulers must never receive a ``ReplayOracle`` instance. The simulator uses
this boundary internally after a job has been selected for launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class OracleRecord:
    """Hidden replay record for one candidate/source pair."""

    y: float
    true_runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.true_runtime_seconds < 0:
            raise ValueError("OracleRecord true_runtime_seconds must be non-negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class ReplayOracle(Protocol):
    """Simulator-only lookup for hidden labels and true runtimes."""

    def query_hidden(self, candidate_id: int, source_id: str) -> OracleRecord:
        """Return the hidden record for an already selected launch job."""
        ...


class InMemoryReplayOracle:
    """Small in-memory oracle useful for tests and synthetic replay fixtures."""

    def __init__(self, records: Mapping[tuple[int, str], OracleRecord]) -> None:
        self._records = dict(records)

    def query_hidden(self, candidate_id: int, source_id: str) -> OracleRecord:
        key = (candidate_id, source_id)
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"No oracle record for candidate/source pair {key!r}.") from exc


__all__ = ["InMemoryReplayOracle", "OracleRecord", "ReplayOracle"]
