"""Runtime prediction models available to schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from hermes.core.types import Backend


@runtime_checkable
class RuntimeModel(Protocol):
    """Scheduler-visible runtime predictor.

    Implementations must use only information available before launch. True
    runtimes from the replay oracle belong to the simulator, not this protocol.
    """

    def predict(self, candidate_id: int, source_id: str, backend_id: str) -> float:
        """Return predicted runtime in seconds."""
        ...

    def predict_cost(self, candidate_id: int, source_id: str, backend_id: str) -> float:
        """Return predicted dollar cost."""
        ...


@dataclass(frozen=True, slots=True)
class SourceMeanRuntimeModel:
    """Predict runtime from source-level means and backend multipliers."""

    source_mean_runtime_seconds: Mapping[str, float]
    backends: Mapping[str, Backend]

    def predict(self, candidate_id: int, source_id: str, backend_id: str) -> float:
        del candidate_id
        if source_id not in self.source_mean_runtime_seconds:
            raise KeyError(f"Unknown source_id for runtime prediction: {source_id!r}")
        if backend_id not in self.backends:
            raise KeyError(f"Unknown backend_id for runtime prediction: {backend_id!r}")

        backend = self.backends[backend_id]
        if source_id not in backend.runtime_multiplier_by_source:
            raise KeyError(
                f"Backend {backend_id!r} has no runtime multiplier for source {source_id!r}"
            )

        source_mean = self.source_mean_runtime_seconds[source_id]
        if source_mean < 0:
            raise ValueError(f"Source mean runtime must be non-negative for source {source_id!r}")

        return (
            source_mean * backend.runtime_multiplier_by_source[source_id]
            + backend.startup_latency_seconds
        )

    def predict_cost(self, candidate_id: int, source_id: str, backend_id: str) -> float:
        runtime_seconds = self.predict(candidate_id, source_id, backend_id)
        return runtime_seconds * self.backends[backend_id].dollar_per_second


__all__ = ["RuntimeModel", "SourceMeanRuntimeModel"]
