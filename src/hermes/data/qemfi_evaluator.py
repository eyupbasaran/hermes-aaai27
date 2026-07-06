"""Hidden QeMFi evaluator for post-run metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from hermes.core.types import CompletedObservation


@dataclass(slots=True)
class QeMFiHiddenEvaluator:
    _target_source_id: str
    _target_utilities: dict[int, float]

    def compute_final_metrics(self, completed_observations: list[CompletedObservation]) -> dict[str, float | int | None]:
        target_observations = [
            obs for obs in completed_observations if obs.job.source_id == self._target_source_id
        ]
        best = max((obs.y for obs in target_observations), default=None)
        return {
            "target_query_count": len(target_observations),
            "best_target_utility": best,
            "top_1_percent_recovery": self.compute_topk_recovery(completed_observations, 0.01),
            "top_5_percent_recovery": self.compute_topk_recovery(completed_observations, 0.05),
        }

    def compute_discovery_curve(self, completions: list[CompletedObservation]) -> list[dict[str, float | int]]:
        best = None
        curve: list[dict[str, float | int]] = []
        for obs in completions:
            if obs.job.source_id != self._target_source_id:
                continue
            best = obs.y if best is None else max(best, obs.y)
            curve.append(
                {
                    "time": obs.finish_time,
                    "target_query_count": len(curve) + 1,
                    "best_target_utility": best,
                }
            )
        return curve

    def compute_topk_recovery(
        self,
        completed_observations: list[CompletedObservation],
        k_fraction: float,
    ) -> float:
        if not self._target_utilities:
            return 0.0
        k = max(1, int(math.ceil(len(self._target_utilities) * k_fraction)))
        top_ids = {
            candidate_id
            for candidate_id, _ in sorted(
                self._target_utilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:k]
        }
        observed_top = {
            obs.job.candidate_id
            for obs in completed_observations
            if obs.job.source_id == self._target_source_id and obs.job.candidate_id in top_ids
        }
        return len(observed_top) / k

    def compute_best_target_score(
        self,
        completed_observations: list[CompletedObservation],
    ) -> float | None:
        target_values = [
            obs.y for obs in completed_observations if obs.job.source_id == self._target_source_id
        ]
        return max(target_values, default=None)


__all__ = ["QeMFiHiddenEvaluator"]

