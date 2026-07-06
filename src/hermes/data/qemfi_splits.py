"""Leakage-safe QeMFi split construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hermes.data.qemfi_schema import QeMFiSplitSpec


@dataclass(frozen=True, slots=True)
class QeMFiSplit:
    split_seed: int
    pool_candidate_ids: tuple[int, ...]
    initial_candidate_ids: tuple[int, ...]
    initial_pairs: tuple[tuple[int, str], ...]
    active_pool_candidate_ids: tuple[int, ...]
    molecule_names: tuple[str, ...]
    source_ids: tuple[str, ...]
    target_source_id: str


def build_qemfi_split(
    *,
    candidate_ids: list[int],
    molecule_names: tuple[str, ...],
    source_ids: tuple[str, ...],
    target_source_id: str,
    spec: QeMFiSplitSpec,
) -> QeMFiSplit:
    if spec.candidate_pool_size > len(candidate_ids):
        raise ValueError("candidate_pool_size exceeds eligible candidate universe.")
    if spec.initial_seed_size > spec.candidate_pool_size:
        raise ValueError("initial_seed_size may not exceed candidate_pool_size.")

    rng = np.random.default_rng(spec.split_seed)
    sorted_ids = np.array(sorted(candidate_ids), dtype=np.int64)
    pool = rng.choice(sorted_ids, size=spec.candidate_pool_size, replace=False)
    initial = rng.choice(pool, size=spec.initial_seed_size, replace=False)

    pool_ids = tuple(int(candidate_id) for candidate_id in sorted(pool.tolist()))
    initial_ids = tuple(int(candidate_id) for candidate_id in sorted(initial.tolist()))
    initial_pairs = tuple(
        (candidate_id, source_id)
        for candidate_id in initial_ids
        for source_id in source_ids
    )
    active_ids = tuple(candidate_id for candidate_id in pool_ids if candidate_id not in set(initial_ids))

    return QeMFiSplit(
        split_seed=spec.split_seed,
        pool_candidate_ids=pool_ids,
        initial_candidate_ids=initial_ids,
        initial_pairs=initial_pairs,
        active_pool_candidate_ids=active_ids,
        molecule_names=molecule_names,
        source_ids=source_ids,
        target_source_id=target_source_id,
    )


__all__ = ["QeMFiSplit", "build_qemfi_split"]

