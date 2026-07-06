from __future__ import annotations

import pytest

from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import synthetic_qemfi_config_dict, write_synthetic_qemfi_npz


def load_split(tmp_path, *, split_seed: int = 0, permute_labels: bool = False):
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20, permute_labels=permute_labels)
    config = QeMFiDatasetConfig.from_dict(
        synthetic_qemfi_config_dict(
            tmp_path,
            candidate_pool_size=10,
            initial_seed_size=2,
            split_seed=split_seed,
        )
    )
    return load_qemfi_from_config(config).public_data.split


def test_split_is_reproducible_by_seed(tmp_path) -> None:
    first = load_split(tmp_path / "a", split_seed=4)
    second = load_split(tmp_path / "a", split_seed=4)

    assert first.pool_candidate_ids == second.pool_candidate_ids
    assert first.initial_pairs == second.initial_pairs


def test_different_seed_changes_split_with_high_probability(tmp_path) -> None:
    first = load_split(tmp_path / "a", split_seed=4)
    second = load_split(tmp_path / "b", split_seed=5)

    assert first.pool_candidate_ids != second.pool_candidate_ids


def test_split_is_independent_of_labels(tmp_path) -> None:
    first = load_split(tmp_path / "plain", split_seed=7, permute_labels=False)
    second = load_split(tmp_path / "permuted", split_seed=7, permute_labels=True)

    assert first.pool_candidate_ids == second.pool_candidate_ids
    assert first.initial_pairs == second.initial_pairs


def test_split_size_validation(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=5)
    too_large_pool = QeMFiDatasetConfig.from_dict(
        synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=99, initial_seed_size=2)
    )
    with pytest.raises(ValueError, match="candidate_pool_size|eligible"):
        load_qemfi_from_config(too_large_pool)

    too_large_seed = QeMFiDatasetConfig.from_dict(
        synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=5, initial_seed_size=6)
    )
    with pytest.raises(ValueError, match="initial_seed_size|candidate_pool_size"):
        load_qemfi_from_config(too_large_seed)

