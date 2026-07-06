from __future__ import annotations

import pytest

from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import synthetic_qemfi_config_dict, write_synthetic_qemfi_npz


def test_public_data_does_not_expose_oracle_or_bulk_hidden_arrays(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
        )
    )

    for name in [
        "replay_oracle",
        "oracle",
        "evaluator",
        "labels",
        "runtimes",
        "target_ranks",
        "topk",
        "query_hidden",
    ]:
        assert not hasattr(result.public_data, name)
        assert name not in result.public_data.metadata


def test_qemfi_replay_oracle_has_only_single_record_public_lookup(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
        )
    )
    pair = result.public_data.split.initial_pairs[0]

    record = result.replay_oracle.query_hidden(*pair)

    assert isinstance(record.y, float)
    assert record.true_runtime_seconds > 0
    with pytest.raises(KeyError):
        result.replay_oracle.query_hidden(-999, "def2tzvp")
    for name in ["get_all_labels", "labels", "target_ranks", "topk"]:
        assert not hasattr(result.replay_oracle, name)

