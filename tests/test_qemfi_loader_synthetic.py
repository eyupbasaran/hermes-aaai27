from __future__ import annotations

from hermes.core.types import CompletedObservation
from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import SOURCE_IDS, synthetic_qemfi_config_dict, write_synthetic_qemfi_npz


def test_synthetic_qemfi_loader_returns_public_private_boundaries(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    config = QeMFiDatasetConfig.from_dict(
        synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
    )

    result = load_qemfi_from_config(config)

    assert result.public_data.num_candidates() == 10
    assert result.public_data.target_source_id == "def2tzvp"
    assert result.public_data.get_source_ids() == SOURCE_IDS
    assert len(result.initial_observations) == 2 * len(SOURCE_IDS)
    assert all(isinstance(obs, CompletedObservation) for obs in result.initial_observations)
    assert result.load_report["candidate_pool_count"] == 10
    assert result.load_report["initial_observation_count"] == 10


def test_property_component_selection_and_utility_transform(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=8)
    raw = synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=4, initial_seed_size=1)
    raw["dataset"]["property"]["component"] = [1]
    raw["dataset"]["property"]["utility_direction"] = "minimize"
    config = QeMFiDatasetConfig.from_dict(raw)

    result = load_qemfi_from_config(config)
    pair = result.public_data.split.initial_pairs[0]
    record = result.replay_oracle.query_hidden(*pair)

    assert record.y > 900.0


def test_invalid_property_source_axis_raises(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=8, n_sources=4)
    config = QeMFiDatasetConfig.from_dict(
        synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=4, initial_seed_size=1)
    )

    try:
        load_qemfi_from_config(config)
    except ValueError as exc:
        assert "source" in str(exc).lower()
    else:
        raise AssertionError("Expected invalid source-axis length to raise.")

