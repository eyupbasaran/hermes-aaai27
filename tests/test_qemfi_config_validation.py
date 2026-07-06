from __future__ import annotations

import pytest

from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import synthetic_qemfi_config_dict


def test_qemfi_config_parses_structured_dataset_config(tmp_path) -> None:
    config = QeMFiDatasetConfig.from_dict(synthetic_qemfi_config_dict(tmp_path))

    assert config.name == "qemfi"
    assert config.target_source_id == "def2tzvp"
    assert [source.source_id for source in config.sources] == [
        "sto3g",
        "321g",
        "631g",
        "def2svp",
        "def2tzvp",
    ]
    assert config.property_spec.key == "fosc"
    assert config.property_spec.component == (0,)
    assert config.runtime_spec.public_runtime_model == "source_mean"


def test_property_key_is_required(tmp_path) -> None:
    raw = synthetic_qemfi_config_dict(tmp_path, property_key=None)

    with pytest.raises(ValueError, match="property.*key"):
        QeMFiDatasetConfig.from_dict(raw)


def test_target_source_must_exist_and_be_unique(tmp_path) -> None:
    raw = synthetic_qemfi_config_dict(tmp_path, target_source_id="missing")

    with pytest.raises(ValueError, match="target.*source"):
        QeMFiDatasetConfig.from_dict(raw)

    raw = synthetic_qemfi_config_dict(tmp_path)
    raw["dataset"]["sources"][0]["is_target"] = True
    with pytest.raises(ValueError, match="exactly one.*target"):
        QeMFiDatasetConfig.from_dict(raw)


def test_candidate_specific_public_runtime_forbidden_in_strict_mode(tmp_path) -> None:
    raw = synthetic_qemfi_config_dict(tmp_path)
    raw["dataset"]["runtime"]["candidate_specific_public_runtime"] = True

    with pytest.raises(ValueError, match="candidate.*runtime|leakage"):
        QeMFiDatasetConfig.from_dict(raw)

