from __future__ import annotations

import numpy as np

from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import (
    SOURCE_IDS,
    assert_public_qemfi_has_no_secrets,
    synthetic_qemfi_config_dict,
    write_synthetic_qemfi_npz,
)


def test_public_data_and_candidate_metadata_have_no_qemfi_secrets(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
        )
    )

    assert_public_qemfi_has_no_secrets(result.public_data)
    for candidate in result.public_data.get_candidates():
        assert set(candidate.metadata) <= {
            "dataset_name",
            "molecule_name",
            "raw_file_name",
            "raw_index",
            "raw_id",
            "conformation_id",
            "num_atoms",
            "formula_or_atom_counts",
        }


def test_features_do_not_equal_synthetic_label_vectors(tmp_path) -> None:
    path = write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
        )
    )

    features = result.public_data.get_features(result.public_data.candidate_ids)
    labels = np.load(path, allow_pickle=True)["fosc"][:, :, 0]
    for source_idx in range(len(SOURCE_IDS)):
        source_labels_by_id = {
            result.load_report["candidate_id_by_raw_index"][idx]: value
            for idx, value in enumerate(labels[:, source_idx])
        }
        vector = np.array([source_labels_by_id[candidate_id] for candidate_id in result.public_data.candidate_ids])
        for col_idx in range(features.shape[1]):
            assert not np.allclose(features[:, col_idx], vector)

