from __future__ import annotations

from hermes.core.types import CompletedObservation, ProposedJob
from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import assert_public_qemfi_has_no_secrets, synthetic_qemfi_config_dict, write_synthetic_qemfi_npz


def test_evaluator_is_private_and_public_split_has_no_rank_or_topk(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
        )
    )

    assert not hasattr(result.public_data, "evaluator")
    assert_public_qemfi_has_no_secrets(result.public_data.split)


def test_hidden_evaluator_computes_post_run_metrics(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=1)
        )
    )
    best_candidate = max(
        result.public_data.candidate_ids,
        key=lambda candidate_id: result.replay_oracle.query_hidden(candidate_id, "def2tzvp").y,
    )
    record = result.replay_oracle.query_hidden(best_candidate, "def2tzvp")
    completed = [
        CompletedObservation(
            job=ProposedJob(best_candidate, "def2tzvp", "cpu"),
            start_time=0.0,
            finish_time=1.0,
            y=record.y,
            runtime_seconds=record.true_runtime_seconds,
            cost=0.0,
        )
    ]

    metrics = result.evaluator.compute_final_metrics(completed)

    assert metrics["best_target_utility"] == record.y
    assert metrics["target_query_count"] == 1
    assert "top_1_percent_recovery" in metrics

