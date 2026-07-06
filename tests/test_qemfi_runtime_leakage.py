from __future__ import annotations

from hermes.core.event_simulator import EventSimulator, ExperimentConfig
from hermes.core.types import Backend, ProposedJob
from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .qemfi_test_utils import synthetic_qemfi_config_dict, write_synthetic_qemfi_npz


def test_source_mean_runtime_model_is_not_candidate_specific(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=2)
        )
    )
    first, second = result.public_data.candidate_ids[:2]

    assert result.runtime_model.predict(first, "def2tzvp", "cpu") == result.runtime_model.predict(
        second, "def2tzvp", "cpu"
    )


def test_true_runtime_revealed_only_after_completion(tmp_path) -> None:
    write_synthetic_qemfi_npz(tmp_path, n_candidates=20)
    result = load_qemfi_from_config(
        QeMFiDatasetConfig.from_dict(
            synthetic_qemfi_config_dict(tmp_path, candidate_pool_size=10, initial_seed_size=1)
        )
    )
    backend = Backend(
        backend_id="cpu",
        name="CPU",
        capacity_slots=2,
        dollar_per_second=0.0,
        startup_latency_seconds=0.0,
        runtime_multiplier_by_source={source_id: 1.0 for source_id in result.public_data.get_source_ids()},
    )
    candidate_id = result.public_data.active_pool_candidate_ids[0]

    class OneJobScheduler:
        name = "one_job"

        def reset(self, seed, initial_observations):
            pass

        def update(self, completed_jobs):
            pass

        def propose(self, state, available_slots):
            if state.current_time == 0.0 and not state.pending:
                return [ProposedJob(candidate_id, "def2tzvp", "cpu")]
            return []

    simulator = EventSimulator(
        replay_oracle=result.replay_oracle,
        runtime_model=result.runtime_model,
        sources=result.public_data.sources,
        backends={"cpu": backend},
    )
    sim_result = simulator.run(
        OneJobScheduler(),
        result.initial_observations,
        ExperimentConfig(horizon_seconds=10_000.0, max_target_queries=100, metadata={"seed": 0}),
    )

    completed = [obs for obs in sim_result.completions if obs.job.candidate_id == candidate_id]
    assert len(completed) == 1
    assert completed[0].runtime_seconds == result.replay_oracle.query_hidden(
        candidate_id, "def2tzvp"
    ).true_runtime_seconds

