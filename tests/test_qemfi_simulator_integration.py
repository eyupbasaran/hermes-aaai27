from __future__ import annotations

from hermes.core.event_simulator import EventSimulator, ExperimentConfig
from hermes.core.types import Backend, ProposedJob
from hermes.data.qemfi_loader import load_qemfi_from_config
from hermes.data.qemfi_schema import QeMFiDatasetConfig

from .conftest import assert_no_scheduler_secrets
from .qemfi_test_utils import (
    assert_public_qemfi_has_no_secrets,
    synthetic_qemfi_config_dict,
    write_synthetic_qemfi_npz,
)


def test_qemfi_load_result_integrates_with_event_simulator_without_scheduler_secrets(tmp_path) -> None:
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

    class PublicDataScheduler:
        name = "public_data_scheduler"

        def __init__(self, public_data, runtime_model):
            assert_public_qemfi_has_no_secrets(public_data)
            self.public_data = public_data
            self.runtime_model = runtime_model
            self.launched = False

        def reset(self, seed, initial_observations):
            assert_no_scheduler_secrets(initial_observations)

        def update(self, completed_jobs):
            assert_no_scheduler_secrets(completed_jobs)

        def propose(self, state, available_slots):
            assert_no_scheduler_secrets(state)
            assert_no_scheduler_secrets(available_slots)
            assert_public_qemfi_has_no_secrets(self.public_data)
            if not self.launched:
                self.launched = True
                return [ProposedJob(candidate_id, "def2tzvp", "cpu")]
            return []

    simulator = EventSimulator(
        replay_oracle=result.replay_oracle,
        runtime_model=result.runtime_model,
        sources=result.public_data.sources,
        backends={"cpu": backend},
    )
    sim_result = simulator.run(
        PublicDataScheduler(result.public_data, result.runtime_model),
        result.initial_observations,
        ExperimentConfig(horizon_seconds=10_000.0, max_target_queries=100, metadata={"seed": 0}),
    )

    metrics = result.evaluator.compute_final_metrics(sim_result.completions)
    assert metrics["target_query_count"] >= 1

