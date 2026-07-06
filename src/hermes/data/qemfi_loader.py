"""Leakage-safe QeMFi loader."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hermes.core.replay_oracle import OracleRecord, ReplayOracle
from hermes.core.runtime_model import SourceMeanRuntimeModel
from hermes.core.types import Backend, Candidate, CompletedObservation, ProposedJob, Source
from hermes.data.qemfi_evaluator import QeMFiHiddenEvaluator
from hermes.data.qemfi_features import build_geometry_basic_features
from hermes.data.qemfi_schema import QeMFiDatasetConfig, QeMFiSourceSpec
from hermes.data.qemfi_splits import QeMFiSplit, build_qemfi_split


@dataclass(frozen=True, slots=True)
class QeMFiPublicData:
    """Scheduler-safe QeMFi data bundle."""

    dataset_name: str
    dataset_version: str
    target_source_id: str
    sources: dict[str, Source]
    candidates: tuple[Candidate, ...]
    candidate_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    split: QeMFiSplit
    public_runtime_summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    _features_by_candidate_id: Mapping[int, np.ndarray] = field(default_factory=dict, repr=False)

    def get_candidates(self, candidate_ids: list[int] | tuple[int, ...] | None = None) -> list[Candidate]:
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        selected_ids = self.candidate_ids if candidate_ids is None else candidate_ids
        return [by_id[int(candidate_id)] for candidate_id in selected_ids]

    def get_features(self, candidate_ids: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        return np.vstack([self._features_by_candidate_id[int(candidate_id)] for candidate_id in candidate_ids])

    def get_source(self, source_id: str) -> Source:
        return self.sources[source_id]

    def get_source_ids(self) -> list[str]:
        return list(self.sources)

    def num_candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def active_pool_candidate_ids(self) -> tuple[int, ...]:
        return self.split.active_pool_candidate_ids


@dataclass(frozen=True, slots=True)
class QeMFiLoadResult:
    public_data: QeMFiPublicData
    replay_oracle: ReplayOracle
    runtime_model: SourceMeanRuntimeModel
    initial_observations: list[CompletedObservation]
    evaluator: QeMFiHiddenEvaluator
    load_report: dict[str, Any]


class QeMFiReplayOracle:
    """Simulator-only QeMFi replay oracle."""

    def __init__(self, records: Mapping[tuple[int, str], OracleRecord]) -> None:
        self._records = dict(records)

    def query_hidden(self, candidate_id: int, source_id: str) -> OracleRecord:
        key = (candidate_id, source_id)
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"No QeMFi oracle record for {key!r}.") from exc


@dataclass(slots=True)
class _LoadedRows:
    candidate_ids: list[int]
    candidates: list[Candidate]
    features: np.ndarray
    feature_names: list[str]
    labels: dict[tuple[int, str], float]
    runtimes: dict[tuple[int, str], float]
    target_utilities: dict[int, float]
    source_mean_seconds: dict[str, float]
    candidate_id_by_raw_index: dict[int, int]


def load_qemfi_from_config(config: QeMFiDatasetConfig) -> QeMFiLoadResult:
    loaded = _load_rows(config)
    split = build_qemfi_split(
        candidate_ids=loaded.candidate_ids,
        molecule_names=config.molecule_names,
        source_ids=config.source_ids,
        target_source_id=config.target_source_id,
        spec=config.split_spec,
    )

    pool_set = set(split.pool_candidate_ids)
    pool_candidates = tuple(candidate for candidate in loaded.candidates if candidate.candidate_id in pool_set)
    features_by_id = {
        candidate.candidate_id: loaded.features[row_idx]
        for row_idx, candidate in enumerate(loaded.candidates)
        if candidate.candidate_id in pool_set
    }

    public_sources = {
        source.source_id: Source(
            source_id=source.source_id,
            name=source.name,
            is_target=source.source_id == config.target_source_id,
            nominal_cost_rank=source.nominal_cost_rank,
        )
        for source in config.sources
    }
    records = {
        pair: OracleRecord(y=label, true_runtime_seconds=loaded.runtimes[pair])
        for pair, label in loaded.labels.items()
        if pair[0] in pool_set
    }
    oracle = QeMFiReplayOracle(records)
    initial_observations = _build_initial_observations(split, oracle)
    evaluator = QeMFiHiddenEvaluator(
        _target_source_id=config.target_source_id,
        _target_utilities={
            candidate_id: value
            for candidate_id, value in loaded.target_utilities.items()
            if candidate_id in pool_set
        },
    )

    runtime_backend = Backend(
        backend_id="cpu",
        name="Neutral CPU",
        capacity_slots=1,
        dollar_per_second=0.0,
        startup_latency_seconds=0.0,
        runtime_multiplier_by_source={source_id: 1.0 for source_id in config.source_ids},
    )
    runtime_model = SourceMeanRuntimeModel(
        source_mean_runtime_seconds=loaded.source_mean_seconds,
        backends={"cpu": runtime_backend},
    )

    public_data = QeMFiPublicData(
        dataset_name=config.name,
        dataset_version=config.version,
        target_source_id=config.target_source_id,
        sources=public_sources,
        candidates=pool_candidates,
        candidate_ids=split.pool_candidate_ids,
        feature_names=tuple(loaded.feature_names),
        split=split,
        public_runtime_summary={
            "model": "source_mean",
            "source_mean_seconds": dict(loaded.source_mean_seconds),
        },
        metadata={
            "dataset_name": config.name,
            "dataset_version": config.version,
            "feature_mode": config.feature_spec.mode,
        },
        _features_by_candidate_id=features_by_id,
    )
    load_report = {
        "dataset_name": config.name,
        "dataset_version": config.version,
        "molecule_names": list(config.molecule_names),
        "source_ids": list(config.source_ids),
        "target_source_id": config.target_source_id,
        "eligible_candidate_count": len(loaded.candidate_ids),
        "candidate_pool_count": len(split.pool_candidate_ids),
        "initial_observation_count": len(initial_observations),
        "feature_dimension": len(loaded.feature_names),
        "candidate_id_by_raw_index": dict(loaded.candidate_id_by_raw_index),
    }
    return QeMFiLoadResult(
        public_data=public_data,
        replay_oracle=oracle,
        runtime_model=runtime_model,
        initial_observations=initial_observations,
        evaluator=evaluator,
        load_report=load_report,
    )


def _load_rows(config: QeMFiDatasetConfig) -> _LoadedRows:
    all_candidates: list[Candidate] = []
    all_feature_rows: list[np.ndarray] = []
    labels: dict[tuple[int, str], float] = {}
    runtimes: dict[tuple[int, str], float] = {}
    target_utilities: dict[int, float] = {}
    source_runtime_values: dict[str, list[float]] = {source_id: [] for source_id in config.source_ids}
    candidate_id_by_raw_index: dict[int, int] = {}
    feature_names: list[str] | None = None
    next_candidate_id = 0
    global_raw_index = 0

    for molecule_name in config.molecule_names:
        path = config.data_root / config.file_pattern.format(molecule=molecule_name)
        if not path.exists():
            raise FileNotFoundError(f"QeMFi file not found: {path}")
        with np.load(path, allow_pickle=True) as npz:
            arrays = {key: npz[key] for key in npz.files}

        property_values = _select_property_values(arrays, config)
        runtime_values = _select_runtime_values(arrays, config)
        coordinates = np.asarray(arrays[config.npz_keys["coordinates"]], dtype=float)
        atomic_numbers = np.asarray(arrays[config.npz_keys["atomic_numbers"]])
        raw_ids = np.asarray(arrays[config.npz_keys["id"]], dtype=object)
        conformations = np.asarray(arrays[config.npz_keys["conformation"]])

        features, names = build_geometry_basic_features(
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            max_atoms=config.feature_spec.max_atoms,
            normalize=config.feature_spec.normalize,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("QeMFi feature dimensions differ across files.")

        for row_idx in range(property_values.shape[0]):
            if not _row_is_eligible(property_values[row_idx], runtime_values[row_idx]):
                global_raw_index += 1
                continue

            candidate_id = next_candidate_id
            next_candidate_id += 1
            metadata = _safe_candidate_metadata(
                config=config,
                molecule_name=molecule_name,
                path=path,
                row_idx=row_idx,
                raw_id=raw_ids[row_idx],
                conformation_id=conformations[row_idx],
                atomic_numbers=atomic_numbers[row_idx],
            )
            candidate = Candidate(
                candidate_id=candidate_id,
                smiles=None,
                features=features[row_idx],
                metadata=metadata,
            )
            all_candidates.append(candidate)
            all_feature_rows.append(features[row_idx])
            candidate_id_by_raw_index[global_raw_index] = candidate_id

            for source in config.sources:
                value = float(property_values[row_idx, source.qemfi_index])
                utility = value if config.property_spec.utility_direction == "maximize" else -value
                runtime = float(runtime_values[row_idx, source.qemfi_index])
                pair = (candidate_id, source.source_id)
                labels[pair] = utility
                runtimes[pair] = runtime
                source_runtime_values[source.source_id].append(runtime)
                if source.source_id == config.target_source_id:
                    target_utilities[candidate_id] = utility

            global_raw_index += 1

    if feature_names is None:
        feature_names = []
    source_mean_seconds = {
        source_id: float(np.mean(values)) if values else 0.0
        for source_id, values in source_runtime_values.items()
    }
    return _LoadedRows(
        candidate_ids=[candidate.candidate_id for candidate in all_candidates],
        candidates=all_candidates,
        features=np.vstack(all_feature_rows) if all_feature_rows else np.empty((0, 0)),
        feature_names=feature_names,
        labels=labels,
        runtimes=runtimes,
        target_utilities=target_utilities,
        source_mean_seconds=source_mean_seconds,
        candidate_id_by_raw_index=candidate_id_by_raw_index,
    )


def _select_property_values(
    arrays: Mapping[str, np.ndarray],
    config: QeMFiDatasetConfig,
) -> np.ndarray:
    key = config.property_spec.key
    if key not in arrays:
        raise ValueError(f"QeMFi property key {key!r} not found. Available keys: {sorted(arrays)}")
    values = np.asarray(arrays[key], dtype=float)
    source_axis_length = values.shape[1] if values.ndim >= 2 else None
    max_source_index = max(source.qemfi_index for source in config.sources)
    if values.ndim < 2 or source_axis_length is None or source_axis_length <= max_source_index:
        raise ValueError("QeMFi property source axis does not cover configured sources.")

    if config.property_spec.component:
        index = (slice(None), slice(None), *config.property_spec.component)
        try:
            values = values[index]
        except IndexError as exc:
            raise ValueError("QeMFi property.component is out of range.") from exc
    if values.ndim != 2:
        raise ValueError("Selected QeMFi property must have shape (n_candidates, n_sources).")
    return values


def _select_runtime_values(
    arrays: Mapping[str, np.ndarray],
    config: QeMFiDatasetConfig,
) -> np.ndarray:
    runtime_key = config.runtime_spec.runtime_key
    if runtime_key is None:
        for candidate_key in config.runtime_spec.runtime_key_candidates:
            if candidate_key in arrays:
                runtime_key = candidate_key
                break
    if runtime_key is None or runtime_key not in arrays:
        if config.runtime_spec.mode == "synthetic_debug" and config.runtime_spec.synthetic_debug_allowed:
            n_candidates = np.asarray(arrays[config.property_spec.key]).shape[0]
            n_sources = max(source.qemfi_index for source in config.sources) + 1
            return np.tile(np.arange(1, n_sources + 1, dtype=float), (n_candidates, 1))
        raise ValueError(
            "runtime.mode=require_real but no runtime key was found. "
            f"Available keys: {sorted(arrays)}"
        )

    values = np.asarray(arrays[runtime_key], dtype=float)
    max_source_index = max(source.qemfi_index for source in config.sources)
    if values.ndim != 2 or values.shape[1] <= max_source_index:
        raise ValueError("QeMFi runtime source axis does not cover configured sources.")
    return values


def _row_is_eligible(property_row: np.ndarray, runtime_row: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(property_row)) and np.all(np.isfinite(runtime_row)))


def _safe_candidate_metadata(
    *,
    config: QeMFiDatasetConfig,
    molecule_name: str,
    path: Path,
    row_idx: int,
    raw_id: Any,
    conformation_id: Any,
    atomic_numbers: np.ndarray,
) -> dict[str, Any]:
    atoms = np.asarray(atomic_numbers)
    present = atoms[atoms > 0]
    atom_counts = {str(int(atom)): int(np.count_nonzero(present == atom)) for atom in sorted(set(present.tolist()))}
    return {
        "dataset_name": config.name,
        "molecule_name": molecule_name,
        "raw_file_name": path.name,
        "raw_index": int(row_idx),
        "raw_id": str(raw_id),
        "conformation_id": int(conformation_id),
        "num_atoms": int(len(present)),
        "formula_or_atom_counts": atom_counts,
    }


def _build_initial_observations(
    split: QeMFiSplit,
    oracle: QeMFiReplayOracle,
) -> list[CompletedObservation]:
    observations: list[CompletedObservation] = []
    for candidate_id, source_id in split.initial_pairs:
        record = oracle.query_hidden(candidate_id, source_id)
        observations.append(
            CompletedObservation(
                job=ProposedJob(candidate_id=candidate_id, source_id=source_id, backend_id="initial_seed"),
                start_time=0.0,
                finish_time=0.0,
                y=record.y,
                runtime_seconds=record.true_runtime_seconds,
                cost=0.0,
            )
        )
    return observations


def inspect_qemfi_config(config: QeMFiDatasetConfig) -> dict[str, Any]:
    result = load_qemfi_from_config(config)
    return dict(result.load_report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a QeMFi dataset config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if not args.inspect:
        raise SystemExit("--inspect is currently the only supported qemfi_loader command.")

    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        config = QeMFiDatasetConfig.from_dict(yaml.safe_load(handle))
    report = inspect_qemfi_config(config)
    for key, value in report.items():
        if key == "candidate_id_by_raw_index":
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


__all__ = [
    "QeMFiLoadResult",
    "QeMFiPublicData",
    "QeMFiReplayOracle",
    "inspect_qemfi_config",
    "load_qemfi_from_config",
]
