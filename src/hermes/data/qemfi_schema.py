"""Validated QeMFi dataset configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class QeMFiSourceSpec:
    source_id: str
    name: str
    qemfi_index: int
    is_target: bool
    nominal_cost_rank: int


@dataclass(frozen=True, slots=True)
class QeMFiPropertySpec:
    key: str
    component: tuple[int, ...]
    utility_direction: str
    allow_nan_policy: str


@dataclass(frozen=True, slots=True)
class QeMFiFeatureSpec:
    mode: str
    max_atoms: int | str
    normalize: bool
    cache_version: str


@dataclass(frozen=True, slots=True)
class QeMFiRuntimeSpec:
    mode: str
    runtime_key: str | None
    runtime_key_candidates: tuple[str, ...]
    public_runtime_model: str
    candidate_specific_public_runtime: bool
    synthetic_debug_allowed: bool
    fallback_source_means: dict[str, float]


@dataclass(frozen=True, slots=True)
class QeMFiSplitSpec:
    split_seed: int
    candidate_pool_size: int
    initial_seed_size: int
    initial_source_policy: str
    require_all_sources: bool
    candidate_id_policy: str


@dataclass(frozen=True, slots=True)
class QeMFiLeakagePolicy:
    forbid_label_like_candidate_metadata: bool
    forbid_runtime_like_candidate_metadata: bool
    forbid_rank_like_candidate_metadata: bool
    expose_source_availability: bool


@dataclass(frozen=True, slots=True)
class QeMFiDatasetConfig:
    name: str
    version: str
    data_root: Path
    cache_root: Path
    molecule_names: tuple[str, ...]
    file_pattern: str
    npz_keys: dict[str, str]
    sources: tuple[QeMFiSourceSpec, ...]
    target_source_id: str
    property_spec: QeMFiPropertySpec
    feature_spec: QeMFiFeatureSpec
    runtime_spec: QeMFiRuntimeSpec
    split_spec: QeMFiSplitSpec
    leakage_policy: QeMFiLeakagePolicy

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QeMFiDatasetConfig":
        data = raw.get("dataset", raw)
        if "property_name" in data and data.get("property_name") is None:
            raise ValueError("QeMFi property.key is required; property_name may not be null.")

        raw_block = data.get("raw", {})
        prop_block = data.get("property", {})
        runtime_block = data.get("runtime", {})
        feature_block = data.get("features", {})
        split_block = data.get("split", {})
        leakage_block = data.get("leakage", {})

        property_key = prop_block.get("key")
        if not property_key:
            raise ValueError("QeMFi property.key is required.")

        sources = tuple(
            QeMFiSourceSpec(
                source_id=str(source["source_id"]),
                name=str(source["name"]),
                qemfi_index=int(source.get("qemfi_index", idx)),
                is_target=bool(source.get("is_target", False)),
                nominal_cost_rank=int(source.get("nominal_cost_rank", idx + 1)),
            )
            for idx, source in enumerate(data.get("sources", ()))
        )
        if not sources:
            raise ValueError("QeMFi config must define sources.")

        target_source_id = str(data.get("target_source", data.get("target_source_id", "")))
        source_ids = {source.source_id for source in sources}
        if target_source_id not in source_ids:
            raise ValueError(f"Configured target source {target_source_id!r} is not in sources.")

        target_specs = [source for source in sources if source.is_target]
        if len(target_specs) != 1:
            raise ValueError("QeMFi config must mark exactly one source as target.")
        if target_specs[0].source_id != target_source_id:
            raise ValueError("Marked target source must match target_source.")

        leakage_policy = QeMFiLeakagePolicy(
            forbid_label_like_candidate_metadata=bool(
                leakage_block.get("forbid_label_like_candidate_metadata", True)
            ),
            forbid_runtime_like_candidate_metadata=bool(
                leakage_block.get("forbid_runtime_like_candidate_metadata", True)
            ),
            forbid_rank_like_candidate_metadata=bool(
                leakage_block.get("forbid_rank_like_candidate_metadata", True)
            ),
            expose_source_availability=bool(leakage_block.get("expose_source_availability", False)),
        )

        runtime_spec = QeMFiRuntimeSpec(
            mode=str(runtime_block.get("mode", "require_real")),
            runtime_key=runtime_block.get("runtime_key"),
            runtime_key_candidates=tuple(runtime_block.get("runtime_key_candidates", ())),
            public_runtime_model=str(runtime_block.get("public_runtime_model", "source_mean")),
            candidate_specific_public_runtime=bool(
                runtime_block.get("candidate_specific_public_runtime", False)
            ),
            synthetic_debug_allowed=bool(runtime_block.get("synthetic_debug_allowed", False)),
            fallback_source_means=dict(runtime_block.get("fallback_source_means", {})),
        )
        if (
            runtime_spec.candidate_specific_public_runtime
            and leakage_policy.forbid_runtime_like_candidate_metadata
        ):
            raise ValueError("candidate-specific public runtime is forbidden in strict leakage mode.")
        if runtime_spec.public_runtime_model != "source_mean":
            raise ValueError("Step 4 supports only source_mean public runtime models.")

        split_spec = QeMFiSplitSpec(
            split_seed=int(split_block.get("split_seed", 0)),
            candidate_pool_size=int(split_block["candidate_pool_size"]),
            initial_seed_size=int(split_block["initial_seed_size"]),
            initial_source_policy=str(
                split_block.get("initial_source_policy", "all_sources_for_seed_candidates")
            ),
            require_all_sources=bool(split_block.get("require_all_sources", True)),
            candidate_id_policy=str(split_block.get("candidate_id_policy", "stable_molecule_row_id")),
        )
        if split_spec.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive.")
        if split_spec.initial_seed_size < 0:
            raise ValueError("initial_seed_size must be non-negative.")
        if split_spec.initial_source_policy != "all_sources_for_seed_candidates":
            raise ValueError("Step 4 supports only all_sources_for_seed_candidates.")

        return cls(
            name=str(data.get("name", "qemfi")),
            version=str(data.get("version", "unknown")),
            data_root=Path(data["data_root"]),
            cache_root=Path(data.get("cache_root", "data/cache/qemfi")),
            molecule_names=tuple(raw_block.get("molecules", ())),
            file_pattern=str(raw_block.get("file_pattern", "QeMFi_{molecule}.npz")),
            npz_keys=dict(
                data.get(
                    "npz_keys",
                    {"id": "ID", "coordinates": "R", "atomic_numbers": "Z", "conformation": "CONF"},
                )
            ),
            sources=sources,
            target_source_id=target_source_id,
            property_spec=QeMFiPropertySpec(
                key=str(property_key),
                component=tuple(int(idx) for idx in prop_block.get("component", ())),
                utility_direction=str(prop_block.get("utility_direction", "maximize")),
                allow_nan_policy=str(
                    prop_block.get(
                        "allow_nan_policy", "drop_candidate_if_any_required_source_missing"
                    )
                ),
            ),
            feature_spec=QeMFiFeatureSpec(
                mode=str(feature_block.get("mode", "geometry_basic_v1")),
                max_atoms=feature_block.get("max_atoms", "auto"),
                normalize=bool(feature_block.get("normalize", True)),
                cache_version=str(feature_block.get("cache_version", "v1")),
            ),
            runtime_spec=runtime_spec,
            split_spec=split_spec,
            leakage_policy=leakage_policy,
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)


__all__ = [
    "QeMFiDatasetConfig",
    "QeMFiFeatureSpec",
    "QeMFiLeakagePolicy",
    "QeMFiPropertySpec",
    "QeMFiRuntimeSpec",
    "QeMFiSourceSpec",
    "QeMFiSplitSpec",
]

