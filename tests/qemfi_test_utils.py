from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_IDS = ["sto3g", "321g", "631g", "def2svp", "def2tzvp"]
TARGET_SOURCE_ID = "def2tzvp"


def write_synthetic_qemfi_npz(
    root: Path,
    *,
    molecule: str = "urea",
    n_candidates: int = 20,
    n_sources: int = 5,
    permute_labels: bool = False,
    include_runtime: bool = True,
    property_key: str = "fosc",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"QeMFi_{molecule}.npz"
    rng = np.random.default_rng(123 if not permute_labels else 999)

    ids = np.array([f"{molecule}_{idx}" for idx in range(n_candidates)], dtype=object)
    conformations = np.arange(n_candidates, dtype=np.int64)
    atomic_numbers = np.tile(np.array([6, 1, 1, 8], dtype=np.int64), (n_candidates, 1))
    coordinates = np.zeros((n_candidates, 4, 3), dtype=float)
    for idx in range(n_candidates):
        coordinates[idx, :, 0] = np.array([0.0, 1.0, 0.0, 0.5]) + idx * 0.01
        coordinates[idx, :, 1] = np.array([0.0, 0.0, 1.0, 0.5])
        coordinates[idx, :, 2] = np.array([0.0, 0.0, 0.0, 1.0])

    labels = np.zeros((n_candidates, n_sources, 2), dtype=float)
    candidate_order = np.arange(n_candidates)
    if permute_labels:
        rng.shuffle(candidate_order)
    for raw_idx, label_idx in enumerate(candidate_order):
        for source_idx in range(n_sources):
            labels[raw_idx, source_idx, 0] = label_idx * 10.0 + source_idx
            labels[raw_idx, source_idx, 1] = -1000.0 - label_idx - source_idx

    payload: dict[str, Any] = {
        "ID": ids,
        "R": coordinates,
        "Z": atomic_numbers,
        "CONF": conformations,
        property_key: labels,
    }
    if include_runtime:
        runtime = np.zeros((n_candidates, n_sources), dtype=float)
        for idx in range(n_candidates):
            for source_idx in range(n_sources):
                runtime[idx, source_idx] = (idx + 1) * (source_idx + 1)
        payload["runtime"] = runtime

    np.savez(path, **payload)
    return path


def synthetic_qemfi_config_dict(
    root: Path,
    *,
    candidate_pool_size: int = 10,
    initial_seed_size: int = 2,
    split_seed: int = 0,
    include_runtime_key: bool = True,
    target_source_id: str = TARGET_SOURCE_ID,
    property_key: str | None = "fosc",
) -> dict[str, Any]:
    runtime_key = "runtime" if include_runtime_key else None
    return {
        "dataset": {
            "name": "qemfi",
            "version": "synthetic_test",
            "data_root": str(root),
            "cache_root": str(root / "cache"),
            "raw": {
                "file_pattern": "QeMFi_{molecule}.npz",
                "molecules": ["urea"],
            },
            "npz_keys": {
                "id": "ID",
                "coordinates": "R",
                "atomic_numbers": "Z",
                "conformation": "CONF",
            },
            "sources": [
                {
                    "source_id": "sto3g",
                    "name": "STO-3G",
                    "qemfi_index": 0,
                    "is_target": False,
                    "nominal_cost_rank": 1,
                },
                {
                    "source_id": "321g",
                    "name": "3-21G",
                    "qemfi_index": 1,
                    "is_target": False,
                    "nominal_cost_rank": 2,
                },
                {
                    "source_id": "631g",
                    "name": "6-31G",
                    "qemfi_index": 2,
                    "is_target": False,
                    "nominal_cost_rank": 3,
                },
                {
                    "source_id": "def2svp",
                    "name": "def2-SVP",
                    "qemfi_index": 3,
                    "is_target": False,
                    "nominal_cost_rank": 4,
                },
                {
                    "source_id": "def2tzvp",
                    "name": "def2-TZVP",
                    "qemfi_index": 4,
                    "is_target": target_source_id == "def2tzvp",
                    "nominal_cost_rank": 5,
                },
            ],
            "target_source": target_source_id,
            "property": {
                "key": property_key,
                "component": [0],
                "utility_direction": "maximize",
                "allow_nan_policy": "drop_candidate_if_any_required_source_missing",
            },
            "runtime": {
                "mode": "require_real",
                "runtime_key": runtime_key,
                "runtime_key_candidates": ["runtime", "runtimes"],
                "public_runtime_model": "source_mean",
                "candidate_specific_public_runtime": False,
                "synthetic_debug_allowed": False,
            },
            "features": {
                "mode": "geometry_basic_v1",
                "max_atoms": "auto",
                "normalize": True,
                "cache_version": "v1",
            },
            "split": {
                "split_seed": split_seed,
                "candidate_pool_size": candidate_pool_size,
                "initial_seed_size": initial_seed_size,
                "initial_source_policy": "all_sources_for_seed_candidates",
                "require_all_sources": True,
                "candidate_id_policy": "stable_molecule_row_id",
            },
            "leakage": {
                "forbid_label_like_candidate_metadata": True,
                "forbid_runtime_like_candidate_metadata": True,
                "forbid_rank_like_candidate_metadata": True,
                "expose_source_availability": False,
            },
        }
    }


def assert_public_qemfi_has_no_secrets(obj: Any) -> None:
    seen: set[int] = set()
    forbidden_fragments = {
        "label",
        "labels",
        "target_y",
        "raw_y",
        "true_runtime",
        "time_seconds",
        "rank",
        "topk",
        "top_k",
        "score",
        "utility",
        "oracle",
        "hidden",
        "evaluator",
    }
    allowed_names = {"target_source_id", "public_runtime_summary", "nominal_cost_rank"}

    def check_name(name: str, path: str) -> None:
        lowered = name.lower()
        if lowered in allowed_names:
            return
        if lowered == "y":
            raise AssertionError(f"Forbidden public QeMFi name at {path}: {name!r}")
        for fragment in forbidden_fragments:
            if fragment in lowered:
                raise AssertionError(f"Forbidden public QeMFi name at {path}: {name!r}")

    def visit(value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(value, "query_hidden"):
            raise AssertionError(f"Public QeMFi object exposes query_hidden at {path}")

        if isinstance(value, dict):
            for key, item in value.items():
                check_name(str(key), f"{path}[{key!r}]")
                visit(item, f"{path}[{key!r}]")
            return
        if isinstance(value, np.ndarray):
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for idx, item in enumerate(value):
                visit(item, f"{path}[{idx}]")
            return
        if is_dataclass(value):
            for field in fields(value):
                check_name(field.name, f"{path}.{field.name}")
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return

    visit(obj, "root")
