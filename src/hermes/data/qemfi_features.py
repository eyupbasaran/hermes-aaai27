"""Public structural feature builders for QeMFi."""

from __future__ import annotations

import numpy as np


DEFAULT_ATOM_HISTOGRAM = (1, 6, 7, 8, 9, 15, 16, 17)


def build_geometry_basic_features(
    *,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    max_atoms: int | str = "auto",
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build public features from geometry only."""

    z = np.asarray(atomic_numbers)
    r = np.asarray(coordinates, dtype=float)
    if z.ndim != 2:
        raise ValueError("atomic_numbers must have shape (n_candidates, n_atoms).")
    if r.ndim != 3 or r.shape[:2] != z.shape or r.shape[2] != 3:
        raise ValueError("coordinates must have shape (n_candidates, n_atoms, 3).")

    n_candidates, n_atoms = z.shape
    if max_atoms == "auto":
        padded_atoms = n_atoms
    else:
        padded_atoms = int(max_atoms)
    padded_atoms = max(padded_atoms, 1)

    rows: list[np.ndarray] = []
    for idx in range(n_candidates):
        zi = z[idx]
        ri = r[idx]
        present = zi > 0
        valid_z = zi[present]
        valid_r = ri[present]
        hist = np.array([np.count_nonzero(valid_z == atom) for atom in DEFAULT_ATOM_HISTOGRAM], dtype=float)
        num_atoms = float(len(valid_z))
        heavy_atoms = float(np.count_nonzero(valid_z > 1))

        if len(valid_r):
            centroid = valid_r.mean(axis=0)
            spread = valid_r.std(axis=0)
            distances = _pairwise_distances(valid_r)
            coulomb_eigs = _coulomb_eigenvalues(valid_z, valid_r, padded_atoms)
        else:
            centroid = np.zeros(3)
            spread = np.zeros(3)
            distances = np.zeros(0)
            coulomb_eigs = np.zeros(padded_atoms)

        if len(distances):
            distance_summary = np.array(
                [distances.min(), distances.max(), distances.mean(), distances.std()],
                dtype=float,
            )
        else:
            distance_summary = np.zeros(4)

        rows.append(
            np.concatenate(
                [
                    hist,
                    np.array([num_atoms, heavy_atoms], dtype=float),
                    centroid,
                    spread,
                    distance_summary,
                    coulomb_eigs,
                ]
            )
        )

    features = np.vstack(rows) if rows else np.empty((0, 0))
    names = (
        [f"atom_count_{atom}" for atom in DEFAULT_ATOM_HISTOGRAM]
        + ["num_atoms", "heavy_atom_count"]
        + ["centroid_x", "centroid_y", "centroid_z"]
        + ["spread_x", "spread_y", "spread_z"]
        + ["pair_distance_min", "pair_distance_max", "pair_distance_mean", "pair_distance_std"]
        + [f"coulomb_eig_{idx}" for idx in range(padded_atoms)]
    )

    if normalize and features.size:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0.0] = 1.0
        features = (features - mean) / std

    return features.astype(float, copy=False), names


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    distances: list[float] = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distances.append(float(np.linalg.norm(coordinates[i] - coordinates[j])))
    return np.array(distances, dtype=float)


def _coulomb_eigenvalues(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    padded_atoms: int,
) -> np.ndarray:
    n_atoms = len(atomic_numbers)
    matrix = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                matrix[i, j] = 0.5 * float(atomic_numbers[i]) ** 2.4
            else:
                distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
                matrix[i, j] = 0.0 if distance == 0.0 else float(atomic_numbers[i] * atomic_numbers[j]) / distance
    eigs = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    padded = np.zeros(padded_atoms, dtype=float)
    count = min(padded_atoms, len(eigs))
    padded[:count] = eigs[:count]
    return padded


__all__ = ["build_geometry_basic_features"]

