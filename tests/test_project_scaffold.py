from pathlib import Path

import hermes


def test_package_imports() -> None:
    assert hermes.__version__ == "0.0.0"


def test_required_scaffold_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "pyproject.toml",
        "environment.yml",
        "configs/dataset/qemfi.yaml",
        "configs/dataset/qemfi_debug.yaml",
        "configs/backend/homogeneous.yaml",
        "configs/backend/heterogeneous_small.yaml",
        "configs/backend/heterogeneous_cloud.yaml",
        "configs/experiment/qemfi_m0_debug.yaml",
        "configs/experiment/qemfi_m0_main.yaml",
        "results/.gitkeep",
    ]
    missing = [path for path in required if not (root / path).exists()]
    assert not missing

