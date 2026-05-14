"""Dataset loader wrapper.

The project keeps compatibility with the original SAND dataset interface while
forcing data files to be read from this repository's ``datasets/`` directory by
default.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the dataset directory.

    Priority:
    1. CLI/config ``data_dir`` argument
    2. Existing ``SAND_DATA_DIR`` environment variable
    3. Repository-local ``datasets/`` directory
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    env_data_dir = os.environ.get("SAND_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).expanduser().resolve()

    return get_project_root() / "datasets"


def _ensure_sand_dataset_alias(dataset_name: str, data_dir: Path) -> None:
    """Make repo-local datasets visible to legacy SAND loaders.

    Some original SAND loaders build file paths relative to
    ``sand/experiments/datasets`` and ignore ``SAND_DATA_DIR``. For example, the
    MICE loader may look for:

    ``sand/experiments/datasets/mice/Data_Cortex_Nuclear.csv``

    while this project keeps files under:

    ``datasets/mice/Data_Cortex_Nuclear.csv``

    To keep the upstream SAND code unchanged, this helper creates a directory
    alias from the legacy location to the repo-local dataset folder. It first
    tries a symlink. If symlink creation is not allowed on Windows, it falls
    back to copying the dataset directory.
    """
    source_dir = data_dir / dataset_name
    if not source_dir.exists():
        return

    try:
        import sand.experiments.datasets.data_loader as sand_data_loader
    except ImportError:
        return

    legacy_datasets_dir = Path(sand_data_loader.__file__).resolve().parent
    target_dir = legacy_datasets_dir / dataset_name

    if target_dir.exists():
        return

    try:
        target_dir.symlink_to(source_dir, target_is_directory=True)
        print(f"Dataset alias created: {target_dir} -> {source_dir}")
    except OSError:
        shutil.copytree(source_dir, target_dir)
        print(f"Dataset copied for legacy SAND loader: {source_dir} -> {target_dir}")


def load_dataset(
    name: str,
    val_ratio: float = 0.125,
    batch_size: int = 32,
    data_dir: str | Path | None = None,
) -> dict:
    """Load a dataset through `sand.experiments.datasets.dataset.get_dataset`.

    The original SAND loader often checks ``SAND_DATA_DIR`` internally. We set
    that variable here before importing/calling the external loader so the data
    can live under this repository's ``datasets/`` folder.
    """
    resolved_data_dir = resolve_data_dir(data_dir)
    os.environ["SAND_DATA_DIR"] = str(resolved_data_dir)

    try:
        from sand.experiments.datasets.dataset import get_dataset
    except ImportError as exc:
        raise ImportError(
            "Could not import `sand.experiments.datasets.dataset.get_dataset`. "
            "Place the original `sand/` directory in the repository root or install it "
            "as a package."
        ) from exc

    _ensure_sand_dataset_alias(name, resolved_data_dir)
    dataset = get_dataset(name, val_ratio, batch_size)
    required_keys = {
        "ds_train",
        "ds_val",
        "ds_test",
        "num_classes",
        "num_features",
        "is_classification",
    }
    missing = required_keys - set(dataset.keys())
    if missing:
        raise KeyError(f"Dataset `{name}` is missing keys: {sorted(missing)}")
    return dataset
