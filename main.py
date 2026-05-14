"""Command-line entry point for Adaptive SAND experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from adaptive_sand.experiments.compare import AVAILABLE_MODELS, compare_models
from adaptive_sand.utils.reproducibility import set_seed


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Adaptive SAND experiments.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=AVAILABLE_MODELS + ["all"],
        help="Model to run.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset root directory. Defaults to ./datasets.",
    )
    parser.add_argument("--initial-k", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def get_value(args, config: dict, key: str, default):
    value = getattr(args, key.replace("-", "_"), None)
    if value is not None:
        return value
    return config.get(key.replace("-", "_"), default)


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset = get_value(args, config, "dataset", "mnist")
    model = get_value(args, config, "model", "hybrid")
    epochs = get_value(args, config, "epochs", 150)
    batch_size = get_value(args, config, "batch-size", 32)
    val_ratio = get_value(args, config, "val-ratio", 0.125)
    data_dir = get_value(args, config, "data-dir", None)
    initial_k = get_value(args, config, "initial-k", None)
    output_dir = get_value(args, config, "output-dir", "outputs")
    seed = get_value(args, config, "seed", 42)
    skip_plots = args.skip_plots or bool(config.get("skip_plots", False))

    model_params = config.get("model_params", {})
    pruning_config = config.get("pruning", {})

    hidden_layers = model_params.get("hidden_layers", [67])
    selected_models = AVAILABLE_MODELS if model == "all" else [model]

    set_seed(seed)

    compare_models(
        dataset_name=dataset,
        models=selected_models,
        epochs=epochs,
        batch_size=batch_size,
        val_ratio=val_ratio,
        data_dir=data_dir,
        initial_k=initial_k,
        output_dir=output_dir,
        skip_plots=skip_plots,
        hidden_layers=hidden_layers,
        learning_rate=model_params.get("learning_rate", 1e-4),
        decay_steps=model_params.get("decay_steps", 250),
        decay_rate=model_params.get("decay_rate", 1.0),
        alpha=model_params.get("alpha", 0.0),
        batch_norm=model_params.get("batch_norm", False),
        original_sand_k=model_params.get("original_sand_k", 60),
        seqatt_k=model_params.get("seqatt_k", 60),
        lasso_k=model_params.get("lasso_k", 60),
        pruning_config=pruning_config,
    )


if __name__ == "__main__":
    main()
