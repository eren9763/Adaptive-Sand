"""Model comparison experiment runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from adaptive_sand.callbacks.pruning import (
    SANDSmartHybridPruningCallback,
    SANDSmartPruningCallback,
)
from adaptive_sand.data.loader import load_dataset
from adaptive_sand.data.transforms import dataset_to_numpy, normalize_dataset
from adaptive_sand.experiments.train import fit_and_measure
from adaptive_sand.models.adaptive import SANDAdaptiveModel
from adaptive_sand.models.baselines import (
    build_lasso_mlp,
    build_original_sand_model,
    build_sequential_attention_model,
)
from adaptive_sand.models.hybrid import SANDAdaptiveModelHybrid
from adaptive_sand.utils.logging import append_result_row
from adaptive_sand.utils.plotting import plot_training_summary


AVAILABLE_MODELS = ["original", "seqatt", "adaptive", "hybrid", "lasso"]


def _prepare_y_for_lasso(y):
    if len(y.shape) > 1:
        return np.argmax(y, axis=1)
    return y


def _result_row(model_name, result, elapsed, peak_mem, selected_k):
    return {
        "model": model_name,
        "accuracy": result.get("accuracy", np.nan),
        "loss": result.get("loss", np.nan),
        "time_sec": elapsed,
        "ram_mb": peak_mem / 1e6,
        "selected_k": selected_k,
    }


def compare_models(
    dataset_name: str = "mnist",
    models: list[str] | None = None,
    epochs: int = 150,
    batch_size: int = 32,
    val_ratio: float = 0.125,
    data_dir: str | Path | None = None,
    initial_k: int | None = None,
    output_dir: str | Path = "outputs",
    skip_plots: bool = False,
    hidden_layers: list[int] | None = None,
    learning_rate: float = 1e-4,
    decay_steps: int = 250,
    decay_rate: float = 1.0,
    alpha: float = 0.0,
    batch_norm: bool = False,
    original_sand_k: int = 60,
    seqatt_k: int = 60,
    lasso_k: int = 60,
    pruning_config: dict | None = None,
):
    """Run selected models on a dataset and save a summary CSV."""
    models = models or ["hybrid"]
    if "all" in models:
        models = AVAILABLE_MODELS

    unknown_models = sorted(set(models) - set(AVAILABLE_MODELS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Available: {AVAILABLE_MODELS}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_layers = hidden_layers or [67]
    pruning_config = pruning_config or {}

    print("=" * 90)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Models : {', '.join(models)}")
    print("=" * 90)

    dataset = load_dataset(
        dataset_name,
        val_ratio=val_ratio,
        batch_size=batch_size,
        data_dir=data_dir,
    )
    ds_train = normalize_dataset(dataset["ds_train"]).cache()
    ds_val = normalize_dataset(dataset["ds_val"]).cache()
    ds_test = normalize_dataset(dataset["ds_test"]).cache()

    is_classification = dataset["is_classification"]
    num_classes = dataset["num_classes"]
    num_features = dataset["num_features"]

    if not is_classification:
        raise NotImplementedError("Current comparison runner is configured for classification.")

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metric_list = ["accuracy"]
    all_histories = {}
    callback_for_plot = None
    rows = []

    needs_numpy = any(model_name in models for model_name in ["seqatt", "lasso"])
    if needs_numpy:
        x_train_np, y_train_np = dataset_to_numpy(ds_train)
        x_val_np, y_val_np = dataset_to_numpy(ds_val)
        x_test_np, y_test_np = dataset_to_numpy(ds_test)
        y_train_labels = _prepare_y_for_lasso(y_train_np)
        y_val_labels = _prepare_y_for_lasso(y_val_np)
        y_test_labels = _prepare_y_for_lasso(y_test_np)
    else:
        x_train_np = y_train_np = x_val_np = y_val_np = x_test_np = y_test_np = None
        y_train_labels = y_val_labels = y_test_labels = None

    if "original" in models:
        print("\n[Original SAND]")
        model = build_original_sand_model(
            num_features=num_features,
            num_classes=num_classes,
            num_inputs_to_select=original_sand_k,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            alpha=alpha,
            batch_norm=batch_norm,
        )
        model.compile(optimizer=model.optimizer, loss=loss_fn, metrics=metric_list)
        history, elapsed, peak_mem = fit_and_measure(
            model, ds_train, ds_val, epochs=epochs, verbose=0
        )
        result = model.evaluate(ds_test, return_dict=True, verbose=0)
        all_histories["Original SAND"] = history.history
        rows.append(_result_row("original", result, elapsed, peak_mem, original_sand_k))

    if "seqatt" in models:
        print("\n[Sequential Attention]")
        steps_per_epoch = int(np.ceil(len(x_train_np) / batch_size))
        num_train_steps = steps_per_epoch * epochs
        model = build_sequential_attention_model(
            num_features=num_features,
            num_classes=num_classes,
            num_inputs_to_select=seqatt_k,
            num_train_steps=num_train_steps,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            alpha=alpha,
            batch_norm=batch_norm,
        )
        model.compile(optimizer=model.optimizer, loss=loss_fn, metrics=metric_list)
        history, elapsed, peak_mem = fit_and_measure(
            model, ds_train, ds_val, epochs=epochs, verbose=0
        )
        result = model.evaluate(ds_test, return_dict=True, verbose=0)
        all_histories["Sequential Attention"] = history.history
        rows.append(_result_row("seqatt", result, elapsed, peak_mem, seqatt_k))

    if "adaptive" in models:
        print("\n[Adaptive SAND]")
        model = SANDAdaptiveModel(
            num_inputs=num_features,
            num_outputs=num_classes,
            initial_k=initial_k or num_features,
            layer_sequence=hidden_layers,
            is_classification=True,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            alpha=alpha,
            batch_norm=batch_norm,
        )
        model.compile(loss=loss_fn, metrics=metric_list)
        callback = SANDSmartPruningCallback(**pruning_config)
        history, elapsed, peak_mem = fit_and_measure(
            model,
            ds_train,
            ds_val,
            epochs=epochs,
            callbacks=[callback],
            verbose=1,
        )
        model.selector.restore_best()
        result = model.evaluate(ds_test, return_dict=True, verbose=0)
        selected_k = model.get_feature_importance()["num_selected"]
        all_histories["Adaptive SAND"] = history.history
        rows.append(_result_row("adaptive", result, elapsed, peak_mem, selected_k))
        callback_for_plot = callback

    if "hybrid" in models:
        print("\n[Hybrid Adaptive SAND]")
        model = SANDAdaptiveModelHybrid(
            num_inputs=num_features,
            num_outputs=num_classes,
            initial_k=initial_k or num_features,
            layer_sequence=hidden_layers,
            is_classification=True,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            alpha=alpha,
            batch_norm=batch_norm,
        )
        model.compile(loss=loss_fn, metrics=metric_list)
        callback = SANDSmartHybridPruningCallback(**pruning_config)
        history, elapsed, peak_mem = fit_and_measure(
            model,
            ds_train,
            ds_val,
            epochs=epochs,
            callbacks=[callback],
            verbose=1,
        )
        model.selector.restore_best()
        result = model.evaluate(ds_test, return_dict=True, verbose=0)
        selected_k = model.get_feature_importance()["num_selected"]
        all_histories["Hybrid Adaptive SAND"] = history.history
        rows.append(_result_row("hybrid", result, elapsed, peak_mem, selected_k))
        callback_for_plot = callback

    if "lasso" in models:
        print("\n[LASSO + MLP]")
        lasso = Lasso(alpha=0.001, max_iter=2000)
        lasso.fit(x_train_np, y_train_labels)

        selector = SelectFromModel(lasso, max_features=lasso_k, prefit=True)
        x_train_sel = selector.transform(x_train_np)
        x_val_sel = selector.transform(x_val_np)
        x_test_sel = selector.transform(x_test_np)
        selected_k = x_train_sel.shape[1]

        model = build_lasso_mlp(
            input_dim=selected_k,
            num_classes=num_classes,
            hidden_dim=hidden_layers[0],
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss_fn,
            metrics=metric_list,
        )

        y_train_cat = tf.keras.utils.to_categorical(y_train_labels, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val_labels, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_labels, num_classes)

        history, elapsed, peak_mem = fit_and_measure(
            model,
            x_train_sel,
            (x_val_sel, y_val_cat),
            epochs=epochs,
            train_targets=y_train_cat,
            callbacks=[],
            verbose=0,
        )
        result = model.evaluate(x_test_sel, y_test_cat, return_dict=True, verbose=0)
        all_histories["LASSO + MLP"] = history.history
        rows.append(_result_row("lasso", result, elapsed, peak_mem, selected_k))

    csv_path = output_dir / "results_summary.csv"
    for row in rows:
        append_result_row(csv_path, {"dataset": dataset_name, **row})

    print("\n" + "=" * 90)
    print("Final comparison report")
    print("=" * 90)
    print(f"{'Model':<20} | {'Acc':<10} | {'Time':<10} | {'RAM (MB)':<10} | {'k'}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['model']:<20} | "
            f"{row['accuracy']:.4f} | "
            f"{row['time_sec']:.2f} | "
            f"{row['ram_mb']:.2f} | "
            f"{row['selected_k']}"
        )

    if not skip_plots and all_histories:
        plot_path = output_dir / "training_summary.png"
        plot_training_summary(all_histories, callback_for_plot, save_path=plot_path)

    print(f"\nSaved results to: {csv_path}")
    return rows, all_histories
