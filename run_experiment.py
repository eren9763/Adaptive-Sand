"""
run_experiment.py — Main entry point for A-SAND / A-SAND-Hybrid experiments.

Usage (local or Colab terminal)
--------------------------------
    python run_experiment.py --dataset mnist --epochs 150 --model both
    python run_experiment.py --dataset isolet --epochs 200 --model asand
    python run_experiment.py --dataset mice   --epochs 150 --model hybrid

For Google Colab usage, see notebooks/colab_run.ipynb.
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

# Allow running from repo root without install
sys.path.insert(0, os.path.dirname(__file__))

from adaptive_sand import (
    SANDAdaptiveModel,
    SANDAdaptiveModelHybrid,
    SANDSmartPruningCallback,
    SANDSmartHybridPruningCallback,
    set_seeds,
    reset_seeds,
    dataset_to_numpy,
    load_prepared_dataset,
    build_classifier_mlp,
)

# ============================================================
# Defaults
# ============================================================

DEFAULT_EPOCHS = 150
DEFAULT_FIXED_K = 60
DEFAULT_LAYER_SEQUENCE = (67,)
DEFAULT_HYBRID_ALPHA = 0.7
DEFAULT_DATASET = "mnist"


# ============================================================
# Memory utilities
# ============================================================

_process = psutil.Process(os.getpid())


def reset_memory_counters():
    gc.collect()
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (ValueError, tf.errors.InvalidArgumentError):
        pass


def get_memory_metrics() -> dict:
    rss_mb = _process.memory_info().rss / (1024 ** 2)
    try:
        gpu_info = tf.config.experimental.get_memory_info("GPU:0")
        gpu_current_mb = gpu_info["current"] / (1024 ** 2)
        gpu_peak_mb = gpu_info["peak"] / (1024 ** 2)
    except (ValueError, tf.errors.InvalidArgumentError):
        gpu_current_mb = np.nan
        gpu_peak_mb = np.nan
    return {
        "gpu_current_mb": gpu_current_mb,
        "gpu_peak_mb": gpu_peak_mb,
        "rss_mb": rss_mb,
    }


# ============================================================
# Per-model runners
# ============================================================

def run_baseline(ds_train, ds_val, ds_test, num_features, num_classes, epochs, seed):
    """Train a plain MLP baseline (no feature selection)."""
    reset_seeds(seed)
    reset_memory_counters()
    tf.keras.backend.clear_session()

    model = build_classifier_mlp(num_features, num_classes, seed=seed)
    t0 = time.time()
    model.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose=0)
    elapsed = time.time() - t0

    res = model.evaluate(ds_test, return_dict=True, verbose=0)
    mem = get_memory_metrics()

    return {
        "model": "Baseline-MLP",
        "test_accuracy": res["accuracy"],
        "final_k": num_features,
        "k_ratio": 1.0,
        "n_prunes": 0,
        "time_sec": elapsed,
        **mem,
    }


def run_asand(
    ds_train, ds_val, ds_test,
    num_features, num_classes,
    epochs, seed,
    fixed_k=DEFAULT_FIXED_K,
    layer_sequence=DEFAULT_LAYER_SEQUENCE,
    prune_rate=0.10,
    lambda_k=0.15,
    cooldown=3,
    plateau_tol=0.003,
):
    """Train A-SAND (first-order pruning)."""
    reset_seeds(seed)
    reset_memory_counters()
    tf.keras.backend.clear_session()

    model = SANDAdaptiveModel(
        num_inputs=num_features,
        num_outputs=num_classes,
        initial_k=num_features,
        layer_sequence=layer_sequence,
        is_classification=True,
        batch_norm=False,
    )
    model.compile(metrics=["accuracy"])

    cb = SANDSmartPruningCallback(
        prune_rate=prune_rate,
        lambda_k=lambda_k,
        cooldown=cooldown,
        plateau_tol=plateau_tol,
    )

    t0 = time.time()
    model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[cb], verbose=0)
    elapsed = time.time() - t0

    res = model.evaluate(ds_test, return_dict=True, verbose=0)
    final_k = model.get_feature_importance()["num_selected"]
    mem = get_memory_metrics()

    return {
        "model": "A-SAND",
        "test_accuracy": res["accuracy"],
        "final_k": final_k,
        "k_ratio": final_k / num_features,
        "n_prunes": len(cb.prune_epochs),
        "time_sec": elapsed,
        **mem,
    }


def run_hybrid(
    ds_train, ds_val, ds_test,
    num_features, num_classes,
    epochs, seed,
    fixed_k=DEFAULT_FIXED_K,
    layer_sequence=DEFAULT_LAYER_SEQUENCE,
    hybrid_alpha=DEFAULT_HYBRID_ALPHA,
    prune_rate=0.10,
    lambda_k=0.15,
    cooldown=3,
    plateau_tol=0.003,
):
    """Train A-SAND-Hybrid (first + second-order pruning)."""
    reset_seeds(seed)
    reset_memory_counters()
    tf.keras.backend.clear_session()

    model = SANDAdaptiveModelHybrid(
        num_inputs=num_features,
        num_outputs=num_classes,
        initial_k=num_features,
        layer_sequence=layer_sequence,
        is_classification=True,
        batch_norm=False,
    )
    model.compile()

    cb = SANDSmartHybridPruningCallback(
        hybrid_alpha=hybrid_alpha,
        prune_rate=prune_rate,
        lambda_k=lambda_k,
        cooldown=cooldown,
        plateau_tol=plateau_tol,
    )

    t0 = time.time()
    model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[cb], verbose=0)
    elapsed = time.time() - t0

    res = model.evaluate(ds_test, return_dict=True, verbose=0)
    final_k = model.get_feature_importance()["num_selected"]
    mem = get_memory_metrics()

    return {
        "model": "A-SAND-Hybrid",
        "test_accuracy": res["accuracy"],
        "final_k": final_k,
        "k_ratio": final_k / num_features,
        "n_prunes": len(cb.prune_epochs),
        "time_sec": elapsed,
        **mem,
    }


# ============================================================
# Main runner
# ============================================================

def run_all(
    dataset_name: str,
    epochs: int = DEFAULT_EPOCHS,
    fixed_k: int = DEFAULT_FIXED_K,
    model_choice: str = "both",
    seed: int = 42,
    results_dir: str = "results",
):
    set_seeds(seed)

    print("=" * 90)
    print(f"DATASET : {dataset_name.upper()}")
    print(f"EPOCHS  : {epochs}   SEED : {seed}   MODEL : {model_choice}")
    print("=" * 90)

    data = load_prepared_dataset(dataset_name, seed=seed)
    ds_train = data["ds_train"]
    ds_val   = data["ds_val"]
    ds_test  = data["ds_test"]
    num_features = data["num_features"]
    num_classes  = data["num_classes"]

    rows = []

    if model_choice in ("baseline", "all"):
        print("\n[1/3] Baseline MLP")
        rows.append(run_baseline(ds_train, ds_val, ds_test, num_features, num_classes, epochs, seed))

    if model_choice in ("asand", "both", "all"):
        print("\n[2/3] A-SAND")
        rows.append(run_asand(ds_train, ds_val, ds_test, num_features, num_classes, epochs, seed, fixed_k=fixed_k))

    if model_choice in ("hybrid", "both", "all"):
        print("\n[3/3] A-SAND-Hybrid")
        rows.append(run_hybrid(ds_train, ds_val, ds_test, num_features, num_classes, epochs, seed, fixed_k=fixed_k))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(df.to_string(index=False))

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"results_{dataset_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    return df


# ============================================================
# CLI
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="Run A-SAND / A-SAND-Hybrid experiments.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["mnist", "isolet", "mice", "coil", "arcene", "har70"],
                        help="Dataset to use.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--fixed_k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--model", type=str, default="both",
                        choices=["baseline", "asand", "hybrid", "both", "all"],
                        help="Which model(s) to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(
        dataset_name=args.dataset,
        epochs=args.epochs,
        fixed_k=args.fixed_k,
        model_choice=args.model,
        seed=args.seed,
        results_dir=args.results_dir,
    )
