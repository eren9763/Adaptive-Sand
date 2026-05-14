"""Reusable training helpers."""

from __future__ import annotations

import time
import tracemalloc


def fit_and_measure(
    model,
    train_data,
    val_data,
    epochs: int,
    train_targets=None,
    callbacks=None,
    verbose: int = 0,
):
    """Fit a model while measuring wall time and peak traced memory."""
    callbacks = callbacks or []
    tracemalloc.start()
    t0 = time.time()
    if train_targets is None:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        history = model.fit(
            train_data,
            train_targets,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
    elapsed = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return history, elapsed, peak_mem
