"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth_curve(points, factor: float = 0.85):
    """Exponential moving average smoothing."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_training_summary(all_histories: dict, pruning_callback=None, save_path=None):
    """Create a 2x2 training and pruning summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Model performance and adaptive pruning", fontsize=16, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax1 = axes[0, 0]
    for i, (model_name, history) in enumerate(all_histories.items()):
        if "val_accuracy" in history:
            y = history["val_accuracy"]
            x = range(1, len(y) + 1)
            color = colors[i % len(colors)]
            ax1.plot(x, y, color=color, alpha=0.2)
            ax1.plot(x, smooth_curve(y), color=color, label=model_name, linewidth=2.5)
    ax1.set_title("Validation accuracy vs. epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = axes[0, 1]
    for i, (model_name, history) in enumerate(all_histories.items()):
        if "val_loss" in history:
            y = history["val_loss"]
            x = range(1, len(y) + 1)
            color = colors[i % len(colors)]
            ax2.plot(x, y, color=color, alpha=0.2)
            ax2.plot(x, smooth_curve(y), color=color, label=model_name, linewidth=2.5)
    ax2.set_title("Validation loss vs. epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax3 = axes[1, 0]
    if pruning_callback is not None and hasattr(pruning_callback, "k_history"):
        y_k = pruning_callback.k_history
        if len(y_k) > 0:
            x_k = list(range(len(y_k)))
            ax3.step(
                x_k,
                y_k,
                where="post",
                color="#d62728",
                linewidth=2.5,
                marker="o",
                markersize=5,
                label="Selected features (k)",
            )
            ax3.text(x_k[0], y_k[0], f"Start: {y_k[0]}", color="black")
            ax3.text(x_k[-1], y_k[-1], f"Final k: {y_k[-1]}", color="black")
    else:
        ax3.text(0.5, 0.5, "No pruning callback available", ha="center")
    ax3.set_title("Adaptive pruning (k)")
    ax3.set_xlabel("Pruning step")
    ax3.set_ylabel("Active features (k)")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    ax4 = axes[1, 1]
    if (
        pruning_callback is not None
        and hasattr(pruning_callback, "acc_history")
        and hasattr(pruning_callback, "k_history")
    ):
        y_acc = np.array(pruning_callback.acc_history)
        y_k = np.array(pruning_callback.k_history)
        min_len = min(len(y_acc), len(y_k) - 1)
        if min_len > 0:
            plot_k = y_k[1 : min_len + 1]
            plot_acc = y_acc[:min_len]
            sc = ax4.scatter(
                plot_k,
                plot_acc,
                c=range(len(plot_k)),
                cmap="viridis",
                s=60,
                edgecolor="k",
                zorder=3,
            )
            ax4.plot(plot_k, plot_acc, linestyle="--", alpha=0.5, color="gray")
            plt.colorbar(sc, ax=ax4).set_label("Pruning steps")
            ax4.invert_xaxis()
        else:
            ax4.text(0.5, 0.5, "Waiting for first pruning...", ha="center")
    else:
        ax4.text(0.5, 0.5, "No pruning callback available", ha="center")
    ax4.set_title("Accuracy vs. feature retention")
    ax4.set_xlabel("Active features (k)")
    ax4.set_ylabel("Validation accuracy")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

