import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


# ======================================================
#  Courbes train / val pour UN modèle
# ======================================================
def plot_training_curves(train_losses, val_losses, train_mae=None, val_mae=None, title=None, save_path=None):
    epochs = np.arange(1, len(train_losses) + 1)
    has_mae = train_mae is not None and val_mae is not None
    ncols = 2 if has_mae else 1
    plt.figure(figsize=(6 * ncols, 5))

    plt.subplot(1, ncols, 1)
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or "Loss over epochs")
    plt.legend()
    plt.grid(alpha=0.3)

    if has_mae:
        plt.subplot(1, ncols, 2)
        plt.plot(epochs, train_mae, label="train MAE")
        plt.plot(epochs, val_mae, label="val MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("MAE over epochs")
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    
    if save_path:  # sauvegarde sur disque
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.show()


# ======================================================
# Comparaison multi-modèles (DEX vs Residual etc.)
# ======================================================
def plot_model_comparison(histories, metric="val_mae"):
    """
    histories: list of dict
    each dict must contain:
        - name
        - metric list (e.g. val_mae)
    """

    plt.figure(figsize=(10, 6))

    for hist in histories:
        if metric not in hist:
            print(f"[WARN] {hist.get('name','unknown')} missing {metric}")
            continue

        values = hist[metric]
        label = hist.get("name", "model")
        plt.plot(values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ======================================================
# Heatmap MAE finale
# ======================================================
def plot_mae_heatmap(results):
    """
    results format:
    {
        "efficientnet_b0": {"dex": 4.2, "residual": 3.9},
        ...
    }
    """
    df = pd.DataFrame(results).T
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("MAE comparison across models and methods")
    plt.tight_layout()
    plt.show()


# =========================================================
# Read tensorboard scalars
# =========================================================
def read_scalars(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        print(f"[WARN] tag '{tag}' not found in {log_dir}")
        return []

    events = ea.Scalars(tag)
    values = [e.value for e in events]
    return values


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--title", type=str, default="Training curves")
    args = parser.parse_args()

    logdir = Path(args.logdir)

    train_log = logdir / "_train"
    val_log = logdir / "_val"

    print("Reading TensorBoard logs...")

    train_loss = read_scalars(train_log, "loss")
    val_loss = read_scalars(val_log, "loss")

    train_mae = read_scalars(train_log, "mae")
    val_mae = read_scalars(val_log, "mae")

    plot_training_curves(
        train_loss,
        val_loss,
        train_mae,
        val_mae,
        title=args.title,
    )


if __name__ == "__main__":
    main()