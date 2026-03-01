import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
import os
from tensorboard.backend.event_processing import event_accumulator


# ======================================================
#  Courbes train / val pour UN modèle
# ======================================================
def plot_training_curves(train_losses, val_losses, train_mae=None, val_mae=None, title=None, save_path=None, save_dir="Images"):
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
       
    os.makedirs(save_dir, exist_ok=True)  # crée le dossier si absent

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
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        print(f"[WARN] tag '{tag}' not found in {log_dir}")
        return []

    events = ea.Scalars(tag)
    values = [e.value for e in events]
    return values


# =========================================================
# Superposer deux méthodes sur le même graphique
# =========================================================
def plot_two_methods(train_logs, val_logs, names, title=None, save_path=None):
    """
    train_logs, val_logs: list of paths
    names: list of labels
    Affiche train/val loss et MAE pour deux méthodes
    """
    plt.figure(figsize=(12,5))
    ea = event_accumulator.EventAccumulator("tf_log/dex_train")
    ea.Reload()
    print(ea.Tags()["scalars"])
    # Loss subplot
    plt.subplot(1,2,1)
    for t_log,  name in zip(train_logs, names):
        train_loss = read_scalars(t_log, "loss")
        #val_loss = read_scalars(v_log, "loss")
        epochs = np.arange(1, len(train_loss)+1)
        plt.plot(epochs, train_loss, linestyle="--", label=f"{name} train")
        #plt.plot(epochs, val_loss, linestyle="-", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss comparison")
    plt.legend()
    plt.grid(alpha=0.3)

    # MAE subplot
    """
    plt.subplot(1,2,2)
    for t_log, v_log, name in zip(train_logs, val_logs, names):
        train_mae = read_scalars(t_log, "mae")
        val_mae = read_scalars(v_log, "mae")
        epochs = np.arange(1, len(train_mae)+1)
        plt.plot(epochs, train_mae, linestyle="--", label=f"{name} train")
        plt.plot(epochs, val_mae, linestyle="-", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE comparison")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    """
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.show()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--title", type=str, default="Training curves")
    parser.add_argument("--compare", action="store_true", help="Comparer trois model DEX vs Laplace vs Gaussian")
    args = parser.parse_args()

    logdir = Path(args.logdir)

    if args.compare:
        # Dossiers TensorBoard pour DEX, Laplace, Gaussian
        dex_train = logdir / "dex_train"
        dex_val   = logdir / "dex_val"

        lap_train = logdir / "MODEL.METHOD_laplace_train"
        lap_val   = logdir / "MODEL.METHOD_laplace_val"

        gaus_train = logdir / "gaussian_train"
        gaus_val   = logdir / "gaussian_val"

        plot_two_methods(   
            train_logs=[dex_train, lap_train, gaus_train],
            val_logs=[dex_val, lap_val, gaus_val],
            names=["DEX", "Laplace", "Gaussian"],
            title=args.title
        )

    else:
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