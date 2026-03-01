import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# ======================================================
#  Courbes train / val pour UN modèle
# ======================================================
def plot_training_curves(train_losses, val_losses, train_mae=None, val_mae=None, title=None):
    epochs = np.arange(1, len(train_losses) + 1)

    has_mae = train_mae is not None and val_mae is not None
    ncols = 2 if has_mae else 1

    plt.figure(figsize=(6 * ncols, 5))

    # ----- LOSS -----
    plt.subplot(1, ncols, 1)
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or "Loss over epochs")
    plt.legend()
    plt.grid(alpha=0.3)

    # ----- MAE -----
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