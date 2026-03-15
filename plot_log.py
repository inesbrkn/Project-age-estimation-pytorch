import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
import os
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from defaults import _C as cfg

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
    print("Nb events:", len(values))
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

    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator("tf_log/MODEL.METHOD_none_val")
    ea.Reload()
    print(ea.Tags()["scalars"])
    # Loss subplot
    plt.subplot(1,2,1)
    for t_log,v_log,  name in zip(train_logs,val_logs, names):
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

    # Acc subplot
    plt.subplot(1,2,2)
    for t_log, v_log, name in zip(train_logs, val_logs, names):
        train_acc = read_scalars(t_log, "acc")
        #val_acc = read_scalars(v_log, "acc")
        epochs = np.arange(1, len(train_acc)+1)
        plt.plot(epochs, train_acc, linestyle="--", label=f"{name} train")
        #plt.plot(epochs, val_acc, linestyle="-", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("ACCURACY")
    plt.title("ACCURACY comparison")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.show()

def plot_uncertainty_by_age(all_std, val_dataset, save_path="Images/uncertainty_by_age.png"):
    """
    Affiche et enregistre l'incertitude du modèle par âge exact (bar chart).
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # récupérer les âges réels
    val_ages = np.array([y.item() for _, y in val_dataset])

    # calcul de l'incertitude moyenne par âge
    ages_unique = np.arange(0, 101)
    mean_std_by_age = []

    for age in ages_unique:
        mask = val_ages == age
        if np.any(mask):
            mean_std_by_age.append(all_std[mask].mean())
        else:
            mean_std_by_age.append(np.nan)

    mean_std_by_age = np.array(mean_std_by_age)

    # ===== BAR PLOT =====
    plt.figure(figsize=(14,5))
    plt.bar(ages_unique, mean_std_by_age)
    plt.xlabel("Âge réel")
    plt.ylabel("Écart-type moyen des prédictions")
    plt.title("Incertitude du modèle par âge")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"=> Graphique d'incertitude par âge enregistré sous {save_path}")


def plot_uncertainty(predictions, targets, std_values, save_dir="Images"):
    """
    Affiche et sauvegarde des plots pour analyser l'incertitude MC Dropout.
    """

    os.makedirs(save_dir, exist_ok=True)

    predictions = np.array(predictions)
    targets = np.array(targets)
    std_values = np.array(std_values)

    errors = np.abs(predictions - targets)

    # Histogramme des incertitudes
    plt.figure()
    plt.hist(std_values, bins=30)
    plt.xlabel("Prediction uncertainty (std)")
    plt.ylabel("Frequency")
    plt.title("Distribution of MC Dropout Uncertainty")
    plt.savefig(f"{save_dir}/uncertainty_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Scatter : incertitude vs erreur
    plt.figure()
    plt.scatter(std_values, errors, alpha=0.4)
    plt.xlabel("Prediction uncertainty (std)")
    plt.ylabel("Absolute Error")
    plt.title("Uncertainty vs Prediction Error")
    plt.savefig(f"{save_dir}/uncertainty_vs_error.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Corrélation
    corr = np.corrcoef(std_values, errors)[0,1]
    print("Correlation between uncertainty and error:", corr)

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
        dex_train = logdir / "n_MODEL.METHOD_dex_train"
        dex_val   = logdir / "n_MODEL.METHOD_dex_val"

        lap_train = logdir / "n_MODEL.METHOD_laplace_train"
        lap_val   = logdir / "n_MODEL.METHOD_laplace_val"

        gaus_train = logdir / "n_MODEL.METHOD_gaussian_train"
        gaus_val   = logdir / "n_MODEL.METHOD_gaussian_val"
        
        none_train = logdir / "essaye_MODEL.METHOD_none_train"
        none_val   = logdir / "essaye_MODEL.METHOD_none_val"

        residual_train = logdir / "MODEL.METHOD_dex_TRAIN.LR_0.1_train"
        residual_val   = logdir / "MODEL.METHOD_dex_TRAIN.LR_0.1_val"
        plot_two_methods(   
            train_logs=[none_train,residual_train],
            val_logs=[ none_val, residual_val],
            names=["Residual"],
            title=args.title
        )
        plot_two_methods(   
            train_logs=[dex_train, lap_train, gaus_train,none_train, residual_train],
            val_logs=[dex_val, lap_val, gaus_val,none_val,  residual_val],
            names=["DEX", "Laplace", "Gaussian","none", "Residual"],
            title=args.title
        )
        """
        plot_two_methods(train_logs=[dex_train, lap_train],val_logs=[dex_val, lap_val, none_val],names=["DEX", "Laplace"],
            title=args.title
        )
        """
        """
        resnet_train = logdir / "MODEL.METHOD_laplace_train"
        resnet_val   = logdir / "MODEL.METHOD_laplace_val"

        resnet50_train = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_se_resnet50_train"
        resnet50_val   = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_se_resnet50_val"

        resnet101_train = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_resnet101_train"
        resnet101_val   = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_resnet101_val"

        efficient_b0_train = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_efficientnet_b0_train"
        efficient_b0_val   = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_efficientnet_b0_val"
        
        efficient_b3_train = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_efficientnet_b3_train"
        efficient_b3__val   = logdir / "MODEL.METHOD_laplace_MODEL.ARCH_efficientnet_b3_val"
        

        plot_two_methods(   
            train_logs=[resnet_train, resnet50_train,resnet101_train,efficient_b0_train,efficient_b3_train],
            val_logs=[resnet_val,resnet50_val,resnet101_val,efficient_b0_val,efficient_b3__val ],
            names=["se_resnext50_32x4d","ResNet50", "ResNet101", "Efficient_b0", "Efficient_b3"],
            title="Comparaison des backbones sur le model utilisant LaplaceLoss"
        ) 
        """   
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