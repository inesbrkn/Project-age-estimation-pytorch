import pandas as pd
import os
import matplotlib.pyplot as plt

def count_examples_by_age(csv_path, age_col='apparent_age', age_min=0, age_max=100):
    """
    Compte le nombre d'exemples pour chaque âge entre age_min et age_max dans un CSV APPA-REAL.
    """
    df = pd.read_csv(csv_path)
    
    # Chercher la colonne d'âge si le nom exact n'existe pas
    if age_col not in df.columns:
        possible_cols = [c for c in df.columns if 'age' in c.lower()]
        if possible_cols:
            age_col = possible_cols[0]
        else:
            raise ValueError(f"Colonne d'âge introuvable dans {csv_path}")
    
    ages = df[age_col].round().astype(int)
    counts = ages.value_counts().sort_index()
    
    all_ages = pd.Series(0, index=range(age_min, age_max+1))
    counts = all_ages.add(counts, fill_value=0).astype(int)
    
    return counts

def get_ages_from_csv(csv_path, age_col='apparent_age', age_min=0, age_max=100):

    df = pd.read_csv(csv_path)

    # trouver la colonne âge automatiquement
    if age_col not in df.columns:
        possible_cols = [c for c in df.columns if 'age' in c.lower()]
        if possible_cols:
            age_col = possible_cols[0]
        else:
            raise ValueError(f"Colonne d'âge introuvable dans {csv_path}")

    ages = df[age_col].round().astype(int)

    return ages.values

def plot_age_distribution(counts, title="Distribution des âges", save_path=None):
    """
    Affiche un bar plot de la distribution des âges et l'enregistre si save_path est fourni.
    """
    plt.figure(figsize=(12,6))
    plt.bar(counts.index, counts.values, color='skyblue')
    plt.xlabel("Âge")
    plt.ylabel("Nombre d'exemples")
    plt.title(title)
    plt.xticks(range(0, 101, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot enregistré sous : {save_path}")
    
    plt.show()

if __name__ == "__main__":
    folder = "/content/app-real-relase/appa-real-release/gt_avg_train.csv"  # chemin vers ton dossier CSV
    files = ["gt_avg_train.csv", "gt_avg_valid.csv", "gt_avg_test.csv"]

    for f in files:
        path = os.path.join(folder, f)
        counts = count_examples_by_age(path)
        print(f"\n--- Distribution des âges pour {f} ---")
        print(counts)
        save_file = f"age_distribution_{f.replace('.csv','')}.png"
        plot_age_distribution(counts, title=f"Distribution des âges pour {f}", save_path=save_file)