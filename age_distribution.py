import pandas as pd
import os
import matplotlib.pyplot as plt

def count_examples_by_age(csv_path, age_col='apparent_age', age_min=0, age_max=100):
    """
    Compte le nombre d'exemples pour chaque âge entre age_min et age_max dans un CSV.
    """
    df = pd.read_csv(csv_path)
    
    # Détection automatique de la colonne d'âge si le nom exact n'existe pas
    if age_col not in df.columns:
        possible_cols = [c for c in df.columns if 'age' in c.lower()]
        if possible_cols:
            age_col = possible_cols[0]
        else:
            raise ValueError(f"Colonne d'âge introuvable dans {csv_path}")
    
    ages = df[age_col].round().astype(int)
    counts = ages.value_counts().sort_index()
    
    # Ajouter les âges manquants avec 0
    all_ages = pd.Series(0, index=range(age_min, age_max+1))
    counts = all_ages.add(counts, fill_value=0).astype(int)
    
    return counts

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

def merge_csv_and_plot(csv_paths, age_cols=None, age_min=0, age_max=100, save_path=None):
    """
    Fusionne plusieurs CSV et crée un graphique unique des âges.
    
    - csv_paths : liste des chemins vers les CSV
    - age_cols : liste des colonnes d'âge correspondantes à chaque CSV (ou None pour auto-détection)
    """
    if age_cols is None:
        age_cols = [None] * len(csv_paths)
    
    merged_counts = pd.Series(0, index=range(age_min, age_max+1))
    
    for csv_path, age_col in zip(csv_paths, age_cols):
        df = pd.read_csv(csv_path)
        
        # Détection automatique de la colonne d'âge si nécessaire
        if age_col is None:
            possible_cols = [c for c in df.columns if 'age' in c.lower()]
            if possible_cols:
                age_col = possible_cols[0]
            else:
                raise ValueError(f"Colonne d'âge introuvable dans {csv_path}")
        
        counts = count_examples_by_age(csv_path, age_col=age_col, age_min=age_min, age_max=age_max)
        merged_counts += counts
    
    plot_age_distribution(merged_counts, title="Distribution des âges (fusion des CSV)", save_path=save_path)
    return merged_counts

# ---------------------- Exemple d'utilisation ----------------------
if __name__ == "__main__":
    folder = "appa-real-release"
    csv_files = ["gt_avg_train.csv", "gt_avg_synthetic.csv"]  # mettre les noms réels de tes fichiers
    csv_paths = [os.path.join(folder, f) for f in csv_files]
    
    merged_counts = merge_csv_and_plot(csv_paths, save_path="age_distribution_fusion.png")
    print("\n--- Distribution fusionnée ---")
    print(merged_counts)