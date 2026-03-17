import os
import shutil
import subprocess
import pandas as pd

# =========================================================
# CONFIGURATION DES CHEMINS (ADAPTÉE À TA STRUCTURE)
# =========================================================

# Chemin vers la racine de ton projet (où se trouve pseudo_labeler.py)
# Fichiers scripts
STYLEGAN_PATH = "/content/Project-age-estimation-pytorch/stylegan3/gen_images.py"
PSEUDO_LBL_PATH = "/content/Project-age-estimation-pytorch/pseudo_labeler.py"

# Fichiers de poids (Checkpoints)
PKL_PATH = "/content/stylegan3-r-ffhqu-1024x1024.pkl"
CHECKPOINT_PTH = "/content/drive/MyDrive/age_estimation/checkpoints_run1/best.pth"

# Dossiers de stockage
GAN_TEMP = "/content/gan_temp"
GAN_FINAL = "/content/drive/MyDrive/age_estimation/gan_images_seniors"

# Paramètres de filtrage
TARGET = 1700
MIN_AGE = 60
MAX_AGE = 100
BATCH_SIZE = 500 

# =========================================================
# PRÉPARATION ET INITIALISATION
# =========================================================

# Création des dossiers nécessaires
os.makedirs(GAN_TEMP, exist_ok=True)
os.makedirs(GAN_FINAL, exist_ok=True)
final_csv_path = os.path.join(GAN_FINAL, "gt_avg_synthetic.csv")

total_kept = 0
seed_start = 0

# Logique de reprise : on compte ce qui est déjà sur le Drive
if os.path.exists(final_csv_path):
    try:
        df_existing = pd.read_csv(final_csv_path)
        total_kept = len(df_existing)
        # On décale les seeds pour ne pas générer les mêmes images
        seed_start = total_kept * 12 
        print(f"🔄 Reprise détectée : {total_kept} images déjà présentes sur le Drive.")
        print(f"🚀 Reprise à partir de la seed {seed_start}")
    except Exception as e:
        print(f"⚠️ Erreur lors de la lecture du CSV existant : {e}")

print(f"🎯 Objectif final : {TARGET} images de seniors ({MIN_AGE}-{MAX_AGE} ans).")

# =========================================================
# BOUCLE PRINCIPALE D'EXTRACTION
# =========================================================

while total_kept < TARGET:
    seed_end = seed_start + BATCH_SIZE - 1
    print(f"\n--- 📦 Lot seeds {seed_start} à {seed_end} ---")
    
    # 1. GÉNÉRATION AVEC STYLEGAN3
    # -----------------------------------------------------
    print("🎨 Génération des visages...")
    cmd_gen = (
        f"python {STYLEGAN_PATH} --outdir={GAN_TEMP} "
        f"--trunc=1 --seeds={seed_start}-{seed_end} --network={PKL_PATH}"
    )
    subprocess.run(cmd_gen, shell=True)
    
    # 2. ANNOTATION AVEC PSEUDO_LABELER
    # -----------------------------------------------------
    print("🧠 Annotation automatique des âges...")
    cmd_lbl = (
        f"python {PSEUDO_LBL_PATH} --gan_dir {GAN_TEMP} --resume {CHECKPOINT_PTH}"
    )
    subprocess.run(cmd_lbl, shell=True)
    
    # 3. FILTRAGE ET TRANSFERT VERS LE DRIVE
    # -----------------------------------------------------
    csv_temp = os.path.join(GAN_TEMP, "gt_avg_synthetic.csv")
    
    if os.path.exists(csv_temp):
        df_batch = pd.read_csv(csv_temp)
        
        # Filtre : strictement entre MIN_AGE et MAX_AGE
        df_good = df_batch[(df_batch['apparent_age_avg'] >= MIN_AGE) & 
                           (df_batch['apparent_age_avg'] <= MAX_AGE)]
        
        num_found = len(df_good)
        print(f"✅ {num_found} seniors identifiés dans ce lot.")

        # Déplacement des fichiers images validés
        for img_name in df_good['file_name']:
            src = os.path.join(GAN_TEMP, f"{img_name}.png")
            dst = os.path.join(GAN_FINAL, f"{img_name}.png")
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # Mise à jour du CSV global sur le Drive
        if os.path.exists(final_csv_path):
            df_final = pd.read_csv(final_csv_path)
            df_final = pd.concat([df_final, df_good]).drop_duplicates(subset=['file_name'])
        else:
            df_final = df_good
            
        # On s'arrête pile à TARGET si on dépasse
        if len(df_final) > TARGET:
            df_final = df_final.head(TARGET)
            
        df_final.to_csv(final_csv_path, index=False)
        total_kept = len(df_final)
        print(f"📊 Progression totale : {total_kept}/{TARGET}")
    else:
        print("❌ Erreur : Le pseudo_labeler n'a pas généré de CSV pour ce lot.")
    
    # 4. NETTOYAGE DU STOCKAGE LOCAL
    # -----------------------------------------------------
    # On vide le dossier temp (qui contient les 500 images du lot) 
    # pour ne pas saturer le disque local de Colab.
    shutil.rmtree(GAN_TEMP)
    os.makedirs(GAN_TEMP, exist_ok=True)
    
    # Passage au lot de seeds suivant
    seed_start += BATCH_SIZE

print(f"\n🎉 Terminé ! Ton dataset de {total_kept} seniors est prêt dans : {GAN_FINAL}")