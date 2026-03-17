import os
import shutil
import subprocess
import pandas as pd

# =========================================================
# CONFIGURATION DES CHEMINS (STRUCTURE NETTOYÉE)
# =========================================================

# Ton dossier principal actuel
ROOT = "/content/Project-age-estimation-pytorch"

# Chemins vers les scripts
# On part du principe que stylegan3 est DIRECTEMENT dans ROOT
STYLEGAN_PATH = os.path.join(ROOT, "stylegan3/gen_images.py")
PSEUDO_LBL_PATH = os.path.join(ROOT, "pseudo_labeler.py")

# Chemins des fichiers de poids
PKL_PATH = "/content/stylegan3-r-ffhqu-1024x1024.pkl"
# ATTENTION : Vérifie si c'est bien 'best.pth' ou 'best_cls.pth' !
CHECKPOINT_PTH = "/content/drive/MyDrive/age_estimation/checkpoints_run1/best.pth"

# Dossiers de stockage
GAN_TEMP = "/content/gan_temp"
GAN_FINAL = "/content/drive/MyDrive/age_estimation/gan_images_seniors"

# Paramètres
TARGET = 1700
MIN_AGE = 60
MAX_AGE = 100
BATCH_SIZE = 500 

# =========================================================
# INITIALISATION
# =========================================================

os.makedirs(GAN_TEMP, exist_ok=True)
os.makedirs(GAN_FINAL, exist_ok=True)
final_csv_path = os.path.join(GAN_FINAL, "gt_avg_synthetic.csv")

total_kept = 0
seed_start = 0

if os.path.exists(final_csv_path):
    df_existing = pd.read_csv(final_csv_path)
    total_kept = len(df_existing)
    seed_start = total_kept * 12 
    print(f"🔄 Reprise : {total_kept} images trouvées. Seed de départ : {seed_start}")

print(f"🎯 Objectif : {TARGET} seniors.")

# =========================================================
# BOUCLE
# =========================================================

while total_kept < TARGET:
    seed_end = seed_start + BATCH_SIZE - 1
    print(f"\n--- Lot {seed_start} à {seed_end} ---")
    
    # 1. Génération
    if not os.path.exists(STYLEGAN_PATH):
        print(f"❌ ERREUR : StyleGAN introuvable ici : {STYLEGAN_PATH}")
        break

    cmd_gen = f"python {STYLEGAN_PATH} --outdir={GAN_TEMP} --trunc=1 --seeds={seed_start}-{seed_end} --network={PKL_PATH}"
    subprocess.run(cmd_gen, shell=True)
    
    # 2. Annotation
    if not os.path.exists(CHECKPOINT_PTH):
        print(f"❌ ERREUR : Checkpoint introuvable ici : {CHECKPOINT_PTH}")
        break

    cmd_lbl = f"python {PSEUDO_LBL_PATH} --gan_dir {GAN_TEMP} --resume {CHECKPOINT_PTH}"
    subprocess.run(cmd_lbl, shell=True)
    
    # 3. Tri
    csv_temp = os.path.join(GAN_TEMP, "gt_avg_synthetic.csv")
    if os.path.exists(csv_temp):
        df_batch = pd.read_csv(csv_temp)
        df_good = df_batch[(df_batch['apparent_age_avg'] >= MIN_AGE) & (df_batch['apparent_age_avg'] <= MAX_AGE)]
        
        for img_name in df_good['file_name']:
            src = os.path.join(GAN_TEMP, f"{img_name}.png")
            dst = os.path.join(GAN_FINAL, f"{img_name}.png")
            if os.path.exists(src):
                shutil.move(src, dst)
        
        if os.path.exists(final_csv_path):
            df_final = pd.read_csv(final_csv_path)
            df_final = pd.concat([df_final, df_good]).drop_duplicates(subset=['file_name'])
        else:
            df_final = df_good
            
        df_final.head(TARGET).to_csv(final_csv_path, index=False)
        total_kept = len(df_final)
        print(f"📊 Progression : {total_kept}/{TARGET}")
    
    # Nettoyage
    shutil.rmtree(GAN_TEMP)
    os.makedirs(GAN_TEMP, exist_ok=True)
    seed_start += BATCH_SIZE

print("\n🎉 Terminé !")