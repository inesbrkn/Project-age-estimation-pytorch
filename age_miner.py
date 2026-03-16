import os
import shutil
import subprocess
import pandas as pd

# --- PARAMÈTRES ---
TARGET = 1700
MIN_AGE = 60
MAX_AGE = 100
BATCH_SIZE = 500  # On génère par paquets de 500 pour ne pas faire planter Colab


gan_temp = "/content/gan_temp"
gan_final = "/content/gan_images_seniors"
pkl_path = "/content/stylegan3-r-ffhqu-1024x1024.pkl"
checkpoint = "/content/drive/MyDrive/age_estimation/checkpoints_run1/best.pth"

# Création des dossiers
os.makedirs(gan_temp, exist_ok=True)
os.makedirs(gan_final, exist_ok=True)
final_csv_path = os.path.join(gan_final, "gt_avg_synthetic.csv")

total_kept = 0
seed_start = 0

# Reprise automatique si le script s'est arrêté
if os.path.exists(final_csv_path):
    df_existing = pd.read_csv(final_csv_path)
    total_kept = len(df_existing)
    seed_start = total_kept * 10  # Approximation pour éviter de refaire les mêmes seeds
    print(f"Reprise : {total_kept} images de 60-100 ans déjà trouvées.")

print(f"Objectif : {TARGET} images. C'est parti !")

while total_kept < TARGET:
    seed_end = seed_start + BATCH_SIZE - 1
    print(f"\n--- Génération du lot seeds {seed_start}-{seed_end} ---")
    
    # 1. Génération avec StyleGAN3
    cmd_gen = f"python stylegan3/gen_images.py --outdir={gan_temp} --trunc=1 --seeds={seed_start}-{seed_end} --network={pkl_path}"
    subprocess.run(cmd_gen, shell=True)
    
    # 2. Annotation avec ton modèle
    cmd_lbl = f"python pseudo_labeler.py --gan_dir {gan_temp} --resume {checkpoint}"
    subprocess.run(cmd_lbl, shell=True)
    
    # 3. Filtrage des âges
    csv_temp = os.path.join(gan_temp, "gt_avg_synthetic.csv")
    if os.path.exists(csv_temp):
        df = pd.read_csv(csv_temp)
        # On filtre strictement entre 60 et 100
        df_good = df[(df['apparent_age_avg'] >= MIN_AGE) & (df['apparent_age_avg'] <= MAX_AGE)]
        
        # Déplacer les bonnes images vers le dossier final
        for img_name in df_good['file_name']:
            src = os.path.join(gan_temp, f"{img_name}.png")
            dst = os.path.join(gan_final, f"{img_name}.png")
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # Mettre à jour le CSV final
        if os.path.exists(final_csv_path):
            df_final = pd.read_csv(final_csv_path)
            df_final = pd.concat([df_final, df_good])
        else:
            df_final = df_good
            
        # Si on dépasse la cible, on coupe
        if len(df_final) > TARGET:
            df_final = df_final.head(TARGET)
            
        df_final.to_csv(final_csv_path, index=False)
        total_kept = len(df_final)
        print(f"-> {len(df_good)} images gardées dans ce lot. Total: {total_kept}/{TARGET}")
    
    # 4. Nettoyage : On supprime les "jeunes" pour libérer la mémoire de Colab
    shutil.rmtree(gan_temp)
    os.makedirs(gan_temp, exist_ok=True)
    
    seed_start += BATCH_SIZE
    
print("\n🎉 Extraction terminée ! Ton dataset équilibré est prêt.")