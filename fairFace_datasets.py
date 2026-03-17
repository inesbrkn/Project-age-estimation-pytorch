import os
from pathlib import Path
import pandas as pd
import shutil
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ----------------------------
# Dossier principal
# ----------------------------
data_dir = Path("FairFace")
data_dir.mkdir(exist_ok=True)
for split_name in ["train", "valid", "test"]:
    (data_dir / split_name).mkdir(exist_ok=True)

# ----------------------------
# Téléchargement depuis HuggingFace
# ----------------------------
print("Téléchargement du dataset FairFace depuis HuggingFace...")
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")  # version 0.25, ~97k images

# ----------------------------
# Conversion en DataFrame
# ----------------------------
df = pd.DataFrame(dataset["train"])  # HuggingFace fournit déjà un split train/validation
df_valid = pd.DataFrame(dataset["validation"])

# Option : merge les deux si tu veux refaire ton split 70/15/15
df_all = pd.concat([df, df_valid]).reset_index(drop=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

train_end = int(len(df_all) * 0.7)
valid_end = int(len(df_all) * 0.85)

df_train = df_all.iloc[:train_end]
df_valid = df_all.iloc[train_end:valid_end]
df_test  = df_all.iloc[valid_end:]

# ----------------------------
# Copier les images localement
# ----------------------------
def copy_images(df_split, split_name):
    split_dir = data_dir / split_name
    for idx, row in df_split.iterrows():
        # HuggingFace stocke les images sous row['image']
        img = row["image"]
        out_path = split_dir / f"{idx}.jpg"
        img.save(out_path)

copy_images(df_train, "train")
copy_images(df_valid, "valid")
copy_images(df_test, "test")

# ----------------------------
# Sauvegarder CSV
# ----------------------------
def save_csv(df_split, split_name):
    df_split[["age", "gender", "race"]].to_csv(data_dir / f"gt_avg_{split_name}.csv", index=False)

save_csv(df_train, "train")
save_csv(df_valid, "valid")
save_csv(df_test, "test")

print("FairFace prêt !")
print(f"Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}")