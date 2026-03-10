import tarfile
from pathlib import Path
import pandas as pd
import shutil

# ----------------------------
# Dossier principal où on extrait UTKFace
# ----------------------------
data_dir = Path("UTKFace")
data_dir.mkdir(exist_ok=True)
extract_dir = data_dir / "extracted"
extract_dir.mkdir(exist_ok=True)

# ----------------------------
# Archives .tar.gz dans le dossier courant
# ----------------------------
parts = ["part1.tar.gz", "part2.tar.gz", "part3.tar.gz"]

for part in parts:
    tar_path = Path(part)  # <- juste le nom du fichier dans le dossier courant
    if not tar_path.is_file():
        print(f"Archive manquante: {tar_path}")
        continue
    print(f"Extraction de {tar_path} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)

# ----------------------------
# Récupération des images
# ----------------------------
image_files = list(extract_dir.rglob("*.jpg"))
print(f"Total images trouvées: {len(image_files)}")

records = []
for img_path in image_files:
    name = img_path.name
    try:
        age, gender, race, _ = name.split("_")
        records.append({
            "file_name": name,
            "apparent_age_avg": int(age),
            "gender": int(gender),
            "race": int(race),
            "path": str(img_path)
        })
    except Exception as e:
        print(f"Skip {name}: {e}")

df = pd.DataFrame(records)
print(f"Total images valides: {len(df)}")

# ----------------------------
# Shuffle et split
# ----------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
train_end = int(n * 0.7)
valid_end = int(n * 0.85)

df_train = df.iloc[:train_end]
df_valid = df.iloc[train_end:valid_end]
df_test  = df.iloc[valid_end:]

# ----------------------------
# Création des dossiers train/valid/test
# ----------------------------
for split in ["train", "valid", "test"]:
    (data_dir / split).mkdir(exist_ok=True)

# ----------------------------
# Copier les images
# ----------------------------
def copy_files(df_split, split_name):
    split_dir = data_dir / split_name
    for _, row in df_split.iterrows():
        src = Path(row["path"])
        dst = split_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)

copy_files(df_train, "train")
copy_files(df_valid, "valid")
copy_files(df_test, "test")

# ----------------------------
# Sauvegarder les CSV
# ----------------------------
df_train[["file_name", "apparent_age_avg", "gender", "race"]].to_csv(data_dir / "gt_avg_train.csv", index=False)
df_valid[["file_name", "apparent_age_avg", "gender", "race"]].to_csv(data_dir / "gt_avg_valid.csv", index=False)
df_test[["file_name", "apparent_age_avg", "gender", "race"]].to_csv(data_dir / "gt_avg_test.csv", index=False)

print("Extraction et préparation terminées !")
print(f"Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}")