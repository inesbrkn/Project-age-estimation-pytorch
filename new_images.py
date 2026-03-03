import os
import pandas as pd

""" 
Ce fichier permet de parser les images qui proviennent de UTKFace car l'age le gender etc sont codés dans le nom de l'image

"""


folder = "UTKFace/"
rows = []

for file in os.listdir(folder):
    if file.endswith(".jpg"):
        age = int(file.split("_")[0])
        rows.append({
            "image": file,
            "age": age
        })

df = pd.DataFrame(rows)
df.to_csv("utkface_labels.csv", index=False)

print("CSV créé :", len(df), "images")