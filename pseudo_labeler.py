import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy
from model import get_model2
from defaults import _C as cfg
from train import _load_state_dict_into_model
from dataset import IMAGENET_MEAN, IMAGENET_STD

def get_args():
    parser = argparse.ArgumentParser(description="Pseudo-labeling GAN images")
    parser.add_argument("--gan_dir", type=str, required=True, help="Dossier contenant les images GAN (.jpg)")
    parser.add_argument("--resume", type=str, required=True, help="Chemin vers best_cls.pth")
    return parser.parse_args()

def process_image(img_path, img_size):
    """Prépare l'image comme dans dataset.py"""
    img = cv2.imread(str(img_path), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32)
    img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0) # Ajout de la dimension batch

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger le modèle (Ordinal ou DEX selon ta config)
    print(f"=> Chargement du modèle {cfg.MODEL.ARCH} ({cfg.MODEL.METHOD})")
    model = get_model2(model_name=cfg.MODEL.ARCH, method=cfg.MODEL.METHOD, pretrained=None)
    #torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    _load_state_dict_into_model(model, checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    gan_dir = Path(args.gan_dir)
    results = []

    # 2. Prédire l'âge pour chaque image
    print("=> Début du pseudo-labeling...")
    image_files = list(gan_dir.rglob("*.jpg"))

    
    with torch.no_grad():
        for img_path in image_files:
            x = process_image(img_path, cfg.MODEL.IMG_SIZE).to(device)
            outputs = model(x)
            
            # Calcul de l'âge selon la méthode
            if cfg.MODEL.METHOD == "ordinal":
                probs = torch.sigmoid(outputs)
                age_est = probs.sum(dim=1).item()
            else: # DEX
                probs = F.softmax(outputs, dim=-1)
                ages = torch.arange(0, 101).to(device)
                age_est = (probs * ages).sum(dim=1).item()
                
            results.append({
                "file_name": img_path.stem, # Nom sans l'extension
                "apparent_age_avg": age_est,
                "apparent_age_std": 1.0 # Bruit fixe pour les données synthétiques
            })

    # 3. Sauvegarder le CSV
    df = pd.DataFrame(results)
    csv_out = gan_dir / "gt_avg_synthetic.csv"
    df.to_csv(csv_out, index=False)
    print(f"=> Fichier sauvegardé : {csv_out} ({len(df)} images annotées)")

if __name__ == "__main__":
    main()