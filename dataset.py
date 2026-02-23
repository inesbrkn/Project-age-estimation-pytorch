import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Normalisation ImageNet (RGB) pour les modèles pré-entraînés
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# =========================================================
# Classe d'augmentation équivalente à imgaug
# =========================================================
class ImgAugTransform:
    def __init__(self):
        self.aug = A.Compose([
            # équivalent OneOf([AdditiveGaussianNoise, GaussianBlur])
            A.OneOf([
                A.GaussNoise(std_range=(0.04, 0.2), mean_range=(0, 0), per_channel=True, p=0.25), # ~0.1*255 ± random
                A.GaussianBlur(blur_limit=(0, 3))
            ], p=0.5),
            
            # affine similaire à iaa.Affine
            A.Affine(
                rotate=(-20, 20),
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                border_mode=cv2.BORDER_REPLICATE
            ),
            
            # Hue & Saturation comme iaa.AddToHueAndSaturation
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=0
            ),
            
            # Gamma contrast équivalent
            A.RandomGamma(gamma_limit=(30, 200)),
            
            # Flip horizontal
            A.HorizontalFlip(p=0.5)
        ])

    def __call__(self, img):
        return self.aug(image=img)["image"]


# =========================================================
# Dataset PyTorch
# =========================================================
class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert data_type in ("train", "valid", "test")
        csv_path = Path(data_dir) / f"gt_avg_{data_type}.csv"
        img_dir = Path(data_dir) / data_type

        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            # Si pas d'augmentation, juste identité
            self.transform = lambda img: img

        self.x = []
        self.y = []
        self.std = []

        df = pd.read_csv(csv_path)

        ignore_path = Path(__file__).resolve().parent / "ignore_list.csv"
        ignore_img_names = []
        if ignore_path.exists():
            ignore_img_names = list(pd.read_csv(ignore_path)["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir / f"{img_name}_face.jpg"
            if not img_path.is_file():
                continue

            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        # ajout de bruit sur l'âge si augmentation
        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)

        # Normalisation ImageNet (pixels 0–255 → 0–1 puis mean/std)
        img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # conversion HWC → CHW pour PyTorch
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), np.clip(round(age), 0, 100)


# =========================================================
# Test rapide
# =========================================================
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    for dt in ["train", "valid", "test"]:
        dataset = FaceDataset(args.data_dir, dt)
        print(f"{dt} dataset len: {len(dataset)}")


if __name__ == '__main__':
    main()
