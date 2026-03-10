import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

# -----------------------------
# Classe d'augmentation d'images
# -----------------------------
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
            ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img

# ----------------------------------------
# Dataset PyTorch pour prédiction d’âge
# ----------------------------------------
class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type="train", img_size=224, augment=False, age_stddev=1.0, dataset_name="UTKFace"):
        assert data_type in ("train", "valid", "test")
        assert dataset_name in ("UTKFace", "appa-real")

        self.dataset_name = dataset_name
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        # chemins
        data_dir = Path(data_dir)
        csv_path = data_dir / f"gt_avg_{data_type}.csv"
        img_dir = data_dir / data_type

        # augmentation
        self.transform = ImgAugTransform() if augment else lambda x: x

        # lecture CSV
        df = pd.read_csv(csv_path)

        # gestion écart-type manquant
        self.std = df["apparent_age_std"].tolist() if "apparent_age_std" in df.columns else [0.0]*len(df)

        # stockage des images et labels
        self.x = []
        self.y = []

        for idx, row in df.iterrows():
            img_name = row["file_name"]

            # chemin complet selon dataset
            img_path = img_dir / img_name
            if not img_path.is_file():
                continue

            # label âge
            age = row["apparent_age_avg"] if "apparent_age_avg" in row else row["age"]

            self.x.append(str(img_path))
            self.y.append(age)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        # ajout bruit selon std
        if self.augment and self.std[idx] > 0:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        # lecture et resize
        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)

        # format PyTorch (C, H, W)
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        return img_tensor, np.clip(round(age), 0, 100)


# -----------------------------
# Exemple d'utilisation
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="UTKFace", choices=["UTKFace", "appa-real"])
    args = parser.parse_args()

    train_dataset = FaceDataset(args.data_dir, "train", dataset_name=args.dataset)
    valid_dataset = FaceDataset(args.data_dir, "valid", dataset_name=args.dataset)
    test_dataset  = FaceDataset(args.data_dir, "test",  dataset_name=args.dataset)

    print(f"Train dataset len: {len(train_dataset)}")
    print(f"Valid dataset len: {len(valid_dataset)}")
    print(f"Test dataset len: {len(test_dataset)}")


if __name__ == "__main__":
    main()