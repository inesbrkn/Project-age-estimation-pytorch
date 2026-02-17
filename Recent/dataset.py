import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================================
# Augmentation moderne avec albumentations
# =========================================================
class ImgAugTransform:
    def __init__(self, img_size):
        self.aug = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
                A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            ], p=0.5),

            A.Affine(
                rotate=(-20, 20),
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                mode=cv2.BORDER_REPLICATE,
                p=1.0
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=0,
                p=1.0
            ),

            A.RandomGamma(gamma_limit=(80, 120), p=1.0),

            A.HorizontalFlip(p=0.5),

            # Normalisation ImageNet (important si backbone pré-entraîné)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
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
            self.transform = ImgAugTransform(img_size)
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

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

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        img = self.transform(img)

        age = np.clip(round(age), 0, 100)

        return img, torch.tensor(age, dtype=torch.float32)


# =========================================================
# Test rapide
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = FaceDataset(args.data_dir, "train")
    print(f"train dataset len: {len(dataset)}")

    dataset = FaceDataset(args.data_dir, "valid")
    print(f"valid dataset len: {len(dataset)}")

    dataset = FaceDataset(args.data_dir, "test")
    print(f"test dataset len: {len(dataset)}")


if __name__ == '__main__':
    main()