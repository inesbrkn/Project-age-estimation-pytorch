import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert data_type in ("train", "valid", "test")

        csv_path = Path(data_dir) / f"gt_avg_{data_type}.csv"
        img_dir = Path(data_dir) / data_type

        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        # Transformations modernes
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

        self.x = []
        self.y = []
        self.std = []

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_name = row["file_name"]
            img_path = img_dir / f"{img_name}_face.jpg"

            if not img_path.is_file():
                continue

            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = self.transform(img)

        return img, torch.tensor(np.clip(round(age), 0, 100), dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = FaceDataset(args.data_dir, "train")
    print("train dataset len:", len(dataset))


if __name__ == "__main__":
    main()
