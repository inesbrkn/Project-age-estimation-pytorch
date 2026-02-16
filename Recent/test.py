import argparse
import torch
from torch.utils.data import DataLoader
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg
from train import validate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(cfg.MODEL.ARCH, pretrained=False)
    model.load_state_dict(torch.load(args.resume, map_location=device))
    model = model.to(device)

    test_dataset = FaceDataset(args.data_dir, "test",
                               img_size=cfg.MODEL.IMG_SIZE,
                               augment=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             shuffle=False)

    mae = validate(test_loader, model, device)

    print("Test MAE:", mae)


if __name__ == "__main__":
    main()
