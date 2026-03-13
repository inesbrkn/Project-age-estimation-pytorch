import argparse
from pathlib import Path

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import get_model2
from dataset import FaceDataset
from defaults import _C as cfg
from train import _load_state_dict_into_model, mae_by_age_group
from train_classification import validate_cls


def get_args():
    parser = argparse.ArgumentParser(
        description="Classification age estimation test (DEX 0-100)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Data root directory"
    )
    parser.add_argument(
        "--resume", type=str, required=True, help="Classification model checkpoint"
    )
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    # Mode classification (DEX ou Ordinal) pour ce script
    method = cfg.MODEL.METHOD
    if method not in ("dex", "ordinal"):
        method = "dex"
    cfg.MODEL.METHOD = method
    cfg.MODEL.TASK = "classification"
    cfg.freeze()

    # create model (classification)
    print(f"=> creating classification model '{cfg.MODEL.ARCH}' (method={cfg.MODEL.METHOD})")
    model = get_model2(model_name=cfg.MODEL.ARCH, method=cfg.MODEL.METHOD, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading classification checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        _load_state_dict_into_model(model, checkpoint["state_dict"])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint.get("epoch", "?")))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    test_dataset = FaceDataset(
        args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.WORKERS,
        drop_last=False,
    )

    print("=> start classification testing")
    criterion = None  # pas de loss en test, uniquement métriques
    _, test_acc, test_mae, preds, gt = validate_cls(
        test_loader, model, criterion, 0, device, cfg.MODEL.METHOD
    )
    print(f"test accuracy (cls): {test_acc:.4f}")
    print(f"test mae (cls, global): {test_mae:.3f}")

    # MAE par tranche d'âge (enfants / adultes / seniors) pour analyse des erreurs
    group_results = mae_by_age_group(preds, gt)
    print("=> MAE by age group (classification):")
    for r in group_results:
        print(
            f"  {r['name']} yrs: mae={r['mae']:.3f}, n={r['count']}, std(err)={r['std']:.3f}"
        )


if __name__ == "__main__":
    main()