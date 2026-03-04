import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
from model import get_model2
from dataset import FaceDataset
from defaults import _C as cfg
from train import (
    set_seed,
    _load_state_dict_into_model,
    AverageMeter,
)




def get_args():
    model_names = sorted(
        name for name in pretrainedmodels.__dict__
        if not name.startswith("__")
        and name.islower()
        and callable(pretrainedmodels.__dict__[name])
    )
    parser = argparse.ArgumentParser(
        description=f"available models: {model_names}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_cls",
        help="Checkpoint directory for classification",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="Tensorboard log directory (classification)",
    )
    parser.add_argument(
        "--multi_gpu", action="store_true", help="Use multi GPUs (data parallel)"
    )
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()
    return args

def train_cls(train_loader, model, criterion, optimizer, epoch, device):
    """
    Boucle d'entraînement dédiée au problème de classification (DEX pur).
    Le modèle renvoie directement des logits de taille [batch_size, num_classes].
    """
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        predicted = logits.argmax(1)
        cur_loss = loss.item()
        correct_num = predicted.eq(y).sum().item()
        sample_num = x.size(0)

        loss_monitor.update(cur_loss, sample_num)
        accuracy_monitor.update(correct_num, sample_num)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_monitor.avg, accuracy_monitor.avg


def validate_cls(validate_loader, model, criterion, epoch, device):
    """
    Validation pour la classification :
    - accuracy (top-1)
    - MAE dérivé en années via l'espérance du softmax sur les classes 0-100.
    """
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    preds = []
    gt = []

    with torch.no_grad():
        for x, y in validate_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predicted = logits.argmax(1)

            probs = F.softmax(logits, dim=-1)
            ages = torch.arange(0, 101).to(device)
            expected_age = (probs * ages).sum(dim=1)

            preds.append(expected_age.cpu().numpy())

            if criterion is not None:
                loss = criterion(logits, y)
                correct_num = predicted.eq(y).sum().item()
                sample_num = x.size(0)

                loss_monitor.update(loss.item(), sample_num)
                accuracy_monitor.update(correct_num, sample_num)

            gt.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    mae = np.abs(preds - gt).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae, preds, gt


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    # Mode classification explicite : on fixe la méthode à DEX
    cfg.MODEL.METHOD = "dex"
    cfg.MODEL.TASK = "classification"
    cfg.freeze()

    set_seed(cfg.TRAIN.SEED)
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model (classification DEX : logits 0-100)
    print("=> creating classification model '{}' (DEX)".format(cfg.MODEL.ARCH))
    model = get_model2(model_name=cfg.MODEL.ARCH, method=cfg.MODEL.METHOD)

    # choisi l'optimizer
    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ----- Reprise depuis un checkpoint (--resume) -----
    resume_path = args.resume
    checkpoint = None

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading classification checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint["epoch"]
            _load_state_dict_into_model(model, checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume_path, checkpoint["epoch"]
                )
            )
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "best_val_mae" in checkpoint:
                best_val_mae = checkpoint["best_val_mae"]
            else:
                best_val_mae = 10000.0
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
            best_val_mae = 10000.0
    else:
        best_val_mae = 10000.0

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.MODEL.LABEL_SMOOTHING).to(
        device
    )

    train_dataset = FaceDataset(
        args.data_dir,
        "train",
        img_size=cfg.MODEL.IMG_SIZE,
        augment=True,
        age_stddev=cfg.TRAIN.AGE_STDDEV,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.WORKERS,
        drop_last=True,
    )

    val_dataset = FaceDataset(
        args.data_dir,
        "valid",
        img_size=cfg.MODEL.IMG_SIZE,
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.WORKERS,
        drop_last=False,
    )

    scheduler = StepLR(
        optimizer,
        step_size=cfg.TRAIN.LR_DECAY_STEP,
        gamma=cfg.TRAIN.LR_DECAY_RATE,
        last_epoch=start_epoch - 1,
    )
    if checkpoint is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("=> loaded scheduler state from classification checkpoint")

    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(
            log_dir=args.tensorboard + "/" + opts_prefix + "_cls_train"
        )
        val_writer = SummaryWriter(
            log_dir=args.tensorboard + "/" + opts_prefix + "_cls_val"
        )

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train_cls(
            train_loader, model, criterion, optimizer, epoch, device
        )

        # validate
        val_loss, val_acc, val_mae, _, _ = validate_cls(
            val_loader, model, criterion, epoch, device
        )

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        scheduler.step()

        # Checkpoint best (classification)
        if val_mae < best_val_mae:
            print(
                f"=> [epoch {epoch:03d}] best val mae (cls) improved from {best_val_mae:.3f} to {val_mae:.3f}"
            )
            best_val_mae = val_mae
            model_state_dict = (
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "arch": cfg.MODEL.ARCH,
                    "state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_val_mae,
                },
                str(checkpoint_dir / "best_cls.pth"),
            )

        # Checkpoint last (classification)
        model_state_dict = (
            model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "arch": cfg.MODEL.ARCH,
                "state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
            },
            str(checkpoint_dir / "last_cls.pth"),
        )

    print("=> classification training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae (cls): {best_val_mae:.3f}")


if __name__ == "__main__":
    main()

