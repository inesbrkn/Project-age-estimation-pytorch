import argparse
from pathlib import Path

import numpy as np
from plot_log import plot_training_curves
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
    AverageMeter,
    _load_state_dict_into_model
)


class OrdinalLoss(nn.Module):
    """
    Loss pour la régression ordinale :
    pour K classes (0..K-1), le modèle prédit K-1 sorties binaires \"age > k ?\".
    On applique une BCEWithLogitsLoss sur ces sorties.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, ages):
        """
        logits: [batch, K-1]
        ages:  [batch] (âges entiers 0..K-1)
        """
        K_minus1 = logits.size(1)
        K = K_minus1 + 1
        thresholds = torch.arange(0, K - 1, device=logits.device).unsqueeze(0)  # [1, K-1]
        ages = ages.unsqueeze(1)  # [batch, 1]
        targets = (ages > thresholds).float()  # [batch, K-1]
        return self.bce(logits, targets)

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
    parser.add_argument(
        "--synth_dir", 
        type=str, default=None, 
        help="Directory containing synthetic images and CSV")
    parser.add_argument(
        "--synth_only", 
        action="store_true", 
        help="Train ONLY on synthetic data")
    return parser.parse_args()

def train_cls(train_loader, model, criterion, optimizer, epoch, device, method):
    """
    Boucle d'entraînement dédiée au problème de classification.
    - method = \"dex\"      : classification DEX (softmax 0-100)
    - method = \"ordinal\" : régression ordinale (K-1 sorties binaires \"age > k ?\")
    """
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        if method == "ordinal":
            logits = outputs
            loss = criterion(logits, y)
            # age estimé = somme des probabilités sigmoid(logits)
            probs = torch.sigmoid(logits)
            age_est = probs.sum(dim=1)
            predicted = age_est.round().clamp(0, 100).long()
        else:  # DEX
            logits = outputs
            loss = criterion(logits, y)
            predicted = logits.argmax(1)
        cur_loss = loss.item()
        correct_num = predicted.eq(y).sum().item()
        sample_num = x.size(0)

        loss_monitor.update(cur_loss, sample_num)
        accuracy_monitor.update(correct_num, sample_num)

        preds.append(age_est.detach().cpu().numpy())
        gt.append(y.detach().cpu().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    mae = np.abs(preds - gt).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def validate_cls(validate_loader, model, criterion, epoch, device, method):
    """
    Validation pour la classification :
    - accuracy (top-1) approximative
    - MAE dérivé en années :
      - DEX : espérance du softmax sur les classes 0-100
      - Ordinal : somme des probabilités sigmoid(logits) (nombre de seuils dépassés)
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

            outputs = model(x)

            if method == "ordinal":
                logits = outputs
                probs = torch.sigmoid(logits)
                expected_age = probs.sum(dim=1)
                if criterion is not None:
                    loss = criterion(logits, y)
                    predicted = expected_age.round().clamp(0, 100).long()
                    correct_num = predicted.eq(y).sum().item()
                    sample_num = x.size(0)
                    loss_monitor.update(loss.item(), sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
            else:  # DEX
                logits = outputs
                predicted = logits.argmax(1)
                probs = F.softmax(logits, dim=-1)
                ages = torch.arange(0, 101).to(device)
                expected_age = (probs * ages).sum(dim=1)
                if criterion is not None:
                    loss = criterion(logits, y)
                    correct_num = predicted.eq(y).sum().item()
                    sample_num = x.size(0)
                    loss_monitor.update(loss.item(), sample_num)
                    accuracy_monitor.update(correct_num, sample_num)

            preds.append(expected_age.cpu().numpy())
            gt.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    mae = np.abs(preds - gt).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae, preds, gt


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    # Mode classification explicite : DEX ou Ordinal selon MODEL.METHOD
    method = cfg.MODEL.METHOD
    if method not in ("dex", "ordinal"):
        method = "ordinal"
    cfg.MODEL.METHOD = method
    cfg.MODEL.TASK = "classification"
    cfg.freeze()

    set_seed(cfg.TRAIN.SEED)
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model (classification : DEX logits 0-100 ou Ordinal K-1 sorties)
    print(f"=> creating classification model '{cfg.MODEL.ARCH}' (method={cfg.MODEL.METHOD})")
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
    
    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            _load_state_dict_into_model(model, checkpoint["state_dict"])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True
    if cfg.MODEL.METHOD == "ordinal":
        criterion = OrdinalLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.MODEL.LABEL_SMOOTHING).to(
            device
        )

    train_dataset = FaceDataset(
        args.data_dir,
        "train",
        img_size=cfg.MODEL.IMG_SIZE,
        augment=True,
        age_stddev=cfg.TRAIN.AGE_STDDEV,
        synth_dir=args.synth_dir,       # <--- NOUVEAU
        synth_only=args.synth_only      # <--- NOUVEAU
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
    best_val_mae = 10000.0
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
        train_loss, train_acc , train_mae= train_cls(
            train_loader, model, criterion, optimizer, epoch, device, cfg.MODEL.METHOD
        )

        # validate
        val_loss, val_acc, val_mae, _, _ = validate_cls(
            val_loader, model, criterion, epoch, device, cfg.MODEL.METHOD
        )
        history = {
            "name": f"{cfg.MODEL.ARCH}-{cfg.MODEL.METHOD}",
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_acc": [],
            "val_acc": [],
        }

   
        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("mae", train_mae, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # ===== save history =====
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        

        if val_mae < best_val_mae:
            
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")



         # adjust learning rate
        scheduler.step()

        
    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")

    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history["train_mae"],
        history["val_mae"],
        title=history["name"],
        save_path=f"Images/training_curves_Dex.png",
    )

if __name__ == '__main__':
    main()

