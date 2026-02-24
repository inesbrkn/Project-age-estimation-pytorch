import numpy as np
np.bool = bool

import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model2
from dataset import FaceDataset
from defaults import _C as cfg


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args

# à appeler pour recup le mode choisi
def get_criterion(mode, alpha=0.5, device="cpu"):
    if mode == "residual":
        return ResidualLoss(alpha=alpha).to(device)
    elif mode == "dex":
        return nn.CrossEntropyLoss().to(device)
    elif mode == "gaussian":
        return GaussianLikelihoodLoss().to(device)
    elif mode == "laplace":
        return LaplaceLikelihoodLoss().to(device) 
    else:
        raise ValueError(f"Unknown mode: {mode}")


"""
Sert à suivre la perte (loss) et la précision (accuracy) pendant l’entraînement et la validation.
Permet de calculer la moyenne cumulée au fil des batches.
"""
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


""" -- > Entraine le modèle

1) Parcourt toutes les images du train_loader.

2) Pour chaque batch :

- Envoie les images et labels sur le GPU (x.to(device)).

- Calcule la sortie du modèle (outputs = model(x)).

- Calcule la loss (criterion(outputs, y)).

- Calcule la précision du batch.

- Fait la rétropropagation (loss.backward()) et met à jour les poids w et b en fonction de alpha et gradient calculé avec loss.backward(optimizer.step()).

Affiche les statistiques en temps réel avec tqdm.

Résultat : la loss et l’accuracy moyenne pour l’epoch.
"""
def run_epoch(loader, model, criterion, optimizer, epoch, device, mode, is_train):
    model.train() if is_train else model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    accN_meter = AverageMeter()  # pour accuracy ±N ans

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    stage = "train" if is_train else "val"

    with ctx, tqdm(loader) as _tqdm:
        for x, y in _tqdm:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            loss = criterion(outputs, y)
            preds = compute_predictions(outputs, mode, device)

            loss_meter.update(loss.item(), x.size(0))
            abs_error = (preds - y.float()).abs()
            mae_meter.update(abs_error.sum().item(), x.size(0))
            N = 3  # ça c'est pour dire on a faut si on c'est trompé de +- 3 ans (écart type)
            within_N = (abs_error <= N).float()
            accN_meter.update(within_N.sum().item(), x.size(0))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _tqdm.set_postfix(OrderedDict(
                stage=stage, epoch=epoch,
                loss=f"{loss_meter.avg:.4f}",
                mae=f"{mae_meter.avg:.4f}",
                accN=f"{accN_meter.avg:.4f}"
            ))

    return loss_meter.avg, mae_meter.avg, accN_meter.avg

def compute_predictions(outputs, mode, device):
    """Extrait les âges prédits depuis les sorties du modèle."""
    if mode == "dex":
        ages = torch.arange(0, 101, device=device).float()
        probs = F.softmax(outputs, dim=-1)
        return (probs * ages).sum(dim=1)
    elif mode == "residual":
        cls_logits, residual = outputs
        return cls_logits.argmax(1).float() + residual.squeeze(1)
    elif mode in ["gaussian", "laplace"]:
        mu, _ = outputs
        return mu.clamp(0, 100)
    else:
        raise ValueError(f"Unknown mode: {mode}")


class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.res_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, outputs, target):
        """
        outputs = (cls_logits, residual)
        target = true age (LongTensor)
        """
        cls_logits, residual = outputs
        # classification loss
        loss_cls = self.cls_loss(cls_logits, target)
        # predicted class
        pred_class = cls_logits.argmax(dim=1)
        # residual target
        residual_target = target.float() - pred_class.float()
        loss_res = self.res_loss(residual, residual_target)
        return loss_cls + self.alpha * loss_res

class GaussianLikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        mu, log_var = outputs
        precision = torch.exp(-log_var)
        return (precision * (target - mu)**2 + log_var).mean()

class LaplaceLikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        mu, log_b = outputs
        b = torch.exp(log_b)
        return ((target - mu).abs() / b + log_b).mean()

def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model2(model_name=cfg.MODEL.ARCH, method=cfg.MODEL.METHOD)

    # choisi l'optimizer 
    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
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

    # choix de la methode de calcul de la loss, modifier la méthode ds defaults.py
    criterion = get_criterion(cfg.MODEL.METHOD, alpha=0.5, device=device)

    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    # Pour que le learning rate diminue pendant l'entrainement
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        train_loss, train_mae , train_acc= run_epoch(train_loader, model, criterion, optimizer, epoch, device, mode=cfg.MODEL.METHOD, is_train=True)
        val_loss, val_mae, val_acc = run_epoch(val_loader, model, criterion, None, epoch, device, mode=cfg.MODEL.METHOD, is_train=False)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("mae", train_mae, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
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


if __name__ == '__main__':
    main()