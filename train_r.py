import numpy as np
np.bool = bool

import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import os
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
from model import get_model2, mc_dropout_predict, compute_predictions, tta_predict, tta_predict2
from losses import get_criterion
from dataset import FaceDataset
from defaults import _C as cfg
import utils as u
from plot_log import plot_training_curves, plot_uncertainty_by_age
from torch.utils.data import WeightedRandomSampler

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

def mae_by_age_group(preds, gt, groups=None):
    """
    Calcule le MAE par tranche d'âge (ex. enfants 0-17, adultes 18-45, seniors 46+).
    groups: liste de (min_age, max_age) inclus. Par défaut cfg.TEST.AGE_GROUPS.
    Retourne une liste de dict avec 'name', 'mae', 'count', 'std' (écart-type des erreurs absolues).
    """
    if groups is None:
        groups = cfg.TEST.AGE_GROUPS
    preds = np.asarray(preds)
    gt = np.asarray(gt)
    errors = np.abs(preds - gt)
    results = []
    for low, high in groups:
        mask = (gt >= low) & (gt <= high)
        n = mask.sum()
        if n == 0:
            results.append({"name": f"{low}-{high}", "mae": np.nan, "count": 0, "std": np.nan})
            continue
        mae = errors[mask].mean()
        std = errors[mask].std()
        results.append({"name": f"{low}-{high}", "mae": float(mae), "count": int(n), "std": float(std)})
    return results

def train_one_epoch(loader, model, criterion, optimizer, epoch, device, mode):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    preds = []
    gt = []

    with torch.enable_grad(), tqdm(loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, y)

            predicted = outputs[0].argmax(1) if mode == "residual" else outputs.argmax(1)

            pred= compute_predictions(outputs, mode, device)
          

            correct_num = (predicted == y).sum().item()
            sample_num = x.size(0)
            
            loss_meter.update(loss.item(), sample_num )
            acc_meter.update(correct_num,sample_num)

    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gt.append(y.cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            _tqdm.set_postfix(
                OrderedDict(
                    stage="train",
                    epoch=epoch,
                    loss=f"{loss_meter.avg:.4f}",
                    accN=f"{acc_meter.avg:.4f}",
                )
            )

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)

    mae = np.abs(preds - gt).mean()
    return loss_meter.avg, acc_meter.avg, mae

def validate_one_epoch(loader, model, criterion,epoch, device, mode, return_preds=False):
    model.eval()

    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    preds = []
    gt = []
    all_std = []

    with torch.no_grad(), tqdm(loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)

            if cfg.TTA > 0:

                if cfg.TTA == 1:
                    mean_pred, std_pred = tta_predict(model, x, mode, device)
                else:
                    mean_pred, std_pred = tta_predict2(model, x, mode, device)

                preds.append(mean_pred.cpu().numpy())
                all_std.append(std_pred.detach().cpu())

            elif cfg.MC_DROPOUT:

                mean_pred, std_pred = mc_dropout_predict(
                    model, x, mode, device, n_samples=30
                )

                preds.append(mean_pred.cpu().numpy())
                all_std.append(std_pred.detach().cpu())

            else:
                preds.append(compute_predictions(outputs, mode, device).cpu().numpy())


            predicted = outputs[0].argmax(1) if mode == "residual" else outputs.argmax(1)

            correct_num = (predicted == y).sum().item()
            sample_num = x.size(0)
            accuracy_monitor.update(correct_num, sample_num)

            if criterion is not None:
                loss = criterion(outputs, y)
                loss_monitor.update(loss.item(), x.size(0))

                _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

            gt.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)

    mae = np.abs(preds - gt).mean()

    if cfg.MC_DROPOUT or cfg.TTA > 0:
        all_std = torch.cat(all_std).numpy()

    if not return_preds:
        preds,gt= None, None

    return loss_monitor.avg, accuracy_monitor.avg, mae, preds, gt, all_std

def build_sampler(dataset):
    """
    Crée un sampler pondéré pour équilibrer les âges.
    Utilise les âges directement depuis le dataset pour éviter un IndexError.
    """
    # récupère les âges depuis le dataset
    ages = []
    for i in range(len(dataset)):
        _, age = dataset[i]  # FaceDataset.__getitem__ retourne (image, age)
        ages.append(age)

    ages = torch.tensor(ages).long()

    # compte le nombre d’occurrences de chaque âge
    counts = torch.bincount(ages, minlength=101)  # suppose que l’âge max est 100

    # poids inversement proportionnels à la fréquence
    weights = 1.0 / counts
    sample_weights = weights[ages]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    #u.set_seed(cfg.TRAIN.SEED)
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
    # ----- Reprise depuis un checkpoint (--resume) -----
    # On charge AVANT d'envelopper le modèle avec DataParallel, pour pouvoir utiliser
    # _load_state_dict_into_model et accepter un checkpoint sauvegardé avec ou sans DataParallel.
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
    
    if cfg.MODEL.balanced_sampler :

        sampler = build_sampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=sampler,      # sampler remplace shuffle
            shuffle=False,
            num_workers=cfg.TRAIN.WORKERS,
            drop_last=True
        )

    else:
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
    
    history = {
        "name": f"{cfg.MODEL.ARCH}-{cfg.MODEL.METHOD}",
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_acc": [],
        "val_acc": [],
    }

    all_std=[]
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        train_loss,  train_acc, train_mae= train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, mode=cfg.MODEL.METHOD)
        val_loss, val_acc, val_mae,preds,gt,std = validate_one_epoch(val_loader, model, criterion, epoch, device, mode=cfg.MODEL.METHOD)

        
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

        if cfg.MC_DROPOUT or cfg.TTA >0 :
                all_std.append(std)

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
    if cfg.MC_DROPOUT or cfg.TTA > 0:
        plot_uncertainty_by_age(
            all_std,        # écarts-types récupérés lors du dernier run_epoch
            val_dataset,    # dataset de validation pour avoir les âges
            save_path="Images/uncertainty_by_age.png"
        )
if __name__ == '__main__':
    main()

