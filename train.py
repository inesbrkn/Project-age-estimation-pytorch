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
from model import get_model2, mc_dropout_predict
from losses import get_criterion
from dataset import FaceDataset
from defaults import _C as cfg
import utils as u
from plot_log import plot_training_curves


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


def compute_predictions(outputs, mode, device):
    
    if mode == "dex":
        ages = torch.arange(0, 101, device=device).float()
        probs = F.softmax(outputs, dim=-1)
        return (probs * ages).sum(dim=1)

    elif mode == "residual":

        cls_logits, residual = outputs

        #ages = torch.arange(0,101, device=device).float()
        #probs = F.softmax(cls_logits, dim=1)

        #soft_class = (probs * ages).sum(dim=1)

        #pred_age = soft_class + residual.squeeze(-1)

        #pred_age = pred_age.clamp(0,100)
        predicted = cls_logits.argmax(1)
        pred_age = predicted.float() + residual.squeeze()

        return pred_age
    
    elif mode in ["gaussian", "laplace"]:
        mu, _ = outputs
        return mu.squeeze(-1).clamp(0, 100)
    elif  mode == "none" : 
        return outputs.argmax(1).float()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_epoch(loader,model,criterion,optimizer,epoch,device,mode,is_train,return_preds=False):
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    accN_meter = AverageMeter()

    all_preds = []
    all_gt = []
    all_std = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    stage = "train" if is_train else "val"

    with ctx, tqdm(loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)


            if not is_train and cfg.TEST.MC_DROPOUT:

                # MC Dropout prediction
                mean_pred, std_pred = mc_dropout_predict(model, x, n_samples=30)

                preds = mean_pred

                # calcul outputs pour la loss normale
                outputs = model(x)
                loss = criterion(outputs, y)
                # je récupère l'écart type 
                all_std.append(std_pred.detach().cpu())
            else:

                outputs = model(x)
                loss = criterion(outputs, y)

                preds = compute_predictions(outputs, mode, device)
            
            
            # ===== MAE correct =====
            abs_error = (preds - y.float()).abs()
            mae_meter.update(abs_error.sum().item(), x.size(0))

            # ===== accuracy ±N ans =====
            if cfg.CLASSIFIER == True:
                predicted_class = outputs[0].argmax(1) if mode == "residual" else outputs.argmax(1)
                correct_num = (predicted_class == y).sum().item()
                accN_meter.update(correct_num, x.size(0))
            else :
                N = int(cfg.N)
                within_N = (abs_error <= N).sum().item()
                accN_meter.update(within_N, x.size(0))

            loss_meter.update(loss.item(), x.size(0))

            # ===== stockage optionnel =====
            if return_preds:
                all_preds.append(preds.detach().cpu())
                all_gt.append(y.detach().cpu())

            # ===== backward =====
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _tqdm.set_postfix(
                OrderedDict(
                    stage=stage,
                    epoch=epoch,
                    loss=f"{loss_meter.avg:.4f}",
                    mae=f"{mae_meter.avg:.4f}",
                    accN=f"{accN_meter.avg:.4f}",
                )
            )

    if cfg.TEST.MC_DROPOUT:
        all_preds = torch.cat(all_preds).numpy()
        all_gt = torch.cat(all_gt).numpy()
        all_std = torch.cat(all_std).numpy()
        return loss_meter.avg, mae_meter.avg, accN_meter.avg, all_preds, all_gt, all_std

    if return_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_gt = torch.cat(all_gt).numpy()
        return loss_meter.avg, mae_meter.avg, accN_meter.avg, all_preds, all_gt

    return loss_meter.avg, mae_meter.avg, accN_meter.avg

def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    u.set_seed(cfg.TRAIN.SEED)
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
    resume_path = args.resume
    checkpoint = None

 
    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']  # prochain epoch à exécuter
            u._load_state_dict_into_model(model, checkpoint['state_dict'])  # gère le préfixe "module." si besoin
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Restaurer best_val_mae évite d'écraser best.pth avec un modèle moins bon après reprise
            if 'best_val_mae' in checkpoint:
                best_val_mae = checkpoint['best_val_mae']
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
    
    if checkpoint is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("=> loaded scheduler state from checkpoint")
    
    train_writer = None
    
    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    best_val_mae = 10000.0
    
    history = {
        "name": f"{cfg.MODEL.ARCH}-{cfg.MODEL.METHOD}",
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_acc": [],
        "val_acc": [],
    }

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

        # ===== save history =====
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # adjust learning rate
        scheduler.step()

        model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
        torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_mae': val_mae,
                },

                str(checkpoint_dir / "last.pth")
        )
        
        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        

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