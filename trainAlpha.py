import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
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


def set_seed(seed):
    """Fixe les seeds pour des runs reproductibles (comparaison DEX vs Residual équitable)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # On laisse cudnn.benchmark à True (défini plus bas en main) pour la vitesse ; pour une reproductibilité
    # stricte sur GPU, mettre cudnn.benchmark = False après cet appel.


def _load_state_dict_into_model(model, state_dict):
    """
    Charge le state_dict dans le modèle en gérant le préfixe DataParallel.

    Pourquoi c'est nécessaire :
    - Avec nn.DataParallel(model), PyTorch enregistre les paramètres sous des clés "module.conv1", etc.
    - En mono-GPU (ex. Colab) le modèle n'a pas ce préfixe, donc load_state_dict() échoue ou ignore des clés.
    - Cette fonction détecte les clés "module.*" et les renomme en retirant "module." (7 caractères),
      pour qu'un même fichier .pth fonctionne après entraînement multi-GPU comme en évaluation mono-GPU.
    """
    if not any(k.startswith("module.") for k in state_dict.keys()):
        model.load_state_dict(state_dict)
        return
    new_state_dict = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)


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

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            # -------------------------
            # RESIDUAL METHOD
            # -------------------------
            if cfg.MODEL.METHOD == "residual":
                cls_logits, residual = outputs
                loss = criterion(outputs, y)
                predicted = cls_logits.argmax(1)

            # -------------------------
            # DEX / CLASSIFICATION
            # -------------------------
            else:
                logits = outputs
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

            _tqdm.set_postfix(
                OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                acc=accuracy_monitor.avg
            )

    return loss_monitor.avg, accuracy_monitor.avg



def validate(validate_loader, model, criterion, epoch, device, method="cls"):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                # -------------------------
                # RESIDUAL METHOD
                # -------------------------
                if method == "residual":
                    cls_logits, residual = outputs
                    predicted = cls_logits.argmax(1)
                    final_age = predicted.float() + residual.squeeze()

                    preds.append(final_age.cpu().numpy())

                    # loss only if training/validation
                    if criterion is not None:
                        loss = criterion(outputs, y)
                        correct_num = predicted.eq(y).sum().item()
                        sample_num = x.size(0)

                        loss_monitor.update(loss.item(), sample_num)
                        accuracy_monitor.update(correct_num, sample_num)

                       
                        _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)
                # -------------------------
                # DEX METHOD
                # -------------------------
                else:
                    logits = outputs
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

                        _tqdm.set_postfix(loss=loss_monitor.avg,
                                          acc=accuracy_monitor.avg)

                # Toujours stocker le GT
                gt.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)

    mae = np.abs(preds - gt).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae, preds, gt


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


class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.5, label_smoothing=0.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    set_seed(cfg.TRAIN.SEED)
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
    checkpoint = None  # gardé pour recharger le scheduler plus bas

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']  # prochain epoch à exécuter
            _load_state_dict_into_model(model, checkpoint['state_dict'])  # gère le préfixe "module." si besoin
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

    method = cfg.MODEL.METHOD  # "dex", "residual", ou None/""

    # choix de la methode de calcule de la loss (label_smoothing améliore souvent la généralisation)
    if method == "residual":
        criterion = ResidualLoss(alpha=0.5, label_smoothing=cfg.MODEL.LABEL_SMOOTHING).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.MODEL.LABEL_SMOOTHING).to(device)

    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    # Scheduler : last_epoch = start_epoch - 1 pour que le learning rate soit correct au prochain step.
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    # À la reprise, restaurer l'état du scheduler évite que le LR reparte de zéro (ex. reprendre à l'epoch 30
    # avec le LR de l'epoch 0). Sans ça, la courbe de LR serait incorrecte après --resume.
    if checkpoint is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("=> loaded scheduler state from checkpoint")
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae, _, _ = validate(val_loader, model, criterion, epoch, device, method=cfg.MODEL.METHOD)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # On fait scheduler.step() avant de sauvegarder pour que le state_dict du scheduler
        # reflète bien "epoch terminé" ; à la reprise, le LR sera cohérent.
        scheduler.step()

        # ----- Checkpoint "best" : un seul fichier best.pth (on écrase l'ancien) -----
        # Évite de remplir le disque (surtout sur Colab). On ne garde que le meilleur modèle selon val MAE.
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            best_val_mae = val_mae
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_mae': best_val_mae,
                },
                str(checkpoint_dir / "best.pth")
            )

        # ----- Checkpoint "last" : à chaque fin d'epoch -----
        # Permet de reprendre avec --resume au bon epoch (modèle + optimizer + scheduler + best_val_mae).
        # Indispensable après une déconnexion Colab : on relance avec --resume checkpoint/last.pth
        # (en pointant le dossier sur Drive) et l'entraînement continue au lieu de repartir de zéro.
        model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
        torch.save(
            {
                'epoch': epoch + 1,
                'arch': cfg.MODEL.ARCH,
                'state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
            },
            str(checkpoint_dir / "last.pth")
        )

        if val_mae >= best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    main()




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
    model = get_model(model_name=cfg.MODEL.ARCH)

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

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate(val_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
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