import argparse
import better_exceptions
from pathlib import Path
from plot_log import plot_uncertainty
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model2
from dataset import FaceDataset
from defaults import _C as cfg
from train import mae_by_age_group, run_epoch, get_criterion




def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, required=True, help="Model weight to be tested")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model2(model_name=cfg.MODEL.ARCH,method=cfg.MODEL.METHOD, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    test_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    criterion = get_criterion(cfg.MODEL.METHOD, alpha=0.5, device=device)
    print("=> start testing")
    if cfg.MC_DROPOUT or cfg.TTA > 0:
        test_loss, test_mae , test_acc, gt, preds, std= run_epoch(test_loader, model, criterion, None, 0, device, mode=cfg.MODEL.METHOD, is_train=False, return_preds=True)
        plot_uncertainty(preds, gt, std)
    else :
        test_loss, test_mae, test_acc, preds, gt = run_epoch(test_loader,model,criterion,None,0,device,mode=cfg.MODEL.METHOD,is_train=False)    
    
    print(f"test loss: {test_loss:.3f}")
    print(f"test mae: {test_mae:.3f}")
    print(f"test acc: {test_acc:.3f}")

     # MAE par tranche d'âge (enfants / adultes / seniors) pour analyse des erreurs
    group_results = mae_by_age_group(preds, gt)
    print("=> MAE by age group:")
    for r in group_results:
        print(f"  {r['name']} yrs: mae={r['mae']:.3f}, n={r['count']}, std(err)={r['std']:.3f}")

if __name__ == '__main__':
    main()