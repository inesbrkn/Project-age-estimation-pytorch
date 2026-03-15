import pandas as pd
import torch
import torch.nn as nn
from age_distribution import count_examples_by_age
from defaults import _C as cfg
import torch.nn.functional as F
"""
class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.2, label_smoothing=0.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.res_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, outputs, target):

        cls_logits, residual = outputs

        # classification loss
        loss_cls = self.cls_loss(cls_logits, target)

        # predicted class
        pred_class = cls_logits.argmax(dim=1).detach()A

        residual_target = target.float() - pred_class.float()

        loss_res = self.res_loss(residual, residual_target)

        return loss_cls + self.alpha * loss_res
"""

class ResidualLoss(nn.Module):

    def __init__(self, alpha=0.5, label_smoothing=0.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.res_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, outputs, target):

        cls_logits, residual = outputs

        class_target = target.long()

        residual_target = target.float() - class_target.float()

        loss_cls = self.cls_loss(cls_logits, class_target)

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

class WeightedCrossEntropy(nn.Module):
    def __init__(self, weights_per_age, device="cpu"):
        super().__init__()
        self.weights_per_age = weights_per_age.to(device)
        self.ce = nn.CrossEntropyLoss(reduction='none')  # per sample

    def forward(self, outputs, target):
        loss_raw = self.ce(outputs, target)         
        weights = self.weights_per_age[target]      
        loss = (loss_raw * weights).mean()
        return loss

class BalancedSoftmaxLoss(nn.Module):

    def __init__(self, class_counts, device="cpu"):
        super().__init__()

        counts = torch.tensor(class_counts, dtype=torch.float32)
        self.log_prior = torch.log(counts)
        self.log_prior = self.log_prior.to(device)

    def forward(self, logits, target):

        logits = logits + self.log_prior

        return F.cross_entropy(logits, target)

# à appeler pour recup le mode choisi
def get_criterion(mode, alpha=0.5, device="cpu"):
    labelSmoothing = cfg.MODEL.LABEL_SMOOTHING
    if mode == "residual":
        return ResidualLoss(alpha=alpha, label_smoothing=labelSmoothing).to(device)
    elif mode == "dex" or mode == "none":
        return nn.CrossEntropyLoss(label_smoothing=labelSmoothing).to(device)
    elif mode == "gaussian":
        return GaussianLikelihoodLoss().to(device)
    elif mode == "laplace":
        return LaplaceLikelihoodLoss().to(device) 
    elif mode == "weightLoss":
        df = pd.read_csv("appa-real-release/gt_avg_train.csv")
        counts = df["apparent_age_avg"].round().value_counts().sort_index()
        counts = counts.reindex(range(101), fill_value=1) + 1  # éviter poids infini

        N = counts.sum()
        K = len(counts)

        weights = N / (K * counts + 1e-6)
        
        class_weights = torch.tensor(weights.values, dtype=torch.float32).to(device)
        return WeightedCrossEntropy(class_weights, device=device)

    elif mode == "balancedSoftmax":

        df = pd.read_csv("appa-real-release/gt_avg_train.csv")

        counts = df["apparent_age_avg"].round().value_counts().sort_index()

        counts = counts.reindex(range(101), fill_value=1)

        return BalancedSoftmaxLoss(counts.values, device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

