import torch
import torch.nn as nn
from defaults import _C as cfg

"""
class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.2, labelSmoothing=0.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        self.res_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, outputs, target):

        cls_logits, residual = outputs
        # classification loss
        loss_cls = self.cls_loss(cls_logits, target)
        # predicted class
        pred_class = cls_logits.argmax(dim=1)
        # residual target
        residual_target = pred_class.float() - target.float()
        loss_res = self.res_loss(residual, residual_target)
        return loss_cls + self.alpha * loss_res

class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.5, labelSmoothing=0.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        self.res_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, outputs, target):
  
        cls_logits, residual = outputs

        # ---------- classification ----------
        loss_cls = self.cls_loss(cls_logits, target)

        # ---------- residual (STABLE VERSION) ----------
        # IMPORTANT: utiliser la vraie classe, pas la prédite
        residual_target = target.float() - target.float()

        # en pratique on veut:
        # residual_target = true_age - true_class
        # mais ici target EST déjà la classe (0–100)

        residual_target = torch.zeros_like(target, dtype=torch.float)

        loss_res = self.res_loss(residual.squeeze(-1), residual_target)

        return loss_cls + self.alpha * loss_res


class ResidualLoss(nn.Module):
    def __init__(self, alpha=0.5, labelSmoothing=0.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        self.res_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, outputs, target):
        cls_logits, residual = outputs

        # classe entière
        target_class = target.long()

        # classification
        loss_cls = self.cls_loss(cls_logits, target_class)

        # résidu cible
        residual_target = target.float() - target_class.float()

        loss_res = self.res_loss(residual.squeeze(-1), residual_target)

        return loss_cls + self.alpha * loss_res

"""

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
    else:
        raise ValueError(f"Unknown mode: {mode}")

