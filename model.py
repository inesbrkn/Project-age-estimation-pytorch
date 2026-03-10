import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import timm
from defaults import _C as cfg
import torch
import torch.nn.functional as F

def compute_predictions(outputs, mode, device):
    
    if mode in ["dex", "weightLoss", "balancedSoftmax"]:
        ages = torch.arange(0, 101, device=device).float()
        probs = F.softmax(outputs, dim=-1)
        return (probs * ages).sum(dim=1)

    elif mode == "residual":

        cls_logits, residual = outputs
        predicted = cls_logits.argmax(1)
        pred_age = predicted.float() + residual.squeeze()

        return pred_age
    
    elif mode in ["gaussian", "laplace"]:
        mu, _ = outputs
        return mu.squeeze(-1).clamp(0, 100)

    elif  mode == "none": 
        return outputs.argmax(1).float()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    
def tta_predict2(model, x, mode, device, n_aug=5):
    preds = []

    model.eval()

    with torch.no_grad():

        for _ in range(n_aug):

            x_aug = x.clone()

            # flip horizontal aléatoire
            if torch.rand(1) < 0.5:
                x_aug = torch.flip(x_aug, dims=[3])

            outputs = model(x_aug)

            pred = compute_predictions(outputs, mode, device)

            preds.append(pred)

    preds = torch.stack(preds)   # shape : [n_aug, batch]

    mean_pred = preds.mean(0)
    std_pred = preds.std(0)

    return mean_pred, std_pred

def tta_predict(model, x, mode, device):
    """
    Test-Time Augmentation robuste.

    x : tensor [B, C, H, W]

    Returns
    -------
    mean_pred : moyenne des prédictions
    std_pred  : incertitude (écart-type)
    """

    tta_transforms = [
        lambda img: img,  # original
        lambda img: torch.flip(img, dims=[3]),  # horizontal flip
        lambda img: torch.rot90(img, 1, dims=[2,3]),  # rotate 90
        lambda img: torch.rot90(img, -1, dims=[2,3]), # rotate -90
        lambda img: F.interpolate(img, scale_factor=0.9, mode="bilinear", align_corners=False),
        lambda img: F.interpolate(img, scale_factor=1.1, mode="bilinear", align_corners=False),
    ]

    preds = []

    for t in tta_transforms:

        x_aug = t(x)

        # si resize change la taille on remet la taille originale
        if x_aug.shape[-1] != x.shape[-1]:
            x_aug = F.interpolate(x_aug, size=x.shape[-2:], mode="bilinear", align_corners=False)

        outputs = model(x_aug)
        pred = compute_predictions(outputs, mode, device)

        preds.append(pred)

    preds = torch.stack(preds)  # [TTA, B, classes]

    mean_pred = preds.mean(0)
    std_pred = preds.std(0)

    return mean_pred, std_pred
    
# ---------------------------
# Modèle Residual Method
# ---------------------------
class ResidualModel(nn.Module):
    def __init__(self, base_model, dim_feats, num_classes=101, p_dropout=0.5):
        super().__init__()
        self.base = base_model
        self.base.last_linear = nn.Linear(dim_feats, num_classes)  # classe principale
        self.residual = nn.Linear(dim_feats, 1)  # résidu
        self.base.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        feats = self.base.features(x)
        feats = self.base.avg_pool(feats).view(feats.size(0), -1)
        if cfg.DROPOUT:
            feats = self.dropout(feats)
        cls_out = self.base.last_linear(feats)
        res_out = self.residual(feats).squeeze(1)
        return cls_out, res_out

# ---------------------------
# Regression Head
# ---------------------------
class RegressionHead(nn.Module):
    def __init__(self, in_features, mode):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)  # mu + log_var/log_b
        self.mode = mode

    def forward(self, x):
        out = self.fc(x)
        mu, scale_param = out[:, 0], out[:, 1]
        if self.mode == "gaussian":
            return mu, scale_param
        elif self.mode == "laplace":
            return mu, scale_param
        else:
            raise ValueError(f"Unknown regression mode: {self.mode}")

# ---------------------------
# Modèle Regression (mu + incertitude)
# ---------------------------
class RegressionModel(nn.Module):
    def __init__(self, base_model, dim_feats, mode, p_dropout=0.5):
        super().__init__()
        self.base = base_model
        self.base.last_linear = nn.Identity()
        self.head = RegressionHead(dim_feats, mode=mode)
        self.dropout = nn.Dropout(p=p_dropout)
        self.base.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feats = self.base.features(x)
        feats = self.base.avg_pool(feats).view(feats.size(0), -1)
        if cfg.DROPOUT:
            feats = self.dropout(feats)
        return self.head(feats)

# ---------------------------
# Fonction pour activer Dropout en inference (MC-Dropout)
# ---------------------------
def enable_dropout(model):
    """Active tous les Dropout pour l'inférence."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_dropout_predict(model, x, mode, device, n_samples=50):
    """Estime la prédiction moyenne et l'incertitude via MC-Dropout."""
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)
            pred = compute_predictions(output, mode, device)
            preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    return mean_pred, std_pred

# ---------------------------
# Fonction pour récupérer le modèle
# ---------------------------
def get_model2(model_name="se_resnext50_32x4d", method=None, num_classes=101, pretrained="imagenet", p_dropout=0.5):
    """
    Retourne un modèle adapté à la méthode choisie :
      - method="dex" : DEX (softmax sur 101 classes)
      - method="residual" : Residual Method
      - method in ["gaussian", "laplace"] : RegressionModel
      - method=None ou "" : modèle classique
    """
    base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = base_model.last_linear.in_features
    base_model.avg_pool = nn.AdaptiveAvgPool2d(1)

    if method == "residual":
        return ResidualModel(base_model, dim_feats, num_classes, p_dropout=p_dropout)

    elif method in ["gaussian", "laplace"]:
        return RegressionModel(base_model, dim_feats, mode=method, p_dropout=p_dropout)

    else:
        base_model.last_linear = nn.Linear(dim_feats, num_classes)
        return base_model

# ---------------------------
# Test / main
# ---------------------------
def main():
    model = get_model2(method="residual")
    print(model)

if __name__ == '__main__':
    main()