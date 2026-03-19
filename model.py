import torch
import torch.nn as nn
import timm
from defaults import _C as cfg
import torch.nn.functional as F

def compute_predictions(outputs, mode, device):
    
    if mode == "dex" or mode == "weightLoss" or mode ==  "balancedSoftmax":
        ages = torch.arange(0, 101, device=device).float()
        probs = F.softmax(outputs, dim=-1)
        return (probs * ages).sum(dim=1)
    elif mode == "ordinal":
            logits = outputs
            # age estimé = somme des probabilités sigmoid(logits)
            probs = torch.sigmoid(logits)
            age_est = probs.sum(dim=1)
            
            return age_est
    elif mode == "residual":
        logits, residual = outputs
        ages = torch.arange(0, 101, device=device).float()
        probs = F.softmax(logits, dim=-1)
        predicted = (probs * ages).sum(dim=1)
        pred_age = predicted.float() + residual.squeeze()

        return pred_age
    
    elif mode in ["gaussian", "laplace"]:
        mu, _ = outputs
        return mu.squeeze(-1).clamp(0, 100)
        
    elif  mode == "none" : 
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

    preds = torch.stack(preds) 

    mean_pred = preds.mean(0)
    std_pred = preds.std(0)

    return mean_pred, std_pred

def tta_predict3(model, x, mode, device):

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

    preds = torch.stack(preds) 

    mean_pred = preds.mean(0)
    std_pred = preds.std(0)

    return mean_pred, std_pred

def tta_predict(model, x, mode, device):
    """
    Test Time Augmentation pour estimation d'âge.
    Retourne :
        mean_pred : prédiction moyenne
        std_pred  : incertitude (écart-type)
    """

    model.eval()

    tta_transforms = [
        lambda img: img,  # original
        lambda img: torch.flip(img, dims=[3]),  # horizontal flip
        lambda img: F.interpolate(img, scale_factor=0.95, mode="bilinear", align_corners=False),  # zoom out
        lambda img: F.interpolate(img, scale_factor=1.05, mode="bilinear", align_corners=False),  # zoom in
    ]

    preds = []

    with torch.no_grad():

        for t in tta_transforms:

            x_aug = t(x)

            # si le resize change la taille, on remet la taille originale
            if x_aug.shape[-2:] != x.shape[-2:]:
                x_aug = F.interpolate(
                    x_aug,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            outputs = model(x_aug)

            pred = compute_predictions(outputs, mode, device)

            preds.append(pred)

    preds = torch.stack(preds)   # [n_aug, batch]

    mean_pred = preds.mean(0)
    std_pred = preds.std(0)

    return mean_pred, std_pred
# =====================================================
# Residual Model
# =====================================================
class ResidualModel(nn.Module):
    def __init__(self, base_model, dim_feats, num_classes=101, p_dropout=0.5):
        super().__init__()
        self.base = base_model
        self.base.classifier = nn.Linear(dim_feats, num_classes)
        self.residual = nn.Linear(dim_feats, 1)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        # features EfficientNet
        feats = self.base.forward_features(x)

        # FIX CRITIQUE
        feats = self.base.global_pool(feats)
        feats = feats.flatten(1)

        if cfg.DROPOUT:
            feats = self.dropout(feats)

        cls_out = self.base.classifier(feats)
        res_out = self.residual(feats).squeeze(1)
        return cls_out, res_out


# =====================================================
# Regression Head (Gaussian / Laplace)
# =====================================================
class RegressionHead(nn.Module):
    def __init__(self, in_features, mode):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)  # mu + scale
        self.mode = mode

    def forward(self, x):
        out = self.fc(x)
        mu, scale_param = out[:, 0], out[:, 1]

        if self.mode in ["gaussian", "laplace"]:
            return mu, scale_param
        else:
            raise ValueError(f"Unknown regression mode: {self.mode}")


# =====================================================
# Regression Model
# =====================================================
class RegressionModel(nn.Module):
    def __init__(self, base_model, dim_feats, mode, p_dropout=0.5):
        super().__init__()
        self.base = base_model
        self.base.classifier = nn.Identity()
        self.head = RegressionHead(dim_feats, mode=mode)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        feats = self.base.forward_features(x)

        # FIX CRITIQUE
        feats = self.base.global_pool(feats)
        feats = feats.flatten(1)

        if cfg.DROPOUT:
            feats = self.dropout(feats)

        return self.head(feats)


# =====================================================
# Enable Dropout at inference (MC Dropout)
# =====================================================
def enable_dropout(model):
    """Active tous les Dropout pour l'inférence."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            print("Dropout layer:", m)
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



# =====================================================
# Model factory
# =====================================================
def get_model2(
    model_name="efficientnet_b0",
    method=None,
    num_classes=101,
    pretrained=True,
    p_dropout=0.5,
):
    """
    Retourne un modèle adapté à la méthode choisie :
      - dex
      - residual
      - gaussian
      - laplace
    """

    # timm propre
    if method in ["dex", "weightLoss", "none", "balancedSoftmax"]:

        base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    else:

        base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
    dim_feats = base_model.num_features

    # =========================
    # DEX
    # =========================
    if method == "dex":
        base_model.classifier = nn.Linear(dim_feats, num_classes)
        return base_model

    # =========================
    # Residual
    # =========================
    elif method == "residual":
        return ResidualModel(
            base_model,
            dim_feats,
            num_classes,
            p_dropout=p_dropout,
        )

    # =========================
    # Gaussian / Laplace
    # =========================
    elif method in ["gaussian", "laplace"]:
        return RegressionModel(
            base_model,
            dim_feats,
            mode=method,
            p_dropout=p_dropout,
        )

    elif method == "ordinal":
        # Ordinal regression : pour K classes (0..K-1), on prédit K-1 sorties binaires \"age > k ?\"
        num_thresholds = num_classes - 1
        base_model.last_linear = nn.Linear(dim_feats, num_thresholds)
        return base_model
    # =========================
    # fallback
    # =========================
    else:
        base_model.classifier = nn.Linear(dim_feats, num_classes)
        return base_model


# =====================================================
# Test
# =====================================================
def main():
    model = get_model2(
        model_name="efficientnet_b0",
        method="residual",
    )

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(" Model OK")
    if isinstance(y, tuple):
        print("cls shape:", y[0].shape)
        print("res shape:", y[1].shape)
    else:
        print("output shape:", y[0].shape)


if __name__ == "__main__":
    main()