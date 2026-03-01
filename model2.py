import torch
import torch.nn as nn
import timm
from defaults import _C as cfg

# ---------------------------
# Modèle Residual Method
# ---------------------------
class ResidualModel(nn.Module):
    def __init__(self, base_model, dim_feats, num_classes=101, p_dropout=0.5):
        super().__init__()
        self.base = base_model
        self.base.classifier = nn.Linear(dim_feats, num_classes)  # classe principale
        self.residual = nn.Linear(dim_feats, 1)  # résidu
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        feats = self.base.forward_features(x)  # EfficientNet: forward_features pour extraire les features
        if cfg.DROPOUT:
            feats = self.dropout(feats)
        cls_out = self.base.classifier(feats)
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
        self.base.classifier = nn.Identity()
        self.head = RegressionHead(dim_feats, mode=mode)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        feats = self.base.forward_features(x)
        if cfg.DROPOUT:
            feats = self.dropout(feats)
        return self.head(feats)

# ---------------------------
# Fonction pour activer Dropout en inference (MC-Dropout)
# ---------------------------
def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_dropout_predict(model, x, n_samples=50):
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)
            if isinstance(output, tuple):  # ResidualModel
                cls_out, res_out = output
                pred = cls_out.argmax(dim=1).float() + res_out
            else:  # RegressionModel
                pred = output[0]  # mu
            preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    return mean_pred, std_pred

# ---------------------------
# Fonction pour récupérer le modèle
# ---------------------------
def get_model2(model_name="efficientnet_b0", method=None, num_classes=101, pretrained=True, p_dropout=0.5):
    """
    Retourne un modèle adapté à la méthode choisie :
      - method="dex" : DEX (softmax sur 101 classes)
      - method="residual" : Residual Method
      - method in ["gaussian", "laplace"] : RegressionModel
    """
    # timm.load_model: efficientNet
    base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0 pour features only
    dim_feats = base_model.num_features

    if method == "dex":
        base_model.classifier = nn.Linear(dim_feats, num_classes)
        return base_model

    elif method == "residual":
        return ResidualModel(base_model, dim_feats, num_classes, p_dropout=p_dropout)

    elif method in ["gaussian", "laplace"]:
        return RegressionModel(base_model, dim_feats, mode=method, p_dropout=p_dropout)

    else:
        base_model.classifier = nn.Linear(dim_feats, num_classes)
        return base_model

# ---------------------------
# Test / main
# ---------------------------
def main():
    model = get_model2(model_name="efficientnet_b0", method="residual")
    print(model)

if __name__ == '__main__':
    main()