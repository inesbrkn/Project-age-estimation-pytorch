import torch
import torch.nn as nn
import timm
from defaults import _C as cfg

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
    base_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",  # important
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