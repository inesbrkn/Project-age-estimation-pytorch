import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils

"""
    crée un modèle CNN pré-entraîné adapté à la prédiction d'âges.

    - les arguments
        model_name (str): nom du modèle pré-entraîné (par défaut SE-ResNeXt50).
        num_classes (int): nombre de classes de sortie (=> ici 101 pour les âges 0-100).
        pretrained (str): type de poids pré-entraînés à charger ("imagenet" ou None).

    - valeur de retour
        model (nn.Module): modèle prêt à l'entraînement.
"""
# ---------------------------
# Classe pour la Residual Method
# ---------------------------
class ResidualModel(nn.Module):
    def __init__(self, base_model, dim_feats, num_classes=101):
        super().__init__()
        self.base = base_model
        self.base.last_linear = nn.Linear(dim_feats, num_classes)  # classe principale
        self.residual = nn.Linear(dim_feats, 1)  # résidu
        self.base.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Extraire les features
        feats = self.base.features(x)
        feats = self.base.avg_pool(feats).view(feats.size(0), -1)
        cls_out = self.base.last_linear(feats)
        res_out = self.residual(feats).squeeze(1)
        return cls_out, res_out
    

def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    # Récupérer la dimension d'entrée de la dernière couche linéaire
    dim_feats = model.last_linear.in_features
    # Remplacer la dernière couche par une nouvelle couche adaptée à notre problème
    # Ici, on veut prédire 101 classes (âges de 0 à 100)
    model.last_linear = nn.Linear(dim_feats, num_classes)
      # Remplacer le pooling global par un pooling adaptatif
    # Cela permet de gérer des images d'entrée de tailles différentes
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

def get_model2(model_name="se_resnext50_32x4d", method=None, num_classes=101, pretrained="imagenet"):
    """
    Retourne un modèle adapté à la méthode choisie :
      - method="dex" : DEX (softmax sur 101 classes)
      - method="residual" : Residual Method
      - method="ordinal" : Ordinal Regression (K-1 sorties binaires \"age > k ?\")
      - method=None ou "" : modèle classique (dernier layer linéaire = num_classes)
    """
    base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = base_model.last_linear.in_features
    base_model.avg_pool = nn.AdaptiveAvgPool2d(1)

    if method == "dex":
        base_model.last_linear = nn.Linear(dim_feats, num_classes)
        return base_model

    elif method == "ordinal":
        # Ordinal regression : pour K classes (0..K-1), on prédit K-1 sorties binaires \"age > k ?\"
        num_thresholds = num_classes - 1
        base_model.last_linear = nn.Linear(dim_feats, num_thresholds)
        return base_model

    elif method == "residual":
        return ResidualModel(base_model, dim_feats, num_classes)

    else:  # comportement par défaut
        base_model.last_linear = nn.Linear(dim_feats, num_classes)
        return base_model
""" 
Le pooling c'est quoi ? une couche intermédiaire du réseau de neurone. 
Son but : réduire la taille des images/features tout en conservant les informations importantes.

average pooling:

Chaque pixel représente la présence d’un motif dans une zone.
exemple pixel 2*2 

le pixel en question:
1 2
3 4

average pooling va faire la moyenne : AVGPooling = (1 +2 + 3 + 4)/4 = 2.5
la zone est résumé par cette valeur là 
"""

def main():
    model = get_model2()
    print(model)


if __name__ == '__main__':
    main()
