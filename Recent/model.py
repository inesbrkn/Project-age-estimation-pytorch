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
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
