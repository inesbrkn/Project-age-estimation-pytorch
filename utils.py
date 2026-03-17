
import torch
import numpy as np
from collections import OrderedDict

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