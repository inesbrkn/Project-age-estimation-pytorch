from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
# Backbone : python train.py -h liste les noms. Ex. se_resnext50_32x4d, resnet18, resnet50, etc.
_C.MODEL.ARCH = "se_resnext50_32x4d"
_C.MODEL.IMG_SIZE = 224
_C.MODEL.METHOD = "ordinal"  # "dex" (classification DEX) ou "residual" (Residual DEX)
_C.MODEL.LABEL_SMOOTHING = 0.1  # label smoothing pour la partie classification (DEX ou Residual)
_C.MODEL.TASK = "residual"  # champ descriptif : "dex", "residual" ou "classification" selon l'expérience
# Train
_C.TRAIN = CN()
_C.TRAIN.SEED = 42  # seed fixe pour runs reproductibles (comparaison DEX vs Residual équitable)
_C.TRAIN.OPT = "adam"  # adam or sgd
_C.TRAIN.WORKERS = 2
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20
_C.TRAIN.LR_DECAY_RATE = 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BATCH_SIZE = 60
_C.TRAIN.EPOCHS = 60
_C.TRAIN.AGE_STDDEV = 1.0

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 128
# Tranches d'âge pour l'analyse des erreurs (MAE par groupe)
_C.TEST.AGE_GROUPS = [(0, 17), (18, 45), (46, 100)]  # enfants, adultes, seniors