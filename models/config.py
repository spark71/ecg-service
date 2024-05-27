from models.nn.xresnet1d import xresnet1d101
from models.nn.inception1d import inception1d
from models.nn.resnet1d import resnet1d_wang
from models.nn.rnn1d import RNN1d

def model_factory(model_name):
    model = None
    if model_name.lower()=='xresnet1d101':
        model = xresnet1d101(input_channels=12, num_classes=5)

    if model_name.lower()=='resnet1d_wang':
        model = resnet1d_wang(input_channels=12, num_classes=5)

    if model_name.lower()=='inception1d_model':
        model = inception1d(input_channels=12, num_classes=5)

    if model_name.lower()=='rnn_1d':
        model = RNN1d(input_channels=12, num_classes=5)

    return model

def load_pretrained_model(model_name):
    pass


class CqtCFG:
    apex = False
    debug = False
    print_freq = 100
    image_size = 224
    num_workers = 2
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','OneCycleLR']
    epochs = 20
    # CosineAnnealingLR params
    cosanneal_params = {
        'T_max': 3,
        'eta_min': 1e-5,
        'last_epoch': -1
    }
    # ReduceLROnPlateau params
    reduce_params = {
        'mode': 'min',
        'factor': 0.2,
        'patience': 4,
        'eps': 1e-6,
        'verbose': True
    }
    # CosineAnnealingWarmRestarts params
    cosanneal_res_params = {
        'T_0': 3,
        'eta_min': 1e-6,
        'T_mult': 1,
        'last_epoch': -1
    }
    onecycle_params = {
        'pct_start': 0.1,
        'div_factor': 1e2,
        'max_lr': 1e-3
    }
    batch_size = 2
    lr = 1e-4
    weight_decay = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_size = 5
    nfolds = 9
    qtransform_params = {"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}
    trn_fold = [9]
    target_col = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    preds_col = ['pred_CD', 'pred_HYP', 'pred_MI', 'pred_NORM', 'pred_STTC']
    model_name = 'tf_efficientnetv2_s'  # 'vit_base_patch32_224_in21k' 'resnext50_32x4d' 'tf_efficientnet_b3' 'resnetv2_50x1_bitm_in21k' 'inception_v4' 'tf_efficientnetv2_s_in21k'
    train = True
    grad_cam = False
    early_stop = True
    fc_dim = 512
    margin = 0.5
    scale = 30
    early_stopping_steps = 5
    seed = 42




# if CqtCFG.debug:
#     CqtCFG.epochs = 1
#     folds = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)



class RawCfg:
    num_workers = 2
    batch_size = 2
    lr = 1e-4
    weight_decay = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_size = 5
    nfolds = 9
    train_folds = [1,2,3,4,5,6,7,8,9]
    target_col = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    preds_col = ['pred_CD', 'pred_HYP', 'pred_MI', 'pred_NORM', 'pred_STTC']
    model_name = 'tf_efficientnetv2_s'  # 'vit_base_patch32_224_in21k' 'resnext50_32x4d' 'tf_efficientnet_b3' 'resnetv2_50x1_bitm_in21k' 'inception_v4' 'tf_efficientnetv2_s_in21k'
    train = True
    grad_cam = False
    early_stop = True
    fc_dim = 512
    margin = 0.5
    scale = 30
    early_stopping_steps = 5
    seed = 42

