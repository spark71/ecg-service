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




if CqtCFG.debug:
    CqtCFG.epochs = 1
    folds = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)



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

