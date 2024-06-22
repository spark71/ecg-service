import os

import torch
from dotenv import load_dotenv

from models.nn.xresnet1d import xresnet1d101
from models.nn.inception1d import inception1d
from models.nn.resnet1d import resnet1d_wang
from models.nn.rnn1d import RNN1d
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.ecg_crnn import ECG_CRNN

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")


def model_factory(model_name):
    model = None
    if model_name.lower()=='xresnet1d101':
        model = xresnet1d101(input_channels=12, num_classes=5)

    if model_name.lower()=='resnet1d':
        # model = resnet1d_wang(input_channels=12, num_classes=5)
        resnet1d_wang_model = resnet1d_wang(input_channels=12, num_classes=5)
        # resnet1d_wang_weights = r'C:\Users\redmi\PycharmProjects\ecg-tool-api\models\pretrained\resnet1d_wang\resnet1d_wang_fold1_16epoch_best_score.pth'
        resnet1d_wang_weights = (ROOT_DIR +
                                 r'/models/pretrained/resnet1d_wang/resnet1d_wang_fold1_16epoch_best_score.pth')
        # xresnet1d_model.load_state_dict(torch.load(xresnet1d_model_weights_path, map_location=torch.device('cpu'))['model'])
        resnet1d_wang_model.load_state_dict(
            torch.load(resnet1d_wang_weights, map_location=torch.device('cpu'))['model'])
        model = resnet1d_wang_model.double().eval()
        # resnet1d_wang_model

    if model_name.lower()=='inception1d':
        # model = inception1d(input_channels=12, num_classes=5)
        inception1d_model = inception1d(input_channels=12, num_classes=5)
        # inception1d_model_weights_path = r'C:\Users\redmi\PycharmProjects\ecg-tool-api\models\pretrained\inception1d\inception1d_fold1_15epoch_best_score.pth'
        inception1d_model_weights_path = (ROOT_DIR +
                                          r'/models/pretrained/inception1d/inception1d_fold1_15epoch_best_score.pth')
        inception1d_model.load_state_dict(
            torch.load(inception1d_model_weights_path, map_location=torch.device('cpu'))['model'])
        model = inception1d_model.double().eval()


    if model_name.lower()=='rnn_1d':
        model = RNN1d(input_channels=12, num_classes=5)

    if model_name.lower() == 'vgg16':
        vgg16_model_weights_path = (ROOT_DIR +
                                    r'/models/pretrained/vgg16/vgg16_fold1_10epoch_best_score.pth')
        config = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG, fs=100)
        config.cnn.name = "vgg16"
        classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
        n_leads = 12
        vgg16_model = ECG_CRNN(classes, n_leads, config)
        vgg16_model.load_state_dict(torch.load(vgg16_model_weights_path, map_location=torch.device('cpu'))['model'])
        model = vgg16_model.double().eval()
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
    model_name = 'tf_efficientnetv2_s'
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
    model_name = 'tf_efficientnetv2_s'
    train = True
    grad_cam = False
    early_stop = True
    fc_dim = 512
    margin = 0.5
    scale = 30
    early_stopping_steps = 5
    seed = 42

if __name__ == '__main__':
    # model = model_factory("vgg16")
    # model = model_factory("inception1d")
    model = model_factory("resnet1d")
    print(model)

