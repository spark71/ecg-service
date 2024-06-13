import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy import signal
from torch_ecg.preprocessors import BaselineRemove
from torch_ecg.utils import normalize

from models.preprocess.GAN_Arch_details import CycleGAN_Unet_Generator
import os

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")

model_weights_path = ROOT_DIR + '\models\preprocess\model_weights_16NQ3.pth'


def gan_preprocess(ecg: torch.Tensor, inference_count: int) -> torch.Tensor:
    '''
  :param ecg: Конкретное отведение ЭКС (4000)
  :inference_count: Количество применений фильтра
  :return Сгенерированный сигнал
  '''
    ecg = signal.resample(ecg, 4000)
    G_basestyle = CycleGAN_Unet_Generator()
    checkpoint = torch.load(model_weights_path)
    G_basestyle.load_state_dict(checkpoint)
    G_basestyle.eval()
    G_basestyle.double()
    base, style = torch.from_numpy(ecg[None, None, :]).to(torch.double), torch.from_numpy(ecg[None, None, :]).to(
        torch.double)
    net = G_basestyle
    net.eval()
    output = base
    with torch.no_grad():
        for i in range(inference_count):
            output = net(output)
    output = signal.resample(output.squeeze().detach().numpy(), 1000)
    print(output.shape)
    return output


def med_filter(ecg) -> torch.Tensor:
    return BaselineRemove(100)(ecg[None, :])[0]


def moving_med(x, w):
    armed = []
    index = w // 2
    it = 0
    while it + w < len(x):
        if w % 2:
            armed.append(sorted(x[it:it + w])[index])
        else:
            armed.append(sum(sorted(x[it:it + w])[index - 1:index + 1]) / 2)
        it += 1
    return armed


def check_baseline(ecg_signal) -> bool:
    norm_signal = normalize(ecg_signal, method='min-max')
    std_med = np.std(moving_med(norm_signal, 66))
    return std_med > 0.1
