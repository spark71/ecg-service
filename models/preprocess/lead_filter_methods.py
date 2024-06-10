import pandas as pd
import torch
from scipy import signal

from models.preprocess.GAN_Arch_details import CycleGAN_Unet_Generator


def gan_preprocess(ecg: torch.Tensor, inference_count: int) -> torch.Tensor:
    '''
  :param ecg: Конкретное отведение ЭКС (4000)
  :inference_count: Количество применений фильтра
  :return Сгенерированный сигнал
  '''
    ecg = signal.resample(ecg, 4000)
    G_basestyle = CycleGAN_Unet_Generator()
    checkpoint = torch.load("model_weights_16NQ3.pth")
    G_basestyle.load_state_dict(checkpoint)
    G_basestyle.eval()
    G_basestyle.double()
    base, style = torch.from_numpy(ecg[None, None, :]).to(torch.double), torch.from_numpy(ecg[None, None, :]).to(
        torch.double)
    net = G_basestyle
    net.eval()
    output = base
    for i in range(inference_count):
        output = net(output)
    return output.squeeze()


def med_filter(ecg: torch.Tensor) -> torch.Tensor:
    #TODO: Медианный фильтр
    pass

def check_baseline(ecg: torch.Tensor) -> bool:
    #TODO Метод определяющий дрейф изолинии
    pass