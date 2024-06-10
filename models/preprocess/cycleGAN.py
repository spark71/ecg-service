from utils import TECGDataModule
from tqdm import tqdm
import pandas as pd
import torch
from GAN_Arch_details import CycleGAN_Unet_Generator

G_basestyle = CycleGAN_Unet_Generator()

checkpoint =torch.load("model_weights_16NQ3.pth")

G_basestyle.load_state_dict(checkpoint)

G_basestyle.eval()


data_dir = "data/npy_signals100/"
batch_size = 8
dm = TECGDataModule(data_dir, batch_size, phase='test')
dm.prepare_data()
dataloader = dm.train_dataloader()
base, style = next(iter(dataloader))
print('Input Shape {}, {}'.format(base.size(), style.size()))
print(type(base))
net = G_basestyle
net.eval()
predicted = []
predicted=pd.DataFrame(data=predicted)
actual = []
actual=pd.DataFrame(data=actual)