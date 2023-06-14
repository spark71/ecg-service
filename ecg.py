import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


class EcgLoader():
    def load_sample_(self):
        pass


class EcgSignal():
    def __int__(self):
        pass


    @staticmethod
    def take_sample_zenodo(self, person_idx: int, zone: str='II', show: bool=False) -> np.ndarray:
        """
        :idx:  индекс записи экг
        :zone: отведение
        :show: график
        ->    значения записи экг
        """
        file = h5py.File('data_zenodo/ecg_tracings.hdf5', 'r')
        file.keys()
        data = file.get('tracings')
        ecg_data = pd.DataFrame(data=data[person_idx]).rename(
            columns={0: 'I', 1: 'II', 2: 'III', 3: 'v1', 4: 'v2', 5: 'v3', 6: 'v4', 7: 'v5', 8: 'v6', 9: 'aVR',
                     10: 'aVL',
                     11: 'aVF'})
        ecg_sample = ecg_data[zone].values

        if show:
            plt.figure(figsize=(18, 8))
            fig, ax = plt.subplots()

            # plt.plot(ecg_data['v2'])
            sns.lineplot(data=ecg_data[zone], color='black')
            plt.title(zone)
            plt.plot()

        return ecg_sample

    def take_ecg_sample_xldb(self):
        pass

    def plot_sample(self):
        pass

    def detect_r_peaks(self):
        pass

    def detect_waves(self):
        pass

    def make_model_signal(self):
        pass

    def overlay_model_signal(ecg_sample, self):
        pass


if __name__ == '__main__':
    sample = EcgSignal.take_sample_zenodo(1, True)
    print(sample)




# file = h5py.File('/content/drive/MyDrive/Дипломная работа/ecg_tracings.hdf5', 'r')
# # file = h5py.File('ecg_tracings.hdf5', 'r')
#
# file.keys()
# data = file.get('tracings')
#
#
# def take_ecg_sample(person_idx, zone='II', show=False):
#     """
#     idx:  индекс записи экг
#     zone: отведение
#     show: график
#     ->    значения записи экг
#     """
#
#     ecg_data = pd.DataFrame(data=data[person_idx])
#     ecg_data = ecg_data.rename(
#         columns={0: 'I', 1: 'II', 2: 'III', 3: 'v1', 4: 'v2', 5: 'v3', 6: 'v4', 7: 'v5', 8: 'v6', 9: 'aVR', 10: 'aVL',
#                  11: 'aVF'})
#     ecg_sample = ecg_data[zone].values
#
#     if show:
#         plt.figure(figsize=(18, 8))
#         # fig, ax = plt.subplots()
#
#         # plt.plot(ecg_data['v2'])
#         sns.lineplot(data=ecg_data['II'], color='black')
#         plt.title('II')
#
#     return ecg_sample
