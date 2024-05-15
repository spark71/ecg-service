import enum
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from enum import Enum
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import h5py
import torch
from torch.utils.data import Dataset
from nnAudio.features.cqt import CQT1992v2
from models.config import CqtCFG, RawCfg
import neurokit2 as nk

# Папка с проектом
#
# ROOT_DIR = r'C:\Users\User\PycharmProjects\ecg-service'
ROOT_DIR = r'C:\Users\redmi\PycharmProjects\ecg-tool-api'



@enum.unique
class Datasets(enum.Enum):
    """
    Enumeration of datasets
    """
    # ptbxl
    ptbxl = 'data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv'
    ptbxl_scp_statements = 'data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv'
    # code-test
    code_test = 'data/CODE-test-12-lead-ecg-dataset/ecg_tracings.hdf5'
    code_test_attrs = 'data/CODE-test-12-lead-ecg-dataset/attributes.csv'
    @property
    def path(self):
        return os.path.join(ROOT_DIR, self.value)


class EcgDataset(Dataset):
    def __init__(self, df, feature="raw", transform=None):
        self.df = df
        self.file_names = df['file_paths'].values
        self.labels = df[RawCfg.target_col].values
        self.wave_transform = CQT1992v2(**CqtCFG.qtransform_params)
        self.transform = transform
        self.feature = feature

    def __len__(self):
        return len(self.df)

    # @classmethod
    def apply_qtransform(self, waves, transform):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def get_feature(self, type: str, signal: np.ndarray):
        """
        :param type: raw, cqt
        :param signal: 12-leads array
        :return: image or signal
        """
        if type.lower() == 'raw':
            return torch.from_numpy(signal.T).to(torch.double)[None, :]

        if type.lower() == 'cqt':
            image = self.apply_qtransform(signal, self.wave_transform)
            return image.squeeze()[None, None, :]

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        signal = np.load(file_path)
        # scaled_signal = apply_scaler(signal, scaler=scaler)
        signal_tensor = self.get_feature(type=self.feature, signal=signal)
        label = torch.tensor(self.labels[idx]).float()
        return signal_tensor, label





def get_transforms(*, data='train'):
    '''
    Return Augmented Image tensor for training dataset
    '''

    if data == 'train':
        return A.Compose(
            [
                # A.Resize(CFG.image_size,CFG.image_size),
                # A.HorizontalFlip(p=0.3),
                # A.VerticalFlip(p=0.3),
                # A.Rotate(limit=180, p=0.3),
                # A.RandomBrightness(limit=0.6, p=0.5),
                # A.Cutout(
                # num_holes=10, max_h_size=12, max_w_size=12,
                # fill_value=0, always_apply=False, p=0.5
                # ),
                # A.ShiftScaleRotate(
                #    shift_limit=0.25, scale_limit=0.1, rotate_limit=0
                # ),
                ToTensorV2(p=1.0),
            ]
        )

    elif data == 'valid':
        return A.Compose([
            ToTensorV2(),
        ])

class EcgLoader:
    def load_sample_(self):
        pass


class EcgSignal:
    def __init__(self):
        pass

    @staticmethod
    def take_sample_codetest( person_idx: int, path: str = Datasets.code_test.path, zone: str = 'II', show: bool = False,
                           prep: bool = True) -> np.ndarray:
        """ CODE-test-12-lead-ecg-dataset
        Описание:

        :idx:  индекс записи экг
        :zone: отведение
        :show: график
        ->    значения записи экг
        """
        file = h5py.File(path, 'r')
        file.keys()
        data = file.get('tracings')
        ecg_data = pd.DataFrame(data=data[person_idx]).rename(
            columns={0: 'I', 1: 'II', 2: 'III', 3: 'v1', 4: 'v2', 5: 'v3', 6: 'v4', 7: 'v5', 8: 'v6', 9: 'aVR',
                     10: 'aVL',
                     11: 'aVF'})
        ecg_sample = ecg_data[zone].values
        if prep:
            ecg_sample = EcgSignal.del_zspan(ecg_sample)
        if show:
            plt.figure(figsize=(18, 8))
            fig, ax = plt.subplots()
            sns.lineplot(data=ecg_data[zone], color='black')
            plt.title(zone)
            plt.plot()
        return ecg_sample

    @staticmethod
    #TODO fix paths
    def take_ecg_sample_xldb(person_idx: int, zone: str = 'II', sr = 500, show = False):
        """
        zone (lead-отведение):  ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        sr (sample rate): 100Hz, 500Hz
        """
        path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records500/'
        if sr == 100:
            path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/'
        if person_idx > 21837 or person_idx < 1:
            raise ValueError('Wrong index, 21837-last')
        len_pidx = abs(len(str(person_idx)) - 5)
        if len_pidx !=  0:
            str_idx = '0' * len_pidx
            str_idx += str(person_idx)
        else:
            str_idx = str(person_idx)

        str_fold = str_idx[:2] + "000"
        path += str_fold + '/' + str_idx
        if sr == 100:
            hea_path = path + '_lr.hea'
        else:
            hea_path = path + '_hr.hea'
        record = ECGRecord.from_wfdb(hea_path)
        time = record.time
        signal = np.array(record.get_lead(zone))
        if show:
            signal_plot(signal)
        return signal

    @staticmethod
    def plot_sample(sample, figsize=(18, 8), leads=None):
        """

        :param ecg_sample: np.array, (ch, lead_values)
        :param figsize:
        :param list(leads) | all | None - plot one lead
        :return: plot
        """
        # fig, ax = plt.subplots(figsize=figsize)
        # plt.rcParams.update({'font.size': 22})
        # ax.set_xlabel('time [n] - отсчёты')
        # ax.set_ylabel('mV')
        # plt.plot(ecg_sample)

        if type(leads) is list:
            bar, axes = plt.subplots(len(leads), 1, figsize=figsize)
            for lead in leads:
                sns.lineplot(x=np.arange(sample.shape[0]), y=sample[:, lead-1], ax=axes[lead-1])
            plt.show()
        elif leads == "all":
        # bar, axes = plt.subplots(sample.shape[1], 1, figsize=(30, 20))
            bar, axes = plt.subplots(sample.shape[1], 1, figsize=figsize)
            plt.rcParams.update({'font.size': 14})
            for i in range(sample.shape[1]):
                sns.lineplot(x=np.arange(sample.shape[0]), y=sample[:, i], ax=axes[i])
            plt.show()
        if leads is None:
            fig, ax = plt.subplots(figsize=figsize)
            plt.rcParams.update({'font.size': 22})
            ax.set_xlabel('time [n] - отсчёты')
            ax.set_ylabel('mV')
            plt.plot(sample)


    @staticmethod
    def del_zspan(ecg_sample):
        """
        Удаление нулевых хвостов
        :param ecg_sample: ecg-lead values
        :return: clear ecg sample without zero tails
        """
        del_fence_start = None
        del_fence_end = None
        for i in range(len(ecg_sample)-1, len(ecg_sample) // 2, -1):
            if ecg_sample[i] != 0:
                del_fence_end = i
                break
        for i in range(0, len(ecg_sample) // 2):
            if ecg_sample[i] != 0:
                del_fence_start = i
        return ecg_sample[del_fence_start:del_fence_end]

    @staticmethod
    def detect_r_peaks(ecg_sample, max_th: float, min_dist: float, show: bool = False) -> list:
        """
        :param ecg_sample: значения ЭКС
        :param max_th: амплитудный порог [0;1]
        :param min_dist: минимальная дистанция RR
        :param show: вывод графика
        :return:
        """
        # амплитудно-пороговый метод нахождения r-пика
        peak_indicies = peakutils.indexes(ecg_sample, thres=max_th, min_dist=min_dist)
        if show:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.set_title('R-peaks')
            ax.set_xlabel('time [n] - отсчёты')
            ax.set_ylabel('Δφ, mV')
            ax.plot(ecg_sample)
            for peak in peak_indicies:
                #   Отображение временных меток (Детекция R-пиков)
                ax.axvline(x=peak, color="r")
        return peak_indicies

    @classmethod
    def detect_waves(self, signal, sr, method='dwt', show=False):
        #TODO: fix
        """
        Детекция особых точек ЭКС
        :self: запись экг
        :method:     метод обнаружения волн ['dwt', 'cwt', 'peak']
        :sr:         sample rate - частота дискретизации


        :return:     _, waves_peak, waves_df, waves_df_rel, mean_pqrst_amp
        ->           словарь индексов pqrst-волн, маркированный график сигнала,
                     словарь относительных точек внутри PPi
        """
        rpeaks = self.detect_r_peaks(signal, max_th=.7, min_dist=100)
        # компенсация шумовых помех
        ecg_sample_clean = nk.ecg_clean(signal, sampling_rate=sr)

        _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=sr, method="dwt", show_type='peaks')

        #-------------------------------------------------------------------------------------------------------
        # Создадим датасет, куда будут входить сами точки их коэф-ты положения внутри PP
        waves_df = pd.DataFrame(data=waves_peak)

        # Добавим R-пики
        waves_df['r_peaks'] = self.detect_r_peaks(signal, max_th=.7, min_dist=100, show=False)

        # Считаем длительности PP в отсчётах
        pon_diff = np.diff(waves_df['ECG_P_Onsets'])
        p_on_diff = np.concatenate((np.array(pon_diff), np.array([pon_diff[-1]])), axis=0)

        # Добавляем в датасет
        waves_df['P_Onsets_diff'] = p_on_diff

        #Коэф-ты положения
        waves_df['p_rel'] = (waves_df['ECG_P_Peaks'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['p_off_rel'] = (waves_df['ECG_P_Offsets'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['q_rel'] = (waves_df['ECG_Q_Peaks'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['r_rel'] = (waves_df['r_peaks'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['s_rel'] = (waves_df['ECG_S_Peaks'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['t_on_rel'] = (waves_df['ECG_T_Onsets'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['t_rel'] = (waves_df['ECG_T_Peaks'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']
        waves_df['t_off_rel'] = (waves_df['ECG_T_Offsets'] - waves_df['ECG_P_Onsets']) / waves_df['P_Onsets_diff']

        # доп коэф-ты для точек q' s' n1 n2 n3 n4
        waves_df['_q_rel'] = (waves_df['p_off_rel'] + waves_df['q_rel']) / 2.2
        waves_df['_s_rel'] = (waves_df['s_rel'] + waves_df['t_on_rel']) / 2.8

        waves_df['n1'] = waves_df['p_rel'] / 1.6
        waves_df['n2'] = (waves_df['p_rel'] + waves_df['p_off_rel'] ) / 2
        waves_df['n3'] = (waves_df['t_on_rel'] + waves_df['t_rel'] ) / 2
        waves_df['n4'] = (waves_df['t_off_rel'] + waves_df['t_rel'] ) / 2
        #Уберём лишние значения и переопределим порядок в датасете с относительными значениями
        waves_df.loc[len(waves_df)-1, 'p_rel':] = 0

        waves_df_rel = waves_df.loc[:, 'ECG_T_Offsets':'n4'].copy().reindex(
            columns=['n1', 'p_rel', 'n2', 'p_off_rel', '_q_rel', 'q_rel', 'r_rel', 's_rel', '_s_rel',
                                   't_on_rel', 'n3', 't_rel', 'n4', 't_off_rel'])
        waves_df_rel['pp_diff'] = p_on_diff
        # Посчитаем среднюю амплитуду волновых точек P Q R S T
        # Находим усреднённые характеристики амплитуд pqrst     (upd 1)
        p_mean_amp = np.mean(signal[ np.array(waves_df[waves_df['ECG_P_Peaks'].notnull()]['ECG_P_Peaks']).astype(int) ])
        q_mean_amp = np.mean(signal[ np.array(waves_df[waves_df['ECG_Q_Peaks'].notnull()]['ECG_Q_Peaks']).astype(int) ])
        r_mean_amp = np.mean(signal[ np.array(waves_df[waves_df['r_peaks'].notnull()]['r_peaks']).astype(int) ])
        s_mean_amp = np.mean(signal[ np.array(waves_df[waves_df['ECG_S_Peaks'].notnull()]['ECG_S_Peaks']).astype(int) ])
        t_mean_amp = np.mean(signal[ np.array(waves_df[waves_df['ECG_T_Peaks'].notnull()]['ECG_T_Peaks']).astype(int) ])

        # список со средними амплитудами для P Q R S T
        mean_pqrst_amp = np.around([p_mean_amp, q_mean_amp, r_mean_amp, s_mean_amp, t_mean_amp], 3)

        #-------------------------------------------------------------------------------------------------------
        # выводим график с маркерами волн, из словаря waves_peak
        if show:
            fig, ax  = plt.subplots(figsize=(15,8))
            ax.set_title('PQST-waves')
            ax.set_xlabel('time [n] - отсчёты')
            ax.set_ylabel('mV')
            ax.plot(signal)

            # Заменим nan на numpy.nan
            for wave in waves_peak.keys():
                for i in range(len(waves_peak['ECG_P_Peaks'])):
                    if pd.isna(waves_peak[wave][i]):
                        waves_peak[wave][i] = np.nan

            for i in range(len(waves_peak['ECG_P_Peaks'])):

                # значения сигнала в определённых индексах волн

                if not pd.isna(waves_peak['ECG_P_Peaks'][i]):
                    peak_p_value = signal[waves_peak['ECG_P_Peaks'][i]]
                else:
                    peak_p_value = np.nan

                if not pd.isna(waves_peak['ECG_P_Onsets'][i]):
                    peak_pon_value = signal[waves_peak['ECG_P_Onsets'][i]]
                else:
                    peak_pon_value = np.nan

                if not pd.isna(waves_peak['ECG_Q_Peaks'][i]):
                    peak_q_value = signal[waves_peak['ECG_Q_Peaks'][i]]
                else:
                    peak_q_value = np.nan

                if not pd.isna(waves_peak['ECG_S_Peaks'][i]):
                    peak_s_value = signal[waves_peak['ECG_S_Peaks'][i]]
                else:
                    peak_s_value = np.nan

                if not pd.isna(waves_peak['ECG_T_Peaks'][i]):
                    peak_t_value = signal[waves_peak['ECG_T_Peaks'][i]]
                else:
                    peak_t_value = np.nan

                # отображение волн точками
                p = ax.scatter(x=waves_peak['ECG_P_Peaks'][i], y=peak_p_value, color='green', alpha=.5, linewidths=3)
                pon = ax.scatter(x=waves_peak['ECG_P_Onsets'][i], y=peak_pon_value, color='#5df656', alpha=.7, linewidths=3)

                q = ax.scatter(x=waves_peak['ECG_Q_Peaks'][i], y=peak_q_value, color='#fc9429', alpha=.7, linewidths=3)
                s = ax.scatter(x=waves_peak['ECG_S_Peaks'][i], y=peak_s_value, color='#fc6b06', alpha=.7, linewidths=3)
                t = ax.scatter(x=waves_peak['ECG_T_Peaks'][i], y=peak_t_value, color='#a87fea', alpha=.7, linewidths=3)

            for i in range(len(rpeaks)):
                peak_r_value = signal[rpeaks[i]]
                ax.scatter(x=rpeaks[i], y=peak_r_value, color='red', alpha=.7, linewidths=5, marker="|")

            plt.legend((pon, p, q, s, t),
               ('P_onset', 'P', 'Q', 'S', 'T'),
               scatterpoints=1,
               loc='upper left',
               ncol=2,
               fontsize=12)

        return _, waves_peak, waves_df, waves_df_rel, mean_pqrst_amp



    def make_model_signal(self):
        pass

    def overlay_model_signal(self, ecg_sample):
        #TODO: fix
        def overlay_signals(self, show=True, bias=0, amp_mode=False, method='spline'):
            """
            :self: сигнал ЭКС
            :show:       график наложения
            :bias:       сдвиг сигнала по оси y
            //:amp_mode:
            :method:     ['spline', 'weib'] - представление модельного сигнала (сплайнами / распределением Вейбулла)
            :return:     ideal_ecg_full, ecg_sample_cut
            ->           [исходный сигал, смоделированный сигнал (равные по длине)]
                                Сигналы накладываются с перовй точки начала P-волны у исходного self

            """

            clean_ecg_sample = del_zspan(self)
            # детекция волновых точек
            _, waves_peak, waves_df, waves_rel, mean_pqrst_amp = waves_detection(self, show=True, sr=400)
            if pd.isna(waves_df.iloc[0]['P_Onsets_diff']):
                waves_rel_copy = waves_rel.iloc[1:].copy()
            else:
                waves_rel_copy = waves_rel.iloc[:].copy()
            # индексы начала p-волны  (???)
            p_onset_ind = list(waves_peak['ECG_P_Onsets'])
            # Удаление nan
            for i in range(len(p_onset_ind) - 1):
                if pd.isna(p_onset_ind[i]):
                    del p_onset_ind[i]

            pp_diff = waves_rel_copy['pp_diff']
            # смоделированнные кардиоциклы
            pp_yi = []
            for i in range(len(pp_diff) - 1):
                p_row = waves_rel_copy.iloc[i]
                pqrst = generate_model_sig_mean(p_row, amp_info=mean_pqrst_amp, method=method)
                #         pqrst = generate_weib(p_row, amp_info=mean_pqrst_amp)
                pp_yi.append(list(pqrst))
            pp_points = tuple(pp_yi)
            # Конкатенация кардиоциклов в смоделированном сигнале
            # Смоделированный идеальный сигнал на основе исходного
            ideal_ecg_full = numpy.concatenate(pp_points, axis=0)
            # end - значение для уравнивания по длине
            end = int(len(ideal_ecg_full) - len(clean_ecg_sample[p_onset_ind[0]:]))
            ecg_len = len(clean_ecg_sample[p_onset_ind[0]:])
            ecg_sample_cut = clean_ecg_sample[p_onset_ind[0]:end]
            if show:
                fig, ax = plt.subplots(figsize=(15, 8))
                #         plt.figure(figsize=(18,8))
                ax.set_xlabel('time [n] - отсчёты')
                ax.set_ylabel('mV')
                plt.plot(ecg_sample_cut)
                plt.plot(ideal_ecg_full + bias, '--', color='red')
            mse = mean_squared_error(ideal_ecg_full + bias, ecg_sample_cut)
            mae = mean_absolute_error(ideal_ecg_full + bias, ecg_sample_cut)
            print('mse: {:.3f}'.format(mse))
            print('mae: {:.3f}'.format(mae))
            return ideal_ecg_full, ecg_sample_cut


if __name__ == '__main__':
    ecg = EcgSignal()
    sample = ecg.take_sample_zenodo(1, show=False)
    ecg.plot_sample(ecg_sample=sample)
    print(sample)



