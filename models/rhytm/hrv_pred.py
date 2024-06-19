import numpy as np
import pandas as pd
from joblib import load
import neurokit2 as nk
import pyhrv.tools as tools
import pyhrv.frequency_domain as fd
import matplotlib
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class ECG_RR():
    def __init__(self,info,ecg_signals_1,name):
        self.ECG = info
        self.ecg_signals_1 = ecg_signals_1
        self.name = name
    def RR(self,sp,RR_d):
        RR_d = RR_d
        if len(self.ECG[self.name]) > 3:
            for i in range(0, len(self.ECG[self.name])-1):
                RR_d = sp(RR_d, list(self.ECG[self.name])[i+1] - list(self.ECG[self.name])[i])
        else :RR_d = 0
        return RR_d
    def R_m(self,sp):
        if len(self.ecg_signals_1[self.ecg_signals_1[self.name]>0].ECG_Raw):
            return sp(self.ecg_signals_1[self.ecg_signals_1[self.name]>0].ECG_Raw)
        else: return 0


def timedomain_pic(ecg_signals_1,info,name,results):
    ECG = ECG_RR(info,ecg_signals_1,"ECG_R_Peaks")
    name = str(name)+"_"
    results[name+'Max_R'] = [ECG.R_m(max)]
    results[name+'Min_R'] = [ECG.R_m(min)]
    results[name+'Max_RR'] = [ECG.RR(max,0)*(1/500)]
    results[name+'Min_RR'] = [ECG.RR(min,500)*(1/500)]
    ECG = ECG_RR(info,ecg_signals_1,"ECG_P_Peaks")
    results[name+'Max_P'] = [ECG.R_m(max)]
    results[name+'Min_P'] = [ECG.R_m(min)]
    ECG = ECG_RR(info,ecg_signals_1,"ECG_Q_Peaks")
    results[name+'Max_Q'] = [ECG.R_m(max)]
    results[name+'Min_Q'] = [ECG.R_m(min)]
    ECG = ECG_RR(info,ecg_signals_1,"ECG_S_Peaks")
    results[name+'Max_S'] = [ECG.R_m(max)]
    results[name+'Min_S'] = [ECG.R_m(min)]
    ECG = ECG_RR(info,ecg_signals_1,"ECG_T_Peaks")
    results[name+'Max_T'] = [ECG.R_m(max)]
    results[name+'Min_T'] = [ECG.R_m(min)]
    return results



def timedomain_HRV(ecg_signals_1,info,name,results):
    try:
        rr_intervals_list = info['ECG_R_Peaks']
        nni_2 = tools.nn_intervals(rr_intervals_list)
        result = fd.welch_psd(nni=nni_2,show=False, show_param=False,legend=False)
        matplotlib.pyplot.close(fig="all")
    except:
        return results
    name = str(name)+"_"
    results[name+'VLF_Peak (Hz)'] = [result[1][0]]
    results[name+'LF_Peak (Hz)'] = [result[1][1]]
    results[name+'HV_Peak (Hz)'] = [result[1][2]]
    results[name+'VLF_Abs (ms2)'] = [result[2][0]]
    results[name+'LF_Abs (ms2)'] = [result[2][1]]
    results[name+'HV_Abs (ms2)'] = [result[2][2]]
    results[name+'VLF_Rel (%)'] = [result[3][0]]
    results[name+'LF_Rel (%)'] = [result[3][1]]
    results[name+'HV_Rel (%)'] = [result[3][2]]
    results[name+'VLF_Log (-)'] = [result[4][0]]
    results[name+'LF_Log (-)'] = [result[4][1]]
    results[name+'HV_Log (-)'] = [result[4][2]]
    results[name+'LF_Norm (-)'] = [result[5][0]]
    results[name+'HV_Norm (-)'] = [result[5][1]]
    results[name+'LF/HF (-)'] = [result[6]]
    results[name+'Total Power (ms)'] = [result[7]]
    return results



def timedomain(ecg_signals_1,info,name,results):
    rr = info['ECG_R_Peaks'][:-1]
    name = str(name)+"_"
    rr_ecg = np.diff(rr)
    rr = rr_ecg*10
    hr = 60000/rr
    results[name+'Mean RR (ms)'] = [np.mean(rr)]
    results[name+'STD RR/SDNN (ms)'] = [np.std(rr)]
    results[name+'Mean HR (Kubios\' style) (beats/min)'] = [60000/np.mean(rr)]
    results[name+'Mean HR (beats/min)'] = [np.mean(hr)]
    results[name+'STD HR (beats/min)'] = [np.std(hr)]
    results[name+'Min HR (beats/min)'] = [np.min(hr)]
    results[name+'Max HR (beats/min)'] = [np.max(hr)]
    results[name+'RMSSD (ms)'] = [np.sqrt(np.mean(np.square(np.diff(rr))))]
    results[name+'NNxx'] = [np.sum(np.abs(np.diff(rr)) > 50)*1]
    results[name+'pNNxx (%)'] = [100 * np.sum((np.abs(np.diff(rr)) > 50)*1) / len(rr)]
    return results


def ecg_HRV_df(ecg_signals_1,info,name):
    results = {}
    results = pd.DataFrame(results)
    results =timedomain(ecg_signals_1,info,name,results)
    results =timedomain_HRV(ecg_signals_1,info,name,results)
    results =timedomain_pic(ecg_signals_1,info,name,results)
    return results



def ecg_HRV_0(ecg_signal,n_0,n_1):
    df_HTV =pd.DataFrame()
    #df = pd.DataFrame(ecg_signal[0])
    df = ecg_signal
    if n_1 - n_0 != 1:
        for j in range(n_0,n_1):
            ecg_signals_1, info = nk.ecg_process(df[j], sampling_rate=500)
            HTV = ecg_HRV_df(ecg_signals_1,info,j)
            df_HTV[HTV.columns] = HTV
    elif n_1 - n_0 == 1:
        ecg_signals_1, info  = nk.ecg_process(df, sampling_rate=500)
        HTV = ecg_HRV_df(ecg_signals_1,info,n_0)
        df_HTV[HTV.columns] = HTV
    return df_HTV



def stardant_N(X_columns,fix_displacement_min, fix_displacement):
    for i in fix_displacement_min.columns:
        X_columns[i] = X_columns[i] - fix_displacement_min[i]["min"]
    for i in fix_displacement.columns:
        X_columns[i] = (X_columns[i] - fix_displacement[i]["mean"])/(fix_displacement[i]["std"])
    return X_columns


name_model = {
    "r":["LGBMClassifier.joblib","LinearSVC.joblib","lstm_model_r.keras"],
    "d":["LGBMClassifier.joblib","ExtraTreesClassifier.joblib","HistGradientBoostingClassifier.joblib","lstm_model_d.keras"]
}


def path_s(led, name_led, name_model, name_d, lan, HRV, path_def) :
    """
    преобразует диагноз или ритм в строковое представление
    :param led:
    :param name_led:
    :param name_model:
    :param name_d:
    :param lan:
    :param HRV:
    :param path_def:
    :return:
    """
    path_n = led + name_model.split(".")[0]
    path_end = str(name_led-int(led.split("_")[0]))+"_"+str(name_led)+"_led_"+name_d + "."+name_model.split(".")[1]
    if name_d == "d":
        uniques = pd.read_csv( path_def +"diagnostic_2/uniques_diagnostic.csv", index_col=0)
        list_corr = pd.read_csv(path_def+"diagnostic_2/list_corr_diagnostic.csv", index_col=0)
        path = path_def + "diagnostic_2/" + path_n+path_end
        path_2 = path_def + "diagnostic_2/"+ path_n + "."+name_model.split(".")[1]
    if name_d == "r":
        uniques = pd.read_csv( path_def +"rhythm_2/uniques_rhythm.csv", index_col=0)
        list_corr = pd.read_csv(path_def+"rhythm_2/list_corr_rhythm.csv", index_col=0)
        path = path_def + "rhythm_2/" + path_n+path_end
        path_2 = path_def + "rhythm_2/"+ path_n + "."+name_model.split(".")[1]
    if "."+name_model.split(".")[1] == ".joblib":
            model = load(path)
            y_predict = model.predict(HRV)
            return uniques[lan][y_predict[0]]
    elif ("."+name_model.split(".")[1] == ".keras")and led == "12_led/":
        model = keras.models.load_model(path_2)
        y_predict = model.predict(HRV[list(list_corr["0"])[:-1]])
        max_in_ = pd.DataFrame(y_predict[0]).idxmax()[0]
        #pred_namber = pd.DataFrame(y_predict[0])*100
        #pred_namber[lan] = uniques[lan]
        return uniques[lan][max_in_]#,#pred_namber




class func_ecg_detect_2():
    def __init__(self, ECG, age, sex):
        self.path_def = r'C:/Users/redmi/PycharmProjects/ecg-tool-api/models/rhytm/'
        self.ECG = ECG
        self.age = age
        self.sex = sex
        self.fix_displacement = pd.read_csv(str(self.path_def + "fix_displacement.csv"), index_col=0)
        self.fix_displacement_min = pd.read_csv(str(self.path_def + "fix_displacement_min.csv"), index_col=0)

    def detect_led(self, count_led, name_led, name_model_1, name_d, lan):
        """
        :param count_led: число отведений
        :param name_led: отведение int 1-12
        :param name_model_1:
        :param name_d:
        :param lan:
        :return:
        """
        self.HRV = ecg_HRV_0(self.ECG, name_led - count_led, name_led)
        self.HRV["age"] = self.age
        self.HRV["sex"] = self.sex
        self.HRV = stardant_N(self.HRV, self.fix_displacement_min[self.HRV.columns],
                              self.fix_displacement[self.HRV.columns])

        for i in name_model[name_d]:
            if name_model_1 in i:
                model = i
        path_2 = path_s(str(count_led) + "_led/", name_led, model, name_d, lan, self.HRV, self.path_def)

        return path_2





