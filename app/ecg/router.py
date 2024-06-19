import base64
import pandas as pd
from fastapi import APIRouter, Depends, Request
import random
from typing import Optional
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from hrvanalysis import get_time_domain_features
import torch
import io
from matplotlib import pyplot as plt
from scipy.signal import resample

from app.ecg.ecg import ROOT_DIR
from app.ecg.ecg import EcgSignal as esig
from app.ecg.form_schema import DataForm, DataBytes

from models.config import model_factory

from models.rhytm.hrv_pred import func_ecg_detect_2


UPLOAD_DIR = ROOT_DIR + 'app/ecg/uploads'
templates = Jinja2Templates(directory="app/templates")

#TODO:
#       - choose a model


router = APIRouter(
    tags=['ЭКГ-сигналы']
)

upload_router = APIRouter(
    tags=['Загрузка ЭКГ']
)

signals = []
latest_signal = []

@router.get('/pages/add_sig_form', response_class=HTMLResponse)
async def get_add_sig(request: Request):
    return templates.TemplateResponse(name="add_form.html", context={'request': request})


@router.post('/pages/add_sig_form', response_class=HTMLResponse)
async def post_add_sig(request: Request, form_data: DataForm = Depends(DataForm.as_form)):
    print(form_data)
    contents = await form_data.leads_values.read()
    decoded_contents = contents.decode('utf-8')  # Декодирование байтов в строку
    data = np.loadtxt(io.StringIO(decoded_contents), dtype=float)
    print(type(data))
    signals.append(dict(form_data))
    latest_signal.append(data)
    return templates.TemplateResponse(name="add_form.html", context={'request': request})

@router.post('/add_sig_bytes')
async def add_sig_bytes(data: DataBytes):
    data_dict = dict(data)
    signal_bytes_str = data_dict["ecg_values"]
    base64_bytes = base64.b64decode(signal_bytes_str)
    signal = np.frombuffer(base64_bytes, dtype=np.float64)
    print("Signal:", signal.shape, type(signal))
    latest_signal.append(signal)
    return data_dict

@router.get('/get_signal_info')
async def get_signal_info():
    signal = latest_signal[-1].reshape(1000, 12).T[np.newaxis, :]
    r_peaks = esig.detect_r_peaks(signal[0, 0], 0.7, 50, False).tolist()
    nn_intervals = np.diff(r_peaks).tolist()
    time_domain_features = get_time_domain_features(nn_intervals)
    time_domain_features = {key: float(value) for key, value in time_domain_features.items()}
    signal_info = {
        'r_peaks': r_peaks,
        'nn_intervals': nn_intervals,
        'time_domain_features': time_domain_features
    }
    return signal_info


@router.get('/predict')
async def predict(nn_model: Optional[str] = None) -> dict:
    classes = np.array(['CD', 'HYP', 'MI', 'NORM', 'STTC'])
    signal = torch.from_numpy(latest_signal[-1].reshape(1000, 12).T[np.newaxis, :]).to(torch.double)
    # print("SHHHHAPE:", signal.shape)
    prediction = model(signal)
    prediction_list = model(signal).detach().tolist()
    prediction_probs_softmax = torch.softmax(prediction, dim=1).detach().numpy()[0]
    th = 0.15
    cls_probs = prediction_probs_softmax[ np.where(prediction_probs_softmax > th)[0] ]
    cls_idx = np.where(prediction_probs_softmax > th)[0]
    cls_pred = classes[cls_idx]
    result = {
        'signal_shape': signal.shape,
        'prediction': prediction_list,
        'prediction_probs_softmax': prediction_probs_softmax.tolist(),
        'cls_pred': cls_pred.tolist(),
        'cls_probs': cls_probs.tolist(),
    }

    return result



@router.get('/predict_by/{nn_model}')
async def predict(nn_model: Optional[str] = None) -> dict:
    classes = np.array(['CD', 'HYP', 'MI', 'NORM', 'STTC'])
    signal = torch.from_numpy(latest_signal[-1].reshape(1000, 12).T[np.newaxis, :]).to(torch.double)
    model = model_factory(nn_model)
    prediction = model(signal)
    prediction_list = model(signal).detach().tolist()
    prediction_probs_softmax = torch.softmax(prediction, dim=1).detach().numpy()[0]
    th = 0.15
    cls_probs = prediction_probs_softmax[ np.where(prediction_probs_softmax > th)[0] ]
    cls_idx = np.where(prediction_probs_softmax > th)[0]
    cls_pred = classes[cls_idx]
    result = {
        'signal_shape': signal.shape,
        'prediction': prediction_list,
        'prediction_probs_softmax': prediction_probs_softmax.tolist(),
        'cls_pred': cls_pred.tolist(),
        'cls_probs': cls_probs.tolist(),
    }

    return result


@router.get('/predict_rhythm_by/{rhytm_model}/age={age}/gender={gender}')
async def predict_rhythm_by(rhytm_model: str, age: int, gender: int):
    # LGBMClassifier, LinearSVC
    count_led = 1
    name_otvedenie = 5
    # диагноз - d, ритм - r
    name_diagnostic = "r"
    name_model_1 = rhytm_model + '.joblib'
    ecg_n_1 = pd.DataFrame(resample(latest_signal[-1].reshape(1000, 12), 5000))
    print("ECG:", ecg_n_1)
    h = func_ecg_detect_2(ecg_n_1[0], age, gender)
    hrv_r = h.detect_led(count_led, name_otvedenie, name_model_1, name_diagnostic, "ru")
    return hrv_r


@router.get('/predict_diagnostic_by/{diagnostic_model}/age={age}/gender={gender}')
async def predict_diagnostic_by(diagnostic_model: str, age: int, gender: int):
    # LGBMClassifier
    count_led = 1
    name_otvedenie = 5
    # диагноз - d, ритм - r
    name_diagnostic = "d"
    name_model_1 = diagnostic_model + '.joblib'
    ecg_n_1 = pd.DataFrame(resample(latest_signal[-1].reshape(1000, 12), 5000))
    esig.plot_sample(ecg_n_1[0])
    print("ECG:", ecg_n_1)
    h = func_ecg_detect_2(ecg_n_1[0], age, gender)
    hrv_d = h.detect_led(count_led, name_otvedenie, name_model_1, name_diagnostic, "ru")
    return hrv_d


@router.get('/preprocess')
async def gan_forward(preprocess_option: str):
    pass



@router.get('/signals')
async def get_signals():
    """
    Получение списка загруженных сигналов
    """
    return signals

