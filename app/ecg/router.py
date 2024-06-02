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

from app.ecg.ecg import ROOT_DIR
from app.ecg.ecg import EcgSignal as esig
from app.ecg.form_schema import DataForm, DataBytes

from models.config import model_factory


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


# Загрузка модели
model = model_factory('resnet1d_wang')
# resnet1d_wang_model = model(input_channels=12, num_classes=5)
resnet1d_wang_weights = r'models\pretrained\resnet1d_wang\resnet1d_wang_fold1_16epoch_best_score.pth'
model.load_state_dict(torch.load(resnet1d_wang_weights, map_location=torch.device('cpu'))['model'])
model.double()
model.eval()
# print(model)

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





@router.get('/signals')
async def get_signals():
    """
    Получение списка загруженных сигналов
    """
    return signals






