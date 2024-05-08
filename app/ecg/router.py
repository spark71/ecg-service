from fastapi import APIRouter, Depends, Request
import random
from typing import Optional
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from hrvanalysis import get_time_domain_features
import torch
import io

from app.ecg.ecg import ROOT_DIR
from app.ecg.ecg import EcgSignal as esig
from app.ecg.form_schema import DataForm

from models.config import model_factory


UPLOAD_DIR = ROOT_DIR + 'app/ecg/uploads'
templates = Jinja2Templates(directory="app/templates")

#TODO:
# 3) Make prediction on uploaded signals /pred_signal/{ecg_id}
#       - choose a model
#       - make pred
# height weight recording_date


#TODO:
# add ecg orm


router = APIRouter(
    tags=['ЭКГ-сигналы']
)

# Загрузка модели
model = model_factory('resnet1d_wang')
# resnet1d_wang_model = model(input_channels=12, num_classes=5)
resnet1d_wang_weights = r'C:\Users\redmi\PycharmProjects\ecg-tool-api\models\pretrained\resnet1d_wang\resnet1d_wang_fold1_16epoch_best_score.pth'
model.load_state_dict(torch.load(resnet1d_wang_weights, map_location=torch.device('cpu'))['model'])
model.double()
model.eval()
# print(model)

signals = []
latest_signal = []


@router.get('/pages/add_sig', response_class=HTMLResponse)
async def get_add_sig(request: Request):
    return templates.TemplateResponse(name="add_form.html", context={'request': request})


@router.post('/pages/add_sig', response_class=HTMLResponse)
async def post_add_sig(request: Request, form_data: DataForm = Depends(DataForm.as_form)):
    print(form_data)
    contents = await form_data.leads_values.read()
    decoded_contents = contents.decode('utf-8')  # Декодирование байтов в строку
    data = np.loadtxt(io.StringIO(decoded_contents), dtype=float)
    print(type(data))
    signals.append(dict(form_data))
    latest_signal.append(data)
    return templates.TemplateResponse(name="add_form.html", context={'request': request})


@router.get('/get_signal_info')
async def get_signal_info():
    r_peaks = esig.detect_r_peaks(latest_signal[-1].T[0], 0.7, 50, False).tolist()
    print(latest_signal)
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
    classes = ['STTC', 'NORM', 'MI', 'HYP', 'CD']
    signal = torch.from_numpy(latest_signal[-1].T).to(torch.double)[None, :]
    prediction = model(signal)
    prediction_list = model(signal).detach().tolist()
    prediction_probs_softmax = torch.softmax(prediction, dim=1).detach().tolist()
    result = {
        'signal_shape': signal.shape,
        'prediction': prediction_list,
        'prediction_probs_softmax': prediction_probs_softmax,
        'predicted_class': 1
    }

    return result





@router.get('/signals')
async def get_signals():
    """
    Получение списка загруженных сигналов
    """
    return signals






