from fastapi import APIRouter, Depends, Request
import random
from typing import Optional
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.ecg.ecg import ROOT_DIR
from app.ecg.form_schema import DataForm
import io
from hrvanalysis import get_time_domain_features
from app.ecg.ecg import EcgSignal as esig


UPLOAD_DIR = ROOT_DIR + 'app/ecg/uploads'
templates = Jinja2Templates(directory="app/templates")

#TODO:
# 1) Загрузка сигнала /upload_signal
# 2) Get signals /get_signals
# 3) Make prediction on uploaded signals /pred_signal/{ecg_id}
#       - choose a model
#       - make pred

router = APIRouter(
    tags=['ЭКГ-сигналы']
)


#TODO:
# add ecg orm
# add /get_signal_info

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
    latest_signal.append(data)
    signals.append(dict(form_data))
    return templates.TemplateResponse(name="add_form.html", context={'request': request})


@router.get('/get_signal_info')
async def get_signal_info() -> dict:
    r_peaks = esig.detect_r_peaks(latest_signal[-1].T[0], 0.7, 50, False).tolist()
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
    if nn_model is None:
        result = {'signal_shape': signals[-1].shape, 'predicted_class': random.choice(classes)}
    else:
        result = nn_model.predict(latest_signal)
    return result


@router.get('/signals')
async def get_signals():
    """
    Получение списка загруженных сигналов
    """
    return signals






