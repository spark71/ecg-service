import random
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.pages.form_schema import DataForm
import io


router = APIRouter(
    prefix='/pages',
    tags=['Фронтенд']
)

templates = Jinja2Templates(directory='app/templates')


#TODO:
# add ecg orm
# add /get_signal_info

signals = []
latest_signal = None

@router.get('/add_sig', response_class=HTMLResponse)
async def get_add_sig(request: Request):
    return templates.TemplateResponse(name="add_form.html", context={'request': request})



@router.post('/add_sig', response_class=HTMLResponse)
async def post_add_sig(request: Request, form_data: DataForm = Depends(DataForm.as_form)):
    print(form_data)
    contents = await form_data.leads_values.read()
    decoded_contents = contents.decode('utf-8')  # Декодирование байтов в строку
    data = np.loadtxt(io.StringIO(decoded_contents))
    signals.append(data)
    print(data)
    print(data.shape)
    return templates.TemplateResponse(name="add_form.html", context={'request': request})



@router.get('/get_signal_info')
async def get_signal_info():
    pass



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
    return signals
