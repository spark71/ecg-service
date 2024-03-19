from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, Request, Response, Form
from pydantic import Json, BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.ecg.ecg import ROOT_DIR
from app.ecg.schemas import EcgSchema
# from app.main import templates

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

signals = []
@router.post('/upload_signal')
async def upload_signal(ecg_metadata: EcgSchema = Depends(), file_upload: UploadFile = File(...)):
    """
    Загрузка сигнала в сервис (БД) в форматах txt, dat, hea, csv
    """
    formats = ['dat', 'hea', 'csv', 'txt']
    data = await file_upload.read()
    await file_upload.close()
    payload = {
        "data": dict(ecg_metadata),
        "file_upload": file_upload.filename,
        "file_data": data,
        "cleaned_file_data": data.decode('utf-8').replace('\n', ' ').replace('\r', ' ')
    }
    signals.append(payload)
    # print(signals)
    return payload


@router.post('/add_signal')
async def upload_signal(ecg_metadata: EcgSchema = Depends(), file_upload: UploadFile = File(...)):
    """
    Загрузка сигнала в сервис (БД) в форматах txt, dat, hea, csv
    """
    formats = ['dat', 'hea', 'csv', 'txt']
    data = await file_upload.read()
    await file_upload.close()
    payload = {
        "data": dict(ecg_metadata),
        "file_upload": file_upload.filename,
        "file_data": data,
        "cleaned_file_data": data.decode('utf-8').replace('\n', ' ').replace('\r', ' ')
    }
    signals.append(payload)
    # print(signals)
    return payload

@router.get('/basic', response_class=HTMLResponse)
def get_basic_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@router.post('/basic', response_class=HTMLResponse)
def post_basic_form(request: Request, username: str = Form(...), password: str = Form(...)):
    print(f'name: {username}')
    print(f'passw: {password}')
    return templates.TemplateResponse("form.html", {"request": request})



# @router.get('/get_signals')
@router.get('/signals')
async def get_signals():
    """
    Получение списка загруженных сигналов
    """
    return signals



@router.get('/predict_signal/{signal_id}')
async def pred_signal(signal):
    """
    Классификация сигнала из списка загруженных по id

    """
    pass


@router.get('/get_signal_info/{signal_id}')
async def get_signal_info():
    """
    Получение информации о сигнале

    """
    pass

# ------------------------------------------------------------------------
class Item(BaseModel):
    name: str
    description: str

@router.post("/files/")
async def create_file_and_item(item: Item = Depends(), files: List[UploadFile] = File(...)):
    return {"item": item, "file_name": [file.filename for file in files]}


