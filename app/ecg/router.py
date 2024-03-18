from typing import List
from fastapi import APIRouter, UploadFile

from app.ecg.schemas import EcgSchema

# 1) Загрузка сигнала /upload_signal
# 2) Get signals /get_signals
# 3) Make prediction on uploaded signals /pred_signal/{ecg_id}
#       - choose a model
#       - make pred
router = APIRouter(
    tags=['ЭКГ-сигналы']
)

@router.post('/upload_signal')
async def upload_signal(ecg_data: EcgSchema, file: UploadFile):
    """
    Загрузка сигнала в сервис (БД) в форматах txt, dat, hea, csv
    """
    # return {"metadata":ecg_data, "file":file}
    return {"metadata":ecg_data}




# @router.get('/get_signals')
@router.get('/signals')
async def get_signals() -> List[EcgSchema]:
    """
    Получение списка загруженных сигналов
    """
    pass



@router.get('/predict_signal/{signal_id}')
async def pred_signal(signal):
    """
    Классификация сигнала из списка загруженных по id

    """
    pass




