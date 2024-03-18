

# 1) Загрузка сигнала /upload_signal
# 2) Get signals /get_signals
# 3) Make prediction on uploaded signals /pred_signal/{ecg_id}
#       - choose a model
#       - make pred


@app.post('/upload_signal')
async def upload_signal():
    """
    Загрузка сигнала в сервис (БД) в форматах txt, dat, hea, csv
    :return:
    """
    pass



@app.get('/get_signals')
@app.get('/signals')
async def get_signals():
    """
    Получение списка загруженных сигналов
    :return:
    """
    pass



@app.get('/predict_signal/{signal_id}')
async def pred_signal(signal):
    """
    Классификация сигнала из списка загруженных по id
    :param signal:
    :return:
    """
    pass




