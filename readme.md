# Установка менеджера пакетов (windows)
1) `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`
2) Добавить `%APPDATA%\Python\Scripts` в системную переменную окружения PATH
3) После установки `poetry --version`
https://python-poetry.org/docs/basic-usage/ 
 

# Переменные окружения
Перед запуском проекта проинициализируйте переменные окружения:
- ROOT_DIR=путь до проекта
- API_HOST=адрес хоста
Это можно сделать в файле .env в корне проекта


# Запуск backend-сервиса
```bash
uvicorn app.main_api:app --port 8000 --reload
```

# Запуск frontend-сервиса
```bash
cd app
streamlit run app.py
```

# Источники данных
1. Zenodo: https://zenodo.org/records/3765780
2. Physionet PTBXL-dataset: https://physionet.org/content/ptb-xl/1.0.1/

