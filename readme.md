# Установка менеджера пакетов (windows)
1) `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`
2) Добавить `%APPDATA%\Python\Scripts` в системную переменную окружения PATH
3) После установки `poetry --version`
https://python-poetry.org/docs/basic-usage/ 
 
# Запуск сервиса
```
uvicorn main:app --reload
```
# Источники данных

1. Zenodo: https://zenodo.org/records/3765780
2. Physionet PTBXL-dataset: https://physionet.org/content/ptb-xl/1.0.1/

