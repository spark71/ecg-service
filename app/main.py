from fastapi import FastAPI, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.ecg.router import router as ecg_router
from app.pages.router import router as page_router
from pathlib import Path

import missingno as msno
from nltk.corpus import stopwords
import spacy

spacy



app = FastAPI()
# app.mount("/static", StaticFiles(directory="app/static"))
# app.mount(r"C:\Users\User\PycharmProjects\ecg-service\app\static", app=StaticFiles(directory=r"\app\static"))
app.include_router(ecg_router)
# app.include_router(page_router)



