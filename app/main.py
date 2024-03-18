from fastapi import FastAPI, Depends
from app.ecg.router import router as ecg_router
from pathlib import Path



app = FastAPI()
app.include_router(ecg_router)


