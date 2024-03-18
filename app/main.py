from typing import Optional, Annotated
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()
app.include_router()


