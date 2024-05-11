from typing import Optional

from fastapi import Form, File, UploadFile
from pydantic import BaseModel


class DataForm(BaseModel):
    id: int
    sample_rate: int
    leads_nums: int
    gender: str
    age: int
    device: str
    leads_values: UploadFile

    @classmethod
    def as_form(
        cls,
        id: int = Form(...),
        sample_rate: int = Form(...),
        leads_nums: int = Form(...),
        gender: str = Form(...),
        age: str = Form(...),
        device: str = Form(...),
        leads_values: UploadFile = File(...)
    ):
        return cls(
            id = id,
            sample_rate = sample_rate,
            leads_nums = leads_nums,
            gender = gender,
            age = age,
            device = device,
            leads_values = leads_values
        )




class DataBytes(BaseModel):
    sample_rate: int
    name: Optional[str] = None
    gender: str
    age: int
    height: int
    weight: float
    device: str
    ecg_values: str
