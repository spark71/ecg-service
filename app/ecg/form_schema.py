from fastapi import Form, File, UploadFile
from pydantic import BaseModel


class DataForm(BaseModel):
    id: int
    leads_nums: int
    gender: str
    age: int
    device: str
    leads_values: UploadFile

    @classmethod
    def as_form(
        cls,
        id: int = Form(...),
        leads_nums: int = Form(...),
        gender: str = Form(...),
        age: str = Form(...),
        device: str = Form(...),
        leads_values: UploadFile = File(...)
    ):
        return cls(
            id = id,
            leads_nums = leads_nums,
            gender = gender,
            age = age,
            device = device,
            leads_values = leads_values
        )