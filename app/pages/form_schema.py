from fastapi import Form, File, UploadFile
from pydantic import BaseModel


# https://stackoverflow.com/a/60670614
class DataForm(BaseModel):
    # username: str
    # password: str
    # file: UploadFile
    id: int
    # leads_nums: List[int | str]
    leads_nums: int
    # leads_values: UploadFile
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