from typing import Optional, List
from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict


class EcgSchema(BaseModel):
    id: int
    # leads_nums: List[int | str]
    leads_nums: int
    # leads_values: UploadFile
    gender: str
    age: int
    device: str
    # metadata: Optional[dict]
    # description: Optional[str] = None
    # dataset: Optional[dict]






# class SEcgAdd(BaseModel):
#     leads: dict
#     metadata: Optional[dict]
#     description: Optional[str] = None
#     gender: str
#     age: int
#     dataset: Optional[dict]
#
#
# class SEcg(SEcgAdd):
#     id: int
#     model_config = ConfigDict(from_attributes=True)