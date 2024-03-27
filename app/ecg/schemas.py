from typing import Optional, List
from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict


class EcgSchema(BaseModel):
    id: int
    sample_rate: int
    leads_nums: List[int]
    # leads_nums: int
    # leads_values: UploadFile
    gender: str
    age: int
    device: str








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