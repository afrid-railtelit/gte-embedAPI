from pydantic import BaseModel
from typing import Optional


class CrossEncoderRerankRequestModel(BaseModel):
    query: str
    docs: list[str]
    returnDocuments:bool = False


class CrossEncoderRerankItemModel(BaseModel):
    index: int
    doc: Optional[str]
    score: float


class CrossEncoderRerankResponseModel(BaseModel):
    results: list[CrossEncoderRerankItemModel]
    query: str
    
