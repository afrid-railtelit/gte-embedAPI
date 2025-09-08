from typing import List
from pydantic import BaseModel


class EmbeddingControllerRequestModel(BaseModel):
    texts: List[str] = []


class EmbeddingItemModel(BaseModel):
    index: int
    embedding: List[float]


class EmbeddingControllerResponseModel(BaseModel):
    results: List[EmbeddingItemModel]
    dim: int


class ErrorResponseModel(BaseModel):
    status: str
    detail: str
