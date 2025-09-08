from typing import List
from pydantic import BaseModel


class EmbedRequestModel(BaseModel):
    texts: List[str] = []


class EmbeddingItemModel(BaseModel):
    index: int
    embedding: List[float]


class EmbeddingResponseModel(BaseModel):
    results: List[EmbeddingItemModel]
    dim: int


class ErrorResponseModel(BaseModel):
    status: str
    detail: str
