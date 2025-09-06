from typing import List
from pydantic import BaseModel


class EmbedRequestModel(BaseModel):
    texts: List[str] = []


class EmbeddingItemModel(BaseModel):
    index: int
    embedding: List[float]


class EmbedResultModel(BaseModel):
    results: List[EmbeddingItemModel]
    dim: int
    modelLatencyMs: float


class ErrorResponseModel(BaseModel):
    status: str
    detail: str
