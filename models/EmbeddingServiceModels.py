from typing import List
from pydantic import BaseModel


class EmbeddingControllerRequestModel(BaseModel):
    texts: List[str] = []


class EmbeddingItemModel(BaseModel):
    index: int
    embedding: List[float]


class EmbeddingControllerResponseModel(BaseModel):
    embeddings: List[EmbeddingItemModel]
    dimensions: int


class ErrorResponseModel(BaseModel):
    status: str
    detail: str
