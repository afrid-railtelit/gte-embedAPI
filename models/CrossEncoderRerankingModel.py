from pydantic import BaseModel


class CrossEncoderRerankRequestModel(BaseModel):
    query: str
    docs: list[str]
    returnDocuments: bool = False
    topN: int = 10


class CrossEncoderRerankItemModel(BaseModel):
    docIndex: int
    doctext: str | None = None
    score: float


class CrossEncoderRerankResponseModel(BaseModel):
    results: list[CrossEncoderRerankItemModel]
    query: str
