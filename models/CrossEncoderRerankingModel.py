from pydantic import BaseModel



class CrossEncoderRerankRequestModel(BaseModel):
    query: str
    docs: list[str]


class CrossEncoderRerankItemModel(BaseModel):
    index: int
    query: str
    doc: str
    score: float


class CrossEncoderRerankResponseModel(BaseModel):
    results: list[CrossEncoderRerankItemModel]
