from pydantic import BaseModel



class CrossEncoderRerankRequestModel(BaseModel):
    query: str
    docs: list[str]
    returnDocuments:bool = False


class CrossEncoderRerankItemModel(BaseModel):
    index: int
    doc: str
    score: float


class CrossEncoderRerankResponseModel(BaseModel):
    results: list[CrossEncoderRerankItemModel]
    query: str
    
