from typing import List, Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from enums import ResponseStatusEnum
from services import CrossEncoderRerankerService
from models import (
    CrossEncoderRerankItemModel,
    CrossEncoderRerankRequestModel,
    CrossEncoderRerankResponseModel,
    ErrorResponseModel,
)

from implementations import CrossEncoderRerankControllerImpl


class CrossEncoderRerankController(CrossEncoderRerankControllerImpl):
    def __init__(self, service: CrossEncoderRerankerService):
        self.service = service
        self.router = APIRouter()
        self.router.add_api_route("/ce/reranker", self.RerankAPI, methods=["POST"])

    async def RerankAPI(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            req = CrossEncoderRerankRequestModel.model_validate(payload)
        except Exception as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.VALIDATION_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=400, content=err.model_dump())

        query: str = req.query
        docs: List[str] = req.docs
        returnDocs: bool = req.returnDocuments
        try:
            ranked_results = await self.service.Rerank(query, docs)
            items: List[Any] = []
            for _, (docIndex, (query, doc), score) in enumerate(ranked_results):
                items.append(
                    CrossEncoderRerankItemModel(
                        index=docIndex, doc=doc, score=score
                    ).model_dump()
                )
                if returnDocs:
                    for item in items:
                        if "doc" in item:
                            del item["doc"]

            resp = CrossEncoderRerankResponseModel(results=items, query=query)
            return JSONResponse(status_code=200, content=resp.model_dump())
        except ValueError as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.VALIDATION_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=400, content=err.model_dump())
        except Exception as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.SERVER_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=500, content=err.model_dump())
