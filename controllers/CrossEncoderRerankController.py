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
            print("Log1")
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
            for docIndex, doc, score in ranked_results:
                if returnDocs:
                    item = CrossEncoderRerankItemModel(index=docIndex, doc=doc, score=score)
                else:
                    item = CrossEncoderRerankItemModel(index=docIndex, score=score)
                items.append(item)

                items.append(item)

            resp = CrossEncoderRerankResponseModel(results=items, query=query)
            return JSONResponse(status_code=200, content=resp.model_dump())
        except ValueError as exc:
            print("Log1")

            err = ErrorResponseModel(
                status=ResponseStatusEnum.VALIDATION_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=400, content=err.model_dump())
        except Exception as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.SERVER_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=500, content=err.model_dump())
