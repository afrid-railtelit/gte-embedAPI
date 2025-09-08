from typing import List, Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from enums import ResponseStatusEnum
from models import (
    EmbeddingControllerRequestModel,
    EmbeddingItemModel,
    EmbeddingControllerResponseModel,
    ErrorResponseModel,
)
from services import EmbeddingService
from implementations import EmbeddingControllerImpl


class EmbeddingController(EmbeddingControllerImpl):
    def __init__(self, service: EmbeddingService):
        self.service = service
        self.router = APIRouter()
        self.router.add_api_route("/embedding", self.EmbedAPI, methods=["POST"])

    async def EmbedAPI(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            req = EmbeddingControllerRequestModel.model_validate(payload)
        except Exception as exc:
            err = ErrorResponseModel(
                status=ResponseStatusEnum.VALIDATION_ERROR.value, detail=str(exc)
            )
            return JSONResponse(status_code=400, content=err.model_dump())

        texts: List[str] = req.texts
        try:
            embeddings = await self.service.EmbedBatched(texts)
            items: List[Any] = []
            for i, emb in enumerate(embeddings):
                items.append(EmbeddingItemModel(index=i, embedding=emb).model_dump())
            resp = EmbeddingControllerResponseModel(embeddings=items, dimensions=len(embeddings[0]) )
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
