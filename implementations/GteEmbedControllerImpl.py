from abc import ABC, abstractmethod
from fastapi import Request
from fastapi.responses import JSONResponse


class GteEmbedControllerImpl(ABC):

    @abstractmethod
    async def EmbedAPI(self, request: Request) -> JSONResponse:
        pass
