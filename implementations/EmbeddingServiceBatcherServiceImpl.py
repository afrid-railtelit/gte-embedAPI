from abc import ABC, abstractmethod
from typing import List, Any


class EmbeddingServiceBatcherServiceImpl(ABC):
    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def submit(self, texts: List[str]) -> List[Any]:
        pass

    @abstractmethod
    async def _runLoop(self) -> None:
        pass
