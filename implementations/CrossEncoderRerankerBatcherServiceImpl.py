from abc import ABC, abstractmethod
from typing import List, Any,Tuple


class CrossEncoderRerankerBatcherServiceImpl(ABC):
    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def submit(self, pairs: List[Tuple[str, str]]) -> List[Any]:
        pass

    @abstractmethod
    async def _runLoop(self) -> None:
        pass
