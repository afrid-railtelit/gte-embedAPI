from abc import ABC, abstractmethod
from typing import List, Tuple


class CrossEncoderRerankerServiceImpl(ABC):
    @abstractmethod
    async def LoadModel(self) -> None:
        pass

    @abstractmethod
    async def ScoreBatched(self, pairs: List[Tuple[str, str]]) -> List[float]:
        pass

    @abstractmethod
    def Score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        pass

    @abstractmethod
    async def Rerank(self, query: str, docs: List[str]) -> List[Tuple[int, str, float]]:
        pass
