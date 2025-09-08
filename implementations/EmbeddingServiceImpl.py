from abc import ABC, abstractmethod
import torch
from typing import  List


class EmbeddingServiceImpl(ABC):
    @abstractmethod
    def LoadModel(self) -> None:
        pass

    @abstractmethod
    def MeanPool(self, lastHidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def Embed(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        pass


    @abstractmethod
    async def GpuKeepAlive(self) -> None:
        pass
