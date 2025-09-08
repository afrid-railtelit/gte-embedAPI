from typing import cast, Any, Dict, List, Tuple
import torch
import asyncio
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from services.CrossEncoderRerankerBatcherService import (
    CrossEncoderRerankerBatcherService,
)
from implementations import CrossEncoderRerankerServiceImpl

modelName: str = "ncbi/MedCPT-Cross-Encoder"
maxLength: int = 512
maxPairs: int = 200
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keepaliveInterval: float = 10

tokenizer: Any = None
model: Any = None
keepaliveTask: Any = None


class CrossEncoderRerankerService(CrossEncoderRerankerServiceImpl):
    def __init__(self):
        self.batcher: CrossEncoderRerankerBatcherService = (
            CrossEncoderRerankerBatcherService(
                self.Score, maxBatchSize=100, maxDelayMs=5
            )
        )
        self._initialized = False
        self._lock = asyncio.Lock()

    async def LoadModel(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            torch.backends.cudnn.benchmark = True
            global tokenizer, model, keepaliveTask
            if tokenizer is not None and model is not None:
                return

            tokenizer = cast(Any, AutoTokenizer).from_pretrained(
                modelName, use_fast=True
            )
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = cast(Any, AutoModelForSequenceClassification).from_pretrained(
                modelName, dtype=dtype
            )
            if torch.cuda.is_available():
                model = model.half().to(device)
            else:
                model = model.to(device)
            model.eval()

            warmPair: Tuple[str, str] = ("hello query " * 8, "hello document " * 8)
            tokWarm: Dict[str, torch.Tensor] = tokenizer(
                [warmPair],
                return_tensors="pt",
                truncation=True,
                max_length=maxLength,
                padding=True,
            )
            inputsWarm: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in tokWarm.items()
            }

            with torch.inference_mode():
                _ = model(**inputsWarm)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            await self.batcher.start()
            self._initialized = True

    async def ScoreBatched(self, pairs: List[Tuple[str, str]]) -> List[float]:
        async with self._lock:
            if not self._initialized:
                await self.LoadModel()
            return await self.batcher.submit(pairs)

    def Score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            raise ValueError("pairs empty")
        if len(pairs) > maxPairs:
            raise ValueError(f"pairs length exceeds {maxPairs}")

        tok: Dict[str, torch.Tensor] = tokenizer(
            pairs,
            return_tensors="pt",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )
        tok = {k: v.pin_memory() for k, v in tok.items()}

        inputs: Dict[str, torch.Tensor] = {
            k: v.to(device, non_blocking=True) for k, v in tok.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                scores = (
                    torch.sigmoid(logits).squeeze(-1).cpu()
                )  # Sigmoid for single logit
            else:
                scores = logits.softmax(dim=-1)[:, 1].cpu()  # Softmax for binary logits

        return cast(Any, scores).detach().to("cpu", non_blocking=True).tolist()

    async def Rerank(self, query: str, docs: List[str]) -> List[Tuple[int, str, float]]:

        if not docs:
            return []

        indices = list(range(len(docs)))
        pairs: List[Tuple[str, str]] = [(query, doc) for doc in docs]

        scores = await self.ScoreBatched(pairs)

        combined = list(zip(indices, docs, scores))
        combined.sort(key=lambda x: x[2], reverse=True)

        return combined
