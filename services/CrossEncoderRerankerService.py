from typing import cast, Any, Dict, List, Tuple
import torch
import asyncio
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from services.CrossEncoderRerankerBatcherService import (
    CrossEncoderRerankerBatcherService,
)
from implementations import CrossEncoderRerankerServiceImpl
from dotenv import load_dotenv
import os
import torch


load_dotenv()

modelName: str = os.getenv(
    "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
maxLength: int = int(os.getenv("MAX_TOKEN_LIMIT_PER_TEXT", 512))
maxPairs: int = int(os.getenv("MAX_RE_RANKER_PAIRS", 200))
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keepaliveInterval: float = 10
tokenizer: Any = None
model: Any = None
keepaliveTask: Any = None


class CrossEncoderRerankerService(CrossEncoderRerankerServiceImpl):
    def __init__(self):
        self.batcher: CrossEncoderRerankerBatcherService = (
            CrossEncoderRerankerBatcherService(
                self.Score,
                maxBatchSize=int(os.getenv("MAX_CE_RE_RANKER_BACTH_SIZE", 100)),
                maxDelayMs=int(os.getenv("MAX_CE_RE_RANKER_BACTH_REQUEST_DELAY", 5)),
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

            warmPair: Tuple[str, str] = ("hello query " * 2, "hello document " * 2)
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

    def Score(
        self, pairs: List[Tuple[str, str]], mode: str = "prob", temp: float = 1.0
    ) -> List[float]:
        if not pairs:
            raise ValueError("pairs empty")
        if len(pairs) > maxPairs:
            raise ValueError(f"pairs length exceeds {maxPairs}")

        tok = tokenizer(
            pairs,
            return_tensors="pt",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )
        tok = {k: v.pin_memory() for k, v in tok.items()}
        inputs = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)

            if logits is not None:
                if logits.shape[-1] == 1:
                    raw = torch.sigmoid(logits.squeeze(-1) / float(temp)).cpu()
                else:
                    probs = logits.softmax(dim=-1)
                    raw = probs[:, 1].cpu()
                scores = raw.detach().to("cpu", non_blocking=True).float()
            else:
                maybe = None
                for name in ("scores", "score", "logits", "predictions"):
                    maybe = getattr(outputs, name, None)
                    if maybe is not None:
                        break
                if maybe is None:
                    if isinstance(outputs, torch.Tensor):
                        maybe = outputs
                    elif (
                        isinstance(outputs, (list, tuple))
                        and len(cast(Any, outputs)) > 0
                        and isinstance(outputs[0], torch.Tensor)
                    ):
                        maybe = outputs[0]
                    else:
                        raise ValueError("Couldn't find score/logits in model output")
                raw_tensor = maybe.squeeze().cpu().float()
                if mode == "prob":
                    scores = torch.sigmoid(raw_tensor / float(temp))
                else:
                    mn = raw_tensor.min()
                    mx = raw_tensor.max()
                    if mx == mn:
                        scores = torch.zeros_like(raw_tensor)
                    else:
                        scores = (raw_tensor - mn) / (mx - mn)

        return cast(Any, scores).tolist()

    async def Rerank(self, query: str, docs: List[str]) -> List[Tuple[int, str, float]]:
        if not docs:
            return []
        indices = list(range(len(docs)))
        pairs: List[Tuple[str, str]] = [(query, doc) for doc in docs]
        scores = await self.ScoreBatched(pairs)
        combined = list(zip(indices, docs, scores))
        combined.sort(key=lambda x: x[2], reverse=True)
        return combined
