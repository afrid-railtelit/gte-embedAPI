import asyncio
from typing import Any, Dict, List, cast

import torch
from transformers import AutoModel, AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService

modelName: str = "abhinand/MedEmbed-large-v0.1"
maxLength: int = 1500
maxTexts: int = 60
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keepaliveInterval: float = 10

tokenizer: Any = None
model: Any = None
keepaliveTask: Any = None


class GteEmbedService(GteEmbedServiceImpl):
    def LoadModel(self) -> None:
        torch.backends.cudnn.benchmark = True
        global tokenizer, model, keepaliveTask
        if tokenizer is not None and model is not None:
            return

        tokenizer = cast(Any, AutoTokenizer).from_pretrained(modelName, use_fast=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = cast(Any, AutoModel).from_pretrained(modelName, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.half().to(device)
        else:
            model = model.to(device)
        model.eval()

        warmText: str = "hello " * 16
        tokWarm: Dict[str, torch.Tensor] = tokenizer(
            warmText, return_tensors="pt", truncation=True, max_length=maxLength
        )
        inputsWarm: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in tokWarm.items()}

        with torch.inference_mode():
            _ = model(**inputsWarm)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        try:
            model = torch.compile(model)
        except Exception:
            pass

        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=40, maxDelayMs=5)

    def MeanPool(self, lastHidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(lastHidden.dtype)
        summed = (lastHidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        return await self.batcher.submit(texts)

    def Embed(self, texts: List[str]) -> List[List[float]]:

        if not texts:
            raise ValueError("texts empty")
        if len(texts) > maxTexts:
            raise ValueError(f"texts length exceeds {maxTexts}")

        tok: Dict[str, torch.Tensor] = tokenizer(
            texts,
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
            out = model(**inputs)
            emb: Any = self.MeanPool(out.last_hidden_state, inputs["attention_mask"])

        emb_cpu = emb.detach().to("cpu", non_blocking=True)

        result: List[List[float]] = [emb_cpu[i].tolist() for i in range(emb_cpu.size(0))]

        return result

    async def GpuKeepAlive(self) -> None:
        while True:
            try:
                if torch.cuda.is_available():
                    a = torch.empty((4, 4), device="cuda")
                    a.add_(1.0)
            except Exception:
                pass
            await asyncio.sleep(keepaliveInterval)
