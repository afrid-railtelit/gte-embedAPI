import asyncio
from typing import Any, Dict, List, cast

import torch
from transformers import AutoModel, AutoTokenizer

from implementations import GteEmbedServiceImpl

modelName: str = "thenlper/gte-large"
maxLength: int = 5000
maxTexts: int = 30
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keepaliveInterval: float = 30

tokenizer: Any = None
model: Any = None
keepaliveTask: Any = None


class GteEmbedService(GteEmbedServiceImpl):
    def LoadModel(self) -> None:
        global tokenizer, model, keepaliveTask
        if tokenizer is not None and model is not None:
            return

        tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            modelName, use_fast=True
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = cast(Any, AutoModel).from_pretrained(modelName, dtype=dtype)
        model.to(device)
        model.eval()

        warmText: str = "hello " * 16
        tokWarm: Dict[str, torch.Tensor] = tokenizer(
            warmText, return_tensors="pt", truncation=True, max_length=maxLength
        )
        inputsWarm: Dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in tokWarm.items()
        }

        with torch.inference_mode():
            _ = model(**inputsWarm)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def MeanPool(self, lastHidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).expand(lastHidden.size()).float()
        return (lastHidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def Embed(self, texts: List[str]) -> List[List[float]]:
        import time
        t0 = time.time()

        if not texts:
            raise ValueError("texts empty")
        if len(texts) > maxTexts:
            raise ValueError(f"texts length exceeds {maxTexts}")

        t_tok0 = time.time()
        tok: Dict[str, torch.Tensor] = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )
        t_tok = time.time() - t_tok0

        t_move0 = time.time()
        inputs: Dict[str, torch.Tensor] = {
            k: v.to(device, non_blocking=True) for k, v in tok.items()
        }
        t_move = time.time() - t_move0

        t_model0 = time.time()
        with torch.inference_mode():
            out = model(**inputs)
            emb: Any = self.MeanPool(
                out.last_hidden_state, inputs["attention_mask"]
            ).cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_model = time.time() - t_model0

        result: List[List[float]] = [emb[i].tolist() for i in range(emb.size(0))]

        total = time.time() - t0
        print(
            f"[embed timings] total={total:.3f}s tok={t_tok:.3f}s "
            f"move={t_move:.3f}s model={t_model:.3f}s"
        )

        return result



    async def GpuKeepAlive(self) -> None:
        while True:
            try:
                # tiny operation that touches the device without heavy compute
                a = torch.empty((64, 64), device="cuda")
                a.add_(1.0)
                # ensure kernels finished
                torch.cuda.synchronize()
            except Exception:
                pass
            await asyncio.sleep(keepaliveInterval)