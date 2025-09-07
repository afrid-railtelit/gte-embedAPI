import asyncio
from typing import Any, Dict, List, cast
import torch
from transformers import AutoModel, AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService
import numpy as np

modelName: str = "abhinand/MedEmbed-large-v0.1"
maxLength: int = 512  # Reduced to typical transformer limit
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
        dtype = torch.float32  # Use full precision to avoid numerical issues
        model = cast(Any, AutoModel).from_pretrained(modelName, dtype=dtype)  # Fixed deprecation
        model = model.to(device)
        model.eval()

        warmText: str = "Patient with hypertension prescribed lisinopril."  # Medical-related warm-up text
        tokWarm: Dict[str, torch.Tensor] = tokenizer(
            warmText, return_tensors="pt", truncation=True, max_length=maxLength
        )
        inputsWarm: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in tokWarm.items()}

        with torch.inference_mode():
            _ = model(**inputsWarm)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=40, maxDelayMs=5)

    def MeanPool(self, lastHidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(lastHidden.dtype)
        summed = (lastHidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / counts
        # L2 normalization to ensure reasonable values
        emb = emb / torch.norm(emb, p=2, dim=1, keepdim=True).clamp(min=1e-9)
        return emb

    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        print(f"Processing batch of {len(texts)} texts")
        result = await self.batcher.submit(texts)
        print(f"Batch result length: {len(result)}")
        return result

    def Embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts empty")
        if len(texts) > maxTexts:
            raise ValueError(f"texts length exceeds {maxTexts}")

        print(f"Input texts (first 2): {texts[:2]}")
        tok: Dict[str, torch.Tensor] = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )
        print(f"Input IDs shape: {tok['input_ids'].shape}")
        print(f"Attention Mask shape: {tok['attention_mask'].shape}")

        tok = {k: v.pin_memory() for k, v in tok.items()}
        inputs = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

        with torch.inference_mode():
            out = model(**inputs)
            print(f"Last hidden state shape: {out.last_hidden_state.shape}")
            print(f"Last hidden state stats: min={out.last_hidden_state.min()}, max={out.last_hidden_state.max()}")
            emb = self.MeanPool(out.last_hidden_state, inputs["attention_mask"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        emb_cpu = emb.detach().to("cpu", non_blocking=True)
        result = [emb_cpu[i].tolist() for i in range(emb_cpu.size(0))]
        emb_np = np.array(result)
        print(f"Embedding stats: min={emb_np.min()}, max={emb_np.max()}, mean={emb_np.mean()}")
        return result

    async def GpuKeepAlive(self) -> None:
        while True:
            try:
                if torch.cuda.is_available():
                    a = torch.empty((4, 4), device="cuda")
                    a.add_(1.0)
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"GpuKeepAlive error: {e}")
            await asyncio.sleep(keepaliveInterval)