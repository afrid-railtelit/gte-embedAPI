import asyncio
from typing import Any, Dict, List, cast
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService

modelName: str = "abhinand/MedEmbed-large-v0.1"
maxLength: int = 500
maxTexts: int = 100
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keepaliveInterval: float = 20  # Increased to reduce overhead
tokenizer: Any = None
model: Any = None
keepaliveTask: Any = None

class GteEmbedService(GteEmbedServiceImpl):
    def __init__(self):
        super().__init__()
        self.LoadModel()  # Preload model at initialization
        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=256, maxDelayMs=1)  # Optimized batching

    def LoadModel(self) -> None:
        global tokenizer, model, keepaliveTask
        if tokenizer is not None and model is not None:
            return
        print(f"Loading model {modelName} on {device}")
        tokenizer = AutoTokenizer.from_pretrained(modelName, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # BF16 for faster inference
        model = AutoModel.from_pretrained(modelName, torch_dtype=dtype)
        model = model.to(device)
        if torch.cuda.is_available():
            model = torch.compile(model, mode="reduce-overhead")  # Compile for optimized execution
        model.eval()
        warmText = "Patient with hypertension prescribed lisinopril."
        tokWarm = tokenizer(warmText, return_tensors="pt", truncation=True, max_length=maxLength)
        inputsWarm = {k: v.to(device, non_blocking=True) for k, v in tokWarm.items()}
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            _ = model(**inputsWarm)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print("Model warmed up successfully")
        if torch.cuda.is_available() and keepaliveTask is None:
            keepaliveTask = asyncio.create_task(self.GpuKeepAlive())

    def MeanPool(self, lastHidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(lastHidden.dtype)
        summed = (lastHidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / counts
        emb = F.normalize(emb, p=2, dim=1)  # Optimized normalization
        return emb

    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        result = await self.batcher.submit(texts)
        return result

    def Embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts empty")
        if len(texts) > maxTexts:
            raise ValueError(f"texts length exceeds {maxTexts}")
        tok = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            out = model(**inputs)
            emb = self.MeanPool(out.last_hidden_state, inputs["attention_mask"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        emb_cpu = emb.detach().to("cpu", non_blocking=True)
        result = [emb_cpu[i].tolist() for i in range(emb_cpu.size(0))]
        return result

    async def GpuKeepAlive(self) -> None:
        if not torch.cuda.is_available():
            return
        while True:
            try:
                torch.cuda.synchronize()  # Minimal keep-alive
            except Exception as e:
                print(f"GpuKeepAlive error: {e}")
            await asyncio.sleep(keepaliveInterval)