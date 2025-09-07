import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, cast
import os
import time
import torch
from transformers import AutoModel, AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService

# --- Configurable parameters (tune for your workload) -----------------------
MODEL_NAME = os.environ.get("GTE_MODEL", "abhinand/MedEmbed-large-v0.1")
MAX_LENGTH = int(os.environ.get("GTE_MAX_LENGTH", "500"))
MAX_TEXTS = int(os.environ.get("GTE_MAX_TEXTS", "100"))
# For very low-latency single requests: smaller batch & tiny delay
DEFAULT_MAX_BATCH_SIZE = int(os.environ.get("GTE_MAX_BATCH_SIZE", "16"))
DEFAULT_MAX_DELAY_MS = int(os.environ.get("GTE_MAX_DELAY_MS", "2"))
KEEPALIVE_INTERVAL = float(os.environ.get("GTE_KEEPALIVE_INTERVAL", "10"))
THREADPOOL_WORKERS = int(os.environ.get("GTE_TPOOL_WORKERS", "4"))
COMPILE_MODEL = os.environ.get("GTE_COMPILE", "1") == "1"
PIN_TOKENIZER_TENSORS = True
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals reused across calls
_tokenizer: Optional[Any] = None
_model: Optional[Any] = None
_batcher: Optional[Any] = None
_tokenize_executor: Optional[ThreadPoolExecutor] = None
_keepalive_task: Optional[asyncio.Task] = None

# ---- Utilities ------------------------------------------------------------
def safe_keep_layernorm_fp32(model: torch.nn.Module) -> None:
    """
    After converting model to half, convert LayerNorm modules back to float32
    to improve numerical stability (common recommendation).
    """
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()
            # convert parameters back to fp32 to avoid unexpected fp16 layernorm behavior
            for p in module.parameters(recurse=False):
                p.data = p.data.float()
                if p.grad is not None:
                    p.grad.data = p.grad.data.float()


def pin_tensor_dict(t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Pin CPU memory for each tensor value in dict (used before .to(device, non_blocking=True)).
    """
    for k, v in list(t.items()):
        if isinstance(v, torch.Tensor) and v.device.type == "cpu":
            try:
                t[k] = v.pin_memory()
            except Exception:
                # pin_memory not supported for some tensor types (safeguard)
                pass
    return t


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool with mask. Compatible with fp16 models; returns normalized vectors.
    """
    # attention_mask might be int64 — convert for multiplication to hidden dtype
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    emb = summed / counts
    # L2 normalize
    norm = torch.norm(emb, p=2, dim=1, keepdim=True).clamp(min=1e-9)
    emb = emb / norm
    return emb


# ---- Core service ---------------------------------------------------------
class GteEmbedService(GteEmbedServiceImpl):
    def __init__(self):
        super().__init__()
        # threadpool for tokenization to avoid blocking event loop
        global _tokenize_executor
        if _tokenize_executor is None:
            _tokenize_executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)

    def LoadModel(self) -> None:
        """
        Load and prepare model for low-latency inference.
        Idempotent: safe to call multiple times.
        """
        global _tokenizer, _model, _batcher, _keepalive_task

        if _tokenizer is not None and _model is not None and _batcher is not None:
            return

        # Basic cudnn tuning
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        print(f"[GteEmbedService] Loading model {MODEL_NAME} -> device {device}")

        # Load tokenizer (fast tokenizer preferred)
        _tokenizer = cast(Any, AutoTokenizer).from_pretrained(MODEL_NAME, use_fast=True)

        # Load model (default dtype). We'll cast to half on CUDA below.
        _model = cast(Any, AutoModel).from_pretrained(MODEL_NAME)
        _model.eval()
        _model.to(device)

        if device.type == "cuda":
            # Convert model weights to FP16 for speed; keep LayerNorm in FP32 for stability.
            try:
                _model.half()
                safe_keep_layernorm_fp32(_model)
                print("[GteEmbedService] Converted model to FP16 with LayerNorm in FP32")
            except Exception as e:
                print("[GteEmbedService] Warning: model.half() failed:", e)

        # Try torch.compile to reduce Python overhead for repeated calls
        if COMPILE_MODEL and hasattr(torch, "compile"):
            try:
                _model = torch.compile(_model, mode="reduce-overhead")
                print("[GteEmbedService] torch.compile applied (reduce-overhead)")
            except Exception as e:
                # compilation can sometimes fail for certain models; fall back gracefully
                print("[GteEmbedService] torch.compile failed or not applicable:", e)

        # Warm-up: several realistic shapes (short + long) to prime kernels
        try:
            warm_texts = [
                "Patient with hypertension prescribed lisinopril.",
                "Short sample.",
                " ".join(["word"] * min(256, MAX_LENGTH))
            ]
            tok_warm = _tokenizer(warm_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
            # pin memory for async copy
            if PIN_TOKENIZER_TENSORS:
                tok_warm = pin_tensor_dict(tok_warm)
            inputs_warm = {k: v.to(device, non_blocking=True) for k, v in tok_warm.items()}

            with torch.inference_mode():
                _ = _model(**inputs_warm)
            print("[GteEmbedService] Warm-up completed")
        except Exception as e:
            print("[GteEmbedService] Warm-up failed:", e)

        # Instantiate batcher tuned for low latency (tune further if needed)
        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=DEFAULT_MAX_BATCH_SIZE, maxDelayMs=DEFAULT_MAX_DELAY_MS)
        _batcher = self.batcher

        # Start keepalive task (if not already running)
        try:
            if _keepalive_task is None:
                loop = asyncio.get_event_loop()
                _keepalive_task = loop.create_task(self.GpuKeepAlive())
                print("[GteEmbedService] Started GPU keepalive task")
        except Exception:
            # In some contexts there may be no running event loop; ignore
            pass

    # Async wrapper to submit to batcher
    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await self.batcher.submit(texts)

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenization runs in a threadpool to avoid blocking asyncio loop.
        Returns tensors on CPU (unpinned unless configured).
        """
        global _tokenizer
        # Using the fast tokenizer with padding/truncation
        tok = _tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        if PIN_TOKENIZER_TENSORS:
            tok = pin_tensor_dict(tok)
        return tok

    def Embed(self, texts: List[str]) -> List[List[float]]:
        """
        Main synchronous embedding path executed by the batcher worker thread.
        Designed to be fast: tokenization happens in threadpool; copies to GPU use non_blocking=True.
        """
        if not texts:
            raise ValueError("texts empty")
        if len(texts) > MAX_TEXTS:
            raise ValueError(f"texts length exceeds {MAX_TEXTS}")

        # Tokenize off the main thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        # run_in_executor expects a callable that returns token dict
        token_future = loop.run_in_executor(_tokenize_executor, self._tokenize_texts, texts)
        # Wait synchronously (we're in a thread used by the batcher; this is fine)
        tok: Dict[str, torch.Tensor] = asyncio.get_event_loop().run_until_complete(token_future)

        # Move inputs to device non-blocking
        inputs = {k: v.to(device, non_blocking=True) for k, v in tok.items()}

        # Ensure input ids are correct dtype (long)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()

        with torch.inference_mode():
            # Forward pass
            out = _model(**inputs)
            # last_hidden_state shape: (batch, seq_len, hidden)
            emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])

            # Do not call torch.cuda.synchronize() here to avoid blocking.
            # If you need to measure, benchmark externally with explicit synchronize calls.

        # Move to CPU non-blocking and convert to float32 for stable JSON serialization
        emb_cpu = emb.detach().to("cpu", non_blocking=True).float()
        results: List[List[float]] = [emb_cpu[i].tolist() for i in range(emb_cpu.size(0))]
        return results

    async def GpuKeepAlive(self) -> None:
        """
        Keep the CUDA context warm. Uses tiny non-blocking kernels and sleeps.
        Keeps GPU context alive without forcing sync.
        """
        while True:
            try:
                if torch.cuda.is_available():
                    # tiny op on GPU; don't sync
                    a = torch.empty((8, 8), device="cuda")
                    a.add_(1.0)
            except Exception as e:
                print("[GteEmbedService][KeepAlive] error:", e)
            await asyncio.sleep(KEEPALIVE_INTERVAL)