# services/GteEmbedService.py
import asyncio
from typing import Any, Dict, List, cast

import torch  # kept for GPU keepalive and original constants
from transformers import AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService

import onnxruntime as ort
import numpy as np

# ----------------------
# Config
# ----------------------
modelName: str = "thenlper/gte-base"
ONNX_MODEL_PATH: str = "/home/alien/gte-embedAPI/services/gte-base.onnx"
maxLength: int = 300
maxTexts: int = 100
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keepaliveInterval: float = 10.0

# Globals (singletons)
tokenizer: Any = None
ort_session: Any = None
_input_names: Any = None
_output_name: Any = None
keepaliveTask: Any = None


class GteEmbedService(GteEmbedServiceImpl):
    def LoadModel(self) -> None:
        """
        Initialize tokenizer and ONNX Runtime session. Warm up once.
        """
        global tokenizer, ort_session, _input_names, _output_name, keepaliveTask
        if tokenizer is not None and ort_session is not None:
            return

        # tokenizer (cast to Any to silence strict type-checkers)
        tokenizer = cast(Any, AutoTokenizer.from_pretrained(modelName, use_fast=True))

        # create ORT session with CUDAExecutionProvider if available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # single-threaded per session to reduce context switching overhead; adjust if needed
        sess_opts.intra_op_num_threads = 1

        try:
            ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_opts, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX Runtime session for {ONNX_MODEL_PATH}: {e}")

        # capture input/output names to map tokenizer outputs to ONNX inputs
        _input_names = [inp.name for inp in ort_session.get_inputs()]
        outputs = ort_session.get_outputs()
        if not outputs:
            raise RuntimeError("ONNX model has no outputs.")
        _output_name = outputs[0].name

        # Warmup (run one short forward to JIT kernels / allocate mem)
        try:
            warmText: str = "hello " * 16
            tokWarm: Dict[str, Any] = tokenizer(
                [warmText],
                return_tensors="np",
                truncation=True,
                max_length=maxLength,
                padding=True,
            )

            onnx_inputs = {}
            # map ort input names to tokenizer outputs (case-insensitive)
            tok_keys_lc = {k.lower(): k for k in tokWarm.keys()}
            for name in _input_names:
                lname = name.lower()
                if lname in tok_keys_lc:
                    arr = tokWarm[tok_keys_lc[lname]]
                    # ensure int64 for ids/masks as ONNX expects
                    if arr.dtype != np.int64:
                        arr = arr.astype(np.int64)
                    onnx_inputs[name] = arr
                else:
                    # if model expects token_type_ids etc. provide zeros
                    if "token_type" in lname or "segment" in lname:
                        sample = next(iter(tokWarm.values()))
                        onnx_inputs[name] = np.zeros(sample.shape, dtype=np.int64)
                    else:
                        # best-effort: attempt to skip unknown optional inputs
                        raise RuntimeError(f"ONNX model expects input '{name}' but tokenizer didn't produce it. Tokenizer keys: {list(tokWarm.keys())}")

            # run warmup
            ort_session.run([_output_name], onnx_inputs)
        except Exception:
            # non-fatal warmup errors shouldn't stop service init
            pass

        # instantiate batcher (maxBatchSize=0 -> immediate dispatch; keep maxDelayMs small)
        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=0, maxDelayMs=5)

    def MeanPool(self, lastHidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Mean pooling for NumPy arrays.
        lastHidden: (batch, seq_len, hidden)
        mask: (batch, seq_len) with 1 for tokens to include, 0 otherwise
        """
        # ensure float type matches lastHidden dtype
        m = np.expand_dims(mask.astype(lastHidden.dtype), axis=-1)  # (batch, seq_len, 1)
        summed = (lastHidden * m).sum(axis=1)  # (batch, hidden)
        denom = m.sum(axis=1).clip(min=1e-9)  # (batch, 1)
        return summed / denom

    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        return await self.batcher.submit(texts)

    def Embed(self, texts: List[str]) -> List[List[float]]:
        """
        Tokenize -> run ONNX -> mean-pool -> return python list embeddings.
        """
        global tokenizer, ort_session, _input_names, _output_name

        if not texts:
            raise ValueError("texts empty")
        if len(texts) > maxTexts:
            raise ValueError(f"texts length exceeds {maxTexts}")

        if tokenizer is None or ort_session is None:
            raise RuntimeError("Model not loaded. Call LoadModel first.")

        # Tokenize to numpy to avoid torch copies; use padding/truncation
        tok: Dict[str, Any] = tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )

        # Build ONNX Runtime input dict mapping names expected by the model
        onnx_inputs: Dict[str, np.ndarray] = {}
        tok_keys_lc = {k.lower(): k for k in tok.keys()}

        for name in _input_names:
            lname = name.lower()
            if lname in tok_keys_lc:
                arr = tok[tok_keys_lc[lname]]
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                onnx_inputs[name] = arr
            else:
                # if input expected is token_type_ids or similar, supply zeros
                if "token_type" in lname or "segment" in lname:
                    sample = next(iter(tok.values()))
                    onnx_inputs[name] = np.zeros(sample.shape, dtype=np.int64)
                else:
                    raise RuntimeError(f"ONNX model expects input '{name}' but tokenizer didn't produce it. Tokenizer keys: {list(tok.keys())}")

        # Run inference (blocking). CUDAExecutionProvider will run on GPU if available.
        ort_outs = ort_session.run([_output_name], onnx_inputs)
        last_hidden = ort_outs[0]  # expected shape: (batch, seq_len, hidden), numpy ndarray

        # get attention_mask (if available) else assume all ones
        attn_key = None
        for k in tok.keys():
            if k.lower() == "attention_mask":
                attn_key = k
                break
        if attn_key is None:
            attention_mask = np.ones(last_hidden.shape[:2], dtype=np.int64)
        else:
            attention_mask = tok[attn_key].astype(np.int64)

        # Mean pool (numpy)
        pooled = self.MeanPool(last_hidden, attention_mask)  # (batch, hidden)

        # Convert to Python list (float) for JSON/transport
        result: List[List[float]] = [pooled[i].astype(float).tolist() for i in range(pooled.shape[0])]
        return result

    async def GpuKeepAlive(self) -> None:
        """
        Periodic GPU keepalive to prevent GPU/driver timeout. Kept simple (uses torch).
        If you don't want torch dependency at all, remove this method or replace with an ORT-based keepalive.
        """
        while True:
            try:
                if torch.cuda.is_available():
                    a = torch.empty((64, 64), device="cuda")
                    a.add_(1.0)
                    # avoid global synchronize here to reduce latency spikes
            except Exception:
                pass
            await asyncio.sleep(keepaliveInterval)
