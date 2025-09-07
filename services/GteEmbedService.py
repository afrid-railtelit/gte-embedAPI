# services/GteEmbedService.py
import asyncio
from typing import Any, Dict, List, cast

import torch  # kept for GPU keepalive and device detection
from transformers import AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService

import onnxruntime as ort
import numpy as np

# ----------------------
# Config
# ----------------------
modelName: str = "thenlper/gte-base"
# Set this to where your ONNX file actually is (absolute path is safest)
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

        # Warmup (run one short forward to compile kernels / allocate mem)
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
                        raise RuntimeError(
                            f"ONNX model expects input '{name}' but tokenizer didn't produce it. "
                            f"Tokenizer keys: {list(tokWarm.keys())}"
                        )

            # run warmup
            ort_session.run([_output_name], onnx_inputs)
        except Exception:
            # non-fatal warmup errors shouldn't stop service init
            pass

        # instantiate batcher (maxBatchSize=0 -> immediate dispatch; keep maxDelayMs small)
        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=0, maxDelayMs=5)

    def MeanPool(self, lastHidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Robust mean pooling that handles:
          - lastHidden already pooled (shape (B, H))
          - lastHidden token-level (B, S, H)
          - mask/lastHidden seq_len mismatches by truncating or padding mask

        lastHidden: np.ndarray, either (B, H) or (B, S, H)
        mask: np.ndarray with shape (B, S_mask) (dtype 0/1)
        returns: np.ndarray (B, H) dtype float32
        """
        # If model already returned pooled embeddings (B, H) -> return directly
        if lastHidden.ndim == 2:
            return lastHidden.astype(np.float32)

        # Expect lastHidden.ndim == 3: (B, seq_len, hidden)
        if lastHidden.ndim != 3:
            raise ValueError(f"Unexpected lastHidden ndim: {lastHidden.ndim}, expected 2 or 3.")

        batch, seq_len, hidden = lastHidden.shape

        # Normalize mask dims
        if mask.ndim != 2:
            # try to squeeze if someone passed (B, S, 1)
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.squeeze(2)
            else:
                raise ValueError(f"Unexpected attention mask shape: {mask.shape}")

        mask_batch, mask_len = mask.shape
        if mask_batch != batch:
            raise ValueError(f"Batch size mismatch between lastHidden ({batch}) and mask ({mask_batch})")

        # If mask length != seq_len, fix by truncating or padding (pad with ones)
        if mask_len != seq_len:
            if mask_len > seq_len:
                # tokenizer produced longer mask than model output seq_len: truncate
                mask = mask[:, :seq_len]
            else:
                # mask shorter than model seq_len: pad with ones (assume remaining tokens valid)
                pad_len = seq_len - mask_len
                pad = np.ones((batch, pad_len), dtype=mask.dtype)
                mask = np.concatenate([mask, pad], axis=1)

        # perform mean pooling safely
        m = np.expand_dims(mask.astype(lastHidden.dtype), axis=-1)  # (B, S, 1)
        summed = (lastHidden * m).sum(axis=1)  # (B, H)
        denom = m.sum(axis=1).clip(min=1e-9)    # (B, 1)
        return (summed / denom).astype(np.float32)

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
                    raise RuntimeError(
                        f"ONNX model expects input '{name}' but tokenizer didn't produce it. Tokenizer keys: {list(tok.keys())}"
                    )

        # Run inference (blocking). CUDAExecutionProvider will run on GPU if available.
        ort_outs = ort_session.run([_output_name], onnx_inputs)
        last_hidden = ort_outs[0]  # expected shape: (batch, seq_len, hidden) or (batch, hidden)

        # get attention_mask (if available) else assume all ones (matching seq dim if possible)
        attn_key = None
        for k in tok.keys():
            if k.lower() == "attention_mask":
                attn_key = k
                break

        if attn_key is None:
            # if model returned sequence dimension, create ones with that seq_len
            if getattr(last_hidden, "ndim", None) == 3:
                attention_mask = np.ones(last_hidden.shape[:2], dtype=np.int64)
            else:
                # last_hidden is (B, H) -> mask not needed, provide dummy
                attention_mask = np.ones((len(texts), 1), dtype=np.int64)
        else:
            attention_mask = tok[attn_key].astype(np.int64)

        # Mean pool (numpy) - MeanPool handles mismatch and already-pooled outputs
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
