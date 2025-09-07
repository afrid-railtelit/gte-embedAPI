# services/GteEmbedService.py
import os
import time
import asyncio
from typing import Any, Dict, List, cast

import torch  # used only for device detection and keepalive
from transformers import AutoTokenizer
from implementations import GteEmbedServiceImpl
from services.GteEmbedBatcherService import GteEmbedBatcherService

import onnxruntime as ort
import numpy as np

# ----------------------
# Config (edit PATH if needed)
# ----------------------
modelName: str = "thenlper/gte-base"
ONNX_MODEL_PATH: str = os.environ.get("ONNX_MODEL_PATH", "/home/alien/gte-embedAPI/services/gte-base.onnx")
maxLength: int = 300
maxTexts: int = 100
device_is_cuda: bool = torch.cuda.is_available()
keepaliveInterval: float = 10.0

# Globals (singletons)
tokenizer: Any = None
ort_session: Any = None
_input_names: List[str] = []
_output_name: str = ""
use_iobinding: bool = False
keepaliveTask: Any = None


def _ensure_onnx_file(path: str) -> str:
    p = os.path.expanduser(path)
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    # try relative to services folder
    here = os.path.dirname(__file__)
    candidate = os.path.join(here, os.path.basename(p))
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    # try cwd
    if os.path.isfile(p):
        return os.path.abspath(p)
    raise FileNotFoundError(
        f"ONNX model file not found at '{path}'. Tried also: '{candidate}'. Set ONNX_MODEL_PATH to absolute path."
    )


class GteEmbedService(GteEmbedServiceImpl):
    def LoadModel(self) -> None:
        """
        Initialize tokenizer and ONNX Runtime session. Ensure CUDA EP is present if torch.cuda.is_available().
        """
        global tokenizer, ort_session, _input_names, _output_name, use_iobinding

        if tokenizer is not None and ort_session is not None:
            return

        # check ONNX exists
        onnx_path = _ensure_onnx_file(ONNX_MODEL_PATH)

        # load tokenizer
        tokenizer = cast(Any, AutoTokenizer.from_pretrained(modelName, use_fast=True))

        # create ONNX session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_opts.intra_op_num_threads = 1  # tune if you want more CPU threads

        # choose providers: prefer CUDA if available
        if device_is_cuda:
            requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            requested_providers = ["CPUExecutionProvider"]

        try:
            ort_session = ort.InferenceSession(onnx_path, sess_opts, providers=requested_providers)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ONNX Runtime session for '{onnx_path}': {e}\n"
                "If you expect GPU usage, install onnxruntime-gpu matching your CUDA version, e.g.:\n"
                "  pip uninstall onnxruntime && pip install onnxruntime-gpu==1.15.1\n"
                "See https://onnxruntime.ai for matching wheels."
            )

        actual_providers = ort_session.get_providers()
        print(f"[ONNX] requested providers: {requested_providers}")
        print(f"[ONNX] actual providers: {actual_providers}")

        if device_is_cuda and "CUDAExecutionProvider" not in actual_providers:
            # Fail fast with clear message because running on CPU will be slow.
            raise RuntimeError(
                "[ONNX] CUDAExecutionProvider not available in ONNX Runtime while torch.cuda.is_available() == True.\n"
                "Install onnxruntime-gpu that matches your CUDA and restart. CPU fallback will be very slow."
            )

        # capture input/output names
        _input_names = [inp.name for inp in ort_session.get_inputs()]
        outputs = ort_session.get_outputs()
        if not outputs:
            raise RuntimeError("ONNX model has no outputs.")
        _output_name = outputs[0].name

        # Decide whether to use IO binding (best for latency). We attempt if CUDA EP active.
        use_iobinding = ("CUDAExecutionProvider" in actual_providers)

        # Warmup: run a very short input to compile kernels / allocate memory
        try:
            warm_text = "hello " * 8
            tokWarm = tokenizer([warm_text], return_tensors="np", truncation=True, max_length=maxLength, padding=True)
            # Map inputs same as in Embed()
            onnx_inputs = {}
            tok_keys_lc = {k.lower(): k for k in tokWarm.keys()}
            for name in _input_names:
                lname = name.lower()
                if lname in tok_keys_lc:
                    arr = tokWarm[tok_keys_lc[lname]]
                    if arr.dtype != np.int64:
                        arr = arr.astype(np.int64)
                    onnx_inputs[name] = arr
                else:
                    # token_type etc -> zeros
                    if "token_type" in lname or "segment" in lname:
                        sample = next(iter(tokWarm.values()))
                        onnx_inputs[name] = np.zeros(sample.shape, dtype=np.int64)

            t0 = time.perf_counter()
            ort_session.run([_output_name], onnx_inputs)
            t1 = time.perf_counter()
            print(f"[ONNX] warmup run finished in {(t1 - t0)*1000:.1f} ms, IO binding enabled: {use_iobinding}")
        except Exception as e:
            print(f"[ONNX] warmup (non-fatal) error: {e}")

        # instantiate batcher in low-latency mode
        self.batcher = GteEmbedBatcherService(self.Embed, maxBatchSize=0, maxDelayMs=5)

    def MeanPool(self, lastHidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Robust pooling that supports both (B, H) and (B, S, H)
        if lastHidden.ndim == 2:
            return lastHidden.astype(np.float32)
        if lastHidden.ndim != 3:
            raise ValueError(f"Unexpected lastHidden.ndim {lastHidden.ndim}")

        b, seq_len, hidden = lastHidden.shape
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(2)
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask.ndim {mask.ndim}")

        if mask.shape[0] != b:
            raise ValueError("Batch size mismatch between hidden and mask")

        # Fix mask length mismatches by truncating/padding
        if mask.shape[1] != seq_len:
            if mask.shape[1] > seq_len:
                mask = mask[:, :seq_len]
            else:
                pad_len = seq_len - mask.shape[1]
                pad = np.ones((b, pad_len), dtype=mask.dtype)
                mask = np.concatenate([mask, pad], axis=1)

        m = np.expand_dims(mask.astype(lastHidden.dtype), axis=-1)
        summed = (lastHidden * m).sum(axis=1)
        denom = m.sum(axis=1).clip(min=1e-9)
        return (summed / denom).astype(np.float32)

    async def EmbedBatched(self, texts: List[str]) -> List[List[float]]:
        return await self.batcher.submit(texts)

    def _bind_inputs_outputs_iobinding(self, sess: ort.InferenceSession, io_binding: ort.IOBinding, onnx_inputs: Dict[str, np.ndarray]):
        """
        Bind inputs and outputs to the session's IO binding for CUDA provider.
        This function uses ortvalue_from_numpy which will create device OrtValue for CUDA.
        """
        # Bind inputs
        for name, arr in onnx_inputs.items():
            # ensure int64 for input ids/mask
            arr = arr.astype(np.int64) if arr.dtype != np.int64 else arr
            # create OrtValue on GPU (this copies to GPU once)
            ort_value = ort.OrtValue.ortvalue_from_numpy(arr, device_type="cuda", device_id=0)
            io_binding.bind_input(name, ort_value)

        # Prepare output binding: infer output shape by running once? We will bind output to device for the known output name.
        # For simplicity, we will not pre-allocate exact sized device buffers — instead request allocation on device by binding None with same device.
        # Many ORT versions allow bind_output with (name, device_type, device_id, elem_type, shape) but Python helper is limited,
        # so we use bind_output with name only and let ORT allocate (works on recent ORT builds).
        io_binding.bind_output(_output_name, device_type="cuda", device_id=0)

    def Embed(self, texts: List[str]) -> List[List[float]]:
        """
        Tokenize -> run ONNX (IOBinding when possible) -> mean-pool -> return python list embeddings.
        Prints timings for visibility.
        """
        global tokenizer, ort_session, _input_names, _output_name, use_iobinding

        if not texts:
            raise ValueError("texts empty")
        if len(texts) > maxTexts:
            raise ValueError(f"texts length exceeds {maxTexts}")
        if tokenizer is None or ort_session is None:
            raise RuntimeError("Model not loaded. Call LoadModel first.")

        t0 = time.perf_counter()
        tok = tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=maxLength,
            padding=True,
        )
        t1 = time.perf_counter()

        # Map tokenizer outputs to ONNX inputs
        onnx_inputs: Dict[str, np.ndarray] = {}
        tok_keys_lc = {k.lower(): k for k in tok.keys()}
        for name in _input_names:
            lname = name.lower()
            if lname in tok_keys_lc:
                arr = tok[tok_keys_lc[lname]]
                # ensure correct dtype for ort
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                onnx_inputs[name] = arr
            else:
                if "token_type" in lname or "segment" in lname:
                    sample = next(iter(tok.values()))
                    onnx_inputs[name] = np.zeros(sample.shape, dtype=np.int64)
                else:
                    raise RuntimeError(f"ONNX model expects input '{name}' but tokenizer didn't produce it. Tokenizer keys: {list(tok.keys())}")

        t2 = time.perf_counter()

        # Run inference (prefer IO binding for CUDA for fewer host->device copies)
        try:
            if use_iobinding:
                # IOBinding path (requires onnxruntime with OrtValue/IOBinding support)
                io_binding = ort_session.io_binding()
                # Bind inputs/outputs (OrtValue will copy numpy -> device once)
                self._bind_inputs_outputs_iobinding(ort_session, io_binding, onnx_inputs)

                ort_start = time.perf_counter()
                ort_session.run_with_iobinding(io_binding)
                ort_end = time.perf_counter()

                # Get outputs from io_binding as numpy on host
                # This returns a list; we assume single output
                outputs = io_binding.get_outputs()
                if not outputs:
                    raise RuntimeError("IOBinding returned no outputs")
                last_hidden = outputs[0].to_numpy()
            else:
                # Normal run() path
                ort_start = time.perf_counter()
                ort_outs = ort_session.run([_output_name], onnx_inputs)
                ort_end = time.perf_counter()
                last_hidden = ort_outs[0]
        except Exception as e:
            raise RuntimeError(f"ONNX runtime error: {e}")

        t3 = time.perf_counter()

        # Get attention mask (if present), else create ones
        attn_key = None
        for k in tok.keys():
            if k.lower() == "attention_mask":
                attn_key = k
                break
        if attn_key is None:
            attention_mask = np.ones(last_hidden.shape[:2], dtype=np.int64) if getattr(last_hidden, "ndim", None) == 3 else np.ones((len(texts), 1), dtype=np.int64)
        else:
            attention_mask = tok[attn_key].astype(np.int64)

        t4 = time.perf_counter()
        pooled = self.MeanPool(last_hidden, attention_mask)
        t5 = time.perf_counter()

        # Convert to list for response
        result = [pooled[i].astype(float).tolist() for i in range(pooled.shape[0])]
        t6 = time.perf_counter()

        print(
            f"[TIMINGS] tokenize {(t1-t0)*1000:.1f}ms | build_inputs {(t2-t1)*1000:.1f}ms | "
            f"ort {(ort_end-ort_start)*1000:.1f}ms | postproc {(t5-t4)*1000:.1f}ms | conv {(t6-t5)*1000:.1f}ms | total {(t6-t0)*1000:.1f}ms"
        )

        return result

    async def GpuKeepAlive(self) -> None:
        """
        Simple keepalive that nudges the GPU occasionally (optional).
        """
        while True:
            try:
                if torch.cuda.is_available():
                    a = torch.empty((16, 16), device="cuda")
                    a.add_(1.0)
            except Exception:
                pass
            await asyncio.sleep(keepaliveInterval)
