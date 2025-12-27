import torch, scipy
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from safetensors import SafetensorError   # ← THIS LINE IS THE FIX (adds missing error type)
import os
from functools import wraps
import re
from scripts.untitled.common import cmn
from scripts.untitled.common import merge_stats
from collections import OrderedDict
import threading
import math
from typing import List, Optional


class WeightsCache:
    def __init__(self, size_mb, max_items=None):
        self.mapping = OrderedDict()
        self.size_cap = min(size_mb, 8192) * 1024 * 1024  # bytes
        self.size = 0
        self.max_items = max_items
        self.lock = threading.Lock()

        # Optional stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __setitem__(self, key, t):
        # Defensive: only cache valid tensors
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return

        # Defensive: key must be truly hashable
        try:
            hash(key)
        except Exception as e:
            raise TypeError(f"WeightsCache key must be hashable: {e}")

        with self.lock:
            # Remove existing entry
            if key in self.mapping:
                old = self.mapping.pop(key)
                self.size -= tensor_size(old)

            # Store CPU, detached tensor
            t_cpu = t.detach().cpu()
            self.mapping[key] = t_cpu
            self.mapping.move_to_end(key)
            self.size += tensor_size(t_cpu)

            # Evict LRU until under limits
            while (
                self.mapping and
                (
                    self.size > self.size_cap or
                    (self.max_items and len(self.mapping) > self.max_items)
                )
            ):
                _, tensor = self.mapping.popitem(last=False)
                self.size -= tensor_size(tensor)
                self.evictions += 1

    def __getitem__(self, key):
        with self.lock:
            if key not in self.mapping:
                self.misses += 1
                raise KeyError(f"WeightsCache miss for key: {key}")

            t = self.mapping[key]
            self.mapping.move_to_end(key)
            self.hits += 1

        return t.clone().to(
            device=cmn.get_device(),
            dtype=cmn.get_dtype()
        )
    
def tensor_size(t):
    """Return tensor size in bytes (safe for all tensor dtypes)."""
    if not isinstance(t, torch.Tensor):
        return 0
    return t.element_size() * t.nelement()


def recurse(operation):
    source_tensors = []

    for idx, source in enumerate(operation.sources):
        try:
            # Resolve source
            if hasattr(source, "merge"):
                t = source.merge()
            else:
                t = torch.as_tensor(source)

            if t is None:
                raise RuntimeError(
                    f"Source resolved to None (index {idx}, source={source})"
                )

            # 🚨 Scalar tensors are forbidden at execution boundary
            if not isinstance(t, torch.Tensor) or t.ndim == 0:
                raise RuntimeError(
                    f"Invalid scalar or non-tensor source at index {idx} "
                    f"for operation '{operation.key}'"
                )

            # Validate base tensor (operator-level policy)
            validated = operation.validate_base(t)
            if validated is None:
                raise RuntimeError(
                    f"Source rejected by validate_base "
                    f"(index {idx}, key={operation.key})"
                )

            t = validated

            # Normalize ONLY floating tensors
            if t.is_floating_point():
                t = t.to(device=cmn.get_device(), dtype=cmn.get_dtype())
            else:
                t = t.to(device=cmn.get_device())

            source_tensors.append(t)

        except Exception as e:
            raise RuntimeError(
                f"Failed resolving source {idx} for operation '{operation.key}' "
                f"(source={source})"
            ) from e

    return operation.oper(*source_tensors)


# Optional debug toggle (global, cheap)
CACHE_DEBUG = False
_cache_lock = threading.Lock()

def multi_cache_operation(func):
    def wrapper(self, *source_tensors):

        # ─────────────────────────────────────────────
        # Optional bypass hook
        # ─────────────────────────────────────────────
        if getattr(self, "disable_cache", False):
            return func(self, *source_tensors)

        # ─────────────────────────────────────────────
        # Fast path: cache hit
        # ─────────────────────────────────────────────
        try:
            result = weights_cache[self]
            if CACHE_DEBUG:
                print(f"[CACHE HIT] {self.__class__.__name__} :: {self.key}")
            return result
        except KeyError:
            if CACHE_DEBUG:
                print(f"[CACHE MISS] {self.__class__.__name__} :: {self.key}")

        # ─────────────────────────────────────────────
        # Compute (outside lock)
        # ─────────────────────────────────────────────
        result = func(self, *source_tensors)

        # ─────────────────────────────────────────────
        # Validation gate (CRITICAL)
        # ─────────────────────────────────────────────
        if result is None or not isinstance(result, torch.Tensor):
            if CACHE_DEBUG:
                print(f"[CACHE SKIP] {self.key} → invalid result")
            return result

        # 🚨 Scalar tensors forbidden
        if result.ndim == 0:
            raise ValueError(
                f"[CACHE ERROR] Scalar tensor detected for key '{self.key}' "
                f"in {self.__class__.__name__}"
            )

        if result.numel() == 0:
            if CACHE_DEBUG:
                print(f"[CACHE SKIP] {self.key} → empty tensor")
            return result

        if result.requires_grad:
            if CACHE_DEBUG:
                print(f"[CACHE SKIP] {self.key} → requires_grad")
            return result

        # ─────────────────────────────────────────────
        # Cross-arch shape enforcement (optional but good)
        # ─────────────────────────────────────────────
        expected_shape = getattr(cmn, "cross_arch_target_shapes", {}).get(self.key)
        if expected_shape is not None and tuple(result.shape) != tuple(expected_shape):
            raise ValueError(
                f"[CACHE ERROR] Shape mismatch for key '{self.key}': "
                f"got {tuple(result.shape)}, expected {tuple(expected_shape)}"
            )

        # ─────────────────────────────────────────────
        # Semantic-critical keys bypass cache
        # ─────────────────────────────────────────────
        forbidden = getattr(self, "FORBIDDEN_PATTERNS", ())
        key = str(self.key)
        if forbidden and any(p in key for p in forbidden):

            if CACHE_DEBUG:
                print(f"[CACHE BYPASS] {self.key} → semantic-critical key")
            return result

        # ─────────────────────────────────────────────
        # Cache store
        # ─────────────────────────────────────────────
        weights_cache[self] = result

        if CACHE_DEBUG:
            print(f"[CACHE STORE] {self.__class__.__name__} :: {self.key}")

        return result

    return wrapper

weights_cache = WeightsCache(4096, max_items=100_000)


###OPERATORS####

class Operation:
    def __init__(self, key, *sources):
        self.key = key
        self.sources = tuple(sources)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self.seed = None
        self.merge_func = recurse  # ← still used by merge()

    def __eq__(self, other):
        return (
            self.key,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.seed,
            self.sources
        ) == (
            other.key,
            other.alpha,
            other.beta,
            other.gamma,
            other.delta,
            other.seed,
            other.sources
        )

    def __hash__(self):
        return hash((
            self.key,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.seed,
            self.sources
        ))

    def oper(self, *tensors) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement oper(self, *tensors)")

    def merge(self):
        out = self.merge_func(self)
        return out
    # -------------------------------------------------
    # CANONICAL: cascade fallback validation helper
    # -------------------------------------------------
    def validate_base(self, base):
        """
        Validate the base tensor for this operation.

        Returns:
          • torch.Tensor → safe to operate on
          • None          → MUST trigger cascade fallback

        Contract:
          • Scalar tensors are NEVER valid
          • Non-tensors are refused
          • Non-float tensors pass through untouched
        """
        if not isinstance(base, torch.Tensor):
            return None

        # Non-floating tensors are safe passthroughs
        if not base.is_floating_point():
            return base

        # 🚨 Scalar tensors must never propagate
        if base.ndim == 0:
            return None

        return base

    # -------------------------------------------------
    # OPTIONAL: shared broadcast-safe math helper
    # -------------------------------------------------
    def safe(self, op, a, b):
        """
        Apply a binary tensor operation safely.
        Prevents broadcasting and catastrophic allocation.
        """
        return cmn.safe_apply(op, a, b, self.key)

class SmartResize(Operation):
    """
    SmartResize:
      - Sacred keys: NEVER interpolate (pad/slice only)
      - Non-sacred:
          * 1D: linear
          * 2D: bilinear (special-case large vocab as row-wise resize)
          * 3D: resize last dim with linear; pad/slice first dims
          * 4D (OIHW conv): resize H/W; pad/slice O/I channels
      - Weird rank mismatch: pad/slice or return as-is

    Fallback contract:
      • If called with no tensors:
          - Use self.source_tensor if provided
          - Otherwise REFUSE (return None)
    """
    def __init__(self, key, target_shape, source_tensor=None, orig_key=None):
        super().__init__(key)
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.source_tensor = source_tensor
        self.orig_key = orig_key if orig_key is not None else key

        # SmartResize is NOT safely cacheable: output depends on input tensor values.
        self.disable_cache = True

    @multi_cache_operation
    def oper(self, *tensors):
        # --------------------------
        # Resolve input tensor
        # --------------------------
        if not tensors:
            if self.source_tensor is None:
                return None
            t = self.source_tensor
        else:
            t = tensors[0]

        t = self.validate_base(t)
        if t is None:
            return None

        target = self.target_shape
        if (not target) or (len(target) == 0) or any(s > 100_000 for s in target):
            return t

        device = t.device
        dtype = t.dtype

        if t.numel() == 0:
            return torch.zeros(target, device=device, dtype=dtype)

        # Sacred check uses canonical key
        is_sacred = cmn.is_sacred_key(self.orig_key)

        # --------------------------
        # Helpers
        # --------------------------
        def _pad_slice_to_target(x: torch.Tensor, tgt_shape: tuple) -> torch.Tensor:
            if x.ndim != len(tgt_shape):
                if x.ndim == 0 and len(tgt_shape) == 0:
                    return x
                return x

            out = torch.zeros(tgt_shape, device=x.device, dtype=x.dtype)
            slices_out = []
            slices_in = []
            for src, tgt in zip(x.shape, tgt_shape):
                n = min(src, tgt)
                slices_out.append(slice(0, n))
                slices_in.append(slice(0, n))
            out[tuple(slices_out)] = x[tuple(slices_in)]
            return out

        def _interp_last_dim(block: torch.Tensor, new_last: int) -> torch.Tensor:
            if block.shape[-1] == new_last:
                return block
            flat = block.reshape(-1, 1, block.shape[-1])
            resized = F.interpolate(flat, size=new_last, mode="linear", align_corners=False)
            return resized.reshape(*block.shape[:-1], new_last)

        def _is_interpolatable(x: torch.Tensor) -> bool:
            # Interpolate supports floating point (not complex) safely here
            return x.is_floating_point()

        # ===============================================================
        # SACRED: NEVER interpolate, only pad/slice
        # ===============================================================
        if is_sacred:
            if t.shape == target:
                return t

            if t.ndim != len(target):
                return t

            result = _pad_slice_to_target(t, target)
            merge_stats.smart_resized += 1
            print(f"[SANCTUARY] Preserved sacred tensor: {self.orig_key} → {tuple(result.shape)}")
            return result

        # ===============================================================
        # NORMAL: interpolation allowed
        # ===============================================================
        if t.ndim != len(target):
            return t

        # Non-float (including complex, int, bool): pad/slice only
        if not _is_interpolatable(t):
            result = _pad_slice_to_target(t, target)
            merge_stats.smart_resized += 1
            return result

        # Promote half/bf16 to fp32 for interpolate stability
        tmp = t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t

        # 1D
        if tmp.ndim == 1:
            if tmp.shape == target:
                return tmp.to(dtype)
            out = F.interpolate(tmp[None, None, :], size=target[0], mode="linear", align_corners=False)[0, 0]
            merge_stats.smart_resized += 1
            return out.to(dtype)

        # 2D
        if tmp.ndim == 2:
            if tmp.shape == target:
                return tmp.to(dtype)

            # Large vocab special-case (vectorized, no Python loop)
            if target[0] > 20_000:
                rows = min(tmp.shape[0], target[0])
                out = torch.zeros(target, device=device, dtype=tmp.dtype)

                # Treat each row as a separate "batch item": (rows, 1, 1, src_w)
                block = tmp[:rows].unsqueeze(1).unsqueeze(2)
                resized = F.interpolate(
                    block,
                    size=(1, target[1]),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(2).squeeze(1)  # -> (rows, target[1])

                out[:rows] = resized
                merge_stats.smart_resized += 1
                return out.to(dtype)

            out = F.interpolate(tmp[None, None, :, :], size=target, mode="bilinear", align_corners=False)[0, 0]
            merge_stats.smart_resized += 1
            return out.to(dtype)

        # 3D
        if tmp.ndim == 3:
            if tmp.shape == target:
                return tmp.to(dtype)

            out = torch.zeros(target, device=device, dtype=tmp.dtype)
            a = min(tmp.shape[0], target[0])
            b = min(tmp.shape[1], target[1])

            block = tmp[:a, :b, :]
            block = _interp_last_dim(block, target[2])
            out[:a, :b, :] = block

            merge_stats.smart_resized += 1
            return out.to(dtype)

        # 4D (OIHW conv)
        if tmp.ndim == 4:
            if tmp.shape == target:
                return tmp.to(dtype)

            tgt_o, tgt_i, tgt_h, tgt_w = target
            src_o, src_i, src_h, src_w = tmp.shape

            inter = torch.zeros((tgt_o, tgt_i, src_h, src_w), device=device, dtype=tmp.dtype)
            o = min(src_o, tgt_o)
            i = min(src_i, tgt_i)
            inter[:o, :i, :, :] = tmp[:o, :i, :, :]

            if (src_h, src_w) != (tgt_h, tgt_w):
                flat = inter.reshape(-1, 1, src_h, src_w)
                flat = F.interpolate(flat, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
                inter = flat.reshape(tgt_o, tgt_i, tgt_h, tgt_w)

            merge_stats.smart_resized += 1
            return inter.to(dtype)

        # Weird rank but matching ndims: pad/slice
        result = _pad_slice_to_target(tmp.to(dtype), target)
        merge_stats.smart_resized += 1
        return result

class CopyPrimary(Operation):
    """
    Authoritative primary fallback.

    Guarantees:
    • No Dual-Soul logic
    • No SmartResize logic
    • No device / dtype normalization
    • No silent shape violations
    • Zero-fill only if explicitly allowed
    # NOTE:
    # CopyPrimary is policy-agnostic by design.
    # Eligibility, SmartResize, sacred handling, and synthetic rules
    # are enforced exclusively by initialize_task().

    """

    def __init__(self, key, primary_path, stats=None, keep_zero_fill=False, bloat_mode=False):
        super().__init__(key)
        self.primary_path = primary_path
        self.stats = stats
        self.keep_zero_fill = keep_zero_fill
        self.bloat_mode = bloat_mode
        self.operation = "CopyPrimary"  # fixes task reuse crash

    def __hash__(self):
        return hash((
            self.key,
            self.primary_path,
            self.keep_zero_fill,
            self.bloat_mode,
            self.operation
        ))

    def __eq__(self, other):
        return (
            isinstance(other, CopyPrimary)
            and self.key == other.key
            and self.primary_path == other.primary_path
            and self.keep_zero_fill == other.keep_zero_fill
            and self.bloat_mode == other.bloat_mode
            and self.operation == other.operation
        )

    def oper(self, *args):
        file = cmn.loaded_checkpoints.get(self.primary_path)
        model_name = os.path.basename(self.primary_path) if self.primary_path else "Unknown"

        # --------------------------------------------------
        # 1. Copy from primary if available
        # --------------------------------------------------
        if file and cmn.has_tensor(file, self.key):
            try:
                t = cmn.get_tensor(file, self.key)


                # Optional bloat mode (never for sacred keys)
                if self.bloat_mode and not cmn.is_sacred_key(self.key):
                    pad = 256
                    shape = t.shape
                    bloated_shape = tuple(s + 2 * pad for s in shape)

                    bloated = torch.zeros(
                        bloated_shape,
                        dtype=t.dtype,
                        device=t.device
                    )
                    slices = tuple(slice(pad, pad + s) for s in shape)
                    bloated[slices] = t
                    t = bloated

                    print(f"[BloatMode] {self.key} ← PADDED to {bloated_shape}")

                # --------------------------------------------------
                # SHAPE AUTHORITY GUARD
                # --------------------------------------------------
                target_shape = cmn.cross_arch_target_shapes.get(self.key)

                if (
                    cmn.smartresize_enabled
                    and target_shape is not None
                    and t.shape != target_shape
                ):
                    # ❗ Never resize here — initialize_task owns that
                    raise RuntimeError(
                        f"[CopyPrimary] Shape mismatch for '{self.key}': "
                        f"{tuple(t.shape)} vs target {tuple(target_shape)}"
                    )

                if self.stats:
                    self.stats.copied_primary += 1

                # NOTE:
                # Do NOT move to device or dtype here.
                # initialize_task is the only normalization point.
                return t

            except Exception as e:
                print(
                    f"[CopyPrimary] FAILED reading '{self.key}' "
                    f"from {model_name}: {e}"
                )

        # --------------------------------------------------
        # 2. Explicit zero-fill (Kitchen-Sink only)
        # --------------------------------------------------
        target_shape = cmn.cross_arch_target_shapes.get(self.key)
        if target_shape and self.keep_zero_fill:
            t = torch.zeros(
                target_shape,
            )
            if self.stats:
                self.stats.zero_filled += 1
            print(f"[KitchenSink] {self.key} ← ZERO-FILLED")
            return t

        # --------------------------------------------------
        # 3. True skip
        # --------------------------------------------------
        if self.stats:
            self.stats.skipped += 1
        print(f"[CopyPrimary:SKIPPED] {self.key}")
        return None



class LoadTensor(Operation):
    """
    Pure tensor loader.

    Guarantees:
    • No resize
    • No zero-fill
    • No Dual-Soul or sacred logic
    • No device / dtype normalization
    • No silent shape violations

    Shape enforcement, SmartResize, sacred handling, and normalization
    live exclusively in initialize_task().
    """

    def __init__(self, key, checkpoint_name):
        super().__init__(key)
        self.checkpoint_name = checkpoint_name or ""
        self.source_checkpoint = checkpoint_name  # attribution only

    def __hash__(self):
        return hash((
            self.key,
            self.checkpoint_name,
        ))

    def __eq__(self, other):
        return (
            isinstance(other, LoadTensor)
            and self.key == other.key
            and self.checkpoint_name == other.checkpoint_name
        )

    @multi_cache_operation
    def oper(self) -> torch.Tensor:
        if not cmn.loaded_checkpoints:
            raise RuntimeError("Checkpoints not loaded")

        file = None
        used_path = None

        # --------------------------------------------------
        # Path normalization (Windows-safe)
        # --------------------------------------------------
        def norm(p):
            try:
                return os.path.normcase(os.path.realpath(os.path.abspath(p)))
            except Exception:
                return None

        req_path = norm(self.checkpoint_name) if self.checkpoint_name else None

        # --------------------------------------------------
        # 1. Explicit checkpoint request → strict match
        # --------------------------------------------------
        if req_path:
            for path, f in cmn.loaded_checkpoints.items():
                if norm(path) == req_path:
                    file = f
                    used_path = path
                    break

            if file is None:
                raise RuntimeError(
                    f"Requested checkpoint not loaded: {self.checkpoint_name}"
                )

        # --------------------------------------------------
        # 2. Implicit request → controlled fallback
        # --------------------------------------------------
        else:
            if cmn.primary and cmn.primary in cmn.loaded_checkpoints:
                used_path = cmn.primary
                file = cmn.loaded_checkpoints[used_path]
            elif cmn.checkpoints_global:
                used_path = cmn.checkpoints_global[0]
                file = cmn.loaded_checkpoints.get(used_path)
            elif cmn.loaded_checkpoints:
                used_path, file = next(iter(cmn.loaded_checkpoints.items()))

        if file is None:
            raise RuntimeError(
                f"No checkpoint available for tensor '{self.key}'"
            )

        # --------------------------------------------------
        # 3. Load tensor (policy-free)
        # --------------------------------------------------
        if not cmn.has_tensor(file, self.key):
            raise RuntimeError(
                f"Key '{self.key}' missing from checkpoint "
                f"{os.path.basename(used_path) if used_path else 'unknown'}"
            )

        t = cmn.get_tensor(file, self.key)


        # --------------------------------------------------
        # SHAPE AUTHORITY GUARD
        # --------------------------------------------------
        target_shape = cmn.cross_arch_target_shapes.get(self.key)

        if (
            cmn.smartresize_enabled
            and target_shape is not None
            and t.shape != target_shape
        ):
            # ❗ Never resize here — initialize_task owns that
            raise RuntimeError(
                f"[LoadTensor] Shape mismatch for '{self.key}': "
                f"{tuple(t.shape)} vs target {tuple(target_shape)}"
            )

        # NOTE:
        # Do NOT move to device or dtype here.
        # initialize_task is the single normalization point.
        return t



# === BASIC OPERATORS (fixed indentation) ===
class Add(Operation):
    """
    Add:
      • Element-wise addition
      • Energy-increasing operation
      • Experimental feature amplification

    Semantic contract:
      • NEVER touches temporal control, noise, CLIP, VAE, attention, normalization
      • Preserves base semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Attention & routing
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",

        # Normalization / scaling
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # Semantic guard
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect compatible contributors (excluding base)
        valid = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
                and torch.any(t != 0)
            )
        ]

        if not valid:
            return base

        # Additive interaction (explicit amplification)
        out = base.clone()
        for t in valid:
            out = out + t

        # Numerical safety (no shape or semantic distortion)
        out = torch.nan_to_num(out, nan=0.0)

        return out.to(base.dtype)


class Sub(Operation):
    """
    Sub:
      • Element-wise subtraction
      • Directional and destructive
      • Experimental feature suppression

    Semantic contract:
      • NEVER touches temporal control, noise, CLIP, VAE, attention, normalization
      • Preserves base semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Attention & routing
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",

        # Normalization / scaling
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # Semantic guard
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect compatible contributors (excluding base)
        valid = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
                and torch.any(t != 0)
            )
        ]

        if not valid:
            return base

        # Destructive subtraction
        out = base.clone()
        for t in valid:
            out = out - t

        # Numerical safety (no semantic distortion)
        out = torch.nan_to_num(out, nan=0.0)

        return out.to(base.dtype)


class Multiply(Operation):
    """
    Multiply:
      • Element-wise multiplicative interaction
      • Extremely aggressive
      • Experimental feature modulation only

    Semantic contract:
      • NEVER touches temporal control, noise, CLIP, VAE, attention, normalization
      • Preserves base semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Attention & routing
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",

        # Normalization / scaling
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # Semantic guard
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect compatible contributors (excluding base)
        valid = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
                and torch.any(t != 0)
            )
        ]

        if not valid:
            return base

        # Multiplicative interaction
        out = base.clone()
        for t in valid:
            out = out * t

        # Numerical safety: kill NaNs / infs without semantic distortion
        out = torch.nan_to_num(out, nan=0.0)

        return out.to(base.dtype)

class Extract(Operation):
    """
    Extract:
      • Computes relative feature interaction between (a - base) and (b - base)
      • Uses cosine agreement to gate interpolation
      • Analysis / feature-isolation operator
      • NEVER overwrites base semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Attention & routing
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # -------------------------------------------------
        # Semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if (
            not isinstance(base, torch.Tensor)
            or not base.is_floating_point()
            or base.ndim < 2        # 🚨 forbid scalars & 1D
            or base.numel() == 0
        ):
            return base

        a = tensors[1] if len(tensors) > 1 else base
        b = tensors[2] if len(tensors) > 2 else base

        if not (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == base.shape == b.shape
            and a.is_floating_point()
            and b.is_floating_point()
        ):
            return base

        # -------------------------------------------------
        # Relative deltas
        # -------------------------------------------------
        base_f = base.float()
        a_f = a.float() - base_f
        b_f = b.float() - base_f

        if not (torch.any(a_f) and torch.any(b_f)):
            return base

        # -------------------------------------------------
        # Cosine agreement (feature-wise, safe)
        # -------------------------------------------------
        c = torch.cosine_similarity(a_f, b_f, dim=-1).clamp(-1.0, 1.0)

        # Ensure non-scalar & broadcast-safe
        if c.ndim == 0:
            return base

        # Shape: [*, 1] for safe broadcast
        while c.ndim < base.ndim:
            c = c.unsqueeze(-1)

        d = ((c + 1.0) * 0.5).pow(self.gamma)

        # -------------------------------------------------
        # Interpolate deltas, re-anchor to base
        # -------------------------------------------------
        delta = torch.lerp(a_f, b_f, self.alpha)
        gate = torch.lerp(d, 1.0 - d, self.beta)

        out = base_f + delta * gate

        return out.to(base.dtype)



class Similarities(Extract):
    """
    Similarities:
      • Delegates to Extract to compute similarity features
      • Non-destructive, analysis-only operator
      • NEVER fabricates tensors
    """

    def __init__(self, key, alpha, beta, gamma, a, b):
        super().__init__(key, alpha, beta, gamma, a, b)

    @multi_cache_operation
    def oper(self, *tensors):
        # -------------------------------------------------
        # Collect valid floating-point tensors
        # -------------------------------------------------
        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.ndim > 0          # 🚨 forbid scalars
                and t.numel() > 0
            )
        ]

        if len(valid) < 2:
            return valid[0] if valid else None

        a, b = valid[0], valid[1]

        # -------------------------------------------------
        # Shape safety
        # -------------------------------------------------
        if a.shape != b.shape:
            return a  # preserve base semantics

        # -------------------------------------------------
        # Delegate to Extract (analysis-only)
        # -------------------------------------------------
        out = super().oper(a, b)

        # Extract should already be safe, but defend anyway
        if not isinstance(out, torch.Tensor):
            return None

        if out.ndim == 0:
            return None

        return out.to(
            dtype=cmn.get_dtype(),
            device=cmn.get_device()
        )


class Clamp(Operation):
    """
    Clamp:
      • Hard value limiter
      • Conditioning / stabilization operator
      • NEVER safe for control, embeddings, or latents
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "timestep",
        "sigma",
        "noise",
        "vae",
        "encoder",
        "decoder",
        "text_model",
        "cond_stage_model",
    )

    def __init__(self, key, min_val=-1.0, max_val=1.0):
        super().__init__(key)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        # Defensive sanity
        if self.min_val > self.max_val:
            self.min_val, self.max_val = self.max_val, self.min_val

    @multi_cache_operation
    def oper(self, *tensors):
        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0] if tensors else None

        if not tensors:
            return None

        t = tensors[0]

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(t, torch.Tensor):
            return t

        if not t.is_floating_point():
            return t

        # 🚨 Scalar tensors forbidden
        if t.ndim == 0:
            return None

        if t.numel() == 0:
            return t

        # Optional: skip pointless work
        if not torch.any(t != 0):
            return t

        out = t.clamp(self.min_val, self.max_val)

        return out.to(
            dtype=cmn.get_dtype(),
            device=cmn.get_device()
        )

    
class Mean(Operation):
    """
    Mean:
      • Uniform linear averaging
      • Safe primitive for feature blending
      • NOT safe for control or routing layers
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "timestep",
        "sigma",
        "noise",
        "attn",
        "attention",
        "skip_connection",
    )

    @multi_cache_operation
    def oper(self, *tensors):
        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0] if tensors else None

        # -------------------------------------------------
        # Collect valid contributors
        # -------------------------------------------------
        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.ndim > 0
                and t.numel() > 0
            )
        ]

        if not valid:
            return None

        base = valid[0]

        # Shape safety (no broadcasting)
        for t in valid[1:]:
            if t.shape != base.shape:
                return base

        # Avoid meaningless averages
        if not torch.any(base != 0):
            return base

        out = sum(valid) / float(len(valid))

        return out.to(
            dtype=cmn.get_dtype(),
            device=cmn.get_device()
        )

    
class Normalize(Operation):
    """
    Normalize:
      • L2 normalizes entire tensor
      • EXPERIMENTAL / DESTRUCTIVE
      • Should never run on production merges
      • NOT cacheable
    """

    FORBIDDEN_PATTERNS = (
        "",  # default deny: forbid everything unless overridden
    )

    def __init__(self, key, *sources):
        super().__init__(key, *sources)
        self.disable_cache = True  # 🚨 mandatory

    def oper(self, *tensors):
        if not tensors:
            return None

        t = tensors[0]

        # -------------------------------------------------
        # Key-level default deny
        # -------------------------------------------------
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return t

        # -------------------------------------------------
        # Tensor safety
        # -------------------------------------------------
        if not isinstance(t, torch.Tensor):
            return t

        if not t.is_floating_point():
            return t

        # 🚨 Scalar tensors forbidden
        if t.ndim == 0:
            return None

        if t.numel() == 0:
            return t

        norm = t.norm()
        if norm <= 1e-8:
            return t

        return (t / norm).to(
            dtype=cmn.get_dtype(),
            device=cmn.get_device()
        )



class ReBasin(Operation):
    """
    ReBasin (Distributional Weight Alignment):

      • Aligns weight distributions via rank-wise interpolation
      • Ignores positional semantics
      • Preserves global statistics, not structure

    Semantic contract:
      • Experimental, highly aggressive
      • Must NEVER touch control, routing, or embeddings
      • Preserves primary semantics on refusal
      • NOT cacheable
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Attention & routing
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",

        # Normalization
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = float(alpha)
        self.disable_cache = True  # 🚨 required

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # -----------------------------
        # Key-level semantic guard
        # -----------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return a

        # -----------------------------
        # Tensor-level safety
        # -----------------------------
        if not isinstance(a, torch.Tensor):
            return a

        if not a.is_floating_point():
            return a

        # 🚨 Scalar tensors forbidden
        if a.ndim == 0:
            return None

        if a.shape != b.shape:
            print(f"[ReBasin] Shape mismatch skipped: {self.key}")
            return a

        if a.numel() < 2:
            return a

        if not torch.any(a != 0):
            return a

        # -----------------------------
        # Alpha safety
        # -----------------------------
        alpha = max(0.0, min(1.0, self.alpha))
        if alpha <= 0.0:
            return a

        # -----------------------------
        # Distributional rebasing
        # -----------------------------
        try:
            a_flat = a.flatten()
            b_flat = b.flatten()

            a_sorted, a_idx = torch.sort(a_flat)
            b_sorted = torch.sort(b_flat).values

            merged_sorted = torch.lerp(a_sorted, b_sorted, alpha)

            # Restore original ordering
            rebased = torch.empty_like(merged_sorted)
            rebased[a_idx] = merged_sorted

            return rebased.view_as(a).to(a.dtype)

        except Exception:
            # Absolute safety: preserve base on any failure
            return a




class DeMe(Operation):
    """
    DeMe (Decoupled Merge):

      • Uses per-feature variance to select dominant contributor
      • Blends result back toward primary tensor
      • Semantic, content-aware merge

    Semantic contract:
      • Useful for feature refinement
      • Dangerous for control signals and embeddings
      • Preserves base semantics on refusal
      • NOT cacheable
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Structural routing
        "skip_connection",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = float(alpha)
        self.disable_cache = True  # 🚨 required

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # -----------------------------
        # Key-level semantic guard
        # -----------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return a

        # -----------------------------
        # Tensor-level safety
        # -----------------------------
        if not isinstance(a, torch.Tensor):
            return a

        if not a.is_floating_point():
            return a

        # 🚨 Scalar tensors forbidden
        if a.ndim == 0:
            return None

        if a.shape != b.shape:
            print(f"[DeMe] Shape mismatch skipped: {self.key}")
            return a

        if a.numel() == 0 or not torch.any(a != 0):
            return a

        # Must have a feature axis
        if a.ndim < 2:
            return a

        # -----------------------------
        # Alpha safety
        # -----------------------------
        alpha = max(0.0, min(1.0, self.alpha))
        if alpha <= 0.0:
            return a

        # -----------------------------
        # Variance-based decoupling
        # -----------------------------
        try:
            var_a = torch.var(a, dim=-1, keepdim=True)
            var_b = torch.var(b, dim=-1, keepdim=True)
        except Exception:
            return a

        decoupled = torch.where(var_a > var_b, a, b)

        # -----------------------------
        # Blend back toward primary
        # -----------------------------
        return torch.lerp(a, decoupled, alpha).to(a.dtype)




class BlockWeighted(Operation):
    """
    Block-aware linear interpolation.

    Semantic contract:
      • Applies ONLY to block-indexed UNet layers
      • Deterministic, depth-aware blending
      • Identity on refusal
      • Never fabricates tensors
      • NOT cacheable
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "sigma",
        "noise",

        # Structural routing
        "skip_connection",

        # Latent encode/decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
    )

    # Allowed UNet block namespaces
    BLOCK_PATTERNS = (
        "input_blocks",
        "output_blocks",
        "down_blocks",
        "up_blocks",
        "middle_block",
        "mid_block",
    )

    def __init__(self, key, alphas, a, b):
        super().__init__(key, a, b)

        self.alphas = list(alphas) if alphas else []
        self.disable_cache = True  # 🚨 required

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # -----------------------------
        # Tensor-level safety
        # -----------------------------
        if not isinstance(a, torch.Tensor):
            return a

        if not a.is_floating_point():
            return a

        # 🚨 Scalar tensors forbidden
        if a.ndim == 0:
            return None

        if a.shape != b.shape:
            print(f"[BlockWeighted] Shape mismatch skipped: {self.key}")
            return a

        # -----------------------------
        # Key-level semantic guard
        # -----------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return a

        key_lower = self.key.lower()

        # Must be a known UNet block
        if not any(p in key_lower for p in self.BLOCK_PATTERNS):
            return a

        # -----------------------------
        # Block index extraction (strict)
        # -----------------------------
        match = re.search(
            r'(input_blocks|output_blocks|down_blocks|up_blocks)[._](\d+)',
            key_lower
        )

        if not match:
            # middle / mid blocks get their own fixed index
            if "middle_block" in key_lower or "mid_block" in key_lower:
                idx = len(self.alphas) // 2 if self.alphas else 0
            else:
                return a
        else:
            idx = int(match.group(2))

        if not self.alphas:
            return a

        alpha = self.alphas[min(idx, len(self.alphas) - 1)]
        alpha = max(0.0, min(1.0, float(alpha)))

        # -----------------------------
        # Linear blend
        # -----------------------------
        return torch.lerp(a, b, alpha).to(a.dtype)




class ToMe(Operation):
    """
    Token Merging (ToMe-style) conditioner.

    Semantic contract:
      • Operates ONLY on token embeddings [N, D]
      • Reduces token count via similarity-based merging
      • Never fabricates tensors
      • Never applies to control or weight keys
      • Preserves base semantics on refusal
      • NOT cacheable
    """

    # Temporal / control only — rely on shape for safety
    FORBIDDEN_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "sigma",
        "noise",
    )

    def __init__(self, key, ratio, tensor):
        super().__init__(key, tensor)

        self.ratio = float(ratio)
        self.disable_cache = True  # 🚨 required

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        x = tensors[0]

        # -----------------------------
        # Basic tensor safety
        # -----------------------------
        if not isinstance(x, torch.Tensor):
            return x

        if not x.is_floating_point():
            return x

        # 🚨 Scalar tensors forbidden
        if x.ndim == 0:
            return None

        # -----------------------------
        # Key-level semantic guard
        # -----------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return x

        # -----------------------------
        # Token shape enforcement
        # -----------------------------
        # Must be [N, D] with N >= 2
        if x.ndim != 2 or x.size(0) < 2:
            return x

        # -----------------------------
        # Ratio validation
        # -----------------------------
        r = max(0.0, min(1.0, self.ratio))
        if r <= 0.0:
            return x

        N, D = x.shape
        k = max(2, int(N * r))

        if k >= N:
            return x

        # -----------------------------
        # ToMe-style global merge
        # -----------------------------
        with torch.no_grad():
            # Normalize for cosine similarity
            normed = F.normalize(x, dim=-1)

            # Mean token as anchor (stable + cheap)
            anchor = normed.mean(dim=0, keepdim=True)  # [1, D]

            # Similarity to anchor
            sim = (normed @ anchor.T).squeeze(1)  # [N]

            # Select top-k most representative tokens
            _, idx = torch.topk(sim, k=k, largest=True)

            merged = x[idx]

        return merged.to(x.dtype)


class AttentionMerge(Operation):
    """
    Attention-only linear merge.

    Semantic contract:
      • Applies ONLY to attention-related keys
      • Linear interpolation between two sources
      • Identity everywhere else
      • Never fabricates tensors
      • Never propagates scalars
      • NOT cached (input + alpha dependent)
    """
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)

        self.alpha = float(alpha)
        self.disable_cache = True  # 🚨 critical: do not cache

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # Type safety
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return a

        # 🚨 Scalar tensors must never propagate
        if a.ndim == 0 or b.ndim == 0:
            return None

        # Only act on attention-related layers
        key_lower = self.key.lower()
        if "attn" not in key_lower and "attention" not in key_lower:
            return a

        # Floating-point only
        if not a.is_floating_point():
            return a

        # Shape safety
        if a.shape != b.shape:
            print(f"[AttentionMerge] Shape mismatch skipped: {self.key}")
            return a

        # Clamp alpha defensively
        alpha = max(0.0, min(1.0, self.alpha))

        # Linear interpolation
        return torch.lerp(a, b, alpha).to(a.dtype)


class Smooth(Operation):
    """
    1D Gaussian smoothing conditioner.

    • Single-tensor only
    • Refuses scalars (never propagate 0-dim tensors)
    • Floating-point only
    • NOT cached (depends on sigma/kernel + input tensor values)
    """
    def __init__(self, key, tensor=None, kernel_size=5, sigma=1.0):
        super().__init__(key, tensor)

        self.disable_cache = True  # 🚨 critical: do NOT cache conditioners

        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        # Enforce sane odd kernel
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel_size = max(3, min(self.kernel_size, 31))

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        tensor = tensors[0]

        if not isinstance(tensor, torch.Tensor):
            return None

        # 🚨 scalar tensors must never propagate
        if tensor.ndim == 0:
            return None

        # Conditioner never modifies non-floating tensors
        if not tensor.is_floating_point():
            return tensor

        # Too small to smooth meaningfully
        if tensor.numel() < 10 or not torch.any(tensor != 0):
            return tensor

        device, dtype = tensor.device, tensor.dtype
        size = self.kernel_size
        sigma = self.sigma
        center = size // 2

        # Build kernel in float32 for numerical stability, then cast
        x = torch.arange(size, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * ((x - center) / (sigma + 1e-12)) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.to(dtype=dtype).view(1, 1, -1)

        orig_shape = tensor.shape

        # Flatten → smooth → restore
        xf = tensor.flatten().unsqueeze(0).unsqueeze(0)
        xf = F.pad(xf, (center, center), mode="replicate")
        smoothed = F.conv1d(xf, kernel).squeeze(0).squeeze(0)

        return smoothed.view(orig_shape).to(dtype)


class SmoothConv(Operation):
    """
    Smart hybrid smoothing (conditioner):

      • Conv2d weights (4D) → 2D Gaussian
      • Linear / attention / other (>=2D) → 1D Gaussian
      • Everything else → refusal / passthrough

    NOTE:
      • Conditioner, not a merge operator
      • Must NOT be cached
      • Must NOT propagate scalars
    """

    def __init__(self, key, sigma=1.0, kernel_size=None, tensor=None):
        super().__init__(key, tensor)

        self.disable_cache = True  # 🚨 critical

        self.sigma = float(sigma)
        self.kernel_size = kernel_size or (max(3, int(4 * self.sigma + 1)) | 1)

        # Safety clamp
        self.kernel_size = min(self.kernel_size, 31)

        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        tensor = tensors[0]

        # Must be tensor
        if not isinstance(tensor, torch.Tensor):
            return None

        # 🚨 Scalar tensors must never propagate
        if tensor.ndim == 0:
            return None

        # Conditioner never modifies non-floating tensors
        if not tensor.is_floating_point():
            return tensor

        if tensor.numel() == 0 or not torch.any(tensor != 0):
            return tensor

        # 4D Conv2d weights → 2D smoothing
        if tensor.ndim == 4 and tensor.shape[2] >= 3 and tensor.shape[3] >= 3:
            return self._smooth_2d(tensor)

        # 2D+ tensors → 1D smoothing
        if tensor.ndim >= 2 and tensor.numel() >= 10:
            return self._smooth_1d(tensor)

        return tensor

    def _smooth_2d(self, tensor):
        device, dtype = tensor.device, tensor.dtype
        size = int(self.kernel_size)
        pad = size // 2

        h, w = tensor.shape[2], tensor.shape[3]
        if pad <= 0 or pad >= h or pad >= w:
            return tensor

        x = torch.arange(size, device=device, dtype=dtype)
        center = size // 2
        kernel_1d = torch.exp(-0.5 * ((x - center) ** 2) / (self.sigma ** 2 + 1e-12))
        kernel_1d /= kernel_1d.sum()

        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, size, size)

        out_c, in_c, _, _ = tensor.shape
        x = tensor.permute(1, 0, 2, 3).reshape(-1, 1, h, w)
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        weight = kernel_2d.expand(x.shape[0], 1, size, size)
        smoothed = F.conv2d(x, weight, groups=x.shape[0])

        smoothed = smoothed.view(in_c, out_c, h, w).permute(1, 0, 2, 3)
        return smoothed.to(dtype)

    def _smooth_1d(self, tensor):
        # Delegate carefully; Smooth must also be non-cached & scalar-safe
        from scripts.untitled.operators import Smooth
        return Smooth(self.key, tensor).oper(tensor)


    
class COPY(Operation):
    """
    N-way COPY operator:

      • Selects ONE tensor to preserve verbatim
      • Authoritative semantic preservation
      • No numeric blending
      • Safe for Sacred keys and non-mergeables
      • Deterministic and cache-safe

    Intended uses:
      • Explicit user choice ("take model B here")
      • Sacred / semantic preservation
      • Final fallback tier
    """

    FORBIDDEN_PATTERNS = (
        "metadata",
        "state_dict",
        "__",
    )

    def __init__(
        self,
        key,
        *sources,
        prefer: int = 0,
        allow_non_floating: bool = True,
    ):
        super().__init__(key, *sources)
        self.prefer = int(prefer)
        self.allow_non_floating = bool(allow_non_floating)

    @multi_cache_operation
    def oper(self, *tensors):
        # -------------------------------------------------
        # Absolute safety: no tensors
        # -------------------------------------------------
        if not tensors:
            return None

        base = tensors[0]

        # -------------------------------------------------
        # Key-level safety (absolute refusal)
        # -------------------------------------------------
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # -------------------------------------------------
        # Preferred index resolution
        # -------------------------------------------------
        idx = self.prefer
        if idx < 0 or idx >= len(tensors):
            return base

        t = tensors[idx]

        # -------------------------------------------------
        # Tensor-level validation
        # -------------------------------------------------
        if not isinstance(t, torch.Tensor):
            return base

        # Scalar tensors must never propagate
        if t.ndim == 0:
            return base

        if not self.allow_non_floating and not t.is_floating_point():
            return base

        # -------------------------------------------------
        # Verbatim preservation
        # (NO clone, NO device, NO dtype)
        # -------------------------------------------------
        return t


class LERP(Operation):
    """
    N-way LERP (Linear Interpolation):

      • Linear weighted blending of contributors
      • Magnitude-preserving
      • Stable for many parameter types
      • Less expressive than SLERP/DARE/WISE

    Semantic contract:
      • Safe default operator
      • Excellent for smooth blending
      • Still dangerous for temporal control and noise math
      • Must NEVER touch timestep conditioning or noise-scale keys
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "conv_in",
        "input_blocks.0",
        "skip_connection",
    )

    def __init__(self, key, weights, *sources):
        super().__init__(key, *sources)

        if not weights:
            self.weights = []
            return

        w = [float(x) for x in weights]
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))

        # Normalize positive weights only
        total = sum(x for x in w if x > 0.0)
        if total <= 0.0:
            self.weights = []
        else:
            self.weights = [x / total if x > 0.0 else 0.0 for x in w]

    # -------------------------------------------------
    # Merge
    # -------------------------------------------------
    @multi_cache_operation
    def oper(self, *tensors):
        # ❗ Never invent tensors
        if not tensors:
            return None

        base = tensors[0]

        # Operator-level semantic guard
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # Must be floating-point tensor
        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        # -------------------------------------------------
        # Collect valid contributors
        # -------------------------------------------------
        valid = []
        wts = []

        for t, w in zip(tensors, self.weights):
            if w <= 0.0:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if not t.is_floating_point():
                continue
            if t.shape != base.shape:
                continue
            if t.numel() == 0:
                continue

            valid.append(t)
            wts.append(w)

        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        # -------------------------------------------------
        # Linear blend
        # -------------------------------------------------
        out = torch.zeros_like(base)
        for t, w in zip(valid, wts):
            out += t * w

        return out.to(base.dtype)


class LERPMEAN(Operation):
    """
    N-way LERP/MEAN hybrid:

      • LERP path: weighted linear blend (identity-preserving)
      • MEAN path: stabilizing average (drift-resistant)
      • Final: lerp(mean, lerp, mix)

    Design goal:
      • Maximum coverage fallback
      • Minimal semantic distortion
      • User can push it into danger by choice
    """

    # Optional “soft guardrails” (set empty tuple if you want NO operator gating)
    FORBIDDEN_PATTERNS = (
        # You can comment these out if you truly want "let it break"
        # "time_embed", "timestep", "sigma", "noise",
    )

    def __init__(
        self,
        key,
        weights,
        *sources,
        mix=1.0,            # 1.0 = pure LERP, 0.0 = pure MEAN
        temperature=1.0,    # 1.0 = raw weights, <1 sharpens, >1 flattens
    ):
        super().__init__(key, *sources)

        self.mix = float(mix)
        self.temperature = float(temperature)

        # -------------------------------------------------
        # Normalize weights safely
        # -------------------------------------------------
        if not weights:
            raise ValueError("LERPMEAN requires weights (at least one)")

        w = [float(x) for x in weights]

        # Pad weights to number of sources
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))

        # Apply temperature reweighting
        if self.temperature != 1.0:
            w = [max(0.0, x) for x in w]
            eps = 1e-12
            t = max(self.temperature, eps)
            w = [(x + eps) ** (1.0 / t) for x in w]

        total = sum(w)
        if total <= 0.0:
            # Primary-dominant fallback
            w = [1.0] + [0.0] * (len(sources) - 1)
            total = 1.0

        self.weights = [x / total for x in w]

    # -------------------------------------------------
    # Merge
    # -------------------------------------------------
    @multi_cache_operation
    def oper(self, *tensors):
        # ❗ NEVER invent tensors — refuse cleanly
        if not tensors:
            return None

        base = tensors[0]

        # Optional operator-level soft gate
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # Must be a floating-point tensor to merge meaningfully
        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        # -------------------------------------------------
        # Collect valid contributors
        # -------------------------------------------------
        valid = []
        wts = []

        for t, w in zip(tensors, self.weights):
            if w <= 0.0:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if not t.is_floating_point():
                continue
            if t.shape != base.shape:
                continue
            if t.numel() == 0:
                continue

            valid.append(t)
            wts.append(w)

        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        # -------------------------------------------------
        # MEAN path (stability)
        # -------------------------------------------------
        mean_out = torch.mean(torch.stack(valid, dim=0), dim=0)

        # -------------------------------------------------
        # LERP path (identity-preserving)
        # -------------------------------------------------
        lerp_out = torch.zeros_like(base)
        for t, w in zip(valid, wts):
            lerp_out += t * w

        # -------------------------------------------------
        # Final blend
        # -------------------------------------------------
        mix = float(max(0.0, min(1.0, self.mix)))
        out = torch.lerp(mean_out, lerp_out, mix)

        return out.to(base.dtype)

    
class AdaptiveLERP(Operation):
    """
    DAREWISE-style AdaptiveLERP (channel-aware, depth-aware):

      • Computes LERP and MEAN candidates
      • Builds per-channel aggression from agreement + variance
      • Applies a gentle UNet depth bias (architecture-agnostic)
      • Final:
          out = mean + (lerp - mean) * mix_channel
    """

    FORBIDDEN_PATTERNS = ()

    def __init__(
        self,
        key,
        weights,
        *sources,
        base_mix=1.0,
        confidence=0.5,
        temperature=1.0,
        mix_min=0.0,
        mix_max=1.0,
        agree_power=1.0,
        var_power=1.0,
        eps=1e-8,
        disable_cache: bool = True,   # ✅ safest default
    ):
        super().__init__(key, *sources)
        self.disable_cache = bool(disable_cache)

        self.base_mix = float(max(0.0, min(1.0, base_mix)))
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.temperature = float(temperature)
        self.mix_min = float(mix_min)
        self.mix_max = float(mix_max)

        self.agree_weight = self.confidence
        self.var_weight = 1.0 - self.confidence

        self.agree_power = float(agree_power)
        self.var_power = float(var_power)
        self.eps = float(eps)

        if weights is None:
            raise ValueError("AdaptiveLERP requires weights")

        w = [float(x) for x in weights]
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))

        # Temperature shaping
        if self.temperature != 1.0:
            w = [max(0.0, x) for x in w]
            t = max(self.temperature, 1e-12)
            w = [(x + 1e-12) ** (1.0 / t) for x in w]

        total = sum(w)
        if total <= 0.0:
            w = [1.0] + [0.0] * (len(sources) - 1)
            total = 1.0

        self.weights = [x / total for x in w]

    # -------------------------------------------------
    # UNet depth bias (architecture-agnostic)
    # -------------------------------------------------
    def _depth_bias_from_key(self, key: str):
        import re
        m = re.search(r'\.(\d+)\.', key)
        if not m:
            if "middle" in key:
                return 1.05
            return 1.0
        idx = int(m.group(1))
        bias = 0.85 + 0.1 * (idx ** 0.5)
        return float(max(0.7, min(1.15, bias)))

    @multi_cache_operation
    def oper(self, *tensors):
        # ✅ Refuse instead of returning scalar tensors
        if not tensors:
            return None

        base = tensors[0]
        base = self.validate_base(base)
        if base is None:
            return None

        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        # Filter valid contributors by shape + weight
        valid, wts = [], []
        for t, w in zip(tensors, self.weights):
            if (
                w > 0.0
                and isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            ):
                valid.append(t)
                wts.append(float(w))

        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        # MEAN candidate
        mean_out = torch.mean(torch.stack(valid, dim=0), dim=0)

        # Weighted LERP candidate
        lerp_out = torch.zeros_like(mean_out)
        for t, w in zip(valid, wts):
            lerp_out = lerp_out + (t * w)

        if self.base_mix <= 0.0:
            return mean_out.to(base.dtype)

        # From here on, base is guaranteed non-scalar by validate_base()

        C = base.shape[0]
        reduce_dims = tuple(range(1, base.ndim)) if base.ndim > 1 else None

        mean_base_f = mean_out.float()
        deltas = [(t.float() - mean_base_f) for t in valid]

        # Agreement: cosine similarity vs ref delta (channel-wise)
        ref = deltas[0]
        ref_flat = ref.flatten(start_dim=1) if base.ndim > 1 else ref.unsqueeze(1)
        ref_norm = ref_flat.norm(dim=-1).clamp_min(self.eps)

        sims = []
        for d in deltas[1:]:
            df = d.flatten(start_dim=1) if base.ndim > 1 else d.unsqueeze(1)
            dn = df.norm(dim=-1).clamp_min(self.eps)
            cos = ((df * ref_flat).sum(dim=-1) / (dn * ref_norm)).clamp(-1.0, 1.0)
            sims.append(cos)

        if sims:
            agree = torch.mean(torch.stack(sims, dim=0), dim=0)
        else:
            agree = torch.ones(C, device=base.device)

        agree = ((agree + 1.0) * 0.5).clamp(0.0, 1.0)
        agree = agree ** self.agree_power

        # Variance safety: higher variance => lower aggression
        stacked = torch.stack(deltas, dim=0)  # [M, ...]
        var = stacked.var(dim=0)
        if reduce_dims:
            var = var.mean(dim=reduce_dims)   # -> [C]

        var_safe = (1.0 / (1.0 + var)).clamp_min(0.0) ** self.var_power

        A = (self.agree_weight * agree + self.var_weight * var_safe).clamp(0.0, 1.0)

        depth_bias = self._depth_bias_from_key(self.key)

        mix_c = (self.base_mix * A * depth_bias).clamp(self.mix_min, self.mix_max)
        mix_view = mix_c.view(C, *([1] * (base.ndim - 1)))

        out = mean_out + (lerp_out - mean_out) * mix_view
        return out.to(base.dtype)
    
class TIES(Operation):
    """
    TIES (Top-Influence Exclusive Selection):

      • Selects ONE globally strongest delta
      • Applies sparse top-k masking
      • Resolves sign conflicts
      • Preserves dominant structure

    Semantic contract:
      • Excellent for structural alignment and decisive feature adoption
      • Extremely aggressive (single-winner semantics)
      • Dangerous for control signals, normalization, and embeddings
      • Must NEVER touch temporal control, noise scale, or semantic glue
    """

    FORBIDDEN_PATTERNS = (
        # Timestep / temporal conditioning
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",

        # Noise / sigma control
        "sigma",
        "noise",

        # Early signal injection
        "conv_in.",
        "input_blocks.0.",

        # Residual routing
        "skip_connection",

        # Latent encode / decode stability
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",

        # Semantic embeddings
        "text_model.",
        "cond_stage_model.",
        "conditioner.",
        "token_embedding",
        "position_embedding",

        # Normalization / scaling layers
        "layer_norm",
        "scale_shift",
        "affine",
        "ln_",
        "norm",
    )

    def __init__(self, key, *sources, density, seed=42):
        super().__init__(key, *sources)
        self.density = float(density)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        # -------------------------------------------------
        # Absolute safety: need a base
        # -------------------------------------------------
        if not tensors:
            return None

        base = tensors[0]
        if base is None:
            return None

        if not isinstance(base, torch.Tensor):
            return base

        # Scalar tensors are forbidden in your pipeline → refuse safely
        if base.ndim == 0 or base.numel() == 0:
            return base

        # -------------------------------------------------
        # Operator-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # Floating-point only
        if not base.is_floating_point():
            return base

        # Density edge cases
        if self.density <= 0.0:
            return base

        # -------------------------------------------------
        # Collect valid same-shape floating contributors (including base)
        # -------------------------------------------------
        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if len(valid) <= 1:
            return base

        others = valid[1:]
        if not others:
            return base

        # Deterministic behavior per key
        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        deltas = [t - base for t in others]

        # If all deltas are effectively zero, refuse
        # (cheap aggregate check)
        delta_norms = torch.stack([d.norm(p=2) for d in deltas])
        if delta_norms.numel() == 0 or float(delta_norms.max().item()) <= 1e-12:
            return base

        # -------------------------------------------------
        # Select ONE winning delta (global)
        # -------------------------------------------------
        winner_idx = int(torch.argmax(delta_norms).item())
        winning_delta = deltas[winner_idx]

        if not torch.any(winning_delta):
            return base

        abs_delta = winning_delta.abs()

        # Top-k mask
        k = max(1, int(self.density * abs_delta.numel()))
        k = min(k, abs_delta.numel())

        threshold = torch.topk(abs_delta.flatten(), k).values[-1]
        mask = abs_delta >= threshold

        if not torch.any(mask):
            return base

        # Keep native sign, apply sparsity
        sparse_delta = winning_delta * mask.to(winning_delta.dtype)

        # -------------------------------------------------
        # Norm preservation (bounded)
        # -------------------------------------------------
        sd_norm = sparse_delta.norm(p=2)
        wd_norm = winning_delta.norm(p=2)

        if sd_norm > 1e-8 and wd_norm > 0.0:
            scale = (wd_norm / (sd_norm + 1e-8)).clamp(max=10.0)
            sparse_delta = sparse_delta * scale

        out = base + sparse_delta
        return out.to(base.dtype)


class WISE(Operation):
    """
    N-way WISE (Winner-Index Sparse Energy):

      • Selects per-element strongest delta among contributors
      • Top-k mask with optional dropout
      • Random scaling on masked entries (intrinsic)
      • Energy preservation (bounded)
    """

    FORBIDDEN_PATTERNS = (
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",
        "sigma",
        "noise",
        "conv_in.",
        "input_blocks.0.",
        "skip_connection",
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",
        "text_model.",
        "cond_stage_model.",
        "conditioner.",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, density, dropout_p=0.3, seed=42, *sources):
        super().__init__(key, *sources)
        self.density = float(density)
        self.dropout_p = float(dropout_p)
        self.seed = int(seed)  # IMPORTANT: this overwrites Operation.seed, so cache uses it

        # Optional: if you want extra safety, you can disable caching:
        # self.disable_cache = True

    # IMPORTANT: include density/dropout in cache identity if caching stays ON
    def __hash__(self):
        return hash((
            self.key,
            self.density,
            self.dropout_p,
            self.seed,
            self.sources,
        ))

    def __eq__(self, other):
        return (
            isinstance(other, WISE)
            and self.key == other.key
            and self.density == other.density
            and self.dropout_p == other.dropout_p
            and self.seed == other.seed
            and self.sources == other.sources
        )

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0]

        base0 = tensors[0]
        base0 = self.validate_base(base0)
        if base0 is None:
            return None

        if not isinstance(base0, torch.Tensor) or not base0.is_floating_point():
            return base0

        density = max(0.0, min(1.0, float(self.density)))
        dropout_p = max(0.0, min(1.0, float(self.dropout_p)))

        if density <= 0.0:
            return base0

        out_dtype = base0.dtype
        device = base0.device

        # Collect valid contributors (same-shape, float, non-empty)
        valid = []
        for t in tensors:
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base0.shape
                and t.numel() > 0
            ):
                valid.append(t)

        if len(valid) <= 1:
            return base0

        # WISE uses contributors beyond base
        contributors = valid[1:]
        if not contributors:
            return base0

        # Work in float32 for stability
        base = base0.float()
        contrib_f = [t.float() for t in contributors]

        # Density >= 1.0: "winner takes all" by largest delta norm
        if density >= 1.0:
            deltas = [t - base for t in contrib_f]
            norms = torch.stack([d.norm(p=2) for d in deltas])
            winner = int(torch.argmax(norms).item())
            return contributors[winner].to(dtype=out_dtype)

        # Thread-safe deterministic RNG
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        # Compute deltas stack: [M, ...]
        deltas = torch.stack([t - base for t in contrib_f], dim=0)
        abs_deltas = deltas.abs()

        # Per-element winning contributor index
        max_mag, best = abs_deltas.max(dim=0)  # max_mag: [...], best: [...]

        # Top-k mask by magnitude
        k = max(1, int(density * max_mag.numel()))
        flat = max_mag.reshape(-1)
        threshold = torch.topk(flat, k, largest=True, sorted=False).values.min()
        mask = (max_mag >= threshold)

        # Optional dropout on mask (thread-safe)
        if dropout_p > 0.0:
            keep = 1.0 - dropout_p
            drop_mask = torch.rand(mask.shape, generator=gen, device=device) < keep
            mask = mask & drop_mask

        if not bool(mask.any().item()):
            return base0

        # Select winning delta per element using gather
        # gather expects index shape with leading dim=1 for dim=0 gather
        winning_delta = deltas.gather(0, best.unsqueeze(0)).squeeze(0)

        # Intrinsic random scaling on masked entries (bounded, deterministic)
        # scale in [0.5, 2.0]
        rand = torch.rand(mask.shape, generator=gen, device=device, dtype=torch.float32)
        scale = 0.5 + 1.5 * rand

        wise_delta = winning_delta * mask.to(winning_delta.dtype) * scale

        # Energy normalization (safety-capped)
        total_energy = deltas.norm(p=2, dim=tuple(range(1, deltas.ndim))).sum()  # scalar
        wd_norm = wise_delta.norm(p=2)

        if wd_norm.item() > 1e-8 and total_energy.item() > 0.0:
            wise_delta = wise_delta * (total_energy / (wd_norm + 1e-8)).clamp(max=10.0)

        out = base + wise_delta
        return out.to(dtype=out_dtype)



class DARE_Nway(Operation):
    """
    True N-way DARE (Delta-Aware Residual Energy merge):
      • Symmetric contributors
      • Per-source sparse deltas
      • Additive (non-competitive)
      • Direction-preserving
      • Energy-stable
    """

    FORBIDDEN_PATTERNS = (
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",
        "sigma",
        "noise",
        "conv_in.",
        "input_blocks.0.",
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",
        "text_model.",
        "cond_stage_model.",
        "conditioner.",
        "token_embedding",
        "position_embedding",
    )

    def __init__(
        self,
        key,
        density,
        dropout_p=0.0,
        seed=42,
        base_mode="mean",
        *sources,
    ):
        super().__init__(key, *sources)
        self.density = float(density)
        self.dropout_p = float(dropout_p)
        self.seed = int(seed)
        self.base_mode = str(base_mode)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        # Semantic guard
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0]

        base0 = tensors[0]
        base0 = self.validate_base(base0)
        if base0 is None:
            return None

        # DARE only applies to floating-point tensors
        if not isinstance(base0, torch.Tensor) or not base0.is_floating_point():
            return base0

        out_dtype = base0.dtype
        device = base0.device

        # Collect valid contributors (same-shape, float, non-empty)
        valid = []
        for t in tensors:
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base0.shape
                and t.numel() > 0
            ):
                valid.append(t)

        if len(valid) <= 1:
            return base0

        # Work in float32 for stability
        valid_f = [t.float() for t in valid]

        # Base selection
        if self.base_mode == "mean":
            base = torch.stack(valid_f, dim=0).mean(dim=0)
        elif self.base_mode == "first":
            base = valid_f[0]
        else:
            return base0  # refuse unknown modes safely

        # Thread-safe deterministic RNG
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        merged_delta = torch.zeros_like(base)
        total_energy = torch.zeros((), device=device, dtype=torch.float32)

        # Clamp density into sane range
        density = max(0.0, min(1.0, self.density))
        dropout_p = max(0.0, min(1.0, self.dropout_p))

        for t in valid_f:
            delta = t - base

            # Skip exact-zero deltas cheaply
            if (delta.abs().max().item() == 0.0):
                continue

            # Per-source sparsity
            k = max(1, int(density * delta.numel()))
            mags = delta.abs().reshape(-1)

            # topk on huge tensors is expensive, but correct.
            # k==numel is fine.
            threshold = torch.topk(mags, k, largest=True, sorted=False).values.min()
            mask = (delta.abs() >= threshold)

            # Optional dropout (uses local generator)
            if dropout_p > 0.0:
                keep = 1.0 - dropout_p
                drop_mask = torch.rand(mask.shape, generator=gen, device=device) < keep
                mask = mask & drop_mask

            sparse_delta = delta * mask.to(delta.dtype)

            merged_delta = merged_delta + sparse_delta
            total_energy = total_energy + delta.norm(p=2)

        # Energy normalization (safety-capped)
        md_norm = merged_delta.norm(p=2)
        if md_norm.item() > 1e-8 and total_energy.item() > 0.0:
            scale = (total_energy / (md_norm + 1e-8)).clamp(max=10.0)
            merged_delta = merged_delta * scale

        out = base + merged_delta
        return out.to(dtype=out_dtype)


class DAREWISE(Operation):
    """
    DARE+WISE Hybrid:

      • DARE provides sparse, additive, energy-stable structure
      • WISE provides competitive, high-contrast detail
      • Key-based or block-based gating decides which dominates
      • Optional soft blending between the two

    Semantic contract:
      • Composite operator: never invents tensors
      • Delegates semantic safety to child operators
      • Refuses cleanly by preserving base semantics
    """

    def __init__(
        self,
        key,
        dare_density,
        dare_dropout,
        wise_density,
        wise_dropout,
        mix=0.5,        # 0 = pure DARE, 1 = pure WISE
        seed=42,
        *sources
    ):
        super().__init__(key, *sources)
        self.dare_density = float(dare_density)
        self.dare_dropout = float(dare_dropout)
        self.wise_density = float(wise_density)
        self.wise_dropout = float(wise_dropout)
        self.mix = float(mix)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        base = tensors[0]

        # Composite operators never invent tensors
        if not tensors or not base.is_floating_point():
            return base

        # Collect valid same-shape floating tensors
        valid = [
            t for t in tensors
            if (
                t.numel() > 0
                and torch.any(t != 0)
                and t.is_floating_point()
                and t.shape == base.shape
            )
        ]
        if len(valid) <= 1:
            return base

        # -------------------------------------------------
        # Step 1: compute both candidate merges
        # (child operators enforce their own safety)
        # -------------------------------------------------
        dare = DARE_Nway(
            self.key,
            self.dare_density,
            self.dare_dropout,
            self.seed,
            *valid
        ).oper(*valid)

        wise = WISE(
            self.key,
            self.wise_density,
            self.wise_dropout,
            self.seed + 1337,
            *valid
        ).oper(*valid)

        # If either child refuses, preserve base
        if dare is base and wise is base:
            return base

        # -------------------------------------------------
        # Step 2: decide dominance
        # -------------------------------------------------
        key_lower = self.key.lower()
        attention_safe = any(
            s in key_lower
            for s in ("attn", "attention", "to_q", "to_k", "to_v", "proj")
        )

        if attention_safe:
            gate = 0.0  # hard-lock to DARE
        else:
            gate = self.mix

        # -------------------------------------------------
        # Step 3: blend (bounded, linear)
        # -------------------------------------------------
        if gate <= 0.0:
            out = dare
        elif gate >= 1.0:
            out = wise
        else:
            out = torch.lerp(dare, wise, gate)

        return out.to(base.dtype)

class AdaptiveDAREWISE(Operation):
    """
    Adaptive DARE+WISE:
      - Computes DARE and WISE candidates
      - Builds an internal aggression scalar A ∈ [0, 1]
      - A decides how much WISE is allowed
      - Attention-ish layers hard-lock to DARE

    Composite contract:
      • Never invents tensors
      • Refuses by preserving base semantics
      • Delegates semantic safety to child operators
    """

    ATTENTION_PATTERNS = (
        "attn", "attention",
        "to_q", "to_k", "to_v",
        "q_proj", "k_proj", "v_proj",
        "proj", "out_proj", "proj_out",
    )

    def __init__(
        self,
        key,
        dare_density,
        dare_dropout,
        wise_density,
        wise_dropout,
        aggression_bias=0.5,
        seed=42,
        *sources
    ):
        super().__init__(key, *sources)

        # 🚫 Safer: do not cache this composite op
        self.disable_cache = True

        self.dare_density = float(dare_density)
        self.dare_dropout = float(dare_dropout)
        self.wise_density = float(wise_density)
        self.wise_dropout = float(wise_dropout)
        self.bias = float(aggression_bias)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        # Must handle empty calls safely
        if not tensors:
            return None

        base = tensors[0]

        # Refusal semantics for invalid bases
        base = self.validate_base(base)
        if base is None:
            return None

        # Non-float passthrough (composite never invents)
        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        # Collect valid same-shape floating tensors
        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if len(valid) <= 1:
            return base

        # -------------------------
        # Step 1: compute mean + deltas in float32 (more stable)
        # -------------------------
        # (keeps memory sane; stack is unavoidable if you want real mean)
        stack = torch.stack([t.float() for t in valid], dim=0)
        mean_base = stack.mean(dim=0)
        deltas = stack - mean_base  # [M, ...]

        # -------------------------
        # Step 2: aggression signals
        # -------------------------
        eps = 1e-8

        # (a) Agreement proxy: average cosine similarity to delta[0]
        # Compute dot/norm via reductions (no giant flatten copies)
        d0 = deltas[0]
        d0_norm = torch.sqrt((d0 * d0).sum() + eps)

        if deltas.shape[0] >= 2:
            sims = []
            for i in range(1, deltas.shape[0]):
                di = deltas[i]
                di_norm = torch.sqrt((di * di).sum() + eps)
                num = (d0 * di).sum()
                cos = (num / (d0_norm * di_norm + eps)).clamp(-1.0, 1.0)
                sims.append(cos)
            agreement = torch.stack(sims).mean()
            A_similarity = float(((agreement + 1.0) * 0.5).clamp(0.0, 1.0))
        else:
            A_similarity = 0.0

        # (b) Variance safety: higher variance => lower aggression
        # Use a scalar variance proxy
        var = deltas.var(dim=0).mean()
        A_variance = float(torch.exp(-var).clamp(0.0, 1.0))

        # (c) Cheap depth heuristic (key-based)
        key_lower = self.key.lower()
        if "down_blocks" in key_lower or "input_blocks" in key_lower:
            depth_scale = 0.3
        elif "mid_block" in key_lower or "middle_block" in key_lower:
            depth_scale = 0.6
        elif "up_blocks" in key_lower or "output_blocks" in key_lower:
            depth_scale = 1.0
        else:
            depth_scale = 0.75

        # -------------------------
        # Step 3: attention override (hard lock to DARE)
        # -------------------------
        is_attention = any(p in key_lower for p in self.ATTENTION_PATTERNS)

        if is_attention:
            A = 0.0
        else:
            # Combine signals (bounded)
            A = (0.45 * A_similarity + 0.35 * A_variance)
            A *= depth_scale
            A *= max(0.0, self.bias)
            A = float(max(0.0, min(1.0, A)))

        # -------------------------
        # Step 4: compute candidates
        # -------------------------
        # IMPORTANT: pass original tensors (not float32 stack) so downstream ops keep dtype behavior
        dare = DARE_Nway(
            self.key,
            self.dare_density,
            self.dare_dropout,
            self.seed,
            *valid
        ).oper(*valid)

        wise = WISE(
            self.key,
            self.wise_density,
            self.wise_dropout,
            self.seed + 1337,
            *valid
        ).oper(*valid)

        # If both refuse, preserve base
        if dare is base and wise is base:
            return base

        # -------------------------
        # Step 5: adaptive blend
        # -------------------------
        if A <= 0.0:
            out = dare
        elif A >= 1.0:
            out = wise
        else:
            out = torch.lerp(dare, wise, A)

        return out.to(dtype=base.dtype)

class SLERP(Operation):
    """
    True N-way spherical linear interpolation on the hypersphere.

    Semantic contract:
      • Excellent for style / direction-bearing weights
      • Dangerous for timestep, noise, and residual routing
      • Must NEVER touch temporal control or noise-scale keys

    Therefore: SLERP enforces its own forbidden-key policy.
    """

    FORBIDDEN_PATTERNS = (
        # Timestep / temporal conditioning
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",

        # Noise / sigma control
        "sigma",
        "noise",

        # Early signal injection
        "conv_in.",
        "input_blocks.0.",

        # Residual routing
        "skip_connection",
    )

    def __init__(self, key, weights, *sources, eps=1e-8):
        super().__init__(key, *sources)

        w = list(weights) if weights is not None else []
        # Pad to match number of sources (same convention as your other ops)
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))
        self.weights = [float(x) for x in w]

        self.eps = float(eps)

        # Optional: normalize if user provided sum > 1 (keeps "base gravity")
        pos_sum = sum(x for x in self.weights[1:] if x > 0.0)
        if pos_sum > 1.0:
            scale = 1.0 / pos_sum
            for i in range(1, len(self.weights)):
                self.weights[i] = max(0.0, self.weights[i]) * scale

    @multi_cache_operation
    def oper(self, *tensors):
        # -------------------------------------------------
        # Absolute safety
        # -------------------------------------------------
        if not tensors:
            return None

        base = tensors[0]
        if base is None:
            return None

        # Preserve non-tensor semantics
        if not isinstance(base, torch.Tensor):
            return base

        # Scalar tensors are forbidden in your system; refuse
        if base.ndim == 0 or base.numel() == 0:
            return None

        # -------------------------------------------------
        # Operator-level semantic guard (authoritative)
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # Floating only
        if not base.is_floating_point():
            return base

        # -------------------------------------------------
        # Collect valid same-shape contributors WITH index-aligned weights
        # -------------------------------------------------
        base_shape = base.shape
        pairs = []  # (tensor, weight)
        for i, (t, w) in enumerate(zip(tensors, self.weights)):
            if i == 0:
                continue
            if w <= 0.0:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if not t.is_floating_point():
                continue
            if t.shape != base_shape:
                continue
            if t.numel() == 0:
                continue
            pairs.append((t, float(w)))

        # Nothing to do → preserve base
        if not pairs:
            return base

        # -------------------------------------------------
        # Proper SLERP on unit sphere, then rescale to base norm
        # -------------------------------------------------
        eps = self.eps
        base_f = base.float()
        base_flat = base_f.flatten()
        base_norm = base_flat.norm().clamp_min(eps)
        base_u = base_flat / base_norm  # unit direction

        def log_map_unit(x: torch.Tensor) -> torch.Tensor:
            x_f = x.float().flatten()
            x_norm = x_f.norm().clamp_min(eps)
            x_u = x_f / x_norm

            cos_theta = (x_u @ base_u).clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)

            sin_theta = torch.sin(theta).clamp_min(eps)

            # factor ~ 1 when theta ~ 0 (avoids item() + avoids blowups)
            factor = torch.where(theta < 1e-6, torch.ones_like(theta), theta / sin_theta)

            # Tangent vector at base_u
            v = (x_u - cos_theta * base_u) * factor
            return v

        # Weighted tangent accumulation
        v = torch.zeros_like(base_u)
        for t, w in pairs:
            v = v + (w * log_map_unit(t))

        v_norm = v.norm()

        # If tiny move, return base (no scalar creation)
        if v_norm <= 1e-6:
            return base

        # Exponential map back to sphere (unit)
        y_u = torch.cos(v_norm) * base_u + torch.sin(v_norm) * (v / v_norm.clamp_min(eps))

        # Preserve base magnitude (important for weight tensors)
        y = (y_u * base_norm).view_as(base_f)

        return y.to(base.dtype)


class TrainDiff(Operation):
    """
    TrainDiff (Training-Delta Aggregation):

      • Approximates training-induced updates in weight space
      • Selects top-K strongest deltas from contributors
      • Soft-weights selected deltas by magnitude
      • Optional drift suppression (channel-aware)
      • Optional strength scaling

    Semantic contract:
      • Directional, additive
      • Safe only for mid / late feature weights
      • Must NEVER touch control, routing, or embeddings
      • Preserves primary semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        # Temporal / noise control
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",

        # Attention & routing
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",

        # Normalization
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",

        # Latent encode / decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",

        # Text / conditioning
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(
        self,
        key,
        a,
        b,
        c,
        *extra_sources,
        top_k: int = 3,
        zero_center: bool = True,
        channelwise_zero_center: bool = True,
        strength: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(key, a, b, c, *extra_sources)
        self.top_k = int(top_k)
        self.zero_center = bool(zero_center)
        self.channelwise_zero_center = bool(channelwise_zero_center)
        self.strength = float(max(0.0, strength))
        self.eps = float(eps)

    @multi_cache_operation
    def oper(self, *tensors):
        # ---------------------------------------------
        # Absolute safety + “preserve base semantics”
        # ---------------------------------------------
        if not tensors:
            return None  # no base exists, let ladder handle
        base = tensors[0]
        if base is None:
            return None

        if not isinstance(base, torch.Tensor):
            return base

        # Scalar tensors are forbidden in your pipeline → refuse safely
        if base.ndim == 0 or base.numel() == 0:
            return base

        # ---------------------------------------------
        # Key-level semantic guard
        # ---------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # Floating-only
        if not base.is_floating_point():
            return base

        # ---------------------------------------------
        # Shape-safe contributors only
        # ---------------------------------------------
        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if not others:
            return base

        # ---------------------------------------------
        # Delta computation
        # ---------------------------------------------
        deltas = [t - base for t in others]

        # If all deltas are trivially zero, preserve base
        # (cheap early-out: check aggregate norm)
        norms = torch.stack([d.norm(p=2) for d in deltas])  # shape [M]
        if norms.numel() == 0:
            return base

        # ---------------------------------------------
        # Top-K selection
        # ---------------------------------------------
        k = max(1, min(self.top_k, int(norms.numel())))
        top_vals, top_idx = torch.topk(norms, k)

        # If top_vals sum is ~0, nothing meaningful to add
        denom = top_vals.sum().clamp_min(self.eps)

        # Soft weights (no CPU sync)
        weights = top_vals / denom  # [k]

        # Combine selected deltas
        combined_delta = torch.zeros_like(base)
        # Iterate over indices without .tolist() (still Python loop, but no forced sync)
        for j in range(k):
            idx = int(top_idx[j].item())  # tiny scalar read; acceptable here
            combined_delta = combined_delta + deltas[idx] * weights[j]

        # ---------------------------------------------
        # Optional drift suppression
        # ---------------------------------------------
        if self.zero_center:
            if self.channelwise_zero_center and combined_delta.ndim > 1:
                dims = tuple(range(1, combined_delta.ndim))
                combined_delta = combined_delta - combined_delta.mean(dim=dims, keepdim=True)
            else:
                combined_delta = combined_delta - combined_delta.mean()

        # ---------------------------------------------
        # Strength scaling
        # ---------------------------------------------
        if self.strength != 1.0:
            combined_delta = combined_delta * self.strength

        return (base + combined_delta).to(base.dtype)



class InterpolateDifference(Operation):
    """
    InterpolateDifference (Stochastic Difference Selector)
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, mode, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.mode = mode
        self.gamma = float(gamma)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # ── Semantic guard ──
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if (
            not isinstance(base, torch.Tensor)
            or not base.is_floating_point()
            or base.ndim < 2            # 🚨 critical
            or base.numel() == 0
        ):
            return base

        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
            )
        ]
        if not others:
            return base

        # Stabilize alpha
        alpha = max(self.alpha, 0.05)

        deltas = [torch.abs(t - base) for t in others]
        delta_stack = torch.stack(deltas, dim=0)
        delta_max = delta_stack.max(dim=0).values

        if not torch.any(delta_max):
            return base

        # Difference vs similarity signal (per-element safe)
        if self.mode == "difference":
            diff = (delta_stack / (delta_max + 1e-8)).max(dim=0).values
        else:
            diff = 1.0 - (delta_max / (delta_max + delta_max.mean() + 1e-8))

        diff = diff.clamp(0.0, 1.0).pow(1.0 / alpha)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Deterministic stochastic mask
        rng = torch.Generator(device=diff.device)
        rng.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        bitmask = torch.bernoulli(diff, generator=rng)
        mask = torch.lerp(bitmask, diff, self.gamma)

        if not torch.any(mask):
            return base

        # Winner selection
        _, best_idx = torch.max(delta_stack, dim=0)

        winning = others[0]
        for i in range(1, len(others)):
            winning = torch.where(best_idx == i, others[i], winning)

        return (base * (1.0 - mask) + winning * mask).to(base.dtype)



class AutoEnhancedInterpolateDifference(Operation):
    """
    AutoEnhancedInterpolateDifference (Adaptive Similarity Band Selector)
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # shaping strength (higher = softer)
        self.beta = float(beta)     # band width (0..1 recommended)
        self.gamma = float(gamma)   # 0=hard bernoulli, 1=soft prob
        self.seed = int(seed)

        # Optional: if you want absolute safety for fallbacks/debugging
        # self.disable_cache = True

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # ── Key-level semantic guard ──
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # ── Tensor-level safety ──
        if (
            not isinstance(base, torch.Tensor)
            or not base.is_floating_point()
            or base.ndim < 2           # 🚨 forbid scalars & 1D
            or base.numel() == 0
        ):
            return base

        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if not others:
            return base

        # ── Clamp knobs ──
        alpha = max(self.alpha, 0.05)
        beta = float(max(0.0, min(1.0, self.beta)))
        gamma = float(max(0.0, min(1.0, self.gamma)))
        eps = 1e-8

        # ─────────────────────────────────────────────
        # PASS 1: elementwise max delta (memory-safe)
        # ─────────────────────────────────────────────
        with torch.no_grad():
            max_overall = None
            for t in others:
                d = (t - base).abs()
                max_overall = d if max_overall is None else torch.maximum(max_overall, d)

        if max_overall is None or not torch.any(max_overall):
            return base

        denom = max_overall.clamp_min(eps)   # per-element denom

        # ─────────────────────────────────────────────
        # PASS 2: winner = least similar (min sim)
        # sim = 1 - d/denom  in [0,1]
        # ─────────────────────────────────────────────
        with torch.no_grad():
            best_sim = None
            best_idx = None

            for i, t in enumerate(others):
                d = (t - base).abs()
                sim = 1.0 - (d / denom)
                sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                if best_sim is None:
                    best_sim = sim
                    best_idx = torch.zeros_like(sim, dtype=torch.long)
                else:
                    take = sim < best_sim
                    best_idx = torch.where(take, torch.full_like(best_idx, i), best_idx)
                    best_sim = torch.where(take, sim, best_sim)

        # If something went sideways, refuse safely
        if best_sim is None or best_idx is None:
            return base

        # ─────────────────────────────────────────────
        # Adaptive similarity band (around mean of best_sim)
        # ─────────────────────────────────────────────
        mean_sim = best_sim.mean()
        lower = mean_sim * (1.0 - beta)
        upper = mean_sim * (1.0 + beta)
        band_mask = (best_sim > lower) & (best_sim < upper)

        # ─────────────────────────────────────────────
        # Shape probability: more different = higher prob
        # diffiness = 1 - sim
        # ─────────────────────────────────────────────
        diffiness = (1.0 - best_sim).clamp(0.0, 1.0)
        shaped = diffiness.pow(1.0 / alpha)
        shaped = torch.nan_to_num(shaped, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        shaped = shaped * band_mask.to(shaped.dtype)

        if not torch.any(shaped):
            return base

        # ─────────────────────────────────────────────
        # Deterministic stochastic gate
        # ─────────────────────────────────────────────
        rng = torch.Generator(device=shaped.device)
        rng.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        bern = torch.bernoulli(shaped, generator=rng)
        interp_mask = torch.lerp(bern, shaped, gamma).clamp(0.0, 1.0)

        if not torch.any(interp_mask):
            return base

        # ─────────────────────────────────────────────
        # Build winning tensor per element (streaming)
        # ─────────────────────────────────────────────
        winning = others[0]
        for i in range(1, len(others)):
            winning = torch.where(best_idx == i, others[i], winning)

        # ─────────────────────────────────────────────
        # Blend base ↔ winner
        # ─────────────────────────────────────────────
        result = base * (1.0 - interp_mask) + winning * interp_mask
        return result.to(base.dtype)


class SingularValueDeOperator(Operation):
    """
    Singular-Value Decomposition based delta reconstruction.

    Contract:
      • Operates ONLY on 2D floating tensors
      • Refuses oversized tensors
      • Avoids large intermediate stacks
      • Preserves base semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",
        "sigma",
        "noise",
        "skip_connection",
        "input_blocks.0.",
        "conv_in.",
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",
        "cond_stage_model.",
        "conditioner.",
        "text_model.",
    )

    def __init__(self, key, alpha, beta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # singular-value threshold multiplier
        self.beta = float(beta)     # top-k fraction to keep
        self.seed = int(seed)

        # SVD is expensive + large; safest to not cache by default
        self.disable_cache = True

    @multi_cache_operation
    def oper(self, *tensors):
        # ─────────────────────────────────────────
        # Basic existence + key guard
        # ─────────────────────────────────────────
        if not tensors:
            return None

        base = tensors[0]

        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # ─────────────────────────────────────────
        # Tensor-level safety
        # ─────────────────────────────────────────
        if not isinstance(base, torch.Tensor):
            return base

        # Forbid scalar + non-float + empty
        if (base.ndim == 0) or (base.numel() == 0) or (not base.is_floating_point()):
            return base

        # Only 2D matrices
        if base.ndim != 2:
            return base

        # Avoid pathological SVDs
        # (tune these limits to taste)
        if base.numel() > 8_000_000 or max(base.shape) > 4096:
            return base

        # Clamp knobs defensively
        alpha = float(max(0.0, self.alpha))
        beta = float(max(0.0, min(1.0, self.beta)))

        # Shape-safe contributors only (same shape, float, non-empty)
        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if not others:
            return base

        # Deterministic behavior (if any internal randomness is introduced later)
        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        try:
            with torch.no_grad():
                # ─────────────────────────────────────────
                # Accumulate delta sum without stacking
                # ─────────────────────────────────────────
                # Use float32 for stability, but keep device
                total_diff = torch.zeros_like(base, dtype=torch.float32)
                base_f = base.to(dtype=torch.float32)

                any_delta = False
                for t in others:
                    d = t.to(dtype=torch.float32) - base_f
                    if torch.any(d):
                        any_delta = True
                        total_diff += d

                if not any_delta or not torch.any(total_diff):
                    return base

                # ─────────────────────────────────────────
                # SVD
                # ─────────────────────────────────────────
                U, S, Vh = torch.linalg.svd(total_diff, full_matrices=False)

                if S.numel() == 0:
                    return base

                s_max = S.max()
                if s_max <= 1e-12:
                    return base

                # ─────────────────────────────────────────
                # Singular value filtering
                # ─────────────────────────────────────────
                threshold = alpha * s_max
                significant = S > threshold

                if beta < 1.0:
                    k = max(1, int(beta * S.numel()))
                    topk_mask = torch.zeros_like(significant, dtype=torch.bool)
                    topk_mask[:k] = True
                    significant = significant & topk_mask

                if not torch.any(significant):
                    return base

                S_filtered = S * significant.to(S.dtype)

                # Reconstruction: (U * S) @ Vh
                reconstructed = (U * S_filtered.unsqueeze(0)) @ Vh

                # Safety cleanup (rare but worth it)
                reconstructed = torch.nan_to_num(
                    reconstructed, nan=0.0, posinf=0.0, neginf=0.0
                )

                out = base_f + reconstructed

                # Return in original dtype (and leave device unchanged)
                return out.to(dtype=base.dtype)

        except Exception:
            return base



class TensorExchange(Operation):
    """
    TensorExchange (Deterministic Swap Operator):

      • Selects ONE alternative tensor with probability α
      • No numeric blending
      • Deterministic per-key behavior
      • Preserves base semantics on refusal
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.seed = int(seed)

        # Hard swaps + stochastic logic should not be cached
        self.disable_cache = True

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # ─────────────────────────────────────────
        # Key-level semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # ─────────────────────────────────────────
        # Tensor-level safety
        # ─────────────────────────────────────────
        if (
            not isinstance(base, torch.Tensor)
            or not base.is_floating_point()
            or base.ndim == 0
            or base.numel() == 0
        ):
            return base

        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
                and torch.any(t != 0)
            )
        ]
        if not others:
            return base

        # ─────────────────────────────────────────
        # Deterministic probability gate
        # ─────────────────────────────────────────
        alpha = max(0.0, min(1.0, self.alpha))
        if alpha <= 0.0:
            return base
        if alpha >= 1.0:
            # Always exchange
            pass
        else:
            # Hash → uniform [0,1)
            seed_val = self.seed ^ (hash(self.key) & 0xFFFFFFFF)
            rnd = ((seed_val * 2654435761) & 0xFFFFFFFF) / 2**32
            if rnd >= alpha:
                return base

        # ─────────────────────────────────────────
        # Deterministic selection among alternatives
        # ─────────────────────────────────────────
        idx = (self.seed + hash(self.key)) % len(others)
        return others[idx].to(base.dtype)


class WeightSumCutoff(Operation):
    """
    WeightSumCutoff (Band-Pass Channel Blend):

      • Computes similarity-to-base per channel
      • Selects channels with moderate difference
      • Blends contributor mean into base for selected channels
      • Preserves base semantics elsewhere
    """

    FORBIDDEN_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "skip_connection",
        "norm",
        "layer_norm",
        "ln_",
        "scale_shift",
        "affine",
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",
        "text_model",
        "cond_stage_model",
        "conditioner",
        "token_embedding",
        "position_embedding",
    )

    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]

        # ─────────────────────────────────────────
        # Key-level semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # ─────────────────────────────────────────
        # Tensor-level safety
        # ─────────────────────────────────────────
        if (
            not isinstance(base, torch.Tensor)
            or not base.is_floating_point()
            or base.ndim < 2
            or base.numel() == 0
        ):
            return base

        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if not others:
            return base

        # ─────────────────────────────────────────
        # Parameter clamps (pure Python)
        # ─────────────────────────────────────────
        alpha = max(0.0, min(1.0, self.alpha))
        beta  = max(0.0, min(1.0, self.beta))
        gamma = max(beta, min(1.0, self.gamma))

        if alpha <= 0.0 or beta >= gamma:
            return base

        # ─────────────────────────────────────────
        # Similarity computation
        # ─────────────────────────────────────────
        deltas = [torch.abs(t - base) for t in others]
        max_delta = torch.max(torch.stack(deltas, dim=0), dim=0).values

        if not torch.any(max_delta):
            return base

        sim = 1.0 - (max_delta / (max_delta.max() + 1e-8))
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # ─────────────────────────────────────────
        # Channel-wise reduction
        # ─────────────────────────────────────────
        reduce_dims = tuple(range(1, sim.ndim))
        channel_sim = sim.mean(dim=reduce_dims, keepdim=True)

        # Band-pass mask
        mask = ((channel_sim > beta) & (channel_sim < gamma)).to(base.dtype)

        if not torch.any(mask):
            return base

        # ─────────────────────────────────────────
        # Contributor mean
        # ─────────────────────────────────────────
        contrib_mean = torch.mean(torch.stack(others, dim=0), dim=0)

        # ─────────────────────────────────────────
        # Selective blend
        # ─────────────────────────────────────────
        out = base * (1.0 - alpha * mask) + contrib_mean * (alpha * mask)

        return out.to(base.dtype)


class HybridCascadeSimple(Operation):
    """
    HybridCascadeSimple (key-aware cascading hybrid merge):

    Goal:
      Automatically choose the safest / most appropriate merge operator
      based on parameter key semantics.

    This class is a *router*, not a math primitive.
    Shape- or dimensionality-specific rules are enforced by child operators.
    """

    # Ultra-hard refuse: never do math here
    FORBIDDEN_PATTERNS = (
        "metadata",
        "state_dict",
        "__",
    )

    # Soft category hints (NOT policy)
    NOISE_TIME_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "conv_in",
        "input_blocks.0",
        "skip_connection",
    )

    ATTENTION_PATTERNS = (
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
    )

    def __init__(
        self,
        key,
        weights,
        *sources,
        prefer_copy: int = 0,

        # Global personality knob (0 stable -> 1 spicy)
        confidence: float = 0.5,

        # CLIP/VAE profile (gentler)
        clip_vae_mix: float = 0.60,
        clip_vae_conf: float = 0.35,
        clip_vae_temp: float = 2.0,

        # Noise/time profile (very gentle)
        noise_mix: float = 0.40,
        noise_conf: float = 0.25,
        noise_temp: float = 2.5,

        # UNet/general profile
        unet_mix: float = 1.0,
        unet_temp: float = 1.0,

        # TrainDiff usage
        use_traindiff: bool = True,
        traindiff_top_k: int = 3,
        traindiff_zero_center: bool = True,
        traindiff_strength: float = 0.50,

        # Optional: sparse agreement pre-pass
        use_ties: bool = False,
        ties_density: float = 0.35,
        ties_seed: int = 42,

        # DAREWISE knobs
        dare_density: float = 0.35,
        dare_dropout: float = 0.10,
        wise_density: float = 0.35,
        wise_dropout: float = 0.30,
        seed: int = 42,
    ):
        super().__init__(key, *sources)

        self.prefer_copy = int(prefer_copy)
        self.confidence = float(max(0.0, min(1.0, confidence)))

        self.clip_vae_mix = float(clip_vae_mix)
        self.clip_vae_conf = float(clip_vae_conf)
        self.clip_vae_temp = float(clip_vae_temp)

        self.noise_mix = float(noise_mix)
        self.noise_conf = float(noise_conf)
        self.noise_temp = float(noise_temp)

        self.unet_mix = float(unet_mix)
        self.unet_temp = float(unet_temp)

        self.use_traindiff = bool(use_traindiff)
        self.traindiff_top_k = int(traindiff_top_k)
        self.traindiff_zero_center = bool(traindiff_zero_center)
        self.traindiff_strength = float(max(0.0, min(1.0, traindiff_strength)))

        self.use_ties = bool(use_ties)
        self.ties_density = float(ties_density)
        self.ties_seed = int(ties_seed)

        self.dare_density = float(dare_density)
        self.dare_dropout = float(dare_dropout)
        self.wise_density = float(wise_density)
        self.wise_dropout = float(wise_dropout)
        self.seed = int(seed)

        # Normalize weights for AdaptiveLERP
        if weights is None:
            raise ValueError("HybridCascadeSimple requires weights")

        w = [float(x) for x in weights]
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))

        total = sum(max(0.0, x) for x in w)
        if total <= 0.0:
            w = [1.0] + [0.0] * (len(sources) - 1)
            total = 1.0

        self.weights = [max(0.0, x) / total for x in w]

    # -------------------------------------------------
    # Key categorization
    # -------------------------------------------------

    def _is_clip_or_vae(self, key_lower: str) -> bool:
        try:
            if cmn.is_clip_key(self.key) or cmn.is_vae_key(self.key):
                return True
        except Exception:
            pass

        return (
            "first_stage_model" in key_lower
            or "vae" in key_lower
            or "encoder" in key_lower
            or "decoder" in key_lower
            or "cond_stage_model" in key_lower
            or "text_model" in key_lower
            or "conditioner" in key_lower
        )

    def _category(self) -> str:
        k = self.key.lower()

        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return "forbidden"
        if self._is_clip_or_vae(k):
            return "clip_vae"
        if any(p in k for p in self.NOISE_TIME_PATTERNS):
            return "noise_time"
        if any(p in k for p in self.ATTENTION_PATTERNS):
            return "attention"
        return "unet_general"

    # -------------------------------------------------
    # Operator wrappers
    # -------------------------------------------------

    def _copy(self, tensors):
        from scripts.untitled.operators import COPY
        return COPY(self.key, *tensors, prefer=self.prefer_copy).oper(*tensors)

    def _adaptive_lerp(self, tensors, *, base_mix, confidence, temperature):
        from scripts.untitled.operators import AdaptiveLERP
        return AdaptiveLERP(
            self.key,
            self.weights,
            *tensors,
            base_mix=base_mix,
            confidence=confidence,
            temperature=temperature,
        ).oper(*tensors)

    def _adaptive_darewise(self, tensors, *, aggression_bias):
        from scripts.untitled.operators import AdaptiveDAREWISE
        return AdaptiveDAREWISE(
            self.key,
            self.dare_density,
            self.dare_dropout,
            self.wise_density,
            self.wise_dropout,
            aggression_bias,
            self.seed,
            *tensors,
        ).oper(*tensors)

    def _maybe_ties(self, tensors):
        if not self.use_ties or len(tensors) < 2:
            return tensors
        try:
            from scripts.untitled.operators import TIES
            out = TIES(
                self.key,
                *tensors,
                density=self.ties_density,
                seed=self.ties_seed,
            ).oper(*tensors)
            return [out] + list(tensors[1:])
        except Exception:
            return tensors

    def _train_diff_then_stabilize(self, tensors):
        if len(tensors) < 3 or self.traindiff_strength <= 0.0:
            return None
        try:
            from scripts.untitled.operators import TrainDiff
            td = TrainDiff(
                self.key,
                tensors[0], tensors[1], tensors[2],
                *tensors[3:],
                top_k=self.traindiff_top_k,
                zero_center=self.traindiff_zero_center,
            ).oper(*tensors)

            stab = self._adaptive_lerp(
                tensors,
                base_mix=self.unet_mix,
                confidence=self.confidence,
                temperature=self.unet_temp,
            )
            return torch.lerp(stab, td.to(stab.dtype), self.traindiff_strength)
        except Exception:
            return None

    # -------------------------------------------------
    # Main
    # -------------------------------------------------

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return None

        base = tensors[0]
        if base is None:
            return None

        if not isinstance(base, torch.Tensor):
            return base

        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not base.is_floating_point():
            return self._copy(tensors)

        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        cat = self._category()

        if cat == "clip_vae":
            return self._adaptive_lerp(
                valid,
                base_mix=self.clip_vae_mix,
                confidence=self.clip_vae_conf,
                temperature=self.clip_vae_temp,
            ).to(base.dtype)

        if cat == "noise_time":
            return self._adaptive_lerp(
                valid,
                base_mix=self.noise_mix,
                confidence=self.noise_conf,
                temperature=self.noise_temp,
            ).to(base.dtype)

        if cat == "attention":
            return self._adaptive_darewise(
                valid,
                aggression_bias=self.confidence,
            ).to(base.dtype)

        # UNet general
        valid2 = self._maybe_ties(valid)

        if self.use_traindiff:
            out = self._train_diff_then_stabilize(valid2)
            if out is not None:
                return out.to(base.dtype)

        if self.confidence >= 0.55:
            return self._adaptive_darewise(
                valid2,
                aggression_bias=self.confidence,
            ).to(base.dtype)

        return self._adaptive_lerp(
            valid2,
            base_mix=self.unet_mix,
            confidence=self.confidence,
            temperature=self.unet_temp,
        ).to(base.dtype)


class HybridCascade(Operation):
    """
    HybridCascade (block-aware, depth-biased cascading hybrid merge):

    Adds:
      • Key-based depth estimation across SD1.5 / SDXL / Flux-like UNet naming.
      • Depth-biased routing:
          - Early blocks: stabilize (favor MEAN / gentle AdaptiveLERP)
          - Mid blocks: balanced
          - Late blocks: more expressive (allow more DAREWISE / TrainDiff)
      • Depth-biased AdaptiveLERP confidence curves (separate from UNet global confidence)
      • Attention-head selective TIES:
          - Optionally sparsify ONLY value projections (to_v / v_proj)
          - (Optionally) include output projections too
      • Learned depth profiles:
          - Derive a "disagreement score" from tensor deltas (variance/magnitude)
          - Blend with key-depth to auto-stabilize fragile layers
    """

    FORBIDDEN_PATTERNS = (
        "metadata",
        "state_dict",
        "__",
    )

    NOISE_TIME_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "conv_in",
        "input_blocks.0",
        "skip_connection",
    )

    ATTENTION_PATTERNS = (
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "q_proj",
        "k_proj",
        "v_proj",
        "proj",
        "out_proj",
    )

    # Attention-selective TIES filters
    ATTENTION_VALUE_PATTERNS = (
        "to_v",
        "v_proj",
    )
    ATTENTION_OUTPROJ_PATTERNS = (
        "out_proj",
        "proj_out",
        "proj",
    )

    def __init__(
        self,
        key,
        weights,
        *sources,
        prefer_copy: int = 0,

        # Global personality knob (0 stable -> 1 spicy)
        confidence: float = 0.5,

        # CLIP/VAE profile (gentler)
        clip_vae_mix: float = 0.60,
        clip_vae_conf: float = 0.35,
        clip_vae_temp: float = 2.0,

        # Noise/time profile (very gentle)
        noise_mix: float = 0.40,
        noise_conf: float = 0.25,
        noise_temp: float = 2.5,

        # UNet/general profile (base)
        unet_mix: float = 1.0,
        unet_temp: float = 1.0,

        # TrainDiff usage
        use_traindiff: bool = True,
        traindiff_top_k: int = 3,
        traindiff_zero_center: bool = True,
        traindiff_strength: float = 0.50,

        # Optional: TIES before detail ops
        use_ties: bool = False,
        ties_density: float = 0.35,
        ties_seed: int = 42,

        # -----------------------------
        # 🆕 Depth-biased TIES controls
        # -----------------------------
        ties_depth_bias_enabled: bool = True,
        ties_density_early: float = 0.45,   # early: stronger sparsity
        ties_density_late: float = 0.15,    # late: gentler sparsity
        ties_depth_curve: float = 1.25,

        # -----------------------------
        # 🆕 Attention-selective TIES
        # -----------------------------
        ties_attention_selective: bool = True,     # only prune value proj (and optional out proj)
        ties_attention_include_outproj: bool = False,
        ties_attention_density_scale: float = 0.75,  # extra gentleness for attention even when active

        # DAREWISE / AdaptiveDAREWISE knobs
        dare_density: float = 0.35,
        dare_dropout: float = 0.10,
        wise_density: float = 0.35,
        wise_dropout: float = 0.30,
        seed: int = 42,

        # -----------------------------
        # 🆕 Depth bias controls
        # -----------------------------
        depth_bias_enabled: bool = True,

        # How much depth changes general "confidence" (detail ops willingness)
        depth_conf_strength: float = 0.35,

        # How much depth changes "mix" (LERP vs MEAN)
        depth_mix_strength: float = 0.25,

        # Optional: scale TrainDiff strength with depth
        depth_traindiff_strength: float = 0.40,

        # Curve shaping ( >1 concentrates deeper; <1 spreads earlier )
        depth_curve: float = 1.25,

        # -----------------------------
        # 🆕 Depth-biased AdaptiveLERP confidence curve (separate from global)
        # -----------------------------
        unet_lerp_conf_early: float = 0.30,   # early blocks: lower confidence = more stabilization
        unet_lerp_conf_late: float = 0.70,    # late blocks: higher confidence = more identity/detail
        unet_lerp_conf_curve: float = 1.35,   # shape

        # -----------------------------
        # 🆕 Learned depth profile (auto-stabilize fragile layers)
        # -----------------------------
        learned_depth_enabled: bool = True,
        learned_depth_blend: float = 0.50,    # 0=only key-depth, 1=only learned profile
        learned_depth_curve: float = 1.20,    # shape learned depth signal
        learned_depth_eps: float = 1e-8,
    ):
        super().__init__(key, *sources)

        self.disable_cache = True

        self.prefer_copy = int(prefer_copy)

        self.confidence = float(max(0.0, min(1.0, confidence)))

        self.clip_vae_mix = float(clip_vae_mix)
        self.clip_vae_conf = float(clip_vae_conf)
        self.clip_vae_temp = float(clip_vae_temp)

        self.noise_mix = float(noise_mix)
        self.noise_conf = float(noise_conf)
        self.noise_temp = float(noise_temp)

        self.unet_mix = float(unet_mix)
        self.unet_temp = float(unet_temp)

        self.use_traindiff = bool(use_traindiff)
        self.traindiff_top_k = int(traindiff_top_k)
        self.traindiff_zero_center = bool(traindiff_zero_center)
        self.traindiff_strength = float(max(0.0, min(1.0, traindiff_strength)))

        self.use_ties = bool(use_ties)
        self.ties_density = float(max(0.0, min(1.0, ties_density)))
        self.ties_seed = int(ties_seed)

        # Depth-biased TIES knobs
        self.ties_depth_bias_enabled = bool(ties_depth_bias_enabled)
        self.ties_density_early = float(max(0.0, min(1.0, ties_density_early)))
        self.ties_density_late = float(max(0.0, min(1.0, ties_density_late)))
        self.ties_depth_curve = float(max(0.1, ties_depth_curve))

        # Attention-selective TIES knobs
        self.ties_attention_selective = bool(ties_attention_selective)
        self.ties_attention_include_outproj = bool(ties_attention_include_outproj)
        self.ties_attention_density_scale = float(max(0.0, min(2.0, ties_attention_density_scale)))

        self.dare_density = float(dare_density)
        self.dare_dropout = float(dare_dropout)
        self.wise_density = float(wise_density)
        self.wise_dropout = float(wise_dropout)
        self.seed = int(seed)

        # Depth bias knobs
        self.depth_bias_enabled = bool(depth_bias_enabled)
        self.depth_conf_strength = float(max(0.0, min(1.0, depth_conf_strength)))
        self.depth_mix_strength = float(max(0.0, min(1.0, depth_mix_strength)))
        self.depth_traindiff_strength = float(max(0.0, min(1.0, depth_traindiff_strength)))
        self.depth_curve = float(max(0.1, depth_curve))

        # Depth-biased AdaptiveLERP confidence curve
        self.unet_lerp_conf_early = float(max(0.0, min(1.0, unet_lerp_conf_early)))
        self.unet_lerp_conf_late = float(max(0.0, min(1.0, unet_lerp_conf_late)))
        self.unet_lerp_conf_curve = float(max(0.1, unet_lerp_conf_curve))

        # Learned depth profile
        self.learned_depth_enabled = bool(learned_depth_enabled)
        self.learned_depth_blend = float(max(0.0, min(1.0, learned_depth_blend)))
        self.learned_depth_curve = float(max(0.1, learned_depth_curve))
        self.learned_depth_eps = float(learned_depth_eps)

        # weights used for AdaptiveLERP / linear-ish blending
        if weights is None:
            raise ValueError("HybridCascade requires weights")

        w = [float(x) for x in weights]
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))

        total = sum(max(0.0, x) for x in w)
        if total <= 0.0:
            w = [1.0] + [0.0] * (len(sources) - 1)
            total = 1.0

        self.weights = [max(0.0, x) / total for x in w]

    # -------------------------------------------------
    # Key helpers
    # -------------------------------------------------
    def _is_clip_or_vae(self, key_lower: str) -> bool:
        try:
            if cmn.is_clip_key(self.key) or cmn.is_vae_key(self.key):
                return True
        except Exception:
            pass

        return (
            "first_stage_model" in key_lower
            or "vae" in key_lower
            or "encoder" in key_lower
            or "decoder" in key_lower
            or "cond_stage_model" in key_lower
            or "text_model" in key_lower
            or "conditioner" in key_lower
        )

    def _is_attention_key(self, key_lower: str) -> bool:
        return any(p in key_lower for p in self.ATTENTION_PATTERNS)

    def _is_attention_value_key(self, key_lower: str) -> bool:
        if any(p in key_lower for p in self.ATTENTION_VALUE_PATTERNS):
            return True
        if self.ties_attention_include_outproj and any(p in key_lower for p in self.ATTENTION_OUTPROJ_PATTERNS):
            return True
        return False

    def _category(self) -> str:
        k = self.key.lower()

        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return "forbidden"

        if self._is_clip_or_vae(k):
            return "clip_vae"

        if any(p in k for p in self.NOISE_TIME_PATTERNS):
            return "noise_time"

        if self._is_attention_key(k):
            return "attention"

        return "unet_general"

    # -------------------------------------------------
    # Depth estimation (SD1.5 / SDXL / Flux-ish)
    # -------------------------------------------------
    def _extract_int(self, s: str, token: str):
        try:
            m = re.search(rf"{re.escape(token)}[\.|_](\d+)", s)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _estimate_depth_norm_from_key(self) -> float:
        k = self.key.lower()

        if "middle_block" in k or "mid_block" in k:
            return 0.85

        ib = self._extract_int(k, "input_blocks")
        ob = self._extract_int(k, "output_blocks")
        if ib is not None:
            return min(1.0, max(0.0, ib / 12.0)) * 0.75
        if ob is not None:
            return 0.75 + min(1.0, max(0.0, ob / 12.0)) * 0.25

        db = self._extract_int(k, "down_blocks")
        ub = self._extract_int(k, "up_blocks")
        if db is not None:
            return min(1.0, max(0.0, db / 4.0)) * 0.75
        if ub is not None:
            return 0.75 + min(1.0, max(0.0, ub / 4.0)) * 0.25

        tb = self._extract_int(k, "transformer_blocks")
        if tb is not None:
            return min(1.0, max(0.0, tb / 24.0))

        db2 = self._extract_int(k, "double_blocks")
        sb2 = self._extract_int(k, "single_blocks")
        if db2 is not None:
            return min(1.0, max(0.0, db2 / 24.0))
        if sb2 is not None:
            return min(1.0, max(0.0, sb2 / 24.0))

        return 0.5

    def _learned_depth_norm(self, tensors) -> float:
        """
        "Learned depth": estimate fragility/disagreement from tensor deltas.

        Output is [0,1]:
          • 0.0 = low disagreement (safe to be expressive)
          • 1.0 = high disagreement (be conservative)
        """
        if not self.learned_depth_enabled or not tensors or len(tensors) < 2:
            return 0.5

        try:
            base = tensors[0]
            if not isinstance(base, torch.Tensor) or not base.is_floating_point():
                return 0.5

            # Use mean as baseline to reduce bias toward any single source
            mean_t = torch.mean(torch.stack([t.float() for t in tensors], dim=0), dim=0)
            deltas = [t.float() - mean_t for t in tensors]

            # Disagreement magnitude proxy:
            # variance across sources, normalized by mean absolute magnitude
            stacked = torch.stack(deltas, dim=0)  # [M, ...]
            var = stacked.var(dim=0)

            # Reduce to a scalar (channel-friendly but cheap)
            if var.ndim > 1:
                var_s = var.mean(dim=tuple(range(1, var.ndim)))
                var_s = var_s.mean()
            else:
                var_s = var.mean()

            mag = mean_t.abs().mean().clamp_min(self.learned_depth_eps)
            score = (var_s / mag).clamp_min(0.0)

            # Map to [0,1] smoothly
            #  score ~0 => 0, big => ->1
            learned = (score / (score + 1.0)).item()
            learned = max(0.0, min(1.0, float(learned)))
            learned = learned ** self.learned_depth_curve
            return learned

        except Exception:
            return 0.5

    def _combined_depth(self, tensors) -> float:
        """
        Depth signal in [0,1], where:
          0.0 = early-ish / safe-ish
          1.0 = late-ish OR fragile (depending on learned blend)
        """
        key_d = self._estimate_depth_norm_from_key()
        key_d = max(0.0, min(1.0, key_d))
        key_d = key_d ** self.depth_curve

        learned = self._learned_depth_norm(tensors)

        # learned is "fragility" (1=fragile=be conservative)
        # Convert to a "depth-like" signal where fragile behaves like early:
        # fragile -> lower effective depth
        learned_depthlike = 1.0 - learned  # fragile => 0, safe => 1
        learned_depthlike = max(0.0, min(1.0, float(learned_depthlike)))

        if not self.learned_depth_enabled or self.learned_depth_blend <= 0.0:
            return key_d

        b = self.learned_depth_blend
        return (1.0 - b) * key_d + b * learned_depthlike

    # -------------------------------------------------
    # Depth-biased parameter shaping
    # -------------------------------------------------
    def _depth_signed(self, tensors) -> float:
        """
        Signed depth in [-1,+1] centered at 0 mid-depth.
        """
        d = self._combined_depth(tensors)
        d = max(0.0, min(1.0, d))
        return (d - 0.5) * 2.0

    def _depth_biased_unet_lerp_conf(self, tensors) -> float:
        """
        Separate curve for AdaptiveLERP confidence on UNet/general layers.

        Early -> unet_lerp_conf_early
        Late  -> unet_lerp_conf_late
        """
        d = self._combined_depth(tensors)
        d = max(0.0, min(1.0, d))
        d = d ** self.unet_lerp_conf_curve
        return (
            self.unet_lerp_conf_early * (1.0 - d)
            + self.unet_lerp_conf_late * d
        )

    def _apply_depth_bias(self, tensors, *, base_mix: float, confidence: float, traindiff_strength: float):
        """
        Returns depth-biased (base_mix, confidence, traindiff_strength).

        Uses combined depth (key-depth blended with learned fragility).
        """
        if not self.depth_bias_enabled:
            return base_mix, confidence, traindiff_strength

        signed = self._depth_signed(tensors)  # -1..+1

        conf = confidence + signed * self.depth_conf_strength
        mix = base_mix + signed * self.depth_mix_strength
        td  = traindiff_strength + signed * self.depth_traindiff_strength

        conf = float(max(0.0, min(1.0, conf)))
        mix  = float(max(0.0, min(1.0, mix)))
        td   = float(max(0.0, min(1.0, td)))
        return mix, conf, td

    def _depth_biased_ties_density(self, tensors) -> float:
        """
        Interpolates TIES density based on combined depth.

        Early blocks -> ties_density_early
        Late blocks  -> ties_density_late
        """
        if not self.ties_depth_bias_enabled:
            return self.ties_density

        d = self._combined_depth(tensors)
        d = max(0.0, min(1.0, d))
        d = d ** self.ties_depth_curve
        return (
            self.ties_density_early * (1.0 - d)
            + self.ties_density_late * d
        )

    # -------------------------------------------------
    # Operator wrappers
    # -------------------------------------------------
    def _copy(self, tensors):
        from scripts.untitled.operators import COPY
        return COPY(self.key, *tensors, prefer=self.prefer_copy).oper(*tensors)

    def _adaptive_lerp(self, tensors, *, base_mix, confidence, temperature):
        from scripts.untitled.operators import AdaptiveLERP
        return AdaptiveLERP(
            self.key,
            self.weights,
            *tensors,
            base_mix=base_mix,
            confidence=confidence,
            temperature=temperature,
        ).oper(*tensors)

    def _adaptive_darewise(self, tensors, *, aggression_bias):
        from scripts.untitled.operators import AdaptiveDAREWISE
        return AdaptiveDAREWISE(
            self.key,
            self.dare_density,
            self.dare_dropout,
            self.wise_density,
            self.wise_dropout,
            aggression_bias,
            self.seed,
            *tensors,
        ).oper(*tensors)

    def _maybe_ties(self, tensors, *, category: str):
        if not self.use_ties or len(tensors) < 2:
            return tensors

        # Attention-selective rule:
        # only apply TIES on value projections (optionally out proj)
        k = self.key.lower()
        if category == "attention" and self.ties_attention_selective:
            if not self._is_attention_value_key(k):
                return tensors

        try:
            from scripts.untitled.operators import TIES

            density = self._depth_biased_ties_density(tensors)

            # Extra gentleness in attention even when allowed
            if category == "attention":
                density = float(max(0.0, min(1.0, density * self.ties_attention_density_scale)))

            out = TIES(
                self.key,
                *tensors,
                density=density,
                seed=self.ties_seed
            ).oper(*tensors)

            return [out] + list(tensors[1:])
        except Exception:
            return tensors

    def _train_diff_then_stabilize(self, tensors, *, td_strength, unet_mix, unet_conf, unet_temp):
        if len(tensors) < 3 or td_strength <= 0.0:
            return None

        try:
            from scripts.untitled.operators import TrainDiff

            td = TrainDiff(
                self.key,
                tensors[0], tensors[1], tensors[2],
                *tensors[3:],
                # If your TrainDiff supports these, great; otherwise remove:
                top_k=self.traindiff_top_k,
                zero_center=self.traindiff_zero_center,
            ).oper(*tensors)

            stab = self._adaptive_lerp(
                tensors,
                base_mix=unet_mix,
                confidence=unet_conf,
                temperature=unet_temp,
            )

            return torch.lerp(stab, td.to(stab.dtype), td_strength)
        except Exception:
            return None

    # -------------------------------------------------
    # Main
    # -------------------------------------------------
    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())

        base = tensors[0]
        if base is None:
            return None
        if not isinstance(base, torch.Tensor):
            return base

        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not base.is_floating_point():
            return self._copy(tensors)

        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            )
        ]
        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        cat = self._category()

        # -----------------------------
        # CLIP / VAE: conservative profile (no depth bias)
        # -----------------------------
        if cat == "clip_vae":
            return self._adaptive_lerp(
                valid,
                base_mix=self.clip_vae_mix,
                confidence=self.clip_vae_conf,
                temperature=self.clip_vae_temp,
            ).to(base.dtype)

        # -----------------------------
        # Noise/time: ultra stable (no depth bias)
        # -----------------------------
        if cat == "noise_time":
            return self._adaptive_lerp(
                valid,
                base_mix=self.noise_mix,
                confidence=self.noise_conf,
                temperature=self.noise_temp,
            ).to(base.dtype)

        # -----------------------------
        # Attention:
        #   • optional TIES only on value proj (and optional out proj)
        #   • depth+learned bias controls how spicy detail ops can get
        # -----------------------------
        if cat == "attention":
            valid2 = self._maybe_ties(valid, category=cat)

            _mix, conf, _td = self._apply_depth_bias(
                valid2,
                base_mix=1.0,
                confidence=self.confidence,
                traindiff_strength=0.0,
            )
            aggression_bias = conf
            return self._adaptive_darewise(valid2, aggression_bias=aggression_bias).to(base.dtype)

        # -----------------------------
        # UNet general:
        #   • depth-biased TIES (optional)
        #   • depth-biased TrainDiff
        #   • depth-biased AdaptiveDAREWISE vs AdaptiveLERP routing
        #   • separate AdaptiveLERP confidence curve (unet_lerp_conf_early/late)
        # -----------------------------
        valid2 = self._maybe_ties(valid, category=cat)

        unet_mix, unet_conf, td_strength = self._apply_depth_bias(
            valid2,
            base_mix=self.unet_mix,
            confidence=self.confidence,
            traindiff_strength=self.traindiff_strength,
        )

        # Separate curve for AdaptiveLERP confidence (this is the big “no cosine required” UX win)
        lerp_conf = self._depth_biased_unet_lerp_conf(valid2)

        # TrainDiff injection (depth-biased)
        if self.use_traindiff:
            out = self._train_diff_then_stabilize(
                valid2,
                td_strength=td_strength,
                unet_mix=unet_mix,
                unet_conf=lerp_conf,   # stabilization uses the lerp-specific curve
                unet_temp=self.unet_temp,
            )
            if out is not None:
                return out.to(base.dtype)

        # If confident (after depth bias), use detail op
        if unet_conf >= 0.55:
            return self._adaptive_darewise(valid2, aggression_bias=unet_conf).to(base.dtype)

        # Otherwise stabilize with AdaptiveLERP (depth-biased mix + lerp-specific confidence curve)
        return self._adaptive_lerp(
            valid2,
            base_mix=unet_mix,
            confidence=lerp_conf,
            temperature=self.unet_temp,
        ).to(base.dtype)

class HybridCascadeLite(Operation):
    """
    HybridCascadeLite (fallback-safe hybrid merge):

    • Key-aware routing
    • Depth-biased confidence & mix
    • Uses ONLY AdaptiveLERP + COPY
    • Deterministic, low-memory, low-compute
    • Safe as fallback AND calc mode
    """

    FORBIDDEN_PATTERNS = (
        "metadata",
        "state_dict",
        "__",
    )

    NOISE_TIME_PATTERNS = (
        "time_embed",
        "time_embedding",
        "timestep",
        "time_in",
        "sigma",
        "noise",
        "conv_in",
        "input_blocks.0",
        "skip_connection",
    )

    ATTENTION_PATTERNS = (
        "attn",
        "attention",
        "to_q",
        "to_k",
        "to_v",
        "proj",
    )

    def __init__(
        self,
        key,
        weights,
        *sources,
        prefer_copy: int = 0,

        # Global personality
        base_mix: float = 1.0,
        confidence: float = 0.5,

        # CLIP / VAE (safe)
        clip_mix: float = 0.6,
        clip_conf: float = 0.35,
        clip_temp: float = 2.0,

        # Noise / timestep (ultra-safe)
        noise_mix: float = 0.4,
        noise_conf: float = 0.25,
        noise_temp: float = 2.5,

        # Depth bias
        depth_bias: float = 0.35,
        depth_curve: float = 1.25,
    ):
        super().__init__(key, *sources)

        # Fallback ops are not safely cacheable across merges unless you salt/clear the cache.
        # This prevents stale-tensor reuse.
        self.disable_cache = True

        self.prefer_copy = int(prefer_copy)

        def _clamp01(x):
            return float(max(0.0, min(1.0, float(x))))

        self.base_mix = _clamp01(base_mix)
        self.confidence = _clamp01(confidence)

        self.clip_mix = _clamp01(clip_mix)
        self.clip_conf = _clamp01(clip_conf)
        self.clip_temp = float(max(0.1, float(clip_temp)))

        self.noise_mix = _clamp01(noise_mix)
        self.noise_conf = _clamp01(noise_conf)
        self.noise_temp = float(max(0.1, float(noise_temp)))

        self.depth_bias = _clamp01(depth_bias)
        self.depth_curve = float(max(0.1, float(depth_curve)))

        # Normalize weights
        if not weights:
            raise ValueError("HybridCascadeLite requires weights")

        w = [max(0.0, float(x)) for x in weights]
        if len(w) < len(sources):
            w += [0.0] * (len(sources) - len(w))
        elif len(w) > len(sources):
            w = w[:len(sources)]

        total = sum(w)
        if total <= 0.0:
            w = [1.0] + [0.0] * (len(sources) - 1)
            total = 1.0

        self.weights = [x / total for x in w]

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _category(self):
        k = (self.key or "").lower()

        if any(p in k for p in self.FORBIDDEN_PATTERNS):
            return "forbidden"

        try:
            if cmn.is_clip_key(self.key) or cmn.is_vae_key(self.key):
                return "clip_vae"
        except Exception:
            pass

        if any(p in k for p in self.NOISE_TIME_PATTERNS):
            return "noise"

        if any(p in k for p in self.ATTENTION_PATTERNS):
            return "attention"

        return "unet"

    def _estimate_depth(self):
        k = (self.key or "").lower()

        for token in ("input_blocks", "output_blocks", "down_blocks", "up_blocks"):
            m = re.search(rf"{token}[._](\d+)", k)
            if m:
                idx = int(m.group(1))
                norm = min(1.0, idx / 12.0)
                return norm ** self.depth_curve

        if "middle_block" in k or "mid_block" in k:
            return 0.85

        return 0.5

    def _depth_adjust(self, mix, conf):
        d = self._estimate_depth()
        signed = (d - 0.5) * 2.0

        mix = mix + signed * self.depth_bias
        conf = conf + signed * self.depth_bias

        mix = float(max(0.0, min(1.0, mix)))
        conf = float(max(0.0, min(1.0, conf)))
        return mix, conf

    def _copy(self, tensors):
        # Avoid importing this module from itself
        return COPY(self.key, *tensors, prefer=self.prefer_copy).oper(*tensors)

    def _lerp(self, tensors, weights, *, mix, conf, temp):
        # Avoid importing this module from itself
        return AdaptiveLERP(
            self.key,
            weights,
            *tensors,
            base_mix=mix,
            confidence=conf,
            temperature=temp,
        ).oper(*tensors)

    # -------------------------------------------------
    # Main
    # -------------------------------------------------
    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        base = tensors[0]
        if not isinstance(base, torch.Tensor):
            return base

        k = (self.key or "").lower()
        if any(p in k for p in self.FORBIDDEN_PATTERNS):
            return base

        if not base.is_floating_point():
            return self._copy(tensors)

        # Filter tensors AND their corresponding weights together
        valid = []
        valid_w = []
        for t, w in zip(tensors, self.weights):
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and t.numel() > 0
            ):
                valid.append(t)
                valid_w.append(float(w))

        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        # Renormalize weights for the valid subset
        total = sum(valid_w)
        if total <= 0.0:
            valid_w = [1.0] + [0.0] * (len(valid) - 1)
        else:
            valid_w = [x / total for x in valid_w]

        cat = self._category()

        if cat == "clip_vae":
            out = self._lerp(valid, valid_w, mix=self.clip_mix, conf=self.clip_conf, temp=self.clip_temp)
            return out.to(base.dtype)

        if cat == "noise":
            out = self._lerp(valid, valid_w, mix=self.noise_mix, conf=self.noise_conf, temp=self.noise_temp)
            return out.to(base.dtype)

        mix, conf = self._depth_adjust(self.base_mix, self.confidence)
        out = self._lerp(valid, valid_w, mix=mix, conf=conf, temp=1.0)
        return out.to(base.dtype)
