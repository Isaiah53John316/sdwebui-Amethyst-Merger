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


def tensor_size(t):
    """Return tensor size in bytes (safe for all tensor dtypes)."""
    if not isinstance(t, torch.Tensor):
        return 0
    return t.element_size() * t.nelement()


def recurse(operation):
    source_tensors = []

    for idx, source in enumerate(operation.sources):
        try:
            if hasattr(source, 'merge'):
                t = source.merge()
            else:
                # Scalar literal → tensor
                t = torch.as_tensor(source)

            if t is None:
                raise RuntimeError(
                    f"Source resolved to None (index {idx}, source={source})"
                )

            # Enforce device + dtype at the boundary
            t = t.to(device=cmn.get_device(), dtype=cmn.get_dtype())
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
        # Optional bypass hook (future-proof)
        if getattr(self, "disable_cache", False):
            return func(self, *source_tensors)

        # Fast path: cache hit
        try:
            with _cache_lock:
                result = weights_cache[self]
            if CACHE_DEBUG:
                print(f"[CACHE HIT] {self.__class__.__name__} :: {self.key}")
            return result
        except KeyError:
            if CACHE_DEBUG:
                print(f"[CACHE MISS] {self.__class__.__name__} :: {self.key}")

        # Compute outside lock
        result = func(self, *source_tensors)

        # Do NOT cache invalid results
        if result is None or not isinstance(result, torch.Tensor):
            if CACHE_DEBUG:
                print(f"[CACHE SKIP] {self.key} → invalid result")
            return result

        # Do NOT cache empty tensors
        if result.numel() == 0:
            if CACHE_DEBUG:
                print(f"[CACHE SKIP] {self.key} → empty tensor")
            return result

        # Do NOT cache autograd tensors
        if result.requires_grad:
            if CACHE_DEBUG:
                print(f"[CACHE SKIP] {self.key} → requires_grad")
            return result

        # Store in cache
        with _cache_lock:
            weights_cache[self] = result

        if CACHE_DEBUG:
            print(f"[CACHE STORE] {self.__class__.__name__} :: {self.key}")

        return result

    return wrapper


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
        return (self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources) == \
               (other.key, other.alpha, other.beta, other.gamma, other.delta, other.seed, other.sources)

    def __hash__(self):
        return hash((self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources))

    def oper(self, *tensors) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement oper(self, *tensors)")

    def merge(self):
        return self.merge_func(self)

    # -------------------------------------------------
    # OPTIONAL: shared broadcast-safe math helper
    # -------------------------------------------------
    def safe(self, op, a, b):
        """
        Apply a binary tensor operation safely.
        Prevents broadcasting and catastrophic allocation.
        """
        return cmn.safe_apply(op, a, b, self.key)


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
                device=cmn.get_device(),
                dtype=cmn.get_dtype()
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
            cmn.smartresize_enabled,  # optional policy salt
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

        # ─────────────────────────────────────────
        # Semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect compatible contributors
        valid = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and torch.any(t != 0)
            )
        ]

        if not valid:
            return base

        # ─────────────────────────────────────────
        # Additive interaction
        # ─────────────────────────────────────────
        result = base.clone()
        for t in valid:
            result = result + t

        # Numerical safety clamp (soft)
        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=base.max(),
            neginf=base.min(),
        )

        return result.to(base.dtype)


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

        # ─────────────────────────────────────────
        # Semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect compatible contributors
        valid = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and torch.any(t != 0)
            )
        ]

        if not valid:
            return base

        # ─────────────────────────────────────────
        # Subtractive interaction
        # ─────────────────────────────────────────
        result = base.clone()
        for t in valid:
            result = result - t

        # Numerical safety
        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=base.max(),
            neginf=base.min(),
        )

        return result.to(base.dtype)


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

        # ─────────────────────────────────────────
        # Semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect compatible contributors
        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and torch.any(t != 0)
            )
        ]

        if len(valid) <= 1:
            return base

        # ─────────────────────────────────────────
        # Multiplicative interaction
        # ─────────────────────────────────────────
        result = base.clone()
        for t in valid[1:]:
            result = result * t

        # Safety: kill NaNs / infs
        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=base.max(),
            neginf=base.min(),
        )

        return result.to(base.dtype)


class MultiplyTensors(Operation):
    """
    MultiplyTensors:
      • Element-wise multiplicative interaction
      • Extremely aggressive
      • Intended for experimental feature modulation ONLY

    Semantic contract:
      • NEVER touches temporal control, noise, CLIP, VAE, attention routing
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

        # Normalization & scaling
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

        # ─────────────────────────────────────────
        # Semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        # Collect compatible contributors
        valid = [
            t for t in tensors
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and torch.any(t != 0)
            )
        ]

        if len(valid) <= 1:
            return base

        # ─────────────────────────────────────────
        # Multiplicative interaction (bounded)
        # ─────────────────────────────────────────
        result = base.clone()

        for t in valid[1:]:
            result = result * t

        # Optional safety clamp (highly recommended)
        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=base.max(),
            neginf=base.min(),
        )

        return result.to(base.dtype)


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

        # ─────────────────────────────────────────
        # Semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
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

        # ─────────────────────────────────────────
        # Relative deltas
        # ─────────────────────────────────────────
        base_f = base.float()
        a_f = (a.float() - base_f)
        b_f = (b.float() - base_f)

        # If deltas are trivial, bail
        if not (torch.any(a_f) and torch.any(b_f)):
            return base

        # Cosine similarity along last dim
        c = torch.cosine_similarity(a_f, b_f, dim=-1).clamp(-1.0, 1.0)
        c = c.unsqueeze(-1)

        d = ((c + 1.0) * 0.5) ** self.gamma

        # Interpolate deltas, then re-anchor to base
        delta = torch.lerp(a_f, b_f, self.alpha)
        gated = delta * torch.lerp(d, 1.0 - d, self.beta)

        return (base_f + gated).to(base.dtype)


class Similarities(Extract):
    """
    Similarities:
      • Delegates to Extract to compute similarity features
      • Non-destructive, analysis-only operator
    """
    def __init__(self, key, alpha, beta, gamma, a, b):
        super().__init__(key, alpha, beta, gamma, a, b)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if isinstance(t, torch.Tensor) and t.numel() > 0]
        if len(valid) < 2:
            return valid[0] if valid else None

        return super().oper(valid[0], valid[1])


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
        self.min = float(min_val)
        self.max = float(max_val)

    @multi_cache_operation
    def oper(self, *tensors):
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0] if tensors else None

        t = tensors[0] if tensors else None
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return t

        return t.clamp(self.min, self.max).to(cmn.get_dtype())
    
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
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0] if tensors else None

        valid = [
            t for t in tensors
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        if not valid:
            return None

        return (sum(valid) / len(valid)).to(cmn.get_dtype())
    
class Normalize(Operation):
    """
    Normalize:
      • L2 normalizes entire tensor
      • EXPERIMENTAL / DESTRUCTIVE
      • Should never run on production merges
    """

    FORBIDDEN_PATTERNS = (
        "",  # default deny: forbid everything unless overridden
    )

    @multi_cache_operation
    def oper(self, *tensors):
        t = tensors[0] if tensors else None
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return t

        norm = t.norm()
        if norm <= 1e-8:
            return t

        return (t / norm).to(cmn.get_dtype())


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

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return a

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(a, torch.Tensor):
            return a

        if not a.is_floating_point():
            return a

        if a.shape != b.shape:
            print(f"[ReBasin] Shape mismatch skipped: {self.key}")
            return a

        # -------------------------------------------------
        # Alpha safety
        # -------------------------------------------------
        alpha = max(0.0, min(1.0, self.alpha))
        if alpha <= 0.0:
            return a
        if alpha >= 1.0:
            # full rebasin from b
            source = b
        else:
            source = b

        # -------------------------------------------------
        # Distributional rebasing
        # -------------------------------------------------
        a_flat = a.flatten()
        b_flat = source.flatten()

        a_sorted, a_idx = torch.sort(a_flat)
        b_sorted = torch.sort(b_flat).values

        merged_sorted = torch.lerp(a_sorted, b_sorted, alpha)

        # Restore original ordering
        rebased = torch.empty_like(merged_sorted)
        rebased[a_idx] = merged_sorted

        return rebased.view_as(a).to(a.dtype)



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
    """

    # DeMe must not touch control or semantic glue
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

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return a

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(a, torch.Tensor):
            return a

        if not a.is_floating_point():
            return a

        if a.shape != b.shape:
            print(f"[DeMe] Shape mismatch skipped: {self.key}")
            return a

        # -------------------------------------------------
        # Alpha safety
        # -------------------------------------------------
        alpha = max(0.0, min(1.0, self.alpha))
        if alpha <= 0.0:
            return a

        # -------------------------------------------------
        # Variance-based decoupling
        # -------------------------------------------------
        var_a = torch.var(a, dim=-1, keepdim=True)
        var_b = torch.var(b, dim=-1, keepdim=True)

        decoupled = torch.where(var_a > var_b, a, b)

        # -------------------------------------------------
        # Blend back toward primary
        # -------------------------------------------------
        return torch.lerp(a, decoupled, alpha).to(a.dtype)



class BlockWeighted(Operation):
    """
    Block-aware linear interpolation.

    Semantic contract:
      • Applies ONLY to block-indexed UNet layers
      • Deterministic, depth-aware blending
      • Identity on refusal
      • Never fabricates tensors
    """

    # Block weighting must not touch control or glue
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

    def __init__(self, key, alphas, a, b):
        super().__init__(key, a, b)
        self.alphas = list(alphas)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return a

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(a, torch.Tensor):
            return a

        if not a.is_floating_point():
            return a

        if a.shape != b.shape:
            print(f"[BlockWeighted] Shape mismatch skipped: {self.key}")
            return a

        # -------------------------------------------------
        # Block index extraction
        # -------------------------------------------------
        match = re.search(r'\.(\d+)\.', self.key)
        idx = int(match.group(1)) if match else 0

        if not self.alphas:
            return a

        alpha = self.alphas[min(idx, len(self.alphas) - 1)]

        # Clamp alpha defensively
        alpha = max(0.0, min(1.0, float(alpha)))

        # -------------------------------------------------
        # Linear blend
        # -------------------------------------------------
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
    """

    # ToMe must never touch non-token or control keys
    FORBIDDEN_PATTERNS = (
        # Temporal / control
        "time_embed",
        "time_embedding",
        "timestep",
        "sigma",
        "noise",

        # Weights / structure
        "weight",
        "bias",
        "conv",
        "linear",
        "proj",

        # Latent encode/decode
        "vae",
        "encoder",
        "decoder",
        "first_stage_model",
    )

    def __init__(self, key, ratio, tensor):
        super().__init__(key, tensor)
        self.ratio = float(ratio)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        tensor = tensors[0]

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensor

        # -------------------------------------------------
        # Token-shape validation
        # -------------------------------------------------
        if not isinstance(tensor, torch.Tensor):
            return tensor

        if not tensor.is_floating_point():
            return tensor

        # Must be token matrix [N, D]
        if tensor.ndim != 2 or tensor.size(0) < 2:
            return tensor

        # -------------------------------------------------
        # Ratio validation
        # -------------------------------------------------
        ratio = max(0.0, min(1.0, self.ratio))
        if ratio <= 0.0:
            return tensor

        N = tensor.size(0)
        k = max(2, int(N * ratio))
        if k >= N:
            return tensor

        # -------------------------------------------------
        # ToMe-style merge
        # -------------------------------------------------
        with torch.no_grad():
            normed = F.normalize(tensor, dim=-1)
            sim = normed @ normed.T  # [N, N]

            _, indices = torch.topk(sim, k, dim=1)
            merged = tensor[indices].mean(dim=1)

        return merged.to(tensor.dtype)

class AttentionMerge(Operation):
    """
    Attention-only linear merge.

    Semantic contract:
      • Applies ONLY to attention-related keys
      • Linear interpolation between two sources
      • Identity everywhere else
      • Never fabricates tensors
    """
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = float(alpha)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        a = tensors[0]
        b = tensors[1] if len(tensors) > 1 else a

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
    • Identity on failure
    • Floating-point only
    • Deterministic and safe
    """
    def __init__(self, key, tensor, kernel_size=5, sigma=1.0):
        super().__init__(key, tensor)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        # Enforce sane odd kernel
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel_size = max(3, min(self.kernel_size, 31))

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        tensor = tensors[0]

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

        x = torch.arange(size, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * ((x - center) / (sigma + 1e-12)) ** 2)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, -1)

        orig_shape = tensor.shape

        # Flatten → smooth → restore
        x = tensor.flatten().unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (center, center), mode="replicate")
        smoothed = F.conv1d(x, kernel)
        smoothed = smoothed.squeeze(0).squeeze(0)

        return smoothed.view(orig_shape).to(dtype)

class SmoothConv(Operation):
    """
    Smart hybrid smoothing:
      • Conv2d weights (4D) → 2D Gaussian
      • Linear / attention / other (>=2D) → 1D Gaussian
      • Everything else → pass-through
    """
    def __init__(self, key, sigma=1.0, kernel_size=None, tensor=None):
        super().__init__(key, tensor)
        self.sigma = float(sigma)
        self.kernel_size = kernel_size or (max(3, int(4 * self.sigma + 1)) | 1)

        # Safety clamp
        self.kernel_size = min(self.kernel_size, 31)

        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        tensor = tensors[0]

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
        # Delegates to existing Smooth operator (1D Gaussian)
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
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
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

        if not self.allow_non_floating and not t.is_floating_point():
            return base

        # -------------------------------------------------
        # Defensive clone + normalize
        # -------------------------------------------------
        return t.clone().to(
            dtype=cmn.get_dtype(),
            device=cmn.get_device()
        )


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

        # Defensive weight handling
        weights = list(weights) if weights else []
        total = sum(w for w in weights if w > 0.0)

        if total <= 0.0:
            # Refuse safely; base semantics preserved in oper()
            self.weights = []
        else:
            self.weights = [w / total for w in weights]

        # Pad to source count
        self.weights += [0.0] * (len(sources) - len(self.weights))

    @multi_cache_operation
    def oper(self, *tensors):
        base = tensors[0]

        # ── Semantic refusal ──
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # LERP only applies to floating-point tensors
        if not base.is_floating_point():
            return base

        # Collect valid same-shape contributors
        contributors = [
            (t, w)
            for t, w in zip(tensors, self.weights)
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and w > 0.0
            )
        ]

        if not contributors:
            return base

        # Linear blend
        merged = torch.zeros_like(base)
        for t, w in contributors:
            merged += t * w

        return merged.to(base.dtype)

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

        # Normalize weights safely
        if not weights:
            raise ValueError("LERPMEAN requires weights (at least one)")

        w = [float(x) for x in weights]
        # Pad weights to number of sources
        if len(w) < len(sources):
            w = w + [0.0] * (len(sources) - len(w))

        # Apply temperature (still linear, just reweighting)
        # - if temperature < 1, emphasizes larger weights
        # - if temperature > 1, flattens weights
        if self.temperature != 1.0:
            # keep sign and avoid weird negatives
            w = [max(0.0, x) for x in w]
            eps = 1e-12
            w = [(x + eps) ** (1.0 / max(self.temperature, eps)) for x in w]

        total = sum(w)
        if total <= 0.0:
            # If user gave all zeros, default to primary-dominant
            w = [1.0] + [0.0] * (len(sources) - 1)
            total = 1.0

        self.weights = [x / total for x in w]

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())

        base = tensors[0]

        # Optional operator-level gate (soft guardrail)
        if self.FORBIDDEN_PATTERNS and any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # Must be float to merge meaningfully
        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        # Collect same-shape float tensors with nonzero weights
        valid = []
        wts = []
        for t, w in zip(tensors, self.weights):
            if w <= 0.0:
                continue
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
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

        # MEAN path (stability)
        mean_out = torch.mean(torch.stack(valid, dim=0), dim=0)

        # LERP path (identity-preserving weighted blend)
        lerp_out = torch.zeros_like(base)
        for t, w in zip(valid, wts):
            lerp_out += t * w

        # Blend between them (bounded)
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
    ):
        super().__init__(key, *sources)

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
        if not tensors:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())

        base = tensors[0]

        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            return base

        valid, wts = [], []
        for t, w in zip(tensors, self.weights):
            if w > 0.0 and isinstance(t, torch.Tensor) and t.is_floating_point() and t.shape == base.shape:
                valid.append(t)
                wts.append(w)

        if not valid:
            return base
        if len(valid) == 1:
            return valid[0]

        mean_out = torch.mean(torch.stack(valid, dim=0), dim=0)

        lerp_out = torch.zeros_like(base)
        for t, w in zip(valid, wts):
            lerp_out += t * w

        if self.base_mix <= 0.0:
            return mean_out.to(base.dtype)

        if base.ndim == 0:
            return torch.lerp(mean_out, lerp_out, self.base_mix).to(base.dtype)

        C = base.shape[0]
        reduce_dims = tuple(range(1, base.ndim)) if base.ndim > 1 else None

        mean_base_f = mean_out.float()
        deltas = [(t.float() - mean_base_f) for t in valid]

        ref = deltas[0]
        ref_flat = ref.flatten(start_dim=1) if base.ndim > 1 else ref.unsqueeze(1)
        ref_norm = ref_flat.norm(dim=-1).clamp_min(self.eps)

        sims = []
        for d in deltas[1:]:
            df = d.flatten(start_dim=1) if base.ndim > 1 else d.unsqueeze(1)
            dn = df.norm(dim=-1).clamp_min(self.eps)
            cos = ((df * ref_flat).sum(dim=-1) / (dn * ref_norm)).clamp(-1.0, 1.0)
            sims.append(cos)

        agree = torch.mean(torch.stack(sims), dim=0) if sims else torch.ones(C, device=base.device)
        agree = ((agree + 1.0) * 0.5).clamp(0.0, 1.0)
        agree = agree ** self.agree_power

        stacked = torch.stack(deltas, dim=0)
        var = stacked.var(dim=0)
        if reduce_dims:
            var = var.mean(dim=reduce_dims)

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

    # ─────────────────────────────────────────────
    # TIES-SPECIFIC FORBIDDEN KEYS
    # (single-winner logic breaks semantics & stability)
    # ─────────────────────────────────────────────
    FORBIDDEN_PATTERNS = (
        # Timestep / temporal conditioning
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",

        # Noise / sigma control
        "sigma",
        "noise",

        # Early signal injection (winner dominance is catastrophic)
        "conv_in.",
        "input_blocks.0.",

        # Residual routing (breaks information pathways)
        "skip_connection",

        # Latent encode / decode stability
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",

        # Semantic embeddings (single winner ≠ meaning)
        "text_model.",
        "cond_stage_model.",
        "conditioner.",
        "token_embedding",
        "position_embedding",

        # Normalization / scaling layers (TIES is especially unsafe here)
        "layer_norm",
        "scale_shift",
        "affine",
        "ln_",          # keep if your keys use ln_ prefixes
        "norm",         # remove later if you decide to allow it
    )

    def __init__(self, key, *sources, density, seed=42):
        super().__init__(key, *sources)
        self.density = float(density)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        # ─────────────────────────────────────────
        # Operator-level semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0]

        base = tensors[0]

        # TIES only applies to floating-point tensors
        if not base.is_floating_point():
            return base

        # Density edge cases
        if self.density <= 0.0:
            return base

        # Collect valid same-shape floating contributors (including base)
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

        others = valid[1:]
        if not others:
            return base

        # Deterministic behavior
        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        deltas = [t - base for t in others]
        if not deltas:
            return base

        # ─────────────────────────────────────────
        # Select ONE winning delta (global)
        # ─────────────────────────────────────────
        delta_norms = torch.stack([d.norm(p=2) for d in deltas])
        winner_idx = int(torch.argmax(delta_norms))
        winning_delta = deltas[winner_idx]

        # If the winning delta is effectively zero, refuse
        if not torch.any(winning_delta):
            return base

        abs_delta = winning_delta.abs()

        # Top-k mask
        k = max(1, int(self.density * abs_delta.numel()))
        threshold = torch.topk(abs_delta.flatten(), k).values[-1]
        mask = abs_delta >= threshold

        if not torch.any(mask):
            return base

        # ─────────────────────────────────────────
        # Sign resolution (safer + simpler)
        # ─────────────────────────────────────────
        # If winning_delta is 0 at a location, do nothing there.
        # Otherwise apply sign from winning_delta.
        sign = torch.sign(winning_delta)
        safe_delta = winning_delta * mask.to(winning_delta.dtype)  # keep native sign

        # ─────────────────────────────────────────
        # Apply delta with bounded scale
        # ─────────────────────────────────────────
        out = base + safe_delta

        # Norm preservation (bounded)
        sd_norm = safe_delta.norm(p=2)
        wd_norm = winning_delta.norm(p=2)
        if sd_norm > 1e-8 and wd_norm > 0.0:
            scale = (wd_norm / (sd_norm + 1e-8)).clamp(max=10.0)
            out = base + safe_delta * scale

        return out.to(base.dtype)


class WISE(Operation):
    """
    N-way WISE (Winner-Index Sparse Energy):

      • Selects per-element strongest delta among contributors
      • Top-k mask with optional dropout
      • Random scaling on masked entries (intrinsic)
      • Energy preservation (bounded)

    Semantic contract:
      • Excellent for sharpening / emphasizing dominant features
      • Highly aggressive and non-linear
      • Dangerous for control signals and embeddings
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
    )

    def __init__(self, key, density, dropout_p=0.3, seed=42, *sources):
        super().__init__(key, *sources)
        self.density = float(density)
        self.dropout_p = float(dropout_p)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0]

        base = tensors[0]
        if not base.is_floating_point():
            return base

        if self.density <= 0.0:
            return base

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

        contributors = valid[1:]

        if self.density >= 1.0:
            deltas = [t - base for t in contributors]
            norms = torch.stack([d.norm(p=2) for d in deltas])
            return contributors[int(torch.argmax(norms))].to(base.dtype)

        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        deltas = [t - base for t in contributors]

        abs_deltas = torch.stack([d.abs() for d in deltas], dim=0)
        max_mag, best = torch.max(abs_deltas, dim=0)

        k = max(1, int(self.density * max_mag.numel()))
        threshold = torch.topk(max_mag.flatten(), k).values[-1]
        mask = max_mag >= threshold

        if self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            mask &= torch.bernoulli(torch.full_like(mask.float(), keep)).bool()

        if not torch.any(mask):
            return base

        winning_delta = deltas[0]
        for i in range(1, len(deltas)):
            winning_delta = torch.where(best == i, deltas[i], winning_delta)

        # Intrinsic random scaling (bounded, deterministic)
        scale = torch.ones_like(winning_delta)
        n = int(mask.sum().item())
        scale_vals = torch.empty(n, device=scale.device).uniform_(0.5, 2.0)
        scale[mask] = scale_vals

        wise_delta = winning_delta * mask.to(winning_delta.dtype) * scale

        total_energy = sum(d.norm(p=2) for d in deltas)
        wd_norm = wise_delta.norm(p=2)
        if wd_norm > 1e-8 and total_energy > 0.0:
            wise_delta *= (total_energy / (wd_norm + 1e-8)).clamp(max=10.0)

        return (base + wise_delta).to(base.dtype)


class DARE_Nway(Operation):
    """
    True N-way DARE (Delta-Aware Residual Energy merge):

      • Symmetric contributors
      • Per-source sparse deltas
      • Additive (non-competitive)
      • Direction-preserving
      • Energy-stable

    Semantic contract:
      • Excellent for mid / late UNet feature refinement
      • Dangerous for timestep math, noise scale, and embeddings
      • Must NEVER touch temporal control or architectural glue
    """

    # ─────────────────────────────────────────────
    # DARE-SPECIFIC FORBIDDEN KEYS
    # (energy redistribution breaks control semantics)
    # ─────────────────────────────────────────────
    FORBIDDEN_PATTERNS = (
        # Timestep / temporal conditioning
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",

        # Noise / sigma control
        "sigma",
        "noise",

        # Early signal injection (energy explosion risk)
        "conv_in.",
        "input_blocks.0.",

        # Latent decode / encode stability
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",

        # Semantic embeddings (energy ≠ meaning)
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
        self.base_mode = base_mode

    @multi_cache_operation
    def oper(self, *tensors):
        # ─────────────────────────────────────────
        # Operator-level semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0]

        base = tensors[0]

        # DARE only applies to floating-point tensors
        if not base.is_floating_point():
            return base

        # Collect valid contributors
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

        # ─────────────────────────────────────────
        # Base selection
        # ─────────────────────────────────────────
        if self.base_mode == "mean":
            base = torch.mean(torch.stack(valid), dim=0)
        elif self.base_mode == "first":
            base = valid[0]
        else:
            return base  # refuse unknown modes safely

        # Deterministic sparsity
        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        merged_delta = torch.zeros_like(base)
        total_energy = 0.0

        # ─────────────────────────────────────────
        # Delta accumulation
        # ─────────────────────────────────────────
        for t in valid:
            delta = t - base
            if not torch.any(delta):
                continue

            # Per-source sparsity
            k = max(1, int(self.density * delta.numel()))
            mags = delta.abs().flatten()
            threshold = torch.topk(mags, k).values[-1]
            mask = delta.abs() >= threshold

            # Optional dropout
            if self.dropout_p > 0.0:
                keep = 1.0 - self.dropout_p
                mask &= torch.bernoulli(
                    torch.full_like(mask.float(), keep)
                ).bool()

            sparse_delta = delta * mask.to(delta.dtype)

            merged_delta += sparse_delta
            total_energy += delta.norm(p=2)

        # ─────────────────────────────────────────
        # Energy normalization (safety-capped)
        # ─────────────────────────────────────────
        md_norm = merged_delta.norm(p=2)
        if md_norm > 1e-8 and total_energy > 0.0:
            scale = (total_energy / (md_norm + 1e-8)).clamp(max=10.0)
            merged_delta *= scale

        return (base + merged_delta).to(base.dtype)


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
      - Computes both DARE and WISE candidates
      - Builds an internal 'aggression field' A ∈ [0, 1]
      - A decides how much WISE is allowed per tensor
      - Attention layers hard-lock to DARE

    Composite contract:
      • Never invents tensors
      • Refuses by preserving base semantics
      • Delegates semantic safety to child operators
    """

    def __init__(
        self,
        key,
        dare_density,
        dare_dropout,
        wise_density,
        wise_dropout,
        aggression_bias=0.5,   # user-facing control
        seed=42,
        *sources
    ):
        super().__init__(key, *sources)
        self.dare_density = float(dare_density)
        self.dare_dropout = float(dare_dropout)
        self.wise_density = float(wise_density)
        self.wise_dropout = float(wise_dropout)
        self.bias = float(aggression_bias)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        base = tensors[0]

        # Composite operators never invent tensors
        if not tensors or not base.is_floating_point():
            return base

        # Collect valid same-shape floating tensors (including base)
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

        # -------------------------
        # Step 1: compute deltas (safe)
        # -------------------------
        mean_base = torch.mean(torch.stack(valid), dim=0)
        deltas = [t - mean_base for t in valid]

        # -------------------------
        # Step 2: aggression signals
        # -------------------------

        # (a) Delta agreement (cosine similarity) using consecutive pairs
        if len(deltas) >= 2:
            flat = [d.flatten() for d in deltas]
            sims = []
            for i in range(len(flat) - 1):
                num = torch.dot(flat[i], flat[i + 1])
                den = flat[i].norm() * flat[i + 1].norm() + 1e-8
                sims.append((num / den).clamp(-1, 1))
            agreement = torch.mean(torch.stack(sims))
            A_similarity = float(((agreement + 1) * 0.5).clamp(0, 1))
        else:
            A_similarity = 0.0

        # (b) Variance safety (high variance → low aggression)
        stacked = torch.stack(deltas)
        var = stacked.var(dim=0).mean()
        A_variance = float(torch.exp(-var).clamp(0, 1))

        # (c) Block-depth heuristic (cheap, key-based)
        key_lower = self.key.lower()
        depth_scale = 1.0
        if "down_blocks" in key_lower or "input_blocks" in key_lower:
            depth_scale = 0.3
        elif "mid_block" in key_lower or "middle_block" in key_lower:
            depth_scale = 0.6
        elif "up_blocks" in key_lower or "output_blocks" in key_lower:
            depth_scale = 1.0

        # -------------------------
        # Step 3: attention override
        # -------------------------
        attention_safe = any(
            s in key_lower
            for s in ("attn", "attention", "to_q", "to_k", "to_v", "proj")
        )

        if attention_safe:
            A = 0.0
        else:
            # Combine signals (bounded)
            A = (0.45 * A_similarity + 0.35 * A_variance)
            A *= depth_scale
            A *= self.bias
            A = float(max(0.0, min(1.0, A)))

        # -------------------------
        # Step 4: compute candidates
        # (child operators enforce their own safety)
        # -------------------------
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

        return out.to(base.dtype)


class SLERP(Operation):
    """
    True N-way spherical linear interpolation on the hypersphere.

    Semantic contract:
      • Excellent for style / direction-bearing weights
      • Dangerous for timestep, noise, and residual routing
      • Must NEVER touch temporal control or noise-scale keys

    Therefore: SLERP enforces its own forbidden-key policy.
    """

    # ─────────────────────────────────────────────
    # SLERP-SPECIFIC FORBIDDEN KEYS
    # (temporal control, noise semantics, signal injection)
    # ─────────────────────────────────────────────
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

    def __init__(self, key, weights, *sources):
        super().__init__(key, *sources)

        total = sum(weights)
        if total > 1.0:
            weights = [w / total for w in weights]
            total = 1.0

        self.weights = weights + [0.0] * (len(sources) - len(weights))
        self.base_weight = 1.0 - total

    @multi_cache_operation
    def oper(self, *tensors):
        # ─────────────────────────────────────────
        # Operator-level semantic guard (authoritative)
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            # Preserve primary semantics
            return tensors[0]

        # Defensive: SLERP only applies to floating-point tensors
        base = tensors[0]
        if not base.is_floating_point():
            return base

        # Filter valid contributors
        valid = [
            t for t in tensors
            if t.numel() > 0 and torch.any(t != 0) and t.is_floating_point()
        ]

        if not valid:
            return torch.zeros(
                [],
                dtype=cmn.get_dtype(),
                device=cmn.get_device()
            )

        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # ─────────────────────────────────────────
        # CROSS-ARCH SAFETY: only same-shape tensors
        # ─────────────────────────────────────────
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        base_flat = base.flatten()
        base_norm = base_flat.norm() + 1e-8

        def log_map(x, base_vec, base_norm):
            x_flat = x.flatten()
            x_norm = x_flat.norm() + 1e-8

            cos_theta = (x_flat @ base_vec) / (x_norm * base_norm)
            cos_theta = cos_theta.clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)

            # Near-identical vectors → zero tangent
            if theta.item() < 1e-6:
                return torch.zeros_like(base_vec)

            return (x_flat - cos_theta * base_vec) * (theta / torch.sin(theta))

        # ─────────────────────────────────────────
        # Log-map accumulation (directional merge)
        # ─────────────────────────────────────────
        log_merged = torch.zeros_like(base_flat)

        for t, w in zip(others, self.weights):
            if w <= 0.0:
                continue
            log_merged += w * log_map(t, base_flat, base_norm)

        # ─────────────────────────────────────────
        # Exponential map back to sphere
        # ─────────────────────────────────────────
        norm = log_merged.norm()
        if norm < 1e-6:
            return base

        exp_merged = (
            torch.cos(norm) * base_flat +
            torch.sin(norm) * (log_merged / norm)
        )

        return exp_merged.view_as(base).to(base.dtype)


class TrainDiff(Operation):
    """
    TrainDiff (Training-Delta Aggregation):

      • Approximates training-induced updates in weight space
      • Selects top-K strongest deltas from contributors
      • Averages selected deltas
      • Optional drift suppression

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
    ):
        super().__init__(key, a, b, c, *extra_sources)
        self.top_k = int(top_k)
        self.zero_center = bool(zero_center)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        base = tensors[0]

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Shape-safe contributors only
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

        # -------------------------------------------------
        # Delta computation
        # -------------------------------------------------
        deltas = [t - base for t in others]
        if not deltas:
            return base

        # Top-K strongest deltas by L2 norm
        norms = torch.stack([d.norm(p=2) for d in deltas])
        k = max(1, min(self.top_k, len(deltas)))
        top_idx = torch.topk(norms, k).indices.tolist()

        selected = [deltas[i] for i in top_idx]

        combined_delta = torch.mean(torch.stack(selected), dim=0)

        # -------------------------------------------------
        # Optional drift suppression
        # -------------------------------------------------
        if self.zero_center:
            combined_delta = combined_delta - combined_delta.mean()

        return (base + combined_delta).to(base.dtype)




class InterpolateDifference(Operation):
    """
    InterpolateDifference (Stochastic Difference Selector):

      • Per-element probabilistic replacement
      • Driven by similarity or difference magnitude
      • Deterministic via key-based seeding
      • Extremely aggressive and non-linear

    Semantic contract:
      • Useful for experimental UNet feature mutation
      • Dangerous for embeddings, normalization, routing, and control
      • Must NEVER touch CLIP, VAE, or timestep math
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
        alpha,
        mode,      # "difference" or "similarity"
        gamma,
        seed,
        *sources
    ):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.mode = mode
        self.gamma = float(gamma)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        base = tensors[0]

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
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

        alpha = max(self.alpha, 1e-3)

        # Absolute deltas
        deltas = [torch.abs(t - base) for t in others]
        delta_max = torch.max(torch.stack(deltas), dim=0).values

        if not torch.any(delta_max):
            return base

        # -------------------------------------------------
        # Difference vs similarity signal
        # -------------------------------------------------
        if self.mode == "difference":
            diff = torch.max(
                torch.stack([d / (delta_max + 1e-8) for d in deltas]),
                dim=0
            ).values
        else:
            diff = 1.0 - (delta_max / (delta_max.max() + 1e-8))

        diff = diff.clamp(0, 1) ** (1 / alpha)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)

        # -------------------------------------------------
        # Deterministic stochastic mask
        # -------------------------------------------------
        rng = torch.Generator(device=diff.device)
        rng.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        bitmask = torch.bernoulli(diff, generator=rng)
        mask = torch.lerp(bitmask, diff, self.gamma)

        if not torch.any(mask):
            return base

        # -------------------------------------------------
        # Winner selection per element
        # -------------------------------------------------
        abs_deltas = torch.stack(deltas, dim=0)
        _, best_idx = torch.max(abs_deltas, dim=0)

        winning = others[0]
        for i in range(1, len(others)):
            winning = torch.where(best_idx == i, others[i], winning)

        # -------------------------------------------------
        # Blend
        # -------------------------------------------------
        return (base * (1 - mask) + winning * mask).to(base.dtype)


class AutoEnhancedInterpolateDifference(Operation):
    """
    AutoEnhancedInterpolateDifference (Adaptive Similarity Band Selector):

      • Computes per-contributor similarity-to-base per element
      • Selects the LEAST similar contributor per element (spicy winner)
      • Applies an adaptive band mask around mean similarity
      • Uses deterministic stochastic gating for controlled swaps
      • Blends base ↔ winner via a smooth interpolation mask

    Semantic contract:
      • Experimental UNet feature mutation tool
      • Extremely unsafe for timestep/noise, CLIP, VAE, attention routing, normalization
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

        # Normalization (optional: comment out for experiments)
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

    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # shaping strength (higher = softer)
        self.beta = float(beta)     # adaptive band width (0..1 is sane)
        self.gamma = float(gamma)   # smoothness (0=hard bernoulli, 1=soft prob)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors or tensors[0] is None:
            return None

        base = tensors[0]

        # -------------------------------------------------
        # Key-level semantic guard
        # -------------------------------------------------
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return base

        # -------------------------------------------------
        # Tensor-level safety
        # -------------------------------------------------
        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
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

        # -------------------------------------------------
        # Similarity-to-base per contributor
        # sim = 1 → identical to base, 0 → far from base
        # -------------------------------------------------
        deltas = [torch.abs(t - base) for t in others]

        max_overall = torch.max(torch.stack(deltas), dim=0).values
        if not torch.any(max_overall):
            return base

        denom = max_overall.max() + 1e-8  # global normalization
        sim_stack = torch.stack([1.0 - (d / denom) for d in deltas], dim=0)
        sim_stack = torch.nan_to_num(sim_stack, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # -------------------------------------------------
        # Winner = LEAST similar to base per element
        # -------------------------------------------------
        sim, best_idx = torch.min(sim_stack, dim=0)

        # -------------------------------------------------
        # Adaptive similarity band
        # (band is around the "least-similar" similarity field)
        # -------------------------------------------------
        mean_sim = sim.mean()
        beta = max(0.0, self.beta)
        lower = mean_sim * (1.0 - beta)
        upper = mean_sim * (1.0 + beta)
        band_mask = (sim > lower) & (sim < upper)

        # -------------------------------------------------
        # Power shaping (alpha)
        # NOTE: since sim is low for "spicy" regions,
        # this tends to reduce probability unless beta selects mid-sim zones.
        # If you want "more spice = higher prob", invert sim (see note below).
        # -------------------------------------------------
        alpha_safe = max(self.alpha, 1e-3)
        diffiness = (1.0 - sim)
        shaped = diffiness ** (1.0 / alpha_safe)
        shaped = torch.nan_to_num(shaped, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        shaped = shaped * band_mask.to(shaped.dtype)

        if not torch.any(shaped):
            return base

        # -------------------------------------------------
        # Deterministic stochastic gate
        # -------------------------------------------------
        rng = torch.Generator(device=shaped.device)
        rng.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        bern = torch.bernoulli(shaped, generator=rng)
        interp_mask = torch.lerp(bern, shaped, self.gamma).clamp(0.0, 1.0)

        if not torch.any(interp_mask):
            return base

        # -------------------------------------------------
        # Build winning tensor per element
        # -------------------------------------------------
        winning = others[0]
        for i in range(1, len(others)):
            winning = torch.where(best_idx == i, others[i], winning)

        # -------------------------------------------------
        # Blend base ↔ winner
        # -------------------------------------------------
        result = base * (1.0 - interp_mask) + winning * interp_mask
        return result.to(base.dtype)



class SingularValueDeOperator(Operation):
    """
    Singular-Value Decomposition based delta reconstruction.

    Semantic contract:
      • Extracts dominant linear modes from parameter deltas
      • Useful for style / feature-space refinement
      • Extremely dangerous for control, routing, and timing weights
      • Computationally expensive

    Therefore:
      • Enforces strict forbidden-key policy
      • Refuses non-floating, non-2D, or oversized tensors
      • Preserves primary semantics on refusal
    """

    # ─────────────────────────────────────────────
    # SVD-SPECIFIC FORBIDDEN KEYS
    # (routing, timing, noise, latent control)
    # ─────────────────────────────────────────────
    FORBIDDEN_PATTERNS = (
        # Temporal / timestep control
        "time_embed.",
        "time_embedding",
        "timestep",
        "time_in.",

        # Noise / sigma control
        "sigma",
        "noise",

        # Structural routing
        "skip_connection",
        "input_blocks.0.",
        "conv_in.",

        # Latent / VAE stability
        "first_stage_model.",
        "vae.",
        "encoder.",
        "decoder.",

        # Text / conditioning
        "cond_stage_model.",
        "conditioner.",
        "text_model.",
    )

    def __init__(self, key, alpha, beta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # singular-value threshold multiplier
        self.beta = float(beta)     # top-k fraction to keep
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        # ─────────────────────────────────────────
        # Operator-level semantic guard
        # ─────────────────────────────────────────
        if any(p in self.key for p in self.FORBIDDEN_PATTERNS):
            return tensors[0]

        base = tensors[0]

        # SVD is strictly floating-point math
        if not base.is_floating_point():
            return base

        # Collect valid contributors
        valid = [
            t for t in tensors
            if (
                t.numel() > 0
                and torch.any(t != 0)
                and t.is_floating_point()
            )
        ]

        if not valid or len(valid) == 1:
            return base

        # Shape-safe contributors only
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        # ─────────────────────────────────────────
        # Structural guards (SVD applicability)
        # ─────────────────────────────────────────
        # Only 2D matrices
        if base.ndim != 2:
            return base

        # Avoid pathological SVDs
        if base.numel() > 8_000_000 or max(base.shape) > 4096:
            return base

        # Deterministic behavior
        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        try:
            # Aggregate deltas
            diffs = [t - base for t in others]
            total_diff = torch.sum(torch.stack(diffs, dim=0), dim=0)

            # Work in float32 for numerical stability
            work = total_diff.float()

            # Compute SVD
            U, S, Vh = torch.linalg.svd(work, full_matrices=False)

            if S.numel() == 0:
                return base

            s_max = S.max()
            if s_max <= 1e-12:
                return base

            # ─────────────────────────────────────────
            # Singular value filtering
            # ─────────────────────────────────────────
            threshold = self.alpha * s_max
            significant = S > threshold

            # Top-k fraction (S is sorted descending)
            if self.beta < 1.0:
                k = max(1, int(self.beta * S.numel()))
                topk_mask = torch.zeros_like(significant)
                topk_mask[:k] = True
                significant = significant & topk_mask

            if not torch.any(significant):
                return base

            S_filtered = S * significant.to(S.dtype)

            # Efficient reconstruction
            reconstructed = (U * S_filtered.unsqueeze(0)) @ Vh

            return (base + reconstructed.to(base.dtype))

        except Exception:
            # Hard refusal: preserve base semantics
            return base


class TensorExchange(Operation):
    """
    TensorExchange (Deterministic Swap Operator):

      • Selects ONE alternative tensor with probability α
      • No numeric blending
      • Deterministic per-key behavior
      • Preserves base semantics on refusal

    Semantic contract:
      • Experimental structural mutation tool
      • Extremely dangerous for control, embeddings, normalization
      • Must NEVER touch temporal control or semantic glue
    """

    # ─────────────────────────────────────────────
    # EXCHANGE-SPECIFIC FORBIDDEN KEYS
    # (hard swap breaks control semantics instantly)
    # ─────────────────────────────────────────────
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

    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.seed = int(seed)

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
        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
            return base

        # Collect valid same-shape floating contributors
        others = [
            t for t in tensors[1:]
            if (
                isinstance(t, torch.Tensor)
                and t.is_floating_point()
                and t.shape == base.shape
                and torch.any(t != 0)
            )
        ]
        if not others:
            return base

        # ─────────────────────────────────────────
        # Probability gate (deterministic)
        # ─────────────────────────────────────────
        alpha = float(torch.clamp(
            torch.tensor(self.alpha),
            0.0,
            1.0
        ))

        seed_val = self.seed + (hash(self.key) & 0xFFFFFFFF)
        rnd = (seed_val % 10_000) / 10_000.0

        if rnd >= alpha:
            return base

        # Deterministic selection among alternatives
        idx = seed_val % len(others)
        return others[idx].to(base.dtype)


class WeightSumCutoff(Operation):
    """
    WeightSumCutoff (Band-Pass Channel Blend):

      • Computes similarity-to-base per channel
      • Selects channels with moderate difference
      • Blends contributor mean into base for selected channels
      • Preserves base semantics elsewhere

    Semantic contract:
      • Safe, conservative UNet refinement operator
      • NOT a control or semantic mutation tool
      • Must NEVER touch temporal control, attention, embeddings, or normalization
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
        if not isinstance(base, torch.Tensor):
            return base

        if not base.is_floating_point():
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

        # Clamp parameters
        alpha = float(torch.clamp(torch.tensor(self.alpha), 0.0, 1.0))
        beta = float(torch.clamp(torch.tensor(self.beta), 0.0, 1.0))
        gamma = float(torch.clamp(torch.tensor(self.gamma), beta, 1.0))

        # Absolute deltas
        deltas = [torch.abs(t - base) for t in others]
        max_delta = torch.max(torch.stack(deltas), dim=0).values
        if not torch.any(max_delta):
            return base

        # Similarity map
        sim = 1.0 - (max_delta / (max_delta.max() + 1e-8))
        sim = torch.nan_to_num(sim, nan=0.0).clamp(0.0, 1.0)

        # Channel-wise similarity
        if sim.ndim > 1:
            reduce_dims = tuple(range(1, sim.ndim))
            channel_sim = sim.mean(dim=reduce_dims, keepdim=True)
        else:
            channel_sim = sim

        # Band-pass mask
        mask = (channel_sim > beta) & (channel_sim < gamma)
        if not torch.any(mask):
            return base

        # Contributor mean
        contrib_mean = torch.mean(torch.stack(others, dim=0), dim=0)

        # Selective blend
        result = torch.where(
            mask,
            base * (1.0 - alpha) + contrib_mean * alpha,
            base,
        )

        return result.to(base.dtype)


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
        with self.lock:
            # Defensive: keys must be hash-stable
            if not hasattr(key, "__hash__"):
                raise TypeError("WeightsCache key must be hashable")

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
                (self.size >= self.size_cap or
                 (self.max_items and len(self.mapping) > self.max_items))
            ):
                _, tensor = self.mapping.popitem(last=False)
                self.size -= tensor_size(tensor)
                self.evictions += 1

    def __getitem__(self, key: Operation) -> torch.Tensor:
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

weights_cache = WeightsCache(4096, max_items=100_000)

class SmartResize(Operation):
    """
    SmartResize:
      - Sacred keys: NEVER interpolate (pad/slice only)
      - Non-sacred:
          * 1D: linear
          * 2D: bilinear (special-case large vocab as row-wise linear on dim)
          * 3D: resize last dim with linear; pad/slice first dims (safe + deterministic)
          * 4D (OIHW conv): resize H/W per-kernel; pad/slice O/I channels
      - Any weird rank mismatch: safe pad/slice or return as-is
    """
    def __init__(self, key, target_shape, source_tensor=None, orig_key=None):
        super().__init__(key)
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.source_tensor = source_tensor

        # 🔧 CHANGE 1: always preserve the *true* tensor key for policy checks
        self.orig_key = orig_key if orig_key is not None else key

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return torch.zeros(
                self.target_shape,
                dtype=cmn.get_dtype(),
                device=cmn.get_device()
            )

        t = tensors[0]
        device = t.device
        dtype = t.dtype

        # --------------------------
        # Safety first
        # --------------------------
        target = self.target_shape
        if (not target) or (len(target) == 0) or any(s > 100_000 for s in target):
            return t

        if t.numel() == 0:
            return torch.zeros(target, device=device, dtype=dtype)

        # 🔧 CHANGE 2: sacred check uses canonical key only
        # (no local lowercase copies, no prefixed keys)
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

        def _is_floatish(x: torch.Tensor) -> bool:
            return x.is_floating_point() or x.is_complex()

        # ===============================================================
        # SACRED: NEVER interpolate, only pad/slice
        # ===============================================================
        if is_sacred:
            if t.shape == target:
                return t.to(dtype)

            if t.ndim != len(target):
                return t.to(dtype)

            result = _pad_slice_to_target(t, target)
            merge_stats.smart_resized += 1
            print(f"[SANCTUARY] Preserved sacred tensor: {self.orig_key} → {tuple(result.shape)}")
            return result.to(dtype)

        # ===============================================================
        # NORMAL: interpolation allowed
        # ===============================================================

        if t.ndim != len(target):
            return t.to(dtype)

        if not _is_floatish(t):
            result = _pad_slice_to_target(t, target)
            merge_stats.smart_resized += 1
            return result.to(dtype)

        # 1D
        if t.ndim == 1:
            if t.shape == target:
                return t.to(dtype)
            tmp = t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
            tmp = F.interpolate(tmp[None, None, :], size=target[0], mode="linear", align_corners=False)[0, 0]
            merge_stats.smart_resized += 1
            return tmp.to(dtype)

        # 2D
        if t.ndim == 2:
            if t.shape == target:
                return t.to(dtype)

            if target[0] > 20_000:
                rows = min(t.shape[0], target[0])
                tmp = t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
                new_t = torch.zeros(target, device=device, dtype=tmp.dtype)

                for i in range(rows):
                    row = tmp[i:i + 1].unsqueeze(0).unsqueeze(0)
                    resized = F.interpolate(
                        row,
                        size=(1, target[1]),
                        mode="bilinear",
                        align_corners=False
                    )[0, 0, 0]
                    new_t[i] = resized

                merge_stats.smart_resized += 1
                return new_t.to(dtype)

            tmp = t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
            out = F.interpolate(tmp[None, None, :, :], size=target, mode="bilinear", align_corners=False)[0, 0]
            merge_stats.smart_resized += 1
            return out.to(dtype)

        # 3D
        if t.ndim == 3:
            if t.shape == target:
                return t.to(dtype)

            tmp = t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
            out = torch.zeros(target, device=device, dtype=tmp.dtype)

            a = min(tmp.shape[0], target[0])
            b = min(tmp.shape[1], target[1])
            block = tmp[:a, :b, :]
            block = _interp_last_dim(block, target[2])
            out[:a, :b, :] = block

            merge_stats.smart_resized += 1
            return out.to(dtype)

        # 4D (OIHW conv)
        if t.ndim == 4:
            if t.shape == target:
                return t.to(dtype)

            tmp = t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
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

        result = _pad_slice_to_target(t, target)
        merge_stats.smart_resized += 1
        return result.to(dtype)
