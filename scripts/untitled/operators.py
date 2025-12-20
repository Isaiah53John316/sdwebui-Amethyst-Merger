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
        if file and self.key in file:
            try:
                t = file.get_tensor(self.key)

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
        print(f"[LeanMode] {self.key} ← SKIPPED")
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
        return hash((self.key, self.checkpoint_name))

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
        if self.key not in file:
            raise RuntimeError(
                f"Key '{self.key}' missing from checkpoint "
                f"{os.path.basename(used_path) if used_path else 'unknown'}"
            )

        t = file.get_tensor(self.key)

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
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        result = valid[0].clone()
        for t in valid[1:]:
            result = self.safe(torch.add, result, t)

        return result.to(cmn.get_dtype())


class Sub(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        result = valid[0].clone()
        for t in valid[1:]:
            result = self.safe(torch.sub, result, t)

        return result.to(cmn.get_dtype())


class Multiply(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        result = valid[0].clone()
        for t in valid[1:]:
            result = self.safe(torch.mul, result, t)

        return result.to(cmn.get_dtype())


class MultiplyTensors(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        result = valid[0].clone()
        for t in valid[1:]:
            result = self.safe(torch.mul, result, t)

        return result.to(cmn.get_dtype())


class Extract(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t is not None and t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # Base / a / b selection
        base = valid[0]
        a = valid[1] if len(valid) > 1 else base
        b = valid[2] if len(valid) > 2 else base

        # -------------------------------------------------
        # HARD SAFETY: all shapes must match
        # -------------------------------------------------
        if not (base.shape == a.shape == b.shape):
            print(f"[Extract] Shape mismatch skipped: {self.key} "
                  f"{base.shape}, {a.shape}, {b.shape}")
            return base

        dtype = base.dtype

        # -------------------------------------------------
        # Pure math — resize & eligibility handled upstream
        # -------------------------------------------------
        base_f = base.float()
        a_f = (a.float() - base_f).contiguous()
        b_f = (b.float() - base_f).contiguous()

        # Cosine similarity along last dim
        c = torch.cosine_similarity(a_f, b_f, dim=-1).clamp(-1, 1).unsqueeze(-1)
        d = ((c + 1) / 2) ** self.gamma

        result = torch.lerp(a_f, b_f, self.alpha) * torch.lerp(d, 1 - d, self.beta)

        return result.to(dtype)


class Similarities(Extract):
    def __init__(self, key, alpha, beta, gamma, a, b):
        super().__init__(key, alpha, beta, gamma, a, b)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t is not None and t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        a = valid[0]
        b = valid[1] if len(valid) > 1 else a

        # Delegate to Extract with no explicit base
        return super().oper(a, b)

class Clamp(Operation):
    def __init__(self, key, min_val=-1.0, max_val=1.0):
        super().__init__(key)
        self.min = min_val
        self.max = max_val

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0]
        if not valid:
            return torch.zeros([], device=cmn.get_device(), dtype=cmn.get_dtype())

        result = valid[0].clone()
        return result.clamp(self.min, self.max).to(cmn.get_dtype())
    
class Mean(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0]
        if not valid:
            return torch.zeros([], device=cmn.get_device(), dtype=cmn.get_dtype())

        result = sum(valid) / len(valid)
        return result.to(cmn.get_dtype())
    
class Normalize(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0]
        if not valid:
            return torch.zeros([], device=cmn.get_device(), dtype=cmn.get_dtype())

        t = valid[0].float()
        norm = t.norm()
        if norm == 0:
            return t.to(cmn.get_dtype())
        return (t / norm).to(cmn.get_dtype())

class ReBasin(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        a = valid[0]
        b = valid[1] if len(valid) > 1 else a

        # HARD SAFETY: shapes must match
        if a.shape != b.shape:
            print(f"[ReBasin] Shape mismatch skipped: {self.key}")
            return a

        a_sorted = torch.sort(a.flatten(), dim=-1).values
        b_sorted = torch.sort(b.flatten(), dim=-1).values

        merged = (1 - self.alpha) * a_sorted + self.alpha * b_sorted

        return merged.view_as(a).to(a.dtype)


class DeMe(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        a = valid[0]
        b = valid[1] if len(valid) > 1 else a

        # HARD SAFETY
        if a.shape != b.shape:
            print(f"[DeMe] Shape mismatch skipped: {self.key}")
            return a

        var_a = torch.var(a, dim=-1, keepdim=True)
        var_b = torch.var(b, dim=-1, keepdim=True)
        decoupled = torch.where(var_a > var_b, a, b)

        return (1 - self.alpha) * a + self.alpha * decoupled



class BlockWeighted(Operation):
    def __init__(self, key, alphas, a, b):
        super().__init__(key, a, b)
        self.alphas = alphas  # list of per-block weights

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        a = valid[0]
        b = valid[1] if len(valid) > 1 else a

        # HARD SAFETY
        if a.shape != b.shape:
            print(f"[BlockWeighted] Shape mismatch skipped: {self.key}")
            return a

        match = re.search(r'\.(\d+)\.', self.key)
        idx = int(match.group(1)) if match else 0
        alpha = self.alphas[min(idx, len(self.alphas) - 1)]

        return (1 - alpha) * a + alpha * b



class ToMe(Operation):
    def __init__(self, key, ratio, tensor):
        super().__init__(key, tensor)
        self.ratio = float(ratio)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())

        tensor = valid[0]

        # Must be 2D tokens [N, D]
        if tensor.ndim != 2 or tensor.size(0) < 2:
            return tensor

        # Fast ToMe (CVPR 2023)
        normed = F.normalize(tensor, dim=-1)
        sim = normed @ normed.T  # [N, N]

        k = max(2, int(tensor.size(0) * self.ratio))
        _, indices = torch.topk(sim, k, dim=1)

        merged = tensor[indices].mean(dim=1)

        return merged.to(tensor.dtype)

class AttentionMerge(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        a = valid[0]
        b = valid[1] if len(valid) > 1 else a

        # Only merge attention-related layers
        if 'attn' in self.key.lower() or 'attention' in self.key.lower():
            if a.shape != b.shape:
                print(f"[AttentionMerge] Shape mismatch skipped: {self.key}")
                return a
            return (1 - self.alpha) * a + self.alpha * b

        # Non-attention → primary only
        return a


class Smooth(Operation):
    def __init__(self, key, tensor):
        super().__init__(key, tensor)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())

        tensor = valid[0]
        if tensor.numel() < 5:
            return tensor

        device, dtype = tensor.device, tensor.dtype
        kernel_size, sigma = 5, 1.0
        center = kernel_size // 2

        x = torch.arange(kernel_size, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        orig_shape = tensor.shape

        x = tensor.flatten().unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (center, center), mode='replicate')
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
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1  # force odd

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t is not None and t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())

        tensor = valid[0]

        # 4D Conv2d weights → 2D smoothing
        if tensor.ndim == 4 and tensor.shape[2] >= 3 and tensor.shape[3] >= 3:
            return self._smooth_2d(tensor)

        # 2D+ tensors → 1D smoothing (flattened)
        if tensor.ndim >= 2 and tensor.numel() >= 10:
            return self._smooth_1d(tensor)

        return tensor

    def _smooth_2d(self, tensor):
        device, dtype = tensor.device, tensor.dtype
        size = int(self.kernel_size)
        pad = size // 2

        # If padding would be invalid for reflect, just bail out
        h, w = tensor.shape[2], tensor.shape[3]
        if pad <= 0 or pad >= h or pad >= w:
            return tensor

        # 1D Gaussian kernel
        x = torch.arange(size, device=device, dtype=dtype)
        center = size // 2
        kernel_1d = torch.exp(-0.5 * ((x - center) ** 2) / (self.sigma ** 2 + 1e-12))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 2D separable kernel
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, size, size)

        orig_shape = tensor.shape
        out_c, in_c = orig_shape[0], orig_shape[1]

        # Flatten to channels: [in*out, 1, h, w]
        x = tensor.permute(1, 0, 2, 3).reshape(-1, 1, h, w)

        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        # Depthwise conv with shared kernel; expand avoids huge allocation
        c = x.shape[0]
        weight = kernel_2d.expand(c, 1, size, size)
        smoothed = F.conv2d(x, weight, groups=c)

        # Restore [out, in, h, w]
        smoothed = smoothed.view(in_c, out_c, h, w).permute(1, 0, 2, 3)

        return smoothed.to(dtype)

    def _smooth_1d(self, tensor):
        # Use the existing Smooth operator with correct signature
        return Smooth(self.key, tensor).oper(tensor)


class TIES(Operation):
    def __init__(self, key, *sources, density, seed=42):
        super().__init__(key, *sources)
        self.density = float(density)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # -------------------------------------------------
        # Density edge cases
        # -------------------------------------------------
        if self.density <= 0.0:
            return base

        # Shape-safe candidates only
        same_shape = [t for t in others if t.shape == base.shape]
        if not same_shape:
            return base

        if self.density >= 1.0:
            deltas = [t - base for t in same_shape]
            norms = torch.stack([d.norm(p=2) for d in deltas])
            return same_shape[int(torch.argmax(norms))]

        # -------------------------------------------------
        # Deterministic seed
        # -------------------------------------------------
        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        deltas = [t - base for t in same_shape]
        if not deltas:
            return base

        # -------------------------------------------------
        # Select ONE winning delta (global, not per-element)
        # -------------------------------------------------
        delta_norms = torch.stack([d.norm(p=2) for d in deltas])
        winner_idx = int(torch.argmax(delta_norms))
        winning_delta = deltas[winner_idx]

        abs_delta = winning_delta.abs()

        # Top-k mask
        k = max(1, int(self.density * abs_delta.numel()))
        threshold = torch.topk(abs_delta.flatten(), k).values[-1]
        mask = abs_delta >= threshold

        # Sign resolution
        sign = torch.sign(winning_delta)
        resolved_sign = torch.where(sign == 0, torch.sign(base), sign)

        elected_delta = winning_delta * mask.to(winning_delta.dtype) * resolved_sign

        # Apply delta safely
        result = torch.where(
            mask & (torch.sign(base) == resolved_sign),
            base,
            base + elected_delta,
        )

        # Norm preservation
        ed_norm = elected_delta.norm(p=2)
        if ed_norm > 1e-8:
            scale = winning_delta.norm(p=2) / (ed_norm + 1e-8)
            result = base + elected_delta * scale.clamp_max(10.0)

        return result.to(base.dtype)

class WISE(Operation):
    """
    N-way WISE:
      - Select per-element strongest delta among contributors
      - Top-k mask with optional dropout
      - Random scaling on masked entries
      - Energy preservation
    """
    def __init__(self, key, density, dropout_p=0.3, seed=42, *sources):
        super().__init__(key, *sources)
        self.density = float(density)
        self.dropout_p = float(dropout_p)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        contributors = valid[1:]

        if self.density <= 0.0:
            return base

        # CROSS-ARCH SAFETY: only same-shape contributors participate
        same_shape = [t for t in contributors if t.shape == base.shape]
        if not same_shape:
            return base

        if self.density >= 1.0:
            deltas = [t - base for t in same_shape]
            norms = torch.stack([d.norm(p=2) for d in deltas])
            strongest = same_shape[int(torch.argmax(norms))]
            return strongest.to(base.dtype)

        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        deltas = [t - base for t in same_shape]
        if not deltas:
            return base

        # Stack absolute deltas → strongest per-element contributor
        abs_deltas = torch.stack([d.abs() for d in deltas], dim=0)  # [M, ...]
        max_magnitude, best_source = torch.max(abs_deltas, dim=0)   # [...], [...]

        # Top-k mask
        k = max(1, int(self.density * max_magnitude.numel()))
        threshold = torch.topk(max_magnitude.flatten(), k).values[-1]
        mask = max_magnitude >= threshold

        # Build winning_delta by selecting per-element best contributor
        winning_delta = deltas[0]
        for i, delta in enumerate(deltas):
            if i == 0:
                continue
            winning_delta = torch.where(best_source == i, delta, winning_delta)

        # Dropout mask
        if self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            dropout_mask = torch.bernoulli(torch.full_like(mask.float(), keep)).bool()
            mask = mask & dropout_mask

        # Random scaling on masked positions
        scale = torch.ones_like(winning_delta)
        if mask.any():
            n = int(mask.sum().item())
            scale_vals = torch.empty(n, device=scale.device).uniform_(0.5, 2.0)
            scale[mask] = scale_vals

        dared_delta = winning_delta * mask.to(winning_delta.dtype) * scale

        # Energy preservation from all contributors
        total_energy = sum(d.norm(p=2) for d in deltas)
        dd_norm = dared_delta.norm(p=2)
        if dd_norm > 1e-8:
            dared_delta = dared_delta * (total_energy / (dd_norm + 1e-8))

        return (base + dared_delta).to(base.dtype)

class DARE_Nway(Operation):
    """
    True N-way DARE:
      - Symmetric contributors
      - Per-source sparse deltas
      - Additive (non-competitive)
      - Direction-preserving
      - Energy-stable
    """
    def __init__(self, key, density, dropout_p=0.0, seed=42, base_mode="mean", *sources):
        super().__init__(key, *sources)
        self.density = float(density)
        self.dropout_p = float(dropout_p)
        self.seed = int(seed)
        self.base_mode = base_mode

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # Base selection
        if self.base_mode == "mean":
            base = torch.mean(torch.stack(valid), dim=0)
        elif self.base_mode == "first":
            base = valid[0]
        else:
            raise ValueError("Unknown base_mode")

        torch.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        merged_delta = torch.zeros_like(base)
        total_energy = 0.0

        for t in valid:
            delta = t - base
            if torch.all(delta == 0):
                continue

            # Per-source sparsity
            k = max(1, int(self.density * delta.numel()))
            mags = delta.abs().flatten()
            threshold = torch.topk(mags, k).values[-1]
            mask = delta.abs() >= threshold

            # Optional dropout
            if self.dropout_p > 0.0:
                keep = 1.0 - self.dropout_p
                drop = torch.bernoulli(
                    torch.full_like(mask.float(), keep)
                ).bool()
                mask &= drop

            sparse_delta = delta * mask.to(delta.dtype)

            merged_delta += sparse_delta
            total_energy += delta.norm(p=2)

        # Energy normalization
        md_norm = merged_delta.norm(p=2)
        if md_norm > 1e-8:
            merged_delta *= total_energy / (md_norm + 1e-8)

        return (base + merged_delta).to(base.dtype)

class DAREWISE(Operation):
    """
    DARE+WISE Hybrid:
      - DARE provides sparse, additive, energy-stable structure
      - WISE provides competitive, high-contrast detail
      - Key-based or block-based gating decides which dominates
      - Optional soft blending between the two
    """
    def __init__(
        self,
        key,
        dare_density,
        dare_dropout,
        wise_density,
        wise_dropout,
        mix=0.5,                 # 0 = pure DARE, 1 = pure WISE
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
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # --- Step 1: Compute both candidate merges ---
        dare = Operation.DARE_Nway(
            self.key,
            self.dare_density,
            self.dare_dropout,
            self.seed,
            *valid
        ).oper(*valid)

        wise = Operation.WISE(
            self.key,
            self.wise_density,
            self.wise_dropout,
            self.seed + 1337,
            *valid
        ).oper(*valid)

        # --- Step 2: Decide which philosophy dominates ---
        # Attention & projections are fragile → favor DARE
        key_lower = self.key.lower()
        attention_safe = any(
            s in key_lower
            for s in ("attn", "attention", "to_q", "to_k", "to_v", "proj")
        )

        if attention_safe:
            gate = 0.0   # pure DARE
        else:
            gate = self.mix

        # --- Step 3: Blend ---
        if gate <= 0.0:
            out = dare
        elif gate >= 1.0:
            out = wise
        else:
            out = torch.lerp(dare, wise, gate)

        return out.to(dare.dtype)

class AdaptiveDAREWISE(Operation):
    """
    Adaptive DARE+WISE:
      - Computes both DARE and WISE candidates
      - Builds an internal 'aggression field' A ∈ [0, 1]
      - A decides how much WISE is allowed per tensor
      - Attention layers hard-lock to DARE
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
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # -------------------------
        # Step 1: compute deltas
        # -------------------------
        base = torch.mean(torch.stack(valid), dim=0)
        deltas = [t - base for t in valid]

        # -------------------------
        # Step 2: aggression signals
        # -------------------------

        # (a) Delta agreement (cosine similarity)
        if len(deltas) >= 2:
            flat = [d.flatten() for d in deltas]
            sims = []
            for i in range(len(flat) - 1):
                num = torch.dot(flat[i], flat[i + 1])
                den = flat[i].norm() * flat[i + 1].norm() + 1e-8
                sims.append((num / den).clamp(-1, 1))
            agreement = torch.mean(torch.stack(sims))
            A_similarity = ((agreement + 1) * 0.5).item()
        else:
            A_similarity = 0.0

        # (b) Variance safety
        stacked = torch.stack(deltas)
        var = stacked.var(dim=0).mean().item()
        A_variance = float(torch.exp(-var))  # high variance → low aggression

        # (c) Block-depth heuristic (cheap, key-based)
        depth_scale = 1.0
        if "down_blocks" in self.key:
            depth_scale = 0.3
        elif "mid_block" in self.key:
            depth_scale = 0.6
        elif "up_blocks" in self.key:
            depth_scale = 1.0

        # -------------------------
        # Step 3: attention override
        # -------------------------
        key_lower = self.key.lower()
        attention_safe = any(
            s in key_lower
            for s in ("attn", "attention", "to_q", "to_k", "to_v", "proj")
        )

        if attention_safe:
            A = 0.0
        else:
            # Combine signals
            A = (
                0.45 * A_similarity +
                0.35 * A_variance
            )
            A *= depth_scale
            A = float(torch.clamp(torch.tensor(A * self.bias), 0.0, 1.0))

        # -------------------------
        # Step 4: compute candidates
        # -------------------------
        dare = Operation.DARE_Nway(
            self.key,
            self.dare_density,
            self.dare_dropout,
            self.seed,
            *valid
        ).oper(*valid)

        wise = Operation.WISE(
            self.key,
            self.wise_density,
            self.wise_dropout,
            self.seed + 1337,
            *valid
        ).oper(*valid)

        # -------------------------
        # Step 5: adaptive blend
        # -------------------------
        if A <= 0.0:
            out = dare
        elif A >= 1.0:
            out = wise
        else:
            out = torch.lerp(dare, wise, A)

        return out.to(dare.dtype)


class SLERP(Operation):
    """
    True N-way spherical linear interpolation on the hypersphere.
    Shape-safe, cross-arch safe, deterministic.
    """
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
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # CROSS-ARCH SAFETY: only same-shape contributors
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        base_flat = base.flatten()
        base_norm = base_flat.norm() + 1e-8

        def log_map(x, base_vec, base_norm):
            x_norm = x.norm() + 1e-8
            cos_theta = (x @ base_vec) / (x_norm * base_norm)
            cos_theta = cos_theta.clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)

            # Near-identical vectors → zero tangent
            if theta.item() < 1e-6:
                return torch.zeros_like(x)

            return (x - cos_theta * base_vec) * (theta / torch.sin(theta))

        # Log map contributors
        log_merged = torch.zeros_like(base_flat)
        for t, w in zip(others, self.weights):
            if w == 0.0:
                continue
            vec = t.flatten()
            log_merged += w * log_map(vec, base_flat, base_norm)

        # Exponential map back to sphere
        norm = log_merged.norm()
        if norm < 1e-6:
            return base

        exp_merged = (
            torch.cos(norm) * base_flat +
            torch.sin(norm) * (log_merged / norm)
        )

        return exp_merged.view_as(base).to(base.dtype)



class TrainDiff(Operation):
    def __init__(self, key, a, b, c, *extra_sources):
        super().__init__(key, a, b, c, *extra_sources)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # CROSS-ARCH SAFETY: only same-shape contributors
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        # Compute deltas from base
        deltas = [t - base for t in others]
        if not deltas:
            return base

        # Top-K strongest deltas by L2 norm
        delta_norms = torch.stack([d.norm(p=2) for d in deltas])
        k = min(3, len(deltas))
        top_k_indices = torch.topk(delta_norms, k).indices.tolist()

        selected_deltas = [deltas[i] for i in top_k_indices]

        combined_delta = torch.mean(torch.stack(selected_deltas), dim=0)

        # Zero-center to prevent drift
        combined_delta = combined_delta - combined_delta.mean()

        return (base + combined_delta).to(base.dtype)



class InterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.beta = int(beta)      # 0 = similarity, 1 = difference
        self.gamma = float(gamma)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # SHAPE SAFETY
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        alpha = max(self.alpha, 1e-3)

        # Absolute deltas
        deltas = [torch.abs(t - base) for t in others]

        # Per-element max delta
        delta_max = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(delta_max == 0):
            return base

        # --------------------------------------------------
        # Difference vs similarity signal
        # --------------------------------------------------
        if self.beta == 1:
            # Difference mode: strongest normalized difference
            diff_per = [d / (delta_max + 1e-8) for d in deltas]
            diff = torch.max(torch.stack(diff_per), dim=0).values
        else:
            # Similarity mode: inverse difference
            diff = 1.0 - (delta_max / (delta_max.max() + 1e-8))

        # Nonlinear shaping
        diff = diff.clamp(0, 1) ** (1 / alpha)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)

        # --------------------------------------------------
        # Deterministic stochastic mask
        # --------------------------------------------------
        rng = torch.Generator(device=diff.device)
        rng.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))

        bitmask = torch.bernoulli(diff, generator=rng)
        mask = torch.lerp(bitmask, diff, self.gamma)

        # --------------------------------------------------
        # Select strongest contributor per element
        # --------------------------------------------------
        abs_deltas = torch.stack(deltas, dim=0)
        _, best_idx = torch.max(abs_deltas, dim=0)

        winning = others[0]
        for i, t in enumerate(others):
            if i == 0:
                continue
            winning = torch.where(best_idx == i, t, winning)

        # --------------------------------------------------
        # Blend
        # --------------------------------------------------
        result = base * (1 - mask) + winning * mask

        return result.to(base.dtype)

    
class AutoEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # interpolation strength
        self.beta = float(beta)     # adaptive band width
        self.gamma = float(gamma)   # smoothness
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # --------------------------------------------------
        # SHAPE SAFETY
        # --------------------------------------------------
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        # --------------------------------------------------
        # Absolute deltas
        # --------------------------------------------------
        deltas = [torch.abs(t - base) for t in others]
        max_delta = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(max_delta == 0):
            return base

        # --------------------------------------------------
        # Per-contributor similarity
        # sim = 1 → identical, 0 → maximally different
        # --------------------------------------------------
        sim_per = [(max_delta - d) / (max_delta + 1e-8) for d in deltas]
        sim_stack = torch.stack(sim_per, dim=0)
        sim_stack = torch.nan_to_num(sim_stack, nan=0.0).clamp(0.0, 1.0)

        # Best contributor per element
        sim, best_idx = torch.max(sim_stack, dim=0)

        # --------------------------------------------------
        # Adaptive similarity band (your key idea)
        # --------------------------------------------------
        mean_sim = sim.mean()
        lower = mean_sim * (1.0 - self.beta)
        upper = mean_sim * (1.0 + self.beta)
        band_mask = (sim > lower) & (sim < upper)

        # --------------------------------------------------
        # Power shaping
        # --------------------------------------------------
        alpha_safe = max(self.alpha, 1e-3)
        shaped = sim ** (1.0 / alpha_safe)
        shaped = torch.nan_to_num(shaped, nan=0.0, posinf=1.0, neginf=0.0)
        shaped = shaped.clamp(0.0, 1.0)

        # Apply adaptive band
        shaped = shaped * band_mask.to(shaped.dtype)

        # --------------------------------------------------
        # Deterministic stochastic gate
        # --------------------------------------------------
        rng = torch.Generator(device=shaped.device)
        rng.manual_seed(self.seed + (hash(self.key) & 0xFFFFFFFF))
        bern = torch.bernoulli(shaped, generator=rng)

        # Smooth interpolation mask
        interp_mask = torch.lerp(bern, shaped, self.gamma).clamp(0.0, 1.0)

        # --------------------------------------------------
        # Select winning tensor per element
        # --------------------------------------------------
        winning = others[0]
        for i, t in enumerate(others):
            if i == 0:
                continue
            winning = torch.where(best_idx == i, t, winning)

        # --------------------------------------------------
        # Final blend: base ↔ winner
        # --------------------------------------------------
        result = base * (1.0 - interp_mask) + winning * interp_mask

        return result.to(base.dtype)


class SingularValueDeOperator(Operation):
    def __init__(self, key, alpha, beta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # threshold multiplier
        self.beta = float(beta)     # top-k fraction to keep
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # Shape-safe contributors only
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        # Practical guard: only do SVD on 2D matrices
        if base.ndim != 2:
            return base

        # Practical guard: avoid huge SVDs
        # (tune as you like; this is a sane default)
        if base.numel() > 8_000_000 or max(base.shape) > 4096:
            return base

        try:
            # Combined difference across contributors
            diffs = [t - base for t in others]
            total_diff = torch.sum(torch.stack(diffs, dim=0), dim=0)

            # Use float32 for numerical stability, then cast back
            work = total_diff.float()

            U, S, Vh = torch.linalg.svd(work, full_matrices=False)

            # Thresholding
            s_max = S.max()
            if s_max <= 1e-12:
                return base

            threshold = self.alpha * s_max
            significant = S > threshold

            # Top-k fraction (S is sorted desc)
            if self.beta < 1.0:
                k = max(1, int(self.beta * S.numel()))
                topk_mask = torch.zeros_like(significant)
                topk_mask[:k] = True
                significant = significant & topk_mask

            S_filtered = S * significant.to(S.dtype)

            # Efficient reconstruction: (U * S) @ Vh
            reconstructed = (U * S_filtered.unsqueeze(0)) @ Vh

            result = base + reconstructed.to(base.dtype)
            return result

        except Exception as e:
            print(f"[SVD] Failed on {self.key}: {e} — using base")
            return base



class TensorExchange(Operation):
    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)    # probability to swap
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # Shape-safe contributors only
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        # Deterministic pseudo-random choice
        seed_val = self.seed + (hash(self.key) & 0xFFFFFFFF)
        rnd = (seed_val % 10_000) / 10_000.0

        # With probability alpha, exchange
        if rnd < self.alpha:
            idx = seed_val % len(others)
            return others[idx]

        return base


class WeightSumCutoff(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)   # merge strength
        self.beta = float(beta)     # lower similarity threshold
        self.gamma = float(gamma)   # upper similarity threshold

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]

        # SHAPE SAFETY
        others = [t for t in valid[1:] if t.shape == base.shape]
        if not others:
            return base

        # Absolute deltas
        deltas = [torch.abs(t - base) for t in others]
        max_delta = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(max_delta == 0):
            return base

        # Similarity: 1 = identical, 0 = max difference
        sim = 1.0 - (max_delta / (max_delta.max() + 1e-8))
        sim = torch.nan_to_num(sim, nan=0.0).clamp(0.0, 1.0)

        # --------------------------------------------------
        # Channel-wise similarity
        # Reduce over all dims except the first (channel/out dim)
        # --------------------------------------------------
        if sim.ndim > 1:
            reduce_dims = tuple(range(1, sim.ndim))
            channel_sim = sim.mean(dim=reduce_dims, keepdim=True)
        else:
            channel_sim = sim

        # Band-pass selection
        mask = (channel_sim > self.beta) & (channel_sim < self.gamma)

        if not mask.any():
            return base

        # Mean of contributors
        contrib_mean = torch.mean(torch.stack(others, dim=0), dim=0)

        # Blend only selected channels
        alpha = self.alpha
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
