import torch, scipy
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from safetensors import SafetensorError   # ← THIS LINE IS THE FIX (adds missing error type)
import os
from functools import wraps
import re
from scripts.untitled.common import cmn

SACRED_SANCTUARY = {
    # === NOISE & TIME ENTRY (breaks sampling if touched) ===
    "conv_in.weight", "conv_in.bias",
    "input_blocks.0.0.weight", "input_blocks.0.0.bias",      # SDXL noise entry
    "time_embed.0.weight", "time_embed.0.bias",
    "time_embed.2.weight", "time_embed.2.bias",
    "time_embedding", "timestep_embed", "timestep_embedding",
    "time_in.weight", "time_in.bias",
    "vector_in.weight", "vector_in.bias",
    "img_in.weight", "img_in.bias",

    # === VAE — NEVER TOUCH (different latent scaling) ===
    "first_stage_model.", "encoder.", "decoder.",
    "quant_conv.", "post_quant_conv.", "vae.",

    # === TEXT ENCODERS — semantics live here ===
    "cond_stage_model.transformer.text_model.embeddings",
    "conditioner.embedders.0.transformer.text_model.embeddings",
    "conditioner.embedders.1.transformer.text_model.embeddings",
    "token_embedding", "position_embedding", "positional_embedding",
    "text_projection", "logit_scale",

    # === FLUX / SD3 / MODERN BLOCKS ===
    "single_blocks", "double_blocks",
    "x_embedder", "context_embedder", "t_embedder",
    "img_proj", "txt_proj",

    # === NOISE AUGMENTORS & OFFSETS ===
    "offset_noise", "noise_offset", "noise_augmentor",
    "learned_sigma", "sigma_embed",

    # === POSITIONAL EMBEDDINGS (any model) ===
    "pos_embed", "position_emb", "position_ids",
}

def tensor_size(t):
    """Return tensor size in bytes — safely handles non-floating point tensors"""
    return t.element_size() * t.nelement() if t.is_floating_point() or t.is_complex() else 0

def recurse(operation):
    source_tensors = []
    for source in operation.sources:
        if hasattr(source, 'merge'):
            source_tensors.append(source.merge())
        else:
            # Scalar value - convert to tensor if needed
            source_tensors.append(torch.tensor(source, device=cmn.get_device(), dtype=cmn.get_dtype()))
    return operation.oper(*source_tensors)

def multi_cache_operation(func):
    def wrapper(self, *source_tensors):
        try:
            return weights_cache[self]
        except KeyError:
            pass

        result = func(self, *source_tensors)
        weights_cache[self] = result
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

class CopyPrimary(Operation):
    def __init__(self, key, primary_path, stats=None, keep_zero_fill=True, bloat_mode=False):
        super().__init__(key)
        self.primary_path = primary_path
        self.stats = stats
        self.keep_zero_fill = keep_zero_fill
        self.bloat_mode = bloat_mode
        self.operation = "CopyPrimary"  # fixes task reuse crash

    @multi_cache_operation
    def oper(self, *args):
        file = cmn.loaded_checkpoints.get(self.primary_path)
        model_name = os.path.basename(self.primary_path) if self.primary_path else "Unknown"

        # Try copying from primary checkpoint
        if file and self.key in file.keys():
            try:
                t = file.get_tensor(self.key)

                # CROSS-ARCH: Resize to target shape
                if cmn.is_cross_arch:
                    target_shape = cmn.cross_arch_target_shapes.get(self.key)
                    if target_shape and t.shape != target_shape:
                        t = SmartResize(f"CopyPrimary_{self.key}", target_shape, t).oper(t)
                        if self.stats:
                            self.stats.smart_resized += 1
                        print(f"[FinalCopy] {self.key} ← RESIZED ({model_name})")

                # BLOAT MODE: Now works in cross-arch — SAFE and CHAOTIC
                if self.bloat_mode:
                    pad_amount = 256
                    current_shape = t.shape
                    bloated_shape = tuple(s + 2 * pad_amount for s in current_shape)
                    
                    bloated = torch.zeros(bloated_shape, dtype=t.dtype, device=t.device)
                    slices = tuple(slice(pad_amount, pad_amount + s) for s in current_shape)
                    bloated[slices] = t
                    
                    t = bloated
                    
                    if self.stats:
                        self.stats.zero_filled += 1
                    print(f"[BloatMode] {self.key} ← PADDED to {bloated_shape} (cross-arch SAFE)")

                if self.stats:
                    self.stats.copied_primary += 1
                return t.to(cmn.get_device(), dtype=cmn.get_dtype())

            except Exception as e:
                print(f"[CopyPrimary] FAILED reading {self.key}: {e}")

        # KITCHEN-SINK ZERO-FILL
        if cmn.is_cross_arch and self.keep_zero_fill:
            target_shape = cmn.cross_arch_target_shapes.get(self.key)
            if target_shape:
                t = torch.zeros(target_shape, dtype=cmn.get_dtype(), device=cmn.get_device())
                if self.stats:
                    self.stats.zero_filled += 1
                print(f"[KitchenSink] {self.key} ← ZERO-FILLED")
                return t

        # Fallback
        if self.stats:
            self.stats.skipped += 1
        print(f"[LeanMode] {self.key} ← SKIPPED")
        return torch.tensor([], dtype=cmn.get_dtype(), device=cmn.get_device())

class LoadTensor(Operation):
    def __init__(self, key, checkpoint_name):
        super().__init__(key)
        self.checkpoint_name = checkpoint_name or ''
        self._resolved_path = checkpoint_name
        self.source_checkpoint = checkpoint_name

    def merge(self) -> torch.Tensor:
        """
        Load a single tensor from the specified checkpoint.
        In kitchen-sink mode: just load it — let initialize_task() handle resizing.
        """
        if cmn.loaded_checkpoints is None:
            raise RuntimeError("Checkpoints not loaded")

        # Find the correct file
        file = None
        for loaded_key in cmn.loaded_checkpoints.keys():
            try:
                if self.checkpoint_name and os.path.samefile(loaded_key, self.checkpoint_name):
                    file = cmn.loaded_checkpoints[loaded_key]
                    break
            except (OSError, FileNotFoundError):
                continue

        # Fallback hierarchy — THE SACRED ORDER
        if file is None:
            # 1. Primary model (set by cross-arch logic)
            if cmn.primary and cmn.primary in cmn.loaded_checkpoints:
                file = cmn.loaded_checkpoints[cmn.primary]
            # 2. First global checkpoint (original order)
            elif cmn.checkpoints_global:
                first_cp = cmn.checkpoints_global[0]
                file = cmn.loaded_checkpoints.get(first_cp)
            # 3. Any loaded checkpoint
            elif cmn.loaded_checkpoints:
                file = next(iter(cmn.loaded_checkpoints.values()))

            if file is None:
                raise RuntimeError(f"No checkpoint available for tensor '{self.key}' (requested: {self.checkpoint_name})")

        # Load the tensor
        try:
            tensor = file.get_tensor(self.key)
        except SafetensorError:
            raise RuntimeError(f"Key {self.key} missing from {self.checkpoint_name} — handled by recovery")

        # ← FINAL FIX: Move to device + dtype
        tensor = tensor.to(device=cmn.get_device(), dtype=cmn.get_dtype())

        # Scalar tensor handling
        if tensor.ndim == 0:
            return tensor

        return tensor

# === BASIC OPERATORS (fixed indentation) ===
class Add(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        # Filter out empty / all-zero tensors (your original logic — keep it)
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # First valid tensor becomes the shape reference
        result = valid[0].clone()           # ← clone so we don't modify cached tensor
        target_shape = result.shape

        for t in valid[1:]:
            if t.shape == target_shape:
                result = result + t
            elif cmn.is_cross_arch:
                # ← THIS IS THE MAGIC LINE THAT FIXES CROSS-ARCH
                resized = SmartResize(self.key, target_shape, t).oper(t)
                result = result + resized
                merge_stats.smart_resized += 1
            else:
                # Same-arch mismatch = real bug → let it crash (old behavior)
                result = result + t

        return result.to(cmn.get_dtype())

class Sub(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # First valid tensor defines the target shape
        result = valid[0].clone()           # ← clone to avoid modifying cached tensor
        target_shape = result.shape

        for t in valid[1:]:
            if t.shape == target_shape:
                result = result - t
            elif cmn.is_cross_arch:
                # ← AUTOMATIC SMART RESIZE FOR CROSS-ARCH
                resized = SmartResize(self.key, target_shape, t).oper(t)
                result = result - resized
                merge_stats.smart_resized += 1
                print(f"[Sub] SmartResize applied → {self.key} ({t.shape} → {target_shape})")
            else:
                # Same-arch shape mismatch = real error → let it fail loudly (original behavior)
                result = result - t

        return result.to(cmn.get_dtype())


class Multiply(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        result = valid[0].clone()           # Clone to prevent in-place mutation of cache
        target_shape = result.shape

        for t in valid[1:]:
            if t.shape == target_shape:
                result = result * t
            elif cmn.is_cross_arch:
                # Auto-resize mismatched tensor to target shape
                resized = SmartResize(self.key, target_shape, t).oper(t)
                result = result * resized
                merge_stats.smart_resized += 1
                # Optional debug (remove if too verbose)
                # print(f"[Multiply] SmartResize applied → {self.key} ({t.shape} → {target_shape})")
            else:
                # Same-arch mismatch = real bug → crash loudly (original behavior)
                result = result * t

        return result.to(cmn.get_dtype())


class MultiplyTensors(Operation):
    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        result = valid[0].clone()
        target_shape = result.shape

        for t in valid[1:]:
            if t.shape == target_shape:
                result = result * t
            elif cmn.is_cross_arch:
                resized = SmartResize(self.key, target_shape, t).oper(t)
                result = result * resized
                merge_stats.smart_resized += 1
            else:
                result = result * t

        return result.to(cmn.get_dtype())

class Extract(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0] if len(valid) >= 1 else None
        a = valid[1] if len(valid) >= 2 else valid[0]
        b = valid[2] if len(valid) >= 3 else valid[0]

        # Determine target shape — base wins, otherwise first tensor
        target_shape = base.shape if base is not None else a.shape
        dtype = base.dtype if base is not None else a.dtype

        # Auto-resize with SmartResize + stat tracking
        def resize_if_needed(tensor, name):
            if tensor.shape != target_shape:
                resized = SmartResize(f"{name}_{self.key}", target_shape, tensor).oper(tensor)
                merge_stats.smart_resized += 1
                return resized
            return tensor

        a = resize_if_needed(a, "extract_a")
        b = resize_if_needed(b, "extract_b")

        base_val = base.float() if base is not None else torch.zeros_like(a.float())
        a_f = (a.float() - base_val).contiguous()
        b_f = (b.float() - base_val).contiguous()

        c = torch.cosine_similarity(a_f, b_f, dim=-1).clamp(-1, 1).unsqueeze(-1)
        d = ((c + 1) / 2) ** self.gamma
        result = torch.lerp(a_f, b_f, self.alpha) * torch.lerp(d, 1 - d, self.beta)

        return result.to(dtype)


class Similarities(Extract):
    def __init__(self, key, alpha, beta, gamma, a, b):
        super().__init__(key, alpha, beta, gamma, a, b)

    @multi_cache_operation
    def oper(self, *tensors):
        # ONE TRUE FILTER — keep your excellent filtering
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # Pure similarity mode → no base model
        a = valid[0]
        b = valid[1] if len(valid) > 1 else valid[0]

        # Let Extract handle everything — including SmartResize
        # We pass None as base → forces pure similarity logic + correct target shape from a
        return super().oper(None, a, b)


class ReBasin(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, *tensors):
        # ← ONE TRUE FILTER — safe for any number of inputs
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # ← ReBasin is always 2-model — take first two valid
        a = valid[0]
        b = valid[1] if len(valid) > 1 else valid[0]

        # Fast permutation alignment via sorting (2025 practical Re-Basin)
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
        # ← ONE TRUE FILTER — safe for everything
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # ← DeMe is strictly 2-model — take first two
        a = valid[0]
        b = valid[1] if len(valid) > 1 else valid[0]

        # Timestep-decoupled merge (2024 ICLR paper) — variance selection
        var_a = torch.var(a, dim=-1, keepdim=True)
        var_b = torch.var(b, dim=-1, keepdim=True)
        decoupled = torch.where(var_a > var_b, a, b)

        return (1 - self.alpha) * a + self.alpha * decoupled


class BlockWeighted(Operation):
    def __init__(self, key, alphas, a, b):
        super().__init__(key, a, b)
        self.alphas = alphas  # list of 12+ values (one per block)

    @multi_cache_operation
    def oper(self, *tensors):
        # ← ONE TRUE FILTER — safe for everything
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # ← BlockWeighted is 2-model — take first two
        a = valid[0]
        b = valid[1] if len(valid) > 1 else valid[0]

        # Extract block index from key: input_blocks.3, output_blocks.7, etc.
        match = re.search(r'\.(\d+)\.', self.key)
        alpha = self.alphas[min(int(match.group(1)), len(self.alphas)-1)] if match else self.alphas[0]

        return (1 - alpha) * a + alpha * b


class ToMe(Operation):
    def __init__(self, key, ratio, tensor):
        super().__init__(key, tensor)
        self.ratio = float(ratio)

    @multi_cache_operation
    def oper(self, *tensors):
        # ← ONE TRUE FILTER — safe for everything
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            tensor = valid[0]
        else:
            # ToMe is single-tensor → take first valid
            tensor = valid[0]

        if tensor.ndim < 2 or tensor.numel() < 10:
            return tensor

        # Fast ToMe (Token Merging) — CVPR 2023
        normed = F.normalize(tensor, dim=-1)
        sim = normed @ normed.T  # [N, N]
        k = max(2, int(tensor.size(0) * self.ratio))
        _, indices = torch.topk(sim, k, dim=1)  # [N, k]

        # Vectorized merge (faster than loop)
        merged = tensor[indices].mean(dim=1)  # [N, D]

        return merged

class AttentionMerge(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, *tensors):
        # ← ONE TRUE FILTER 
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        # ← AttentionMerge is 2-model — take first two
        a = valid[0]
        b = valid[1] if len(valid) > 1 else valid[0]

        # Only merge attention-related layers (Q, K, V, proj, etc.)
        if 'attention' in self.key.lower() or 'attn' in self.key.lower():
            return (1 - self.alpha) * a + self.alpha * b

        # Everything else stays from Model A
        return a


class Smooth(Operation):
    def __init__(self, key, tensor):
        super().__init__(key, tensor)

    @multi_cache_operation
    def oper(self, *tensors):
        # Take first valid tensor (should always be exactly one)
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        tensor = valid[0]

        if tensor.numel() < 5:
            return tensor
        device, dtype = tensor.device, tensor.dtype
        kernel_size, sigma = 5, 1.0
        center = kernel_size // 2

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)  # [1,1,5]

        orig_shape = tensor.shape

        # Flatten and prepare for conv1d
        x = tensor.flatten().unsqueeze(0).unsqueeze(0)  # [1,1,N]
        x = F.pad(x, (center, center), mode='replicate')
        smoothed = F.conv1d(x, kernel)                  # [1,1,N]
        smoothed = smoothed.squeeze(0).squeeze(0)       # [N] — safe squeeze

        # Restore original shape — SAFE for all cases
        return smoothed.view(orig_shape).to(dtype)

class SmoothConv(Operation):
    """
    Smart hybrid smoothing:
    • Conv2d weights (4D) → 2D Gaussian (beautiful, edge-preserving)
    • Linear / attention / other (1D/2D) → 1D Gaussian (clean)
    • Everything else → untouched
    • Zero-safe, cross-arch safe, kitchen-sink safe
    • Immortal
    """
    def __init__(self, key, sigma=1.0, kernel_size=None, tensor=None):
        super().__init__(key, tensor)
        self.sigma = float(sigma)
        self.kernel_size = kernel_size or max(3, int(4 * sigma + 1) | 1)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1  # Force odd size

    @multi_cache_operation
    def oper(self, *tensors):
        # ← ONE TRUE FILTER — the shield of the gods
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if not valid:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        tensor = valid[0]

        # 4D Conv2d weights → 2D Gaussian smoothing
        if tensor.ndim == 4 and tensor.shape[2] >= 3 and tensor.shape[3] >= 3:
            return self._smooth_2d(tensor)

        # 2D/1D tensors (Linear, attention, embeddings) → 1D Gaussian
        if tensor.ndim >= 2 and tensor.numel() >= 10:
            return self._smooth_1d(tensor)

        # Everything else (small, scalar, metadata) → pass through
        return tensor

    def _smooth_2d(self, tensor):
        device, dtype = tensor.device, tensor.dtype
        size = self.kernel_size
        center = size // 2

        # Create 1D Gaussian
        x = torch.arange(size, device=device, dtype=dtype)
        kernel_1d = torch.exp(-0.5 * ((x - center) ** 2) / (self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Make 2D separable kernel: outer product
        kernel_2d = kernel_1d.unsqueeze(0).T @ kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.view(1, 1, size, size)

        orig_shape = tensor.shape
        h, w = orig_shape[2], orig_shape[3]

        # Reshape to [in_channels * out_channels, 1, h, w]
        x = tensor.permute(1, 0, 2, 3).reshape(-1, 1, h, w)

        pad = size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')

        # Depthwise conv: each channel smoothed independently
        smoothed = F.conv2d(x, kernel_2d.repeat(x.shape[0], 1, 1, 1), groups=x.shape[0])

        # Restore original shape: [out, in, h, w]
        smoothed = smoothed.view(orig_shape[1], orig_shape[0], h, w)
        smoothed = smoothed.permute(1, 0, 2, 3)

        return smoothed.to(dtype)

    def _smooth_1d(self, tensor):
        # Reuse your battle-tested 1D Smooth — perfect
        return Smooth(self.key, tensor=tensor).oper(tensor)

class TIES(Operation):
    def __init__(self, key, *sources, density, seed=42):
        super().__init__(key, *sources)
        self.density = float(density)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        if self.density <= 0.0:
            return base
        if self.density >= 1.0:
            deltas = [t - base for t in others]
            norms = [d.norm(p=2) for d in deltas]
            return others[torch.argmax(torch.stack(norms))]

        torch.manual_seed(self.seed + hash(self.key) % (2**32 - 1))

        # ← CROSS-ARCH SAFETY: Only use tensors with same shape
        same_shape_others = [t for t in others if t.shape == base.shape]
        if not same_shape_others:
            return base

        deltas = [t - base for t in same_shape_others]
        if not deltas:
            return base

        # Stack and find strongest signal
        abs_deltas = torch.stack([d.abs() for d in deltas], dim=0)
        max_magnitude, best_source = torch.max(abs_deltas, dim=0)

        # Top-k
        k = max(1, int(self.density * max_magnitude.numel()))
        threshold = torch.topk(max_magnitude.flatten(), k).values[-1]
        mask = max_magnitude >= threshold

        # ← FINAL FIX: Safe indexing
        if best_source.ndim == 0:
            best_source_idx = best_source.item()
        else:
            # For 1D tensors (bias), take first element
            best_source_idx = best_source.flatten()[0].item()

        winning_delta = deltas[best_source_idx]

        # Rest of TIES logic (unchanged)
        sign = torch.sign(winning_delta)
        resolved_sign = torch.where(sign == 0, torch.sign(base), sign)
        elected_delta = winning_delta * mask.to(winning_delta.dtype) * resolved_sign

        result = torch.where(mask & (torch.sign(base) == resolved_sign), base, base + elected_delta)

        if elected_delta.norm(p=2) > 1e-8:
            scale = winning_delta.norm(p=2) / (elected_delta.norm(p=2) + 1e-8)
            result = base + elected_delta * scale.clamp_max(10.0)

        return result.to(base.dtype)

class DARE(Operation):
    """
    The One True DARE — works with 2, 3, 4, 10, 100 models.
    DARE (2-way): strongest signal from first contributor
    DARE3/N-way: strongest signal from ANY contributor
    Unified. Perfect. Immortal.
    """
    def __init__(self, key, density, dropout_p=0.3, seed=42, *sources):
        super().__init__(key, *sources)
        self.density = float(density)
        self.dropout_p = float(dropout_p)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        contributors = valid[1:]

        if self.density <= 0.0:
            return base
        if self.density >= 1.0:
            # Pick strongest contributor by L2 norm of delta
            deltas = [t - base for t in contributors]
            norms = torch.stack([d.norm(p=2) for d in deltas])
            strongest = contributors[torch.argmax(norms)]
            return strongest

        torch.manual_seed(self.seed + hash(self.key) % (2**32 - 1))

        # Compute deltas from base
        deltas = [t - base for t in contributors]
        if not deltas:
            return base

        # Stack absolute deltas → find strongest signal from ANY contributor
        abs_deltas = torch.stack([d.abs() for d in deltas], dim=0)
        max_magnitude, best_source = torch.max(abs_deltas, dim=0)

        # Top-k selection
        k = max(1, int(self.density * max_magnitude.numel()))
        threshold = torch.topk(max_magnitude.flatten(), k).values.min()
        mask = max_magnitude >= threshold

        # Pick winning delta from any contributor
        winning_delta = deltas[0]
        for i, delta in enumerate(deltas):
            winning_delta = torch.where(best_source == i, delta, winning_delta)

        # Dropout
        if self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            dropout_mask = torch.bernoulli(torch.full_like(mask.float(), keep)).bool()
            mask = mask & dropout_mask

        # Random scaling
        scale = torch.ones_like(winning_delta)
        if mask.any():
            scale[mask] = torch.empty(mask.sum(), device=scale.device).uniform_(0.5, 2.0)

        dared_delta = winning_delta * mask.to(winning_delta.dtype) * scale

        # L2 energy preservation from all contributors
        total_energy = sum(d.norm(p=2) for d in deltas)
        if dared_delta.norm(p=2) > 1e-8:
            dared_delta *= total_energy / (dared_delta.norm(p=2) + 1e-8)

        return (base + dared_delta).to(base.dtype)

class SLERP(Operation):
    """
    True N-way spherical linear interpolation on the hypersphere.
    Replaces SLERP (2-way) and SLERP3 (3-way) forever.
    Works with 2, 3, 4, 10, 100 models.
    Mathematically perfect.
    Immortal.
    """
    def __init__(self, key, weights, *sources):
        """
        weights: list[float] — weights for each source (including base)
                 Must sum to <= 1.0, remainder goes to base
                 e.g. [0.3, 0.2] → 0.5 to base, 0.3 to model B, 0.2 to model C
        """
        super().__init__(key, *sources)
        total = sum(weights)
        if total > 1.0:
            weights = [w / total for w in weights]
            total = 1.0
        # Pad with zero weights if fewer than sources
        self.weights = weights + [0.0] * (len(sources) - len(weights))
        self.base_weight = 1.0 - total

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # Flatten
        base_flat = base.flatten()
        others_flat = [t.flatten() for t in others]

        # Log map all contributors to tangent space at base
        def log_map(x, base_vec):
            cos_theta = (x @ base_vec) / (x.norm() * base_vec.norm() + 1e-8)
            cos_theta = cos_theta.clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)
            if theta < 1e-6:
                return torch.zeros_like(x)
            return (x - cos_theta * base_vec) * (theta / torch.sin(theta))

        log_contribs = [log_map(vec, base_flat) for vec in others_flat]

        # Weighted sum in tangent space
        log_merged = torch.zeros_like(base_flat)
        for log_vec, weight in zip(log_contribs, self.weights):
            log_merged += weight * log_vec

        # Add base contribution (always included)
        log_merged = self.base_weight * torch.zeros_like(log_merged) + log_merged

        # Exp map back
        norm = torch.norm(log_merged)
        if norm < 1e-6:
            return base

        exp_merged = torch.cos(norm) * base_flat + torch.sin(norm) * (log_merged / norm)
        return exp_merged.view_as(base).to(base.dtype)


class TrainDiff(Operation):
    def __init__(self, key, a, b, c, *extra_sources):
        super().__init__(key, a, b, c, *extra_sources)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # Compute all deltas from base
        deltas = [other - base for other in others]
        if not deltas:
            return base

        # ← THE TOP-K ENHANCEMENT — THIS IS WHERE IT GOES
        delta_norms = torch.stack([d.norm(p=2) for d in deltas])
        k = min(3, len(deltas))  # top 3 strongest changes
        top_k_indices = torch.topk(delta_norms, k).indices
        selected_deltas = [deltas[i] for i in top_k_indices]

        combined_delta = torch.mean(torch.stack(selected_deltas), dim=0)
        combined_delta = combined_delta - combined_delta.mean()  # zero-center

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
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        alpha = max(self.alpha, 0.001)

        # Compute absolute differences from base
        deltas = [torch.abs(other - base) for other in others]
        if not deltas:
            return base

        # Max delta across all contributors
        delta_max = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(delta_max == 0):
            return base  # all identical

        # Per-contributor normalized difference
        if self.beta != 1:
            # Similarity mode: higher = more similar
            diff = ((delta_max - delta_max) / (delta_max + 1e-8)) ** (1 / alpha - 1)
        else:
            # Difference mode: higher = more different
            diff_per = [delta / (delta_max + 1e-8) for delta in deltas]
            # Take the strongest difference signal from any model
            diff = torch.max(torch.stack(diff_per), dim=0).values ** (1 / alpha - 1)

        diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Key-specific seeded Bernoulli
        rng = torch.Generator(device=diff.device)
        rng.manual_seed(self.seed + hash(self.key) % (2**32 - 1))
        bitmask = torch.bernoulli(torch.clamp(diff, 0.0, 1.0), generator=rng)

        # Smooth interpolation
        mask = torch.lerp(bitmask, diff, self.gamma)

        # Apply strongest contributor where mask is high
        result = base
        for other in others:
            result = torch.where(mask > 0.5, other, result)  # or use mask directly

        # Or: linear blend with mask strength
        result = base * (1 - mask) + result * mask

        return result.to(base.dtype)

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
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        alpha = max(self.alpha, 0.001)

        # Compute absolute differences from base
        deltas = [torch.abs(other - base) for other in others]
        if not deltas:
            return base

        delta_max = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(delta_max == 0):
            return base

        # Difference mode: higher = more different
        # Similarity mode: higher = more similar
        if self.beta != 1:
            # Similarity: invert difference
            diff = ((delta_max - delta_max) / (delta_max + 1e-8)) ** (1 / alpha - 1)
        else:
            diff_per = [delta / (delta_max + 1e-8) for delta in deltas]
            diff = torch.max(torch.stack(diff_per), dim=0).values ** (1 / alpha - 1)

        diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Seeded Bernoulli + smooth interpolation
        rng = torch.Generator(device=diff.device)
        rng.manual_seed(self.seed + hash(self.key) % (2**32 - 1))
        bitmask = torch.bernoulli(torch.clamp(diff, 0.0, 1.0), generator=rng)
        mask = torch.lerp(bitmask, diff, self.gamma)

        # ← FINAL N-WAY WEIGHTED BLEND — THIS IS WHERE IT GOES
        weighted_sum = base * (1.0 - mask)
        total_weight = 1.0 - mask

        for other in others:
            weighted_sum += other * mask
            total_weight += mask

        result = weighted_sum / (total_weight + 1e-8)
        return result.to(base.dtype)
    
class AutoEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)      # Interpolation strength
        self.beta = float(beta)         # Threshold adjustment factor
        self.gamma = float(gamma)       # Smoothness factor
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # Compute absolute differences from base
        deltas = [torch.abs(other - base) for other in others]
        if not deltas:
            return base

        # Global max delta across all contributors
        max_delta = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(max_delta == 0):
            return base

        # Normalize: 1 = no difference, 0 = max difference
        diff_per = [(max_delta - delta) / (max_delta + 1e-8) for delta in deltas]
        diff = torch.max(torch.stack(diff_per), dim=0).values
        diff = torch.nan_to_num(diff, nan=0.0)

        # Global mean difference — adaptive center
        mean_diff = torch.mean(diff)

        # Dynamic thresholds — your genius
        lower_threshold = mean_diff * (1 - self.beta)
        upper_threshold = mean_diff * (1 + self.beta)

        # Keep values near the mean difference
        mask = torch.logical_and(lower_threshold < diff, diff < upper_threshold)

        # Power curve shaping
        alpha_safe = max(self.alpha, 0.001)
        powered_diff = diff ** (1 / alpha_safe - 1)
        powered_diff = torch.nan_to_num(powered_diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Apply mask
        masked_diff = powered_diff * mask.float()

        # Seeded random mask
        rng = torch.Generator(device=base.device)
        rng.manual_seed(self.seed + hash(self.key) % (2**32 - 1))
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0.0, 1.0), generator=rng)

        # Final smooth interpolation
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.gamma)

        # ← N-WAY WEIGHTED BLEND — The Final Form
        weighted_sum = base * (1.0 - interpolated_mask)
        total_weight = 1.0 - interpolated_mask

        for other in others:
            weighted_sum += other * interpolated_mask
            total_weight += interpolated_mask

        result = weighted_sum / (total_weight + 1e-8)
        return result.to(base.dtype)

class SingularValueDeOperator(Operation):
    def __init__(self, key, alpha, beta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)      # Threshold multiplier (0.01–0.1)
        self.beta = float(beta)         # Top-k fraction to keep (0.1–0.5)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # Only works on 2D+ tensors
        if base.ndim < 2:
            return base

        try:
            # Compute all differences from base
            diffs = [other - base for other in others]
            if not diffs:
                return base

            # Stack and compute SVD on the *sum* of differences
            # This captures shared interference across all models
            total_diff = torch.sum(torch.stack(diffs), dim=0)

            # SVD on the combined difference
            U, S, Vh = torch.linalg.svd(total_diff, full_matrices=False)

            # Threshold: significant singular values
            threshold = self.alpha * S.max()
            significant = S > threshold

            # Keep top-k fraction
            if self.beta < 1.0:
                k = max(1, int(self.beta * len(S)))
                top_k = torch.zeros_like(significant)
                top_k[:k] = True
                significant = significant & top_k

            # Zero out noise
            S_filtered = S * significant.float()

            # Reconstruct cleaned difference
            reconstructed = torch.matmul(U, torch.matmul(torch.diag(S_filtered), Vh))

            # Add back to base
            result = base + reconstructed

            return result.to(base.dtype)

        except Exception as e:
            # Nuclear-grade fallback
            print(f"[SVD] Failed on {self.key}: {e} — using base")
            return base


class TensorExchange(Operation):
    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)    # Probability to pick a random other model
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # Key + seed deterministic randomness
        import hashlib
        hash_input = f"{self.key}{self.seed}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        choice_val = (hash_val % 10000) / 10000.0

        # With probability alpha, pick a random other model
        if choice_val < self.alpha and others:
            # Pick one contributor deterministically
            choice_idx = hash_val % len(others)
            return others[choice_idx]

        # Otherwise keep base
        return base


class WeightSumCutoff(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)      # Merge strength
        self.beta = float(beta)         # Lower similarity threshold
        self.gamma = float(gamma)       # Upper similarity threshold

    @multi_cache_operation
    def oper(self, *tensors):
        valid = [t for t in tensors if t.numel() > 0 and torch.any(t != 0)]
        if len(valid) == 0:
            return torch.zeros([], dtype=cmn.get_dtype(), device=cmn.get_device())
        if len(valid) == 1:
            return valid[0]

        base = valid[0]
        others = valid[1:]

        # Compute max absolute difference across all contributors
        deltas = [torch.abs(other - base) for other in others]
        if not deltas:
            return base

        delta_max = torch.max(torch.stack(deltas), dim=0).values
        if torch.all(delta_max == 0):
            return base

        # Normalized similarity: 1 = identical, 0 = max difference
        diff = (delta_max - delta_max) / (delta_max + 1e-8)
        diff = torch.nan_to_num(diff, nan=0.0)

        # Mean similarity per channel
        mean_sim = torch.mean(diff, dim=0, keepdim=True)  # [1, C]

        # Keep channels where similarity is BETWEEN beta and gamma
        mask = (mean_sim > self.beta) & (mean_sim < self.gamma)
        mul = self.alpha * mask.float()

        # ← FINAL N-WAY BLEND — ONLY ON SELECTED CHANNELS
        result = base.clone()
        if mask.any():
            weighted = torch.zeros_like(base)
            count = torch.zeros_like(base)

            for other in others:
                contrib = other * mul
                weighted += contrib
                count += mul

            # Blend only on masked channels: weighted average of contributors + base
            result = torch.where(mask, weighted / (count + 1e-8) + base * (1 - mul), result)

        return result.to(base.dtype)

class WeightsCache:
    def __init__(self, size):
        self.mapping = OrderedDict()
        self.size_cap = min(size, 8192)*1024*1024
        self.size = 0

    def __setitem__(self, key, t):
        if key in self.mapping:
            self.mapping.move_to_end(key)
        else:
            t = t.detach().cpu()
            self.mapping[key] = t
            self.size += tensor_size(t)
            while self.size >= self.size_cap:
                _ , tensor = self.mapping.popitem(last=False)
                self.size -= tensor_size(tensor)

    def __getitem__(self, key: Operation) -> torch.Tensor:
        t = self.mapping[key]
        self.mapping.move_to_end(key)
        return t.clone().to(cmn.get_device()).type(cmn.get_dtype())
    

weights_cache = WeightsCache(4096)

class SmartResize(Operation):
    def __init__(self, key, target_shape, source_tensor=None):
        super().__init__(key)
        self.target_shape = target_shape
        self.source_tensor = source_tensor

    @multi_cache_operation
    def oper(self, *tensors):
        if not tensors:
            return torch.zeros(self.target_shape, dtype=cmn.get_dtype(), device=cmn.get_device())
        t = tensors[0]

        device = t.device
        dtype = t.dtype

        # === SAFETY FIRST ===
        if not self.target_shape or len(self.target_shape) == 0 or any(s > 100000 for s in self.target_shape):
            return t
        if t.numel() == 0:
            return torch.zeros(self.target_shape, device=device, dtype=dtype)

        target = self.target_shape
        key = self.key.lower()

        # ===================================================================
        # SACRED SANCTUARY — These must NEVER be resized or interpolated
        # ===================================================================
        SANCTUARY = {
            # ——— NOISE & TIME ENTRY (breaks first step, haze, refiner death) ———
            "conv_in.weight", "conv_in.bias",
            "input_blocks.0.0.weight", "input_blocks.0.0.bias",
            "time_embed.0.weight", "time_embed.0.bias",
            "time_embed.2.weight", "time_embed.2.bias",
            "time_embedding", "timestep_embed", "timestep_embedding",
            "time_in.weight", "time_in.bias",
            "vector_in.weight", "vector_in.bias",
            "img_in.weight", "img_in.bias",
            "offset_noise", "noise_offset", "noise_augmentor",
            "learned_sigma", "sigma_embed",

            # ——— VAE (encoder/decoder) — NEVER interpolate latents ———
            "first_stage_model.", "encoder.", "decoder.", "quant_conv.", "post_quant_conv.",
            "vae.",

            # ——— CLIP / TEXT ENCODERS — Token embeddings & positional ———
            "cond_stage_model.transformer.text_model.embeddings",
            "conditioner.embedders.0.transformer.text_model.embeddings",
            "conditioner.embedders.1.transformer.text_model.embeddings",
            "token_embedding", "position_embedding", "positional_embedding",
            "text_projection", "logit_scale",

            # ——— FLUX / SD3 / SPECIAL BLOCKS ———
            "single_blocks", "double_blocks", "img_proj", "txt_proj",
            "x_embedder", "context_embedder", "t_embedder",

            # ——— POS EMBEDDINGS (any model) ———
            "pos_embed", "position_emb", "position_ids",
        }

        # Fast prefix/suffix check
        if any(pat in key for pat in SANCTUARY):
            # Sacred tensor — preserve exactly
            if t.shape == target:
                return t.to(dtype)
            else:
                # Pad or truncate — NEVER interpolate
                pad_total = [max(0, tgt - src) for tgt, src in zip(target, t.shape)]
                if any(pad_total):
                    pad_before = [p // 2 for p in pad_total]
                    pad_after = [p - p//2 for p in pad_total]
                    pad_tuple = tuple(p for pair in zip(pad_before[::-1], pad_after[::-1]) for p in pair)
                    t = F.pad(t, pad_tuple)
                slices = tuple(slice(0, s) for s in target)
                result = t[slices]
                merge_stats.smart_resized += 1
                print(f"[SANCTUARY] Preserved sacred tensor: {self.key} → {result.shape}")
                return result.to(dtype)

        # ===================================================================
        # NORMAL SMART RESIZE — Everything else gets full treatment
        # ===================================================================

        # 1D tensors (e.g. biases, norms)
        if t.ndim == 1:
            if len(target) >= 1 and t.shape[0] != target[0]:
                t = F.interpolate(t.unsqueeze(0).unsqueeze(0), size=(target[0],), mode='linear', align_corners=False)[0, 0]
            return t.to(dtype)

        # 2D tensors — Linear layers, attention, FFN
        if t.ndim == 2:
            if len(target) < 2 or target[0] >= 100000 or target[1] >= 100000:
                return t

            # Token embeddings (e.g. 49408,768 → 49408,1280)
            if target[0] > 20000:
                new_t = torch.zeros(target, device=device, dtype=dtype)
                min_rows = min(t.shape[0], target[0])
                for i in range(min_rows):
                    row = t[i:i+1]
                    resized = F.interpolate(row.unsqueeze(0).unsqueeze(0), size=(target[1],), mode='linear', align_corners=False)[0, 0]
                    new_t[i] = resized
                merge_stats.smart_resized += 1
                return new_t.to(dtype)
            else:
                result = F.interpolate(t.unsqueeze(0).unsqueeze(0), size=target, mode='bilinear', align_corners=False)[0, 0]
                merge_stats.smart_resized += 1
                return result.to(dtype)

        # 3D tensors (positional embeddings, etc.)
        if t.ndim == 3:
            if len(target) >= 3 and all(s < 100000 for s in target):
                result = F.interpolate(t.unsqueeze(0), size=target, mode='trilinear', align_corners=False).squeeze(0)
                merge_stats.smart_resized += 1
                return result.to(dtype)

        # 4D tensors — Conv2d weights
        if t.ndim == 4:
            cout = target[0] if len(target) >= 4 else t.shape[0]
            cin = target[1] if len(target) >= 4 else t.shape[1]
            h, w = target[2:4] if len(target) >= 4 else t.shape[2:]

            if h > 1000 or w > 1000 or cout > 10000 or cin > 10000:
                return t

            if (h, w) != t.shape[2:]:
                t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)

            if t.shape[0] != cout or t.shape[1] != cin:
                if t.shape[0] != cout:
                    if t.shape[0] > cout:
                        t = t[:cout]
                    else:
                        pad = torch.zeros((cout - t.shape[0], t.shape[1], h, w), device=device, dtype=dtype)
                        t = torch.cat([t, pad], dim=0)
                if t.shape[1] != cin:
                    if t.shape[1] > cin:
                        t = t[:, :cin]
                    else:
                        pad = torch.zeros((t.shape[0], cin - t.shape[1], h, w), device=device, dtype=dtype)
                        t = torch.cat([t, pad], dim=1)
            merge_stats.smart_resized += 1
            return t.to(dtype)

        # Final fallback — pad + slice
        pad_total = [max(0, tgt - src) for tgt, src in zip(target, t.shape)]
        if any(pad_total):
            pad_before = [p // 2 for p in pad_total]
            pad_after = [p - p//2 for p in pad_total]
            pad_tuple = tuple(p for pair in zip(pad_before[::-1], pad_after[::-1]) for p in pair)
            t = F.pad(t, pad_tuple)
        slices = tuple(slice(0, s) for s in target)
        result = t[slices]
        merge_stats.smart_resized += 1
        return result.to(dtype)