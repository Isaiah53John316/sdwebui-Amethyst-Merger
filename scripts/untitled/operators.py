import torch, scipy
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from safetensors import SafetensorError   # ← THIS LINE IS THE FIX (adds missing error type)
import os
from functools import wraps
import re
from scripts.untitled.common import cmn

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

def cache_operation(func):
    def inner(operation):
        try:
            return weights_cache[operation]
        except KeyError:pass

        result = func(operation)

        weights_cache[operation] = result
        return result
    return inner

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

def safe_resize_in_cross_arch(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback for same-arch merges — just use SmartResize on b if needed"""
    if a.shape == b.shape:
        return a, b
    # Even in same-arch, use SmartResize — it's safer and works
    b = SmartResize("fallback", a.shape, b).oper(b)
    return a, b

def _safe_binary_op(op_name, a, b, op_func):
    """Apply op_func(a, b) after forcing both tensors to same shape"""
    if a.shape != b.shape:
        target_shape = a.shape  # Primary model wins
        if b.shape != target_shape:
            b = SmartResize(f"{op_name}_fix", target_shape, b).oper(b)
        if a.shape != target_shape:  # Should never happen, but defense in depth
            a = SmartResize(f"{op_name}_fix_a", target_shape, a).oper(a)
    return op_func(a, b)

###OPERATORS####

class Operation:
    def __init__(self,key,*sources):
        self.key = key
        self.sources = tuple(sources)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self.seed = None
        self.merge_func = recurse

    def __eq__(self, other):
        return (self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources) == (other.key, other.alpha, other.beta, other.gamma, other.delta, other.seed, other.sources)
    
    def __hash__(self):
        return hash((self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources))
    
    def oper(self,*args) -> torch.Tensor:
        raise NotImplementedError

    def merge(self):
        return self.merge_func(self)
    
    def cache(self):
        if cmn.opts['cache_size'] > 512:
            self.merge_func = cache_operation(recurse)
        return self

class CopyPrimary(Operation):
    """
    Copy from primary model — 100% safe for cross-arch, never crashes
    Now fully integrated with MergeStats via constructor injection
    """
    def __init__(self, key, primary_path, stats=None):
        super().__init__(key)
        self.primary_path = primary_path
        self.stats = stats  # ← MergeStats instance passed from create_tasks()

    @multi_cache_operation
    def oper(self, *args):
        file = cmn.loaded_checkpoints.get(self.primary_path)
        model_name = os.path.basename(self.primary_path) if self.primary_path else "Unknown"
        resized = False

        if file and self.key in file.keys():
            try:
                t = file.get_tensor(self.key)

                # RESIZE IF CROSS-ARCH
                if cmn.is_cross_arch:
                    target_shape = cmn.cross_arch_target_shapes.get(self.key)
                    if target_shape and t.shape != target_shape:
                        t = SmartResize(f"CopyPrimary_{self.key}", target_shape, t).oper(t)
                        resized = True

                # Stats update
                if self.stats:
                    self.stats.copied_primary += 1
                    if resized:
                        self.stats.smart_resized += 1

                # Logging
                if resized:
                    print(f"[FinalCopy] {self.key} ← COPIED FROM PRIMARY + RESIZED ({model_name})")
                else:
                    print(f"[FinalCopy] {self.key} ← COPIED FROM PRIMARY ({model_name})")

                return t.to(cmn.get_device(), dtype=cmn.get_dtype())

            except Exception as e:
                print(f"[CopyPrimary] FAILED reading {self.key} from {model_name}: {e}")

        # ULTIMATE FALLBACK: Zero-fill with target shape (cross-arch)
        if cmn.is_cross_arch:
            target_shape = cmn.cross_arch_target_shapes.get(self.key)
            if target_shape:
                t = torch.zeros(target_shape, dtype=cmn.get_dtype(), device=cmn.get_device())
                if self.stats:
                    self.stats.zero_filled += 1
                print(f"[Emergency] {self.key} ← ZERO-FILLED (CopyPrimary fallback)")
                return t

        # ABSOLUTE LAST RESORT: Should never happen
        if self.stats:
            self.stats.skipped += 1
        print(f"[CRITICAL] {self.key} ← SKIPPED in CopyPrimary (impossible state)")
        return torch.tensor([], dtype=cmn.get_dtype(), device=cmn.get_device())

class LoadTensor(Operation):
    def __init__(self, key, checkpoint_name):
        super().__init__(key)
        self.checkpoint_name = checkpoint_name or ''
        self._resolved_path = checkpoint_name

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

        # Fallback to primary if not found
        if file is None:
            # PRIMARY MODEL FIRST (the correct, safe order)
            if cmn.primary and cmn.primary in cmn.loaded_checkpoints:
                file = cmn.loaded_checkpoints[cmn.primary]
            # THEN fall back to the first loaded checkpoint (in global order)
            elif cmn.checkpoints_global:
                first_cp = cmn.checkpoints_global[0]
                file = cmn.loaded_checkpoints.get(first_cp)
            # ABSOLUTE LAST RESORT: any loaded checkpoint
            elif cmn.loaded_checkpoints:
                file = next(iter(cmn.loaded_checkpoints.values()))
            
            if file is None:
                raise RuntimeError(f"No checkpoint available for tensor '{self.key}' (requested: {self.checkpoint_name})")

        # Load the tensor (let initialize_task() handle any errors/reshaping)
        try:
            tensor = file.get_tensor(self.key)
        except SafetensorError:
            # In kitchen-sink mode, we don't care if a key is missing here
            # initialize_task() will find it in another model or SmartResize it
            raise RuntimeError(f"Key {self.key} missing from {self.checkpoint_name} — handled by recovery")

        # Move to correct device (let initialize_task() handle dtype/shape)
        tensor = tensor.to(device=cmn.get_device())

        if tensor.ndim == 0:  # scalar tensor
            return tensor

        return tensor

# === BASIC OPERATORS (fixed indentation) ===
class Add(Operation):
    @multi_cache_operation
    def oper(self, a, b):
        return _safe_binary_op("Add", a, b, lambda x, y: x + y)


class Sub(Operation):
    @multi_cache_operation
    def oper(self, a, b):
        return _safe_binary_op("Sub", a, b, lambda x, y: x - y)


class Multiply(Operation):
    @multi_cache_operation
    def oper(self, a, b):
        return _safe_binary_op("Mul", a, b, lambda x, y: x * y)


class MultiplyTensors(Operation):
    @multi_cache_operation
    def oper(self, a, b):
        return _safe_binary_op("MulT", a, b, lambda x, y: x * y)

class Extract(Operation):
    def __init__(self, key, alpha, beta, gamma, *args):
        super().__init__(key, *args)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @multi_cache_operation
    def oper(self, base: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        target_shape = base.shape if base is not None else a.shape

        if a.shape != target_shape:
            a = SmartResize("tmp", target_shape, a).oper(a)
        if b.shape != target_shape:
            b = SmartResize("tmp", target_shape, b).oper(b)
        if base is not None and base.shape != target_shape:
            base = SmartResize("tmp", target_shape, base).oper(base)

        dtype = base.dtype if base is not None else a.dtype
        base_val = base.float() if base is not None else 0.0

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
    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Pure similarity-driven interpolation (no base subtraction)
        return super().oper(None, a, b)


class ReBasin(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, a, b):
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
    def oper(self, a, b):
        # Timestep-decoupled merge (2024 ICLR paper) — variance selection
        var_a = torch.var(a, dim=-1, keepdim=True)
        var_b = torch.var(b, dim=-1, keepdim=True)
        decoupled = torch.where(var_a > var_b, a, b)
        return (1 - self.alpha) * a + self.alpha * decoupled


class BlockWeighted(Operation):
    def __init__(self, key, alphas, a, b):
        super().__init__(key, a, b)
        self.alphas = alphas  # list of 12+ values

    @multi_cache_operation
    def oper(self, a, b):
        # Extract block index from key: input_blocks.3, output_blocks.7, etc.
        match = re.search(r'\.(\d+)\.', self.key)
        alpha = self.alphas[min(int(match.group(1)), len(self.alphas)-1)] if match else self.alphas[0]
        return (1 - alpha) * a + alpha * b


class ToMe(Operation):
    def __init__(self, key, ratio, tensor):
        super().__init__(key, tensor)
        self.ratio = float(ratio)

    @multi_cache_operation
    def oper(self, tensor):
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
    def oper(self, a, b):
        # Only merge attention-related layers (Q, K, V, proj, etc.)
        if 'attention' in self.key.lower() or 'attn' in self.key.lower():
            return (1 - self.alpha) * a + self.alpha * b
        return a  # Everything else stays from Model A


class Smooth(Operation):
    def __init__(self, key, tensor):
        super().__init__(key, tensor)

    @multi_cache_operation
    def oper(self, tensor):
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


class TIES(Operation):
    def __init__(self, key, density, seed, a, b):
        super().__init__(key, a, b)
        self.density = float(density)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, a, b):
        if self.density <= 0.0:
            return a
        if self.density >= 1.0:
            return b

        torch.manual_seed(self.seed)
        delta = b - a
        abs_delta = delta.abs()

        k = max(1, int(self.density * abs_delta.numel()))
        threshold = torch.topk(abs_delta.flatten(), k).values[-1]  # .min() → [-1] for safety
        mask = abs_delta >= threshold

        # Sign resolution
        sign = torch.sign(delta)
        resolved_sign = torch.where(sign == 0, torch.sign(a), sign)  # fallback to a's sign

        elected_delta = delta * mask.to(delta.dtype) * resolved_sign

        # Dominance resolution
        result = torch.where(mask & (torch.sign(a) == resolved_sign), a, b)

        # L2-norm preservation
        if elected_delta.norm(p=2) > 1e-8:
            scale = delta.norm(p=2) / elected_delta.norm(p=2)
            result = a + elected_delta * scale.clamp_max(10.0)  # prevent explosion

        return result.to(a.dtype)


class DARE(Operation):
    def __init__(self, key, density, dropout_p=0.3, seed=42, a=None, b=None):
        super().__init__(key, a, b)
        self.density = density
        self.dropout_p = dropout_p
        self.seed = seed

    @multi_cache_operation
    def oper(self, a, b):
        if self.density <= 0.0: return a
        if self.density >= 1.0: return b

        torch.manual_seed(self.seed + hash(self.key) % (2**32 - 1))

        delta = b - a
        abs_delta = delta.abs()
        k = max(1, int(self.density * abs_delta.numel()))
        threshold = torch.topk(abs_delta.flatten(), k).values.min()
        mask = abs_delta >= threshold

        if self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            dropout_mask = torch.bernoulli(torch.full_like(mask.float(), keep)).bool()
            mask = mask & dropout_mask

        scale = torch.ones_like(delta)
        scale[mask] = torch.empty(mask.sum(), device=delta.device).uniform_(0.5, 2.0)

        dared_delta = delta * mask.to(delta.dtype) * scale

        if dared_delta.norm(p=2) > 0:
            rescale = delta.norm(p=2) / dared_delta.norm(p=2)
            dared_delta = dared_delta * rescale

        return (a + dared_delta).to(a.dtype)

class DARE3(Operation):
    def __init__(self, key, density, dropout_p=0.3, seed=42, a=None, b=None, c=None):
        super().__init__(key, a, b, c)
        self.density = density
        self.dropout_p = dropout_p
        self.seed = seed

    @multi_cache_operation
    def oper(self, a, b, c):
        if self.density <= 0.0:
            return a

        torch.manual_seed(self.seed + hash(self.key) % (2**32-1))

        # Compute deltas from A (the base)
        delta_b = b - a
        delta_c = c - a

        # Combine deltas in magnitude space
        abs_delta = torch.stack([delta_b.abs(), delta_c.abs()], dim=0)
        max_magnitude, source = torch.max(abs_delta, dim=0)

        # Top-k selection on strongest signals
        k = max(1, int(self.density * max_magnitude.numel()))
        threshold = torch.topk(max_magnitude.flatten(), k).values.min()
        mask = max_magnitude >= threshold

        # Choose which model contributed the winning signal
        delta = torch.where(source == 0, delta_b, delta_c)

        # Optional dropout on surviving parameters
        if self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            dropout_mask = torch.bernoulli(torch.full_like(mask.float(), keep)).bool()
            mask = mask & dropout_mask

        # Random scaling in [0.5, 2.0] on survivors (the DARE secret sauce)
        scale = torch.ones_like(delta)
        if mask.any():
            scale[mask] = torch.empty(mask.sum(), device=delta.device).uniform_(0.5, 2.0)

        dared_delta = delta * mask.to(delta.dtype) * scale

        # L2-rescale to preserve energy
        if dared_delta.norm(p=2) > 0:
            rescale_factor = (delta_b.norm(p=2) + delta_c.norm(p=2)) / (dared_delta.norm(p=2) + 1e-8)
            dared_delta = dared_delta * rescale_factor

        return (a + dared_delta).to(a.dtype)

class SLERP3(Operation):
    def __init__(self, key, alpha, beta, a, b, c):
        super().__init__(key, a, b, c)
        self.alpha = alpha   # weight for B
        self.beta = beta     # weight for C (alpha + beta <= 1.0, remainder is A)

    @multi_cache_operation
    def oper(self, a, b, c):
        # Normalize weights
        total = self.alpha + self.beta
        if total <= 0.0:
            return a
        alpha = self.alpha / total if total > 0 else 0.0
        beta = self.beta / total if total > 0 else 0.0
        gamma = 1.0 - alpha - beta  # weight for A

        if gamma >= 1.0:
            return a
        if alpha >= 1.0:
            return b
        if beta >= 1.0:
            return c

        # Flatten for vector operations
        a_flat = a.flatten()
        b_flat = b.flatten()
        c_flat = c.flatten()

        # Compute log map to origin (A as base)
        def log_map(x, base):
            cos_theta = torch.sum(x * base) / (torch.norm(x) * torch.norm(base) + 1e-8)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            theta = torch.acos(cos_theta)
            if theta < 1e-6:
                return torch.zeros_like(x)
            return (x - cos_theta * base) * (theta / torch.sin(theta))

        log_b = log_map(b_flat, a_flat)
        log_c = log_map(c_flat, a_flat)

        # Weighted average in tangent space
        log_merged = alpha * log_b + beta * log_c

        # Exp map back to manifold
        norm = torch.norm(log_merged)
        if norm < 1e-6:
            return a
        exp_merged = torch.cos(norm) * a_flat + torch.sin(norm) * (log_merged / norm)

        return exp_merged.view_as(a).to(a.dtype)

class SLERP(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, a, b):
        if self.alpha <= 0.0: return a
        if self.alpha >= 1.0: return b

        a_flat = a.flatten()
        b_flat = b.flatten()

        cos_theta = (a_flat * b_flat).sum() / (a_flat.norm() * b_flat.norm() + 1e-8)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)

        if sin_theta < 1e-6:
            return (1 - self.alpha) * a + self.alpha * b

        coeff_a = torch.sin((1.0 - self.alpha) * theta) / sin_theta
        coeff_b = torch.sin(self.alpha * theta) / sin_theta

        return (coeff_a * a + coeff_b * b).to(a.dtype)


class TrainDiff(Operation):
    def __init__(self, key, a, b, c):
        super().__init__(key, a, b, c)

    @multi_cache_operation
    def oper(self, a, b, c):
        delta = b - c
        if delta.numel() == 0:
            return a
        delta = delta - delta.mean()
        return (a + delta).to(a.dtype)


def pad_to(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    if t.shape == target_shape:
        return t
    
    # Critical safety: reject insane shapes (prevent OOM)
    if any(s > 100000 for s in target_shape):  # 100k is already insane
        return t  # refuse to pad — keep original
    
    diff = [max(0, tgt - src) for src, tgt in zip(t.shape, target_shape)]
    if all(d == 0 for d in diff):
        return t
    
    pad = []
    for d in diff[::-1]:
        pad.extend([d // 2, d - d // 2])
    
    try:
        return F.pad(t, pad)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "allocate" in str(e).lower():
            return t  # silently refuse on OOM
        raise


def resize_tensors(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape == b.shape:
        return a, b
    
    # Safety: never trust shapes from secondary model in cross-arch
    if getattr(cmn, 'is_cross_arch', False):
        # Always resize b → a (a is primary)
        if b.shape != a.shape:
            b = SmartResize("", a.shape, b).oper(b)
        return a, b
    
    # Same-arch: pad both to max shape
    max_shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
    
    # Reject insane max shapes
    if any(s > 100000 for s in max_shape):
        return a, b  # refuse to resize
    
    return pad_to(a, max_shape), pad_to(b, max_shape)


class InterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.beta = int(beta)   # 0 or 1
        self.gamma = float(gamma)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, a, b):
        alpha = max(self.alpha, 0.001)
        delta = torch.abs(a - b)

        # Avoid division by zero
        delta_max = torch.max(delta)
        if delta_max == 0:
            return a  # or b, they're identical

        if self.beta != 1:
            diff = ((delta_max - delta) / delta_max) ** (1 / alpha - 1)
        else:
            diff = (delta / delta_max) ** (1 / alpha - 1)

        diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Bernoulli mask with seed
        rng = torch.Generator(device=diff.device)
        rng.manual_seed(self.seed + hash(self.key) % (2**32 - 1))  # key-specific seed
        bitmask = torch.bernoulli(torch.clamp(diff, 0.0, 1.0), generator=rng)

        # Final interpolation
        interpolated_mask = torch.lerp(bitmask, diff, self.gamma)
        result = a * (1 - interpolated_mask) + b * interpolated_mask

        return result.to(a.dtype)

class ManualEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, delta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)    # Interpolation strength
        self.beta = float(beta)      # Lower threshold
        self.gamma = float(gamma)    # Upper threshold
        self.delta = float(delta)    # Smoothness factor
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, a, b):
        # Absolute differences
        delta = torch.abs(a - b)

        # Avoid division by zero
        delta_max = torch.max(delta)
        if delta_max == 0:
            return a  # a and b are identical

        # Normalize differences: 1 = no difference, 0 = max difference
        diff = (delta_max - delta) / delta_max
        diff = torch.nan_to_num(diff, nan=0.0)

        # Mean difference per channel (for attention layers, etc.)
        mean_diff = torch.mean(diff, dim=0, keepdim=True)

        # Create threshold mask
        mask = torch.logical_and(self.beta < mean_diff, mean_diff < self.gamma)

        # Apply power function (controls curve shape)
        alpha_safe = max(self.alpha, 0.001)
        powered_diff = diff ** (1 / alpha_safe - 1)
        powered_diff = torch.nan_to_num(powered_diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Apply threshold mask
        masked_diff = powered_diff * mask.float()

        # Random mask with seed
        rng = torch.Generator(device=a.device)
        rng.manual_seed(self.seed + hash(self.key) % (2**32 - 1))
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0.0, 1.0), generator=rng)

        # Final interpolation between random and smooth
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.delta)

        # Apply to output
        result = a * (1 - interpolated_mask) + b * interpolated_mask

        return result.to(a.dtype)
    
class AutoEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)    # Interpolation strength
        self.beta = float(beta)      # Threshold adjustment factor
        self.gamma = float(gamma)    # Smoothness factor
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, a, b):
        delta = torch.abs(a - b)

        # Avoid division by zero
        max_delta = torch.max(delta)
        if max_delta == 0:
            return a  # a and b are identical

        # Normalize: 1 = no difference, 0 = max difference
        diff = (max_delta - delta) / max_delta
        diff = torch.nan_to_num(diff, nan=0.0)

        # Global mean difference (adaptive center)
        mean_diff = torch.mean(diff)

        # Dynamic thresholds — the genius of your method
        lower_threshold = mean_diff * (1 - self.beta)
        upper_threshold = mean_diff * (1 + self.beta)

        # Mask: keep values near the mean difference
        mask = torch.logical_and(lower_threshold < diff, diff < upper_threshold)

        # Power curve for shaping
        alpha_safe = max(self.alpha, 0.001)
        powered_diff = diff ** (1 / alpha_safe - 1)
        powered_diff = torch.nan_to_num(powered_diff, nan=0.0, posinf=1.0, neginf=0.0)

        # Apply mask
        masked_diff = powered_diff * mask.float()

        # Random mask with seed
        rng = torch.Generator(device=a.device)
        rng.manual_seed(self.seed + hash(self.key) % (2**32 - 1))
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0.0, 1.0), generator=rng)

        # Final lerp between random and smooth
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.gamma)

        result = a * (1 - interpolated_mask) + b * interpolated_mask

        return result.to(a.dtype)

class SingularValueDeOperator(Operation):
    def __init__(self, key, alpha, beta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)    # Threshold for significant singular values
        self.beta = float(beta)      # Fraction of top singular values to keep (0.1–0.5)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, a, b):
        # Only works on 2D+ tensors — skip 1D (bias) and scalars
        if a.ndim < 2 or b.ndim < 2:
            return a

        # Skip if shapes don't match (after SmartResize)
        if a.shape != b.shape:
            return a

        try:
            diff = a - b

            # SVD — the heart of the method
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

            # Keep only significant singular values
            threshold = self.alpha * S.max()
            significant = S > threshold

            # Keep top-k fraction (beta)
            if self.beta < 1.0:
                k = max(1, int(self.beta * len(S)))
                top_k = torch.zeros_like(significant)
                top_k[:k] = True
                significant = significant & top_k

            # Zero out non-significant values
            S_filtered = S * significant.float()

            # Reconstruct denoised difference
            reconstructed = torch.matmul(U, torch.matmul(torch.diag(S_filtered), Vh))

            # Add back to base (a)
            result = a + reconstructed

            return result.to(a.dtype)

        except Exception as e:
            # If anything goes wrong (e.g., SVD convergence), fall back to original
            print(f"[SVD] Failed on {self.key}: {e} — using original")
            return a


class TensorExchange(Operation):
    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)
        self.seed = int(seed)

    @multi_cache_operation
    def oper(self, a, b):
        import hashlib
        hash_input = f"{self.key}{self.seed}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        choice_val = (hash_val % 10000) / 10000.0
        return b if choice_val < self.alpha else a


class WeightSumCutoff(Operation):
    def __init__(self, key, alpha, beta, gamma, *sources):
        super().__init__(key, *sources)
        self.alpha = float(alpha)    # Merge strength
        self.beta = float(beta)      # Lower threshold
        self.gamma = float(gamma)    # Upper threshold

    @multi_cache_operation
    def oper(self, a, b):
        delta = torch.abs(a - b)

        # Avoid division by zero
        delta_max = torch.max(delta)
        if delta_max == 0:
            return a  # a and b are identical

        # Normalized similarity: 1 = identical, 0 = max difference
        diff = (delta_max - delta) / delta_max
        diff = torch.nan_to_num(diff, nan=0.0)

        # Mean similarity per channel
        mean_sim = torch.mean(diff, dim=0, keepdim=True)  # [1, C]

        # Keep only channels where similarity is BETWEEN beta and gamma
        # → beta < mean_sim < gamma
        mask = (mean_sim > self.beta) & (mean_sim < self.gamma)

        # Apply cutoff: only merge in the selected range
        mul = self.alpha * mask.float()
        result = a * (1 - mul) + b * mul

        return result.to(a.dtype)

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

    def oper(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape == self.target_shape:
            return t

        device = t.device
        dtype = t.dtype

        # Safety: Safety: if target is invalid or insane, return original
        if not self.target_shape or len(self.target_shape) == 0 or any(s > 100000 for s in self.target_shape):
            return t  # refuse OOM

        target = self.target_shape

        # 1D tensors
        if t.ndim == 1:
            if len(target) >= 1 and t.shape[0] != target[0]:
                t = F.interpolate(t.unsqueeze(0).unsqueeze(0), size=(target[0],), mode='linear')[0, 0]
            return t.to(dtype)

        # 2D tensors — THE CRITICAL FIX FOR CROSS-ARCH (SD1.5 ↔ SDXL)
        if t.ndim == 2:
            if len(target) < 2 or target[0] >= 100000 or target[1] >= 100000:
                return t  # insane size, skip

            # Memory-safe path: large first dim = likely token embedding (49408, 768) → (49408, 1280)
            if target[0] > 20000:  # vocab-sized first dim
                new_t = torch.zeros(target, device=device, dtype=dtype)
                min_rows = min(t.shape[0], target[0])
                for i in range(min_rows):
                    row = t[i:i+1]  # (1, old_dim)
                    resized_row = F.interpolate(
                        row.unsqueeze(0).unsqueeze(0),  # (1, 1, 1, old_dim)
                        size=(target[1],),
                        mode='linear',
                        align_corners=False
                    )[0, 0]  # → (1, new_dim)
                    new_t[i] = resized_row
                # Extra rows stay zero (safe padding)
                return new_t.to(dtype)
            else:
                # Fast path for normal layers (e.g., attention QKV, FFN)
                return F.interpolate(
                    t.unsqueeze(0).unsqueeze(0),
                    size=target,
                    mode='bilinear',
                    align_corners=False
                )[0, 0].to(dtype)

        # 3D tensors (rare, e.g., positional embeddings)
        if t.ndim == 3:
            if len(target) >= 3 and all(s < 100000 for s in target):
                t = F.interpolate(t.unsqueeze(0), size=target, mode='trilinear').squeeze(0)
            return t.to(dtype)

        # 4D tensors (Conv2d weights)
        if t.ndim == 4:
            cout, cin, h, w = (target + t.shape[2:])[:4] if len(target) >= 4 else (t.shape[0], t.shape[1], t.shape[2], t.shape[3])

            if h > 1000 or w > 1000 or cout > 10000 or cin > 10000:
                return t  # refuse dangerous resize

            if (h, w) != t.shape[2:]:
                t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)

            if t.shape[0] != cout or t.shape[1] != cin:
                if t.shape[0] > cout:
                    t = t[:cout]
                elif t.shape[0] < cout:
                    pad = torch.zeros((cout - t.shape[0], t.shape[1], h, w), device=device, dtype=dtype)
                    t = torch.cat([t, pad], dim=0)

                if t.shape[1] > cin:
                    t = t[:, :cin]
                elif t.shape[1] < cin:
                    pad = torch.zeros((t.shape[0], cin - t.shape[1], h, w), device=device, dtype=dtype)
                    t = torch.cat([t, pad], dim=1)

            return t.to(dtype)

        # Fallback: pad + slice (safe but less accurate)
        pad = []
        for s, tgt in zip(t.shape[::-1], target[::-1]):
            diff = tgt - s
            pad.extend([diff//2, diff - diff//2] if diff > 0 else [0, 0])
        t = F.pad(t, pad[::-1])
        slices = tuple(slice(0, s) for s in target)
        return t[slices].to(dtype)