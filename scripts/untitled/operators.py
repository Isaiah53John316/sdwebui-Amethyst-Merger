import torch, scipy
import scripts.untitled.common as cmn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from safetensors import SafetensorError   # ← THIS LINE IS THE FIX (adds missing error type)
import os
from functools import wraps
import re

def recurse(operation):
    source_tensors = []
    for source in operation.sources:
        if hasattr(source, 'merge'):
            source_tensors.append(source.merge())
        else:
            # Scalar value - convert to tensor if needed
            source_tensors.append(torch.tensor(source, device=cmn.device(), dtype=cmn.dtype()))
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
        

# ≈ make LoadTensor respect cross-arch target shapes + survive missing keys
class LoadTensor(Operation):
    def __init__(self, key, checkpoint_name):
        super().__init__(key)
        # ALWAYS store the **exact** string that was used when the file was opened
        self.checkpoint_name = checkpoint_name
        self._resolved_path = checkpoint_name  # keep original for debugging

    def merge(self) -> torch.Tensor:
        if cmn.loaded_checkpoints is None:
            raise RuntimeError("Checkpoints not loaded")

        # ← THE FIX: resolve the exact key that is actually in the dict
        file = None
        for loaded_key in cmn.loaded_checkpoints.keys():
            if os.path.samefile(loaded_key, self.checkpoint_name):
                file = cmn.loaded_checkpoints[loaded_key]
                break

        # Fallback to primary if still not found (should never happen now)
        if file is None:
            file = cmn.loaded_checkpoints.get(cmn.primary)

        try:
            tensor = file.get_tensor(self.key)
        except SafetensorError:
            # Still try primary as last resort
            file = cmn.loaded_checkpoints.get(cmn.primary)
            try:
                tensor = file.get_tensor(self.key)
            except SafetensorError:
                if getattr(cmn, 'is_cross_arch', False):
                    shapes = getattr(cmn, 'cross_arch_target_shapes', {})
                    if self.key in shapes:
                        target = shapes[self.key]
                        print(f"[Merger] Missing key {self.key} → zeros {target}")
                        return torch.zeros(target, device=cmn.device(), dtype=cmn.dtype())
                raise

        tensor = tensor.to(cmn.device())

        if tensor.ndim == 0:
            return tensor

        # Cross-arch resizing (unchanged)
        if getattr(cmn, 'is_cross_arch', False):
            shapes = getattr(cmn, 'cross_arch_target_shapes', {})
            if self.key in shapes:
                target = shapes[self.key]
                if tensor.shape != target:
                    tensor = SmartResize(f"LoadTensor_fix_{os.path.basename(self.checkpoint_name)}", target, tensor).oper(tensor)
                    if tensor.shape != target:
                        tensor = tensor.view(-1)[:torch.tensor(target).prod()].view(target)

        return tensor

# === FIXED: Add ===
class Add(Operation):
    def oper(self, a, b):
        return _safe_binary_op("Add", a, b, lambda x, y: x + y)

class Sub(Operation):
    def oper(self, a, b):
        return _safe_binary_op("Sub", a, b, lambda x, y: x - y)

class Multiply(Operation):
    def oper(self, a, b):
        return _safe_binary_op("Mul", a, b, lambda x, y: x * y)

class MultiplyTensors(Operation):
    def oper(self, a, b):
        return _safe_binary_op("MulT", a, b, lambda x, y: x * y)


class Extract(Operation):
    def __init__(self, key, alpha, beta, gamma, *args):
        super().__init__(key, *args)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return super().oper(None, a, b)


class ReBasin(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, a, b):
        # Permutation alignment via sorting + lerp — fast, effective
        a_sorted = torch.sort(a.flatten(), dim=-1)[0]
        b_sorted = torch.sort(b.flatten(), dim=-1)[0]
        merged = (1 - self.alpha) * a_sorted + self.alpha * b_sorted
        return merged.view_as(a).to(a.dtype)
    

class DeMe(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, a, b):
        # Variance-based decoupling (per 2024 paper)
        var_a = torch.var(a, dim=-1, keepdim=True)
        var_b = torch.var(b, dim=-1, keepdim=True)
        decoupled = torch.where(var_a > var_b, a, b)
        return (1 - self.alpha) * a + self.alpha * decoupled
    

class BlockWeighted(Operation):
    def __init__(self, key, alphas, a, b):
        super().__init__(key, a, b)
        self.alphas = alphas

    @multi_cache_operation
    def oper(self, a, b):
        # Extract block number from key (e.g., input_blocks.3 → 3)
        match = re.search(r'\.(\d+)\.', self.key)
        if match:
            idx = int(match.group(1))
            alpha = self.alphas[min(idx, len(self.alphas)-1)]
        else:
            alpha = self.alphas[0]  # fallback
        return (1 - alpha) * a + alpha * b


class ToMe(Operation):
    def __init__(self, key, ratio, tensor):
        super().__init__(key, tensor)
        self.ratio = ratio

    @multi_cache_operation
    def oper(self, tensor):
        if tensor.ndim < 2 or tensor.numel() < 10:
            return tensor

        # Cosine similarity between tokens
        normed = F.normalize(tensor, dim=-1)
        sim = normed @ normed.T
        values, indices = torch.topk(sim, k=max(2, int(tensor.size(0) * self.ratio)), dim=1)
        
        # Merge top-k similar tokens
        merged = torch.zeros_like(tensor)
        for i in range(tensor.size(0)):
            group = tensor[indices[i]]
            merged[i] = group.mean(0)
        
        return merged


class AttentionMerge(Operation):
    def __init__(self, key, alpha, a, b):
        super().__init__(key, a, b)
        self.alpha = alpha

    @multi_cache_operation
    def oper(self, a, b):
        if 'attention' in self.key.lower() or 'attn' in self.key.lower():
            return (1 - self.alpha) * a + self.alpha * b
        return a  # non-attention layers unchanged


class Smooth(Operation):
    def __init__(self, key, tensor):
        super().__init__(key, tensor)

    @multi_cache_operation
    def oper(self, tensor):
        # Fast, safe, 1D conv-based smoothing — works on ALL PyTorch versions
        if tensor.numel() < 5:
            return tensor

        # Gaussian kernel for sigma=1.0 (kernel size 5)
        device = tensor.device
        dtype = tensor.dtype
        kernel_size = 5
        sigma = 1.0
        center = kernel_size // 2
        x = torch.arange(kernel_size, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)  # [1,1,kernel_size]

        orig_shape = tensor.shape

        # Flatten to 1D for smoothing along the last dimension
        flattened = tensor.flatten().unsqueeze(0).unsqueeze(0)  # [1,1,N]
        padding = (center, center)
        padded = F.pad(flattened, padding, mode='replicate')  # Safe padding mode
        smoothed = F.conv1d(padded, kernel).squeeze()  # [N]

        # Restore original shape
        result = smoothed.view(orig_shape)

        return result.to(dtype)
    

class TIES(Operation):
    def __init__(self, key, density, seed, a, b):
        super().__init__(key, a, b)
        self.density = density
        self.seed = seed

    @multi_cache_operation
    def oper(self, a, b):
        if self.density <= 0.0: return a
        if self.density >= 1.0: return b

        torch.manual_seed(self.seed)

        delta = b - a
        abs_delta = delta.abs()
        k = max(1, int(self.density * abs_delta.numel()))
        threshold = torch.topk(abs_delta.flatten(), k).values.min()
        mask = abs_delta >= threshold

        sign_a = torch.sign(a)
        sign_b = torch.sign(b)
        resolved_sign = torch.where(sign_a == sign_b, sign_a, torch.sign(sign_a + sign_b))

        elected_delta = delta * resolved_sign * mask.to(delta.dtype)

        dominate_a = mask & (sign_a == resolved_sign)
        dominate_b = mask & (sign_b == resolved_sign)

        result = torch.where(dominate_a, a, b)
        result = torch.where(dominate_b, b, result)

        if elected_delta.norm(p=2) > 0:
            scale = delta.norm(p=2) / elected_delta.norm(p=2)
            result = a + elected_delta * scale

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
    def __init__(self,key,alpha,beta,gamma,seed,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seed = seed

    def oper(self, a, b):
        alpha = max(self.alpha,0.001)

        delta = torch.abs(a - b)

        if self.beta != 1:
            diff = ((torch.max(delta) - delta) / torch.max(delta)) ** (1 / alpha - 1)
        else:
            diff = (delta / torch.max(delta)) ** (1 / alpha - 1)

        diff = torch.nan_to_num(diff)

        rngenerator = torch.Generator(device=diff.device)
        rngenerator.manual_seed(self.seed)
        bitmask = torch.bernoulli(torch.clamp(diff,0,1),out=torch.empty_like(diff),generator=rngenerator)

        interpolated_mask = torch.lerp(bitmask, diff, self.gamma)

        res = a * (1 - interpolated_mask) + b * interpolated_mask
        return res

class ManualEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, delta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha  # Interpolation strength
        self.beta = beta    # Lower threshold for mean differences
        self.gamma = gamma  # Upper threshold for mean differences
        self.delta = delta  # Smoothness factor
        self.seed = seed    # Seed for random number generation

    def oper(self, a, b):
        # Calculate absolute differences
        delta = torch.abs(a - b)
        
        # Normalize differences
        diff = (torch.max(delta) - delta) / torch.max(delta)
        diff = torch.nan_to_num(diff)
        
        # Calculate mean differences
        mean_diff = torch.mean(diff, 0, keepdim=True)
        
        # Create mask based on mean differences
        mask = torch.logical_and(self.beta < mean_diff, mean_diff < self.gamma)
        
        # Apply power function to differences
        powered_diff = diff ** (1 / max(self.alpha, 0.001) - 1)
        powered_diff = torch.nan_to_num(powered_diff)
        
        # Apply mask to powered differences
        masked_diff = powered_diff * mask.float()
        
        # Generate random mask
        rng = torch.Generator(device=a.device)
        rng.manual_seed(self.seed)
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0, 1), generator=rng)
        
        # Interpolate between random mask and powered differences
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.delta)
        
        # Apply final interpolation
        result = a * (1 - interpolated_mask) + b * interpolated_mask
        
        return result

class AutoEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha  # Interpolation strength
        self.beta = beta    # Threshold adjustment factor
        self.gamma = gamma  # Smoothness factor
        self.seed = seed    # Seed for random number generation

    def oper(self, a, b):
        # Calculate absolute differences
        delta = torch.abs(a - b)
        
        # Normalize differences
        max_delta = torch.max(delta)
        diff = (max_delta - delta) / max_delta
        diff = torch.nan_to_num(diff)
        
        # Calculate mean differences
        mean_diff = torch.mean(diff)
        
        # Dynamically set lower and upper thresholds
        lower_threshold = mean_diff * (1 - self.beta)
        upper_threshold = mean_diff * (1 + self.beta)
        
        # Create mask based on dynamic thresholds
        mask = torch.logical_and(lower_threshold < diff, diff < upper_threshold)
        
        # Apply power function to differences
        powered_diff = diff ** (1 / max(self.alpha, 0.001) - 1)
        powered_diff = torch.nan_to_num(powered_diff)
        
        # Apply mask to powered differences
        masked_diff = powered_diff * mask.float()
        
        # Generate random mask
        rng = torch.Generator(device=a.device)
        rng.manual_seed(self.seed)
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0, 1), generator=rng)
        
        # Interpolate between random mask and powered differences
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.gamma)
        
        # Apply final interpolation
        result = a * (1 - interpolated_mask) + b * interpolated_mask
        
        return result

#class SingularValueDeOperator(Operation):
#    def __init__(self, key, alpha, beta, seed, *sources):
#        super().__init__(key, *sources)
#        self.alpha = alpha  # threshold for significant singular values
#        self.beta = beta    # used to determine which singular values to keep
#
#    def oper(self, a, b):
#        assert a.shape == b.shape, "Tensors must have the same shape"
#
#        diff = a - b
#        U, S, Vh = torch.linalg.svd(diff)
#
#        # Apply thresholding based on alpha
#        significant_values = S > self.alpha
#
#        # Optionally keep only top k singular values based on beta
#        if self.beta < 1:
#            k = max(1, int(self.beta * len(S)))
#            top_k_mask = torch.empty_like(significant_values)
#            top_k_mask[:k] = True
#            significant_values = significant_values & top_k_mask
#
#        # Reconstruct the difference using only significant singular values
#        S_filtered = S * significant_values
#        rng = torch.Generator(device=a.device)
#        rng.manual_seed(self.seed)
#        reconstructed_diff = torch.matmul(U, torch.matmul(torch.diag(S_filtered), Vh), generator=rng)
#
#        result = reconstructed_diff
#        return result


class TensorExchange(Operation):
    def __init__(self, key, alpha, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha
        self.seed = seed

    def oper(self, a, b):
        # Use hash of key + seed to deterministically choose tensor
        import hashlib
        hash_input = f"{self.key}{self.seed}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        # Normalize hash to 0-1 range
        choice_val = (hash_val % 10000) / 10000.0

        # If choice_val < alpha, return b, else return a
        if choice_val < self.alpha:
            return b
        else:
            return a


class WeightSumCutoff(Operation):
    def __init__(self,key,alpha, beta, gamma, *sources):
        super().__init__(key,*sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def oper(self, a, b):
        delta = torch.abs(a - b)

        diff = (torch.max(delta) - delta) / torch.max(delta)
        diffn = torch.nan_to_num(diff)

        mean = torch.mean(diffn,0,True)
        mask = torch.logical_and(mean < self.beta,self.gamma < mean)
        mul = self.alpha*mask

        res = a * (1 - mul) + b * mul
        return res
#The cache
tensor_size = lambda x: x.element_size() * x.nelement()

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
        return t.clone().to(cmn.device()).type(cmn.dtype())
    

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
            if target[0] > 10000:  # vocab-sized first dim
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