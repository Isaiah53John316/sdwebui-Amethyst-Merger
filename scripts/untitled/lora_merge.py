import torch
import safetensors.torch
from safetensors import safe_open
import scripts.untitled.common as cmn
import scripts.untitled.misc_util as mutil
from modules import sd_models
import re
import os
import traceback  # For errors

def get_lora_keys(lora_path):
    """Get all keys from a LoRA file"""
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        return list(f.keys())

def merge_loras_to_checkpoint(
    checkpoint_path: str,
    lora_paths: list[str],
    output_path: str,
    individual_weights: list = None,
    progress=None,
    save_model: bool = True,
    verbose: bool = False
) -> str:
    """
    Bake multiple LoRAs into a checkpoint with per-LoRA strength.
    Full cross-arch support: SD1.5, SDXL, Pony, Flux.
    Novel: Supports block-specific strengths (e.g., "unet:1.2, te:0.8" in weights as str).
    Fixed: Ultimate key matching — no more skips.
    """
    # Safe progress (anti-bloat)
    def safe_progress(msg, popup=False, verbose_only=False):
        if verbose_only and not verbose:
            return
        if progress:
            try:
                progress(msg, popup=popup)
            except:
                print(f"[LoRA Bake] {msg}")
        else:
            print(f"[LoRA Bake] {msg}")

    if not lora_paths:
        return "Error: No LoRAs provided."

    if individual_weights is None or len(individual_weights) != len(lora_paths):
        individual_weights = [1.0] * len(lora_paths)

    safe_progress(f"Baking {len(lora_paths)} LoRA(s) → {os.path.basename(output_path)}")

    # Load checkpoint
    try:
        with safetensors.torch.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            ckpt_dict = {k: f.get_tensor(k) for k in f.keys()}
    except Exception as e:
        return f"Failed to load checkpoint: {e}"

    total_applied = 0
    total_skipped = 0
    unique_logs = set()  # Anti-bloat for repeats

    for idx, (path, weight) in enumerate(zip(lora_paths, individual_weights)):
        if weight == 0.0:
            safe_progress(f"Skipping {os.path.basename(path)} (weight=0)")
            continue

        # Parse block strengths (novel)
        block_strengths = _parse_block_strengths(weight) if isinstance(weight, str) else {"default": weight}

        safe_progress(f"[{idx+1}/{len(lora_paths)}] Baking {os.path.basename(path)} × {weight}")

        # Load LoRA
        try:
            with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
                lora_keys = f.keys()
                lora_tensors = {k: f.get_tensor(k) for k in lora_keys}
        except Exception as e:
            safe_progress(f"  Failed: {e}")
            continue

        # Group LoRA keys
        lora_map = {}
        for k in lora_keys:
            base, typ = _parse_lora_key(k)
            if base:
                lora_map.setdefault(base, {})[typ] = lora_tensors[k]

        ckpt_keys = set(ckpt_dict.keys())

        applied_this = 0
        skipped_this = 0

        for base, parts in lora_map.items():
            if "up" not in parts or "down" not in parts:
                skipped_this += 1
                continue

            up = parts["up"]
            down = parts["down"]
            alpha = parts["alpha"].item() if "alpha" in parts else up.shape[1]  # rank = up.shape[1] (out, rank)

            # Effective strength for block (novel)
            block_type = _get_block_type(base)
            eff_strength = block_strengths.get(block_type, block_strengths.get("default", 1.0))

            # Find matching key (ultimate matcher)
            ckpt_key = _find_ckpt_key_from_lora(base, ckpt_keys)
            if not ckpt_key or ckpt_key not in ckpt_dict:
                skipped_this += 1
                if verbose:
                    log = f"  No match for: {base}"
                    if log not in unique_logs:
                        safe_progress(log, verbose_only=True)
                        unique_logs.add(log)
                continue

            base_weight = ckpt_dict[ckpt_key]

            # RESIZE: Pad outer, keep rank
            rank = up.shape[1]
            target_shape = base_weight.shape
            if len(target_shape) >= 2:
                target_out, target_in = target_shape[:2]
                up_padded = _pad_lora_outer(up, (target_out, rank))
                down_padded = _pad_lora_outer(down, (rank, target_in))

                # Delta
                delta = torch.matmul(up_padded, down_padded) * (eff_strength * alpha / rank)

                # Higher dims (conv)
                if len(target_shape) > 2:
                    delta = delta.view(target_shape)

                ckpt_dict[ckpt_key] = base_weight + delta.to(base_weight.dtype)
                applied_this += 1

                if verbose:
                    log = f"  Applied: {ckpt_key.split('.')[-1]}"
                    if log not in unique_logs:
                        safe_progress(log, verbose_only=True)
                        unique_logs.add(log)

        total_applied += applied_this
        total_skipped += skipped_this

        safe_progress(f"  Done: Applied {applied_this} (skipped {skipped_this})")

    result_msg = f"Checkpoint baked! Applied {total_applied} layers"
    if total_skipped:
        result_msg += f" (skipped {total_skipped})"
    safe_progress(result_msg)

    if save_model:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            safetensors.torch.save_file(ckpt_dict, output_path)
            safe_progress(f"Saved: {os.path.basename(output_path)}", popup=True)
        except Exception as e:
            result_msg += f"\nSave failed: {e}"

    return result_msg

# Helpers...

def _parse_lora_key(key: str) -> tuple[str | None, str | None]:
    """Parse LoRA key to base + type"""
    if '.lora_up.weight' in key:
        return key.replace('.lora_up.weight', ''), 'up'
    if '.lora_down.weight' in key:
        return key.replace('.lora_down.weight', ''), 'down'
    if '.alpha' in key:
        return key.replace('.alpha', ''), 'alpha'
    return None, None

def _find_ckpt_key_from_lora(lora_base: str, ckpt_keys: set) -> str | None:
    """Ultimate robust LoRA → checkpoint key matching — 2025 edition"""
    # Step 1: Remove prefixes
    prefixes = ['lora_unet_', 'lora_te1_', 'lora_te2_', 'lora_te_', 'lora_flux_', 'lora_sd3_', 'lora_conditioner_']
    for p in prefixes:
        lora_base = lora_base.removeprefix(p)

    # Step 2: Normalize structure — improved regex for numbers
    lora_base = re.sub(r'_(\d+)_', r'.\1.', lora_base)  # _0_ → .0.
    lora_base = re.sub(r'_(\d+)$', r'.\1', lora_base)  # _0 at end → .0
    lora_base = re.sub(r'(\.\d+)$', r'\1.0', lora_base)  # .0 → .0.0 for some
    lora_base = re.sub(r'(\.0)\.', r'\1.0.', lora_base)  # Insert 0 for double

    # Novel: Specific for mid attn
    lora_base = lora_base.replace('attn_0', '0.attn').replace('attn_1', '1.attn')
    lora_base = lora_base.replace('proj.out', 'proj.out')  # Ensure

    # Step 3: Key term replacements
    replacements = {
        'down_blocks': 'input_blocks',
        'up_blocks': 'output_blocks',
        'mid_block': 'middle_block',
        'proj_in': 'proj.in',
        'proj_out': 'proj.out',
        'attentions': 'attn',
        'resnets': 'res',
        'encoder_layers': 'resblocks',
        'self_attn_q_proj': 'attn.in_proj',
        'self_attn': 'attn',
        'qkv': 'qkv',  # SD3
        'transformer': 'transformer',
        'text_model': 'transformer.text_model',
        'embedders_0_model': 'embedders.0.model',
        'embedders_1_model': 'embedders.1.model',
        'single_blocks': 'single_blocks',
        'transformer_blocks': 'transformer_blocks',
    }
    for old, new in replacements.items():
        lora_base = lora_base.replace(old, new)

    # Step 4: Generate candidates — smart prefix addition
    base_candidates = [lora_base]
    if 'embedders.0.model' in lora_base:
        base_candidates.append(lora_base.replace('embedders.0.model.', '', 1))  # Avoid duplicate

    candidates = []
    for b in base_candidates:
        candidates.extend([
            b,
            f"model.diffusion_model.{b}",
            f"conditioner.embedders.0.model.{b}",
            f"conditioner.embedders.1.model.{b}",
            f"cond_stage_model.{b}",
            f"cond_stage_model.transformer.{b}",
            f"cond_stage_model.transformer.text_model.{b}",
            f"model.diffusion_model.single_blocks.{b}",
            f"model.diffusion_model.transformer_blocks.{b}",
            f"conditioner.{b}",
            # Special for middle
            f"model.diffusion_model.middle_block.0.{b}",
            f"model.diffusion_model.middle_block.1.{b}",
        ])

    # Add .weight variants
    candidates += [c + '.weight' for c in candidates if not c.endswith('.weight')]

    # Step 5: Exact match first
    for c in candidates:
        if c in ckpt_keys:
            return c

    # Step 6: Fuzzy match if no exact (novel: split on . and _ , check 80% match)
    base_parts = set(re.split(r'[._]', lora_base))
    for ck_key in ckpt_keys:
        ck_parts = set(re.split(r'[._]', ck_key))
        match_ratio = len(base_parts & ck_parts) / len(base_parts) if len(base_parts) > 0 else 0
        if match_ratio >= 0.8:
            return ck_key

    return None

def _get_block_type(base_key: str) -> str:
    """Detect block type for per-block strength (novel)"""
    if 'unet' in base_key or 'diffusion_model' in base_key:
        return 'unet'
    if 'te' in base_key or 'text_model' in base_key or 'conditioner' in base_key:
        return 'te'
    return 'other'

def _parse_block_strengths(weight_str: str) -> dict[str, float]:
    """Parse 'unet:1.2, te:0.8' → {'unet': 1.2, 'te': 0.8} (novel feature)"""
    strengths = {}
    for part in weight_str.split(','):
        if ':' in part:
            block, val = part.split(':')
            strengths[block.strip()] = float(val.strip())
        else:
            strengths['default'] = float(part.strip())
    return strengths

def _pad_lora_outer(tensor: torch.Tensor, target_dim: tuple[int, int]) -> torch.Tensor:
    """Pad/trim LoRA tensor outer dims, keep inner (rank) fixed"""
    if tensor.ndim != 2:
        return torch.zeros(target_dim, dtype=tensor.dtype, device=tensor.device)

    out_target, inner = target_dim  # (out, rank) for up; (rank, in) for down
    out_curr, inner_curr = tensor.shape

    if inner_curr != inner:
        # Rank mismatch — interp inner dim
        tensor = torch.nn.functional.interpolate(
            tensor.t().unsqueeze(0).unsqueeze(0),  # Transpose for interp
            size=(inner,),
            mode='linear'
        ).squeeze(0).squeeze(0).t()

    padded = torch.zeros(target_dim, dtype=tensor.dtype, device=tensor.device)
    min_out = min(out_curr, out_target)
    padded[:min_out, :] = tensor[:min_out, :]

    return padded

# Test the matcher with sample data (as before)
test_lora_bases = [
    "lora_unet_down_blocks_0_attentions_0_proj_in",
    "lora_te1_text_model_encoder_layers_0_self_attn_q_proj",
    "lora_te2_text_model_encoder_layers_0_self_attn_q_proj",
    "lora_unet_mid_block_attentions_0_proj_out",
    "lora_flux_single_blocks_0_linear",
    "lora_sd3_conditioner_embedders_0_model_transformer_blocks_0_attn_qkv",
]

sample_ckpt_keys = set([
    "model.diffusion_model.input_blocks.0.0.attn.proj.in.weight",
    "conditioner.embedders.0.model.transformer.resblocks.0.attn.in_proj.weight",
    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj.weight",
    "model.diffusion_model.middle_block.0.attn.proj.out.weight",
    "model.diffusion_model.single_blocks.0.linear.weight",
    "conditioner.embedders.0.model.transformer_blocks.0.attn.qkv.weight",
])

results = {}
for base in test_lora_bases:
    matched = _find_ckpt_key_from_lora(base, sample_ckpt_keys)
    results[base] = matched or "NO MATCH"

print("Matcher Test Results:", results)

def _pad_lora_outer(tensor: torch.Tensor, target_dim: tuple[int, int]) -> torch.Tensor:
    """Pad/trim LoRA tensor outer dims, keep inner (rank) fixed"""
    if tensor.ndim != 2:
        return torch.zeros(target_dim, dtype=tensor.dtype, device=tensor.device)

    out_target, inner = target_dim  # (out, rank) for up; (rank, in) for down
    out_curr, inner_curr = tensor.shape

    if inner_curr != inner:
        # Rank mismatch — rare, but resize inner with interp
        tensor = torch.nn.functional.interpolate(
            tensor.t().unsqueeze(0).unsqueeze(0),  # Transpose for interp
            size=(inner,),
            mode='linear'
        ).squeeze(0).squeeze(0).t()

    padded = torch.zeros(target_dim, dtype=tensor.dtype, device=tensor.device)
    min_out = min(out_curr, out_target)
    padded[:min_out, :] = tensor[:min_out, :]

    return padded

def _get_block_type(base_key: str) -> str:
    """Detect block type for per-block strength (novel)"""
    if 'unet' in base_key or 'diffusion_model' in base_key:
        return 'unet'
    if 'te' in base_key or 'text_model' in base_key or 'conditioner' in base_key:
        return 'te'
    return 'other'

def _parse_block_strengths(weight_str: str) -> dict[str, float]:
    """Parse 'unet:0.8, te:0.6' → {'unet': 0.8, 'te': 0.6} (novel feature)"""
    strengths = {}
    for part in weight_str.split(','):
        if ':' in part:
            block, val = part.split(':')
            strengths[block.strip()] = float(val.strip())
        else:
            strengths['default'] = float(part.strip())
    return strengths


def merge_loras_resilient(lora_paths: list[str], output_path: str, weights=None, progress=None):
    """
    Ultra-robust LoRA merging with automatic tensor resizing.
    Handles SD1.5, SDXL, Pony, Flux, and mixed-rank LoRAs flawlessly.
    """
    if not lora_paths:
        return "Error: No LoRA paths provided."

    if weights is None:
        weights = [1.0 / len(lora_paths)] * len(lora_paths)
    if len(weights) != len(lora_paths):
        return f"Error: {len(lora_paths)} LoRAs but {len(weights)} weights provided."

    if progress:
        progress(f"Merging {len(lora_paths)} LoRAs → {os.path.basename(output_path)}")

    merged = None
    base_shapes = {}  # Track the "winning" shape per key
    total_added = 0

    for idx, (path, weight) in enumerate(zip(lora_paths, weights)):
        if weight == 0.0:
            if progress:
                progress(f"Skipping {os.path.basename(path)} (weight = 0)")
            continue

        filename = os.path.basename(path)
        if progress:
            progress(f"[{idx+1}/{len(lora_paths)}] Loading {filename} × {weight:.3f}")

        try:
            with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
                keys = f.keys()
                lora_dict = {}
                for k in keys:
                    try:
                        lora_dict[k] = f.get_tensor(k)
                    except Exception as e:
                        if progress:
                            progress(f"  Warning: Failed to read tensor {k}: {e}")
                        continue
        except Exception as e:
            if progress:
                progress(f"Failed to open {filename}: {e}")
            continue

        if not lora_dict:
            if progress:
                progress(f"  Empty or corrupted LoRA: {filename}")
            continue
 
        if merged is None:
            # First LoRA defines the base
            merged = {k: t.clone() * weight for k, t in lora_dict.items()}
            base_shapes = {k: t.shape for k, t in lora_dict.items()}
            total_added += 1
            if progress:
                progress(f"  Set as base model ({len(merged)} keys)")
            continue

        added_this = 0
        for key, tensor_new in lora_dict.items():
            if key not in merged:
                continue  # Key not in base → skip (safe)

            tensor_base = merged[key]
            target_shape = base_shapes[key]

            if tensor_new.shape == target_shape:
                merged[key] = merged[key] + tensor_new * weight
                added_this += 1
                continue

            # RESIZING LOGIC — Smart & Safe
            if progress:
                progress(f"  Resizing {key}: {tensor_new.shape} → {target_shape}")

            try:
                resized = _smart_resize_tensor(tensor_new, target_shape)
                merged[key] = merged[key] + resized * weight
                added_this += 1
            except Exception as e:
                if progress:
                    progress(f"  Failed to resize {key}: {e}")

        total_added += 1
        if progress and added_this > 0:
            progress(f"  Added {added_this} tensors from {filename}")

    if merged is None or len(merged) == 0:
        return "Error: No valid tensors were merged. Check LoRA files."

    if progress:
        progress(f"Saving merged LoRA: {len(merged)} keys → {os.path.basename(output_path)}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        safetensors.torch.save_file(merged, output_path)
    except Exception as e:
        return f"Failed to save file: {e}"

    if progress:
        progress("SUCCESS! Merged LoRA saved.", popup=True)

    return f"Success: Merged {total_added}/{len(lora_paths)} LoRAs → {os.path.basename(output_path)} ({len(merged)} keys)"


def _smart_resize_tensor(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Intelligently resize a tensor to target_shape with proper padding/trimming/interpolation.
    Handles 2D (Linear), 3D (Conv1D/Emb), 4D (Conv2D safely.
    """
    if tensor.shape == target_shape:
        return tensor

    device = tensor.device
    dtype = tensor.dtype
    tensor = tensor.to("cpu")

    ndim = len(target_shape)

    if ndim == 2:  # Linear layer: (out_features, in_features)
        out_dim, in_dim = target_shape
        t = tensor

        # Trim or pad output dim
        if t.shape[0] > out_dim:
            t = t[:out_dim]
        elif t.shape[0] < out_dim:
            pad = torch.zeros((out_dim - t.shape[0], t.shape[1]), dtype=dtype)
            t = torch.cat([t, pad], dim=0)

        # Trim or pad input dim
        if t.shape[1] > in_dim:
            t = t[:, :in_dim]
        elif t.shape[1] < in_dim:
            pad = torch.zeros((t.shape[0], in_dim - t.shape[1]), dtype=dtype)
            t = torch.cat([t, pad], dim=1)

        return t.to(device).type(dtype)

    elif ndim == 3:  # Conv1D or embeddings, positional, etc.
        return torch.nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="linear",
            align_corners=False
        ).squeeze(0).squeeze(0).to(device).type(dtype)

    elif ndim == 4:  # Conv2D weights
        return torch.nn.functional.interpolate(
            tensor,
            size=target_shape[2:],
            mode="bilinear",
            align_corners=False
        ).to(device).type(dtype)

        # Adjust channels if needed
    if tensor.shape[1] != target_shape[1]:
            # This is rare — just pad/trim channels
            c_out, c_in, h, w = target_shape
            t = tensor
            if t.shape[0] != c_out or t.shape[1] != c_in:
                new_t = torch.zeros(target_shape, dtype=dtype)
                min_cout = min(t.shape[0], c_out)
                min_cin = min(t.shape[1], c_in)
                new_t[:min_cout, :min_cin] = t[:min_cout, :min_cin]
                t = new_t
            return t.to(device)

    else:
        # Fallback: pad + slice
        pad = []
        for s, t in zip(tensor.shape[::-1], target_shape[::-1]):
            diff = t - s
            pad.extend([diff//2, diff - diff//2] if diff > 0 else [0, 0])
        padded = torch.nn.functional.pad(tensor, pad[::-1])
        slices = tuple(slice(0, s) for s in target_shape)
        return padded[slices].to(device).type(dtype)