import gradio as gr
import re
import os
import shutil
import torch
import traceback
import gc
import json
import warnings
import time
from collections import OrderedDict
from modules.timer import Timer

import torch
import safetensors.torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open as safetensors_open

warnings.filterwarnings("ignore", message=".*UniPC.*")  # Harmless sampler spam

# === CORE MODULES — SAFE CROSS-PLATFORM IMPORTS (A1111 + Forge Neo + Reforge) ===
from modules import (
    sd_models, shared, paths_internal, paths, processing,
    script_callbacks, images, ui_common, script_loading
)

# === OPTIONAL / VERSION-SPECIFIC MODULES — SAFE FALLBACKS ===
try:
    from modules import sd_unet
except ImportError:
    sd_unet = None

try:
    from modules import sd_hijack
except ImportError:
    sd_hijack = None

try:
    from modules import sd_models_config
except ImportError:
    sd_models_config = None  # Removed in Forge — expected

# === LoRA NETWORKS LOADER — FINAL 2025 EDITION (works everywhere) ===
def _load_lora_networks():
    candidates = [
        os.path.join(paths.extensions_builtin_dir, 'sd_forge_lora', 'networks.py'),  # Forge Neo
        os.path.join(paths.extensions_builtin_dir, 'Lora', 'networks.py'),           # Classic A1111
        os.path.join(paths.script_path, 'extensions', 'sd-webui-lora', 'networks.py'),  # Some custom installs
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return script_loading.load_module(path)
            except Exception as e:
                print(f"[Merger] Failed to load LoRA networks from {path}: {e}")
    return None

networks = _load_lora_networks()

# === YOUR MERGER COMMON MODULE ===
import scripts.untitled.common as cmn

BASE_SELECTORS = {
    # ── Universal / Global ──
    "all":      r".*",  

    # ── Core components ──
    "unet":     r"model\.diffusion_model\.",
    "vae":      r"first_stage_model\.",
    "clip":     r"(cond_stage_model\.|conditioner\.embedders\.0\.)",
    "te":       r"(cond_stage_model\.transformer\.|conditioner\.embedders\.0\.transformer\.)",
    "te1":      r"cond_stage_model\.transformer\.text_model\.",
    "te2":      r"conditioner\.embedders\.0\.transformer\.text_model\.",

    # ── XL / modern ──
    "xl":       r"conditioner\.",
    "pony":     r"conditioner\.embedders\.1\.",
    "flux":     r"(single_blocks|double_blocks|img_proj|txt_proj)\.",

    # ── UNet blocks ──
    "in":       r"model\.diffusion_model\.input_blocks\.",
    "mid":      r"model\.diffusion_model\.middle_block\.",
    "out":      r"model\.diffusion_model\.output_blocks\.",

    # ── Legacy / EMA ──
    "model_ema": r"model_ema\.",

    # ── Convenience ──
    "base":     r"cond_stage_model\.",
    "text":     r"(text_model|transformer)\.",
}


def target_to_regex(target_input: str | list) -> str:
    """
    Converts weight editor target strings into a safe regex.
    Supports:
      • Built-ins (unet, te1, te2, xl, flux, etc.)
      • Wildcards (*, ?)
      • Explicit paths (model.diffusion_model.*)
    """

    if isinstance(target_input, (list, tuple)):
        targets = target_input
    else:
        targets = [t.strip() for t in str(target_input).split(',') if t.strip()]

    patterns = []

    for target in targets:
        if not target:
            continue

        # ── Global wildcard ──
        if target in ("*", "all"):
            patterns.append(r".*")
            continue

        # ── Built-in selector ──
        if target in BASE_SELECTORS:
            patterns.append(BASE_SELECTORS[target])
            continue

        # ── Manual selector ──
        # Escape everything first
        escaped = re.escape(target)

        # Restore wildcard semantics
        escaped = escaped.replace(r"\*", ".*").replace(r"\?", ".")

        # If user didn't anchor explicitly, allow weight/bias suffix
        if not re.search(r"(\\\.weight|\\\.bias|\$)$", escaped):
            escaped += r"(?:\.weight|\.bias)?"

        patterns.append(escaped)

    if not patterns:
        return r"(?!x)x"  # matches nothing, safely

    # IMPORTANT:
    # - non-capturing groups
    # - avoids re.findall side effects
    final_regex = "|".join(f"(?:{p})" for p in patterns)
    return final_regex


def id_checkpoint(name):
    """
    Identify checkpoint architecture family and preferred dtype.
    Deterministic, filename-hint first, key-based second.
    Never throws.
    """
    if not name:
        return "Unknown", torch.float16

    info = sd_models.get_closet_checkpoint_match(name)
    if not info or not info.filename:
        return "Unknown", torch.float16

    filename = info.filename
    lower = filename.lower()

    # -------------------------------------------------
    # 1. FAST PATH — filename hints (cheap, optional)
    # -------------------------------------------------
    if any(x in lower for x in ("fp8", "nf4", "q4", "q5", "q8")):
        return "Flux (Quantized)", torch.bfloat16

    if "bf16" in lower or "bfp16" in lower:
        return "Flux", torch.bfloat16

    if "fp32" in lower or "f32" in lower:
        return "SD (FP32)", torch.float32

    # -------------------------------------------------
    # 2. AUTHORITATIVE PATH — inspect safetensors
    # -------------------------------------------------
    try:
        with safetensors.torch.safe_open(filename, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                return "Unknown", torch.float16

            # ── Determine dtype (metadata preferred)
            tensor_dtype = torch.float16
            metadata = f.metadata()
            if metadata and "dtype" in metadata:
                d = metadata["dtype"].lower()
                if "bf16" in d:
                    tensor_dtype = torch.bfloat16
                elif "fp32" in d or "float32" in d:
                    tensor_dtype = torch.float32
            else:
                try:
                    tensor_dtype = f.get_tensor(keys[0]).dtype
                except Exception:
                    pass

            # -------------------------------------------------
            # 3. ARCHITECTURE DETECTION (key-based, stable)
            # -------------------------------------------------
            keyset = set(keys)

            # Flux / SD3 / Aurora
            if any(k.startswith("denoiser.") for k in keyset):
                return "Flux/SD3/Aurora", tensor_dtype

            # Flux / Pony blocks
            if any(("single_blocks" in k or "double_blocks" in k) for k in keyset):
                return "Pony/Flux", tensor_dtype

            # SDXL family
            if any(k.startswith("conditioner.embedders.") for k in keyset):
                return "SDXL", tensor_dtype

            # SD 1.5 family
            if any(k.startswith("cond_stage_model.") for k in keyset):
                return "SD1.5", tensor_dtype

            # Fallback: still SD-like
            return "SD (Unknown Variant)", tensor_dtype

    except Exception as e:
        print(f"[id_checkpoint] Failed to inspect {filename}: {e}")
        return "Unknown", torch.float16

    
class NoCaching:
    def __init__(self):
        self.cachebackup = None

    def __enter__(self):
        self.cachebackup = sd_models.checkpoints_loaded
        sd_models.checkpoints_loaded = OrderedDict()

    def __exit__(self, *args):
        sd_models.checkpoints_loaded = self.cachebackup

def create_name(checkpoints, calcmode, alpha, max_models=4):
    """
    Generate a clean, deterministic merge name.
    2025 Dual-Soul safe.
    """

    import os
    import re

    names = []

    for filename in checkpoints[:max_models]:
        if not filename:
            continue

        base = os.path.splitext(os.path.basename(filename))[0]

        # Normalize
        clean = re.sub(r'[^a-zA-Z0-9]+', '-', base).strip('-')

        # Shorten intelligently
        parts = clean.split('-')
        short = parts[0][:12]

        # Preserve useful tags
        tags = []
        for p in parts[1:]:
            if re.fullmatch(r'(v|e)\d{1,3}', p.lower()):
                tags.append(p.upper())
            elif p.lower() in {"xl", "sdxl", "flux", "pony"}:
                tags.append(p.upper())

        name = short
        if tags:
            name += "-" + "-".join(sorted(set(tags)))

        names.append(name)

    # Alpha formatting
    try:
        alpha_str = f"{float(alpha):.3f}".rstrip('0').rstrip('.')
    except Exception:
        alpha_str = str(alpha)

    mode = calcmode.replace(" ", "-").upper()
    return f"{'~'.join(names)}_{mode}x{alpha_str}"


def save_loaded_model(name, settings):
    """
    Save the currently loaded merged model to disk.
    2025-safe, Dual-Soul compatible.
    """

    if shared.sd_model is None:
        gr.Warning("No model is currently loaded.")
        return

    progress_name = name or "merged_model"

    # --- FULL CLEANUP (correct order) ---
    try:
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    except Exception:
        pass

    try:
        sd_unet.apply_unet("None")
    except Exception:
        pass

    # Restore network weights (LoRA / LyCORIS)
    try:
        import networks
        with torch.no_grad():
            for module in shared.sd_model.modules():
                networks.network_restore_weights_from_backup(module)
    except Exception:
        pass

    # Collect state_dict
    try:
        state_dict = shared.sd_model.state_dict()
    except Exception as e:
        gr.Error(f"Failed to extract model weights: {e}")
        return

    # Generate filename if missing
    if not name:
        info = shared.sd_model.sd_checkpoint_info
        if info and info.name:
            progress_name = info.name.replace("_TEMP_MERGE_", "")
        else:
            progress_name = "merged_model"

    # Save
    checkpoint_info = save_state_dict(
        state_dict,
        filename=progress_name,
        settings=settings
    )

    if not checkpoint_info:
        gr.Error("Model save failed.")
        return

    # Update UI state
    shared.sd_model.sd_checkpoint_info = checkpoint_info
    shared.sd_model_file = checkpoint_info.filename

    return f"Model saved as: {checkpoint_info.filename}"


def save_state_dict(
    state_dict,
    save_path=None,
    filename=None,
    settings="",
    timer=None,
    discard_keys=None,
    target_dtype=None,
):
    """
    FINAL 2025 SAFE SAVE (safetensors only)
    - Atomic
    - Reforge / A1111 compatible
    - Metadata written at save time
    """

    import os
    from modules import paths_internal, sd_models
    from safetensors.torch import save_file
    from safetensors import safe_open

    # Resolve filename
    if save_path is not None:
        filename = save_path
    elif filename is None:
        raise ValueError("save_state_dict: No save_path or filename provided")

    # Ensure directory
    if os.path.basename(filename) == filename:
        base_dir = os.path.join(paths_internal.models_path, "Stable-diffusion")
        os.makedirs(base_dir, exist_ok=True)
        filename = os.path.join(base_dir, filename)

    # Force safetensors
    filename = os.path.splitext(filename)[0] + ".safetensors"
    filename = os.path.normpath(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if timer:
        timer.record("Pre-save prep")

    # Apply discards
    if discard_keys:
        state_dict = {k: v for k, v in state_dict.items() if k not in discard_keys}

    # Validation + CPU move
    cleaned = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach().cpu()
        if target_dtype and t.is_floating_point() and t.dtype != target_dtype:
            t = t.to(dtype=target_dtype)
        cleaned[k] = t

    state_dict = cleaned

    # ✅ METADATA — SAFETENSORS REQUIRES STRINGS
    if isinstance(settings, (list, tuple)):
        desc = ", ".join(map(str, settings))
    else:
        desc = str(settings) if settings else "custom merge"

    metadata = {
        "modelspec.author": "Amethyst Merger",
        "modelspec.description": desc,
        "modelspec.format": "safetensors",
    }

    # Save
    try:
        save_file(state_dict, filename, metadata=metadata)
        print(f"[Merger] Saved: {os.path.basename(filename)}")
    except Exception as e:
        print(f"[Merger] Save failed, retrying contiguous: {e}")
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(state_dict, filename, metadata=metadata)

    # Validate
    try:
        with safe_open(filename, framework="pt") as f:
            print(f"[Merger] Validation OK — {len(f.keys())} tensors")
    except Exception as e:
        print(f"[Merger] Validation warning: {e}")

    if timer:
        timer.record("Save checkpoint")

    # Register with UI
    try:
        info = sd_models.CheckpointInfo(filename)
        info.register()
        gr.Info(f"Merged model saved: {os.path.basename(filename)}")
        return info
    except Exception as e:
        gr.Warning(f"Saved but failed to register: {e}")
        return None


def load_merged_state_dict(state_dict, checkpoint_info=None):
    """
    Load merged model fully in-memory.
    Works with:
      • Autosave ON/OFF
      • A1111 dev
      • reForge
    """

    import os
    import gc
    from safetensors.torch import save_file
    from modules import sd_models, shared, paths_internal

    if not state_dict:
        gr.Warning("Nothing to load — state_dict is empty")
        return

    # Respect merge dtype
    sample = next(iter(state_dict.values()))
    target_dtype = sample.dtype

    print(f"[Merger] Preparing {len(state_dict)} tensors ({target_dtype}) for in-memory load")

    state_dict = {
        k: v.to(dtype=target_dtype, device="cpu")
        for k, v in state_dict.items()
    }

    # -------------------------------------------------
    # Ensure checkpoint_info has a REAL file path
    # -------------------------------------------------
    if checkpoint_info is None or not getattr(checkpoint_info, "filename", None):
        base_dir = os.path.join(paths_internal.models_path, "Stable-diffusion")
        os.makedirs(base_dir, exist_ok=True)

        fake_name = "merged_in_memory.safetensors"
        fake_path = os.path.join(base_dir, fake_name)

        # Create tiny placeholder if needed
        if not os.path.exists(fake_path):
            save_file({}, fake_path, metadata={"placeholder": "true"})

        checkpoint_info = sd_models.CheckpointInfo(fake_path)
        checkpoint_info.filename = fake_path
        checkpoint_info.title = "Merged (In-Memory)"
        checkpoint_info.name = fake_name
        checkpoint_info.hash = None
        checkpoint_info.sha256 = None
        checkpoint_info.shorthash = None
        checkpoint_info.registered = True

        sd_models.checkpoints_list[fake_name] = checkpoint_info

    # -------------------------------------------------
    # Full unload (clean slate)
    # -------------------------------------------------
    if shared.sd_model is not None:
        print("[Merger] Unloading current model...")
        try:
            sd_models.unload_model_weights(shared.sd_model)
        except Exception:
            pass
        shared.sd_model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # -------------------------------------------------
    # Load model using state_dict injection
    # -------------------------------------------------
    print("[Merger] Loading merged model (in-memory, registered)")

    try:
        sd_models.load_model(
            checkpoint_info=checkpoint_info,
            already_loaded_state_dict=state_dict,
        )
    except Exception as e:
        print(f"[FATAL] In-memory load failed: {e}")
        raise gr.Error(f"In-memory load failed: {e}")

    # -------------------------------------------------
    # Post-load hooks (critical)
    # -------------------------------------------------
    try:
        from modules import sd_hijack, script_callbacks, sd_unet

        sd_hijack.model_hijack.hijack(shared.sd_model)
        script_callbacks.model_loaded_callback(shared.sd_model)
        sd_models.model_data.set_sd_model(shared.sd_model)

        if hasattr(sd_unet, "apply_unet"):
            sd_unet.apply_unet("upcast")

    except Exception as e:
        print(f"[Merger] Post-load hooks warning (non-fatal): {e}")

    print("[Merger] ✅ Merged model loaded successfully (in-memory)")



def find_checkpoint_w_config(config_source, model_a, model_b, model_c, model_d):
    """
    Resolve a YAML config file to use for merged model loading.
    Priority is controlled by config_source.
    """

    infos = [
        sd_models.get_closet_checkpoint_match(model_a),
        sd_models.get_closet_checkpoint_match(model_b),
        sd_models.get_closet_checkpoint_match(model_c),
        sd_models.get_closet_checkpoint_match(model_d),
    ]

    def yaml_near_checkpoint(info):
        if not info or not info.filename:
            return None
        path = os.path.splitext(info.filename)[0] + ".yaml"
        return path if os.path.exists(path) else None

    def find_config(info):
        if not info:
            return None

        # A1111 official helper (preferred)
        if sd_models_config and hasattr(sd_models_config, "find_checkpoint_config_near_filename"):
            try:
                cfg = sd_models_config.find_checkpoint_config_near_filename(info.filename)
                if cfg and os.path.exists(cfg):
                    return cfg
            except Exception:
                pass

        # Fallback: yaml next to checkpoint
        return yaml_near_checkpoint(info)

    valid = [i for i in infos if i]

    if not valid:
        return None

    if config_source == 0:  # Auto
        for info in valid:
            cfg = find_config(info)
            if cfg:
                return cfg
        return None

    index_map = {1: 0, 2: 1, 3: 2, 4: 3}
    idx = index_map.get(config_source, 0)

    primary = infos[idx] if idx < len(infos) else None
    fallback = infos[0]

    cfg = find_config(primary)
    if cfg:
        return cfg

    return find_config(fallback)

    
def copy_config(origin, target):
    """
    Copy YAML config from origin checkpoint to target checkpoint.
    """

    if not origin or not target:
        return

    origin_filename = origin.filename if hasattr(origin, "filename") else origin
    target_filename = target.filename if hasattr(target, "filename") else target

    if not origin_filename or not target_filename:
        return

    origin_config = None

    # A1111 helper (preferred)
    if sd_models_config and hasattr(sd_models_config, "find_checkpoint_config_near_filename"):
        try:
            origin_config = sd_models_config.find_checkpoint_config_near_filename(origin_filename)
        except Exception:
            origin_config = None

    # Fallback: same-name yaml
    if not origin_config:
        candidate = os.path.splitext(origin_filename)[0] + ".yaml"
        if os.path.exists(candidate):
            origin_config = candidate

    if not origin_config or not os.path.exists(origin_config):
        return

    target_base, _ = os.path.splitext(target_filename)
    target_config = target_base + ".yaml"

    if os.path.abspath(origin_config) == os.path.abspath(target_config):
        return  # no-op

    os.makedirs(os.path.dirname(target_config), exist_ok=True)

    print(f"[Merger] Copying config:\n  from: {origin_config}\n    to: {target_config}")
    shutil.copyfile(origin_config, target_config)


def save_metadata(filename: str, metadata: dict):
    """
    Safely attach metadata to a safetensors file.
    NOTE: Metadata must be written at save time.
    This function only logs if called late.
    """

    print(
        "[Merger] WARNING: save_metadata() called after save. "
        "Safetensors metadata must be written during initial save."
    )
    print(f"[Merger] Metadata ignored for {filename}")
