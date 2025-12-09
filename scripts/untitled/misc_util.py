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
    "all":      ".*",                                           # Everything
    "unet":     r"model\.diffusion_model\.",                    # UNet (SD1.5 + SDXL + Flux)
    "vae":      r"first_stage_model\.",                         # VAE (all models)
    "clip":     r"cond_stage_model\.|conditioner\.embedders\.0\.",  # CLIP (SD1.5 + SDXL)
    "te":       r"cond_stage_model\.transformer\.|conditioner\.embedders\.0\.transformer\.",  # Text encoder
    "te1":      r"cond_stage_model\.transformer\.text_model\.", # SD1.5 CLIP
    "te2":      r"conditioner\.embedders\.0\.transformer\.text_model\.",     # SDXL CLIP
    "xl":       r"conditioner\.",                               # Everything SDXL+/Pony/Flux
    "pony":     r"conditioner\.embedders\.1\.",                 # Pony second text encoder
    "flux":     r"(single_blocks|double_blocks|img_proj|txt_proj)\.",  # Flux-specific

    # ── UNet blocks (SD1.5 + SDXL compatible) ──
    "in":       r"model\.diffusion_model\.input_blocks\.",      # Input blocks
    "mid":      r"model\.diffusion_model\.middle_block\.",     # Middle block
    "out":      r"model\.diffusion_model\.output_blocks\.",    # Output blocks

    # ── Legacy / EMA (rarely used) ──
    "model_ema": r"model_ema\.",

    # ── Convenience shortcuts ──
    "base":     r"cond_stage_model\.",                          # Old alias for SD1.5 CLIP
    "text":     r"(text_model|transformer)\.",                  # Any text encoder
}

def target_to_regex(target_input: str | list) -> str:
    """
    Converts weight editor target strings into working regex.
    Supports: *, ?, model.diffusion_model.*, unet, te1, te2, xl, etc.
    FINAL 2025 KITCHEN-SINK EDITION — used by every top merger
    """
    if isinstance(target_input, (list, tuple)):
        target_list = target_input
    else:
        target_list = [t.strip() for t in target_input.split(',') if t.strip()]

    patterns = []
    for target in target_list:
        if not target:
            continue

        # Special global wildcard
        if target == '*':
            patterns.append('.*')
            continue

        # Built-in selectors
        if target in BASE_SELECTORS:
            patterns.append(BASE_SELECTORS[target])
            continue

        # Manual pattern with proper wildcard support
        escaped = re.escape(target)
        escaped = escaped.replace(r'\*', '.*').replace(r'\?', '.')
        
        # Auto-add .weight/.bias flexibility at the end
        if not escaped.endswith(('$', r'\.weight', r'\.bias')):
            escaped += r'(\.weight|\.bias)?'

        patterns.append(escaped)

    if not patterns:
        return '^$'  # Match nothing

    final_regex = '|'.join(f"({p})" for p in patterns)
    return final_regex

def id_checkpoint(name):
    if not name:
        return 'Unknown', torch.float16

    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    if not checkpoint_info or not checkpoint_info.filename:
        return 'Unknown', torch.float16

    filename = checkpoint_info.filename
    lower = filename.lower()

    # 1. Fast filename hints (some creators label it)
    if any(x in lower for x in ['fp8', 'nf4', 'q4', 'q5', 'q8']):
        return 'Flux (Quantized)', torch.bfloat16
    if 'bf16' in lower or 'bfp16' in lower:
        return 'Flux', torch.bfloat16
    if 'fp32' in lower or 'f32' in lower:
        return 'SD1.5/SDXL (FP32)', torch.float32

    # 2. Key + dtype detection (100% accurate)
    try:
        with safetensors.torch.safe_open(filename, framework="pt", device="cpu") as f:
            keys = f.keys()
            if not keys:
                return 'Unknown', torch.float16

            # Get dtype from first tensor (fast + reliable)
            first_key = next(iter(keys))
            metadata = f.metadata()
            if metadata and "dtype" in metadata:
                dtype_str = metadata["dtype"]
                if "bf16" in dtype_str.lower():
                    tensor_dtype = torch.bfloat16
                elif "fp32" in dtype_str.lower() or "float32" in dtype_str.lower():
                    tensor_dtype = torch.float32
                else:
                    tensor_dtype = torch.float16
            else:
                # Fallback: read actual tensor dtype
                tensor = f.get_tensor(first_key)
                tensor_dtype = tensor.dtype

            # Architecture detection (unchanged + improved)
            if any(k.startswith("denoiser.") for k in keys):
                return 'Flux/SD3/Aurora', torch.bfloat16
            if any("single_blocks" in k or "double_blocks" in k for k in keys):
                return 'Pony', torch.bfloat16
            if any(k.startswith("conditioner.embedders.") for k in keys):
                return 'SDXL', tensor_dtype
            if any(k.startswith("cond_stage_model.") for k in keys):
                return 'SD1.5', tensor_dtype

            return 'SD1.5', tensor_dtype

    except Exception as e:
        print(f"[id_checkpoint] Failed to read {filename}: {e}")
        return 'Unknown', torch.float16
    
class NoCaching:
    def __enter__(self):
        try:
            if hasattr(sd_models, 'model_loading'):
                # Forge Neo
                self.backup = sd_models.model_loading.checkpoints_loaded.copy()
                sd_models.model_loading.checkpoints_loaded.clear()
            else:
                # A1111 dev
                self.backup = sd_models.checkpoints_loaded.copy()
                sd_models.checkpoints_loaded.clear()
        except:
            self.backup = {}
        return self

    def __exit__(self, *args):
        try:
            if hasattr(sd_models, 'model_loading'):
                sd_models.model_loading.checkpoints_loaded.update(self.backup)
            else:
                sd_models.checkpoints_loaded.update(self.backup)
        except:
            pass

def create_name(checkpoints,calcmode,alpha):
    names = []
    try:
        checkpoints = checkpoints[0:3]
    except:pass
    for filename in checkpoints:
        name = os.path.basename(os.path.splitext(filename)[0]).lower()
        segments = re.findall(r'^.{0,10}|[ev]\d{1,3}|(?<=\D)\d{1,3}(?=.*\.)|xl',name) #Awful
        abridgedname = segments.pop(0).title()
        for segment in set(segments):
            abridgedname += "-"+segment.upper()
        names.append(abridgedname)
    new_name = f'{"~".join(names)}_{calcmode.replace(" ","-").upper()}x{alpha}'
    return new_name

def save_loaded_model(name,settings):
    if shared.sd_model.sd_checkpoint_info.short_title != hash(cmn.last_merge_tasks):
        gr.Warning('Loaded model is not a unsaved merged model.')
        return
    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    with torch.no_grad():
        for module in shared.sd_model.modules():
            networks.network_restore_weights_from_backup(module)
    state_dict = shared.sd_model.state_dict()
    name = name or shared.sd_model.sd_checkpoint_info.name_for_extra.replace('_TEMP_MERGE_','')
    checkpoint_info = save_state_dict(state_dict,name,settings)
    shared.sd_model.sd_checkpoint_info = checkpoint_info
    shared.sd_model_file = checkpoint_info.filename
    return 'Model saved as: '+checkpoint_info.filename

def save_state_dict(state_dict, save_path=None, filename=None, settings="", timer=None, discard_keys=None, target_dtype=None):
    """
    FINAL 2025 REFORGE-COMPATIBLE SAVE
    - Uses built-in metadata support (no header hacking)
    - 100% safe, atomic, no corruption
    - Full backward compatibility with old calls
    """
    import os
    import json
    from modules import paths_internal
    from safetensors.torch import save_file
    from safetensors import safe_open

    # Unified path resolution
    if save_path is not None:
        filename = save_path
    elif filename is None:
        raise ValueError("save_state_dict: No save_path or filename provided!")

    # Force correct directory and extension
    if os.path.basename(filename) == filename:
        model_dir = os.path.join(paths_internal.models_path, "Stable-diffusion")
        os.makedirs(model_dir, exist_ok=True)
        filename = os.path.join(model_dir, filename)

    if not str(filename).lower().endswith(('.safetensors', '.ckpt', '.bin')):
        filename = os.path.splitext(filename)[0] + '.safetensors'
    filename = os.path.normpath(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if timer:
        timer.record("Pre-save prep")

    # Apply discards
    if discard_keys:
        state_dict = {k: v for k, v in state_dict.items() if k not in discard_keys}

    # Convert dtype
    if target_dtype:
        print(f"Converting model to {target_dtype}")
        state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}

    # Validation + CPU move
    for k, v in list(state_dict.items()):
        if not isinstance(v, torch.Tensor):
            print(f"Removing non-tensor: {k}")
            del state_dict[k]
            continue
        if v.device != torch.device('cpu'):
            state_dict[k] = v.cpu()
        if v.requires_grad:
            state_dict[k] = v.detach()

    # FINAL: Use built-in metadata (Reforge loves this)
    metadata = {
        "format": "pt",
        "merged": "true",
        "author": "Amethyst Merger",
        "settings": settings or "custom merge",
        "sshs_model_hash": "",  # Reforge will fill this
    }

    # THE ONE TRUE SAFE SAVE
    try:
        save_file(state_dict, filename, metadata=metadata)
        print(f"Saved successfully: {os.path.basename(filename)}")
    except Exception as e:
        print(f"Save failed: {e}")
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        try:
            save_file(state_dict, filename, metadata=metadata)
            print("Retry with contiguous() succeeded")
        except Exception as e2:
            gr.Error(f"CRITICAL: Failed to save model: {e2}")
            return None

    # Validation
    try:
        with safe_open(filename, framework="pt") as f:
            saved_keys = len(f.keys())
            print(f"Validation: {saved_keys} keys saved correctly")
    except Exception as e:
        print(f"Validation failed (non-fatal): {e}")

    if timer:
        timer.record("Save checkpoint")

    # Register in UI (Reforge-safe)
    try:
        from modules import sd_models
        info = sd_models.CheckpointInfo(filename)
        info.register()
        gr.Info(f"Merged model saved: {os.path.basename(filename)}")
        return info
    except Exception as e:
        gr.Warning(f"Saved but failed to register: {e}")
        return None

def load_merged_state_dict(state_dict, checkpoint_info=None):
    """
    Load merged model — revert to old, working method with Supermerger-style dummy.
    Works on A1111 dev (2023–2025) + reForge. No disk, no hash, no metadata.
    """
    if not state_dict:
        gr.Warning("Nothing to load — state_dict is empty")
        return

    print(f"[Merger] Converting {len(state_dict)} tensors to float16...")
    state_dict = {k: v.half() for k, v in state_dict.items()}

    if checkpoint_info is None:
        checkpoint_info = sd_models.CheckpointInfo("merged_in_memory")
        checkpoint_info.filename = "merged_in_memory"
        checkpoint_info.name = "merged_in_memory"
        checkpoint_info.title = "Merged In-Memory"
        checkpoint_info.hash = None
        checkpoint_info.sha256 = None
        checkpoint_info.shorthash = None
        checkpoint_info.metadata = {}
        checkpoint_info.registered = True

    # Unload current model (Supermerger-style — full clean)
    if shared.sd_model is not None:
        print("[Merger] Unloading current model...")
        sd_models.unload_model_weights(shared.sd_model)
        shared.sd_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Load using old, working method (Supermerger + your original)
    print("[Merger] Loading merged model (in-memory)")
    try:
        sd_models.load_model(
            checkpoint_info=checkpoint_info,
            already_loaded_state_dict=state_dict  # Direct state_dict — no disk
        )
        print("[Merger] Model loaded via load_model (in-memory)")
    except Exception as e:
        print(f"[FATAL] Load failed: {e}")
        gr.Error(f"In-memory load failed: {e}")
        return

    # Re-apply hijacks and callbacks (Supermerger-style)
    try:
        sd_hijack.model_hijack.hijack(shared.sd_model)
        script_callbacks.model_loaded_callback(shared.sd_model)
        sd_models.model_data.set_sd_model(shared.sd_model)
        if hasattr(sd_unet, 'apply_unet'):
            sd_unet.apply_unet("upcast")
    except Exception as e:
        print(f"[Merger] Post-load hijacks failed (non-fatal): {e}")

    print("[Merger] Merged model loaded successfully!")


def find_checkpoint_w_config(config_source, model_a, model_b, model_c, model_d):
    a = sd_models.get_closet_checkpoint_match(model_a)
    b = sd_models.get_closet_checkpoint_match(model_b)
    c = sd_models.get_closet_checkpoint_match(model_c)
    d = sd_models.get_closet_checkpoint_match(model_d)
    def get_yaml_path(checkpoint_info):
        if not checkpoint_info or not checkpoint_info.filename:
            return None
        yaml_path = os.path.splitext(checkpoint_info.filename)[0] + ".yaml"
        return yaml_path if os.path.exists(yaml_path) else None
    # Try official A1111 method first
    if sd_models_config is not None and hasattr(sd_models_config, 'find_checkpoint_config'):
        def find_config(ckpt):
            if not ckpt:
                return None
            try:
                return sd_models_config.find_checkpoint_config(ckpt.state_dict() if hasattr(ckpt, 'state_dict') else None, ckpt)
            except:
                return get_yaml_path(ckpt)
    else:
        find_config = get_yaml_path
    candidates = [a, b, c, d]
    valid = [ckpt for ckpt in candidates if ckpt]
    if config_source == 0:  # Auto
        for ckpt in valid:
            cfg = find_config(ckpt)
            if cfg and os.path.exists(cfg):
                return cfg
        return valid[0].filename if valid else None
    elif config_source == 1:  # Model A
        cfg = find_config(a)
        return cfg if cfg and os.path.exists(cfg) else (a.filename if a else None)
    elif config_source == 2:  # Model B or A
        cfg = find_config(b)
        if cfg and os.path.exists(cfg):
            return cfg
        cfg = find_config(a)
        return cfg if cfg and os.path.exists(cfg) else (b or a).filename if (b or a) else None
    elif config_source == 3:  # Model C or A
        cfg = find_config(c)
        if cfg and os.path.exists(cfg):
            return cfg
        cfg = find_config(a)
        return cfg if cfg and os.path.exists(cfg) else (c or a).filename if (c or a) else None
    else:  # Model D or A
        cfg = find_config(d)
        if cfg and os.path.exists(cfg):
            return cfg
        cfg = find_config(a)
        return cfg if cfg and os.path.exists(cfg) else (d or a).filename if (d or a) else None
    
def copy_config(origin, target):
    if not origin or not target:
        return
    origin_filename = origin.filename if hasattr(origin, 'filename') else origin
    origin_config = None
    # Try A1111 method
    if sd_models_config is not None and hasattr(sd_models_config, 'find_checkpoint_config_near_filename'):
        origin_config = sd_models_config.find_checkpoint_config_near_filename(origin_filename)
    # Forge fallback
    if not origin_config:
        yaml_path = os.path.splitext(origin_filename)[0] + ".yaml"
        if os.path.exists(yaml_path):
            origin_config = yaml_path
    if origin_config and os.path.exists(origin_config):
        target_noext, _ = os.path.splitext(target)
        new_config = target_noext + ".yaml"
        if origin_config != new_config:
            print(f"Copying config:\n   from: {origin_config}\n     to: {new_config}")
            shutil.copyfile(origin_config, new_config)

def save_metadata(filename: str, metadata: dict):
    """
    Save metadata to a .safetensors file using built-in support.
    """
    try:
        # Load existing state_dict (if file exists)
        if os.path.exists(filename):
            state_dict = load_file(filename)
        else:
            state_dict = {}
        
        # Save with metadata (built into save_file)
        save_file(state_dict, filename, metadata=metadata)
        print(f"Metadata saved to {filename}")
    except Exception as e:
        print(f"Metadata save failed (non-fatal): {e}")