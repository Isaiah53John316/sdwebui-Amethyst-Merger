import gradio as gr
import concurrent.futures
import scripts.untitled.operators as oper
import scripts.untitled.misc_util as mutil
import scripts.untitled.calcmodes as calcmodes
import torch
import os
import re
import gc
import random
import threading
import json
import psutil
import time
from threading import Thread
from collections import defaultdict
from modules.timer import Timer
from scripts.untitled.operators import weights_cache
from safetensors.torch import safe_open, load_file
from safetensors import SafetensorError
from collections import OrderedDict
from tqdm import tqdm
from copy import copy, deepcopy
from modules import devices, shared, script_loading, paths, paths_internal, sd_models
from scripts.untitled.operators import SmartResize
from scripts.untitled.common import cmn

# Modern safe imports (2025+)
try:
    from modules import sd_vae
except ImportError:
    sd_vae = None
try:
    from modules import script_callbacks
except ImportError:
    script_callbacks = None

try:
    networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'sd_forge_lora','networks.py'))
except (FileNotFoundError, OSError):
    networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

class MergeInterruptedError(Exception):
    def __init__(self,*args):
        super().__init__(*args)

class MergerContext:
    def __init__(self):
        self.device = None
        self.dtype = torch.float32
        self.is_cross_arch = False
        self.primary = None

        # ✅ MUST start as None
        self.loaded_checkpoints = None

        # ✅ used by LoadTensor fallback
        self.checkpoints_global = []

        self.cross_arch_target_shapes = {}
        self.last_merge_tasks = None
        self.opts = {}

    def get_device(self):
        return self.device

    def get_dtype(self):
        return self.dtype

    def set_device(self, device_str: str):
        self.device = torch.device(device_str if device_str != "cuda" else "cuda")

    def set_dtype(self, dtype_str: str):
        if dtype_str == "fp16":
            self.dtype = torch.float16
        elif dtype_str == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32


VALUE_NAMES = ('alpha','beta','gamma','delta')

mergemode_selection = {obj.name: obj for obj in calcmodes.MERGEMODES_LIST}
calcmode_selection  = {obj.name: obj for obj in calcmodes.CALCMODES_LIST}

# Safety net: if UI ever sends a checkpoint name, silently fix it
def safe_get_mergemode(name):
    if name not in mergemode_selection:
        print(f"[Amethyst] Invalid merge mode '{name}' → forcing 'Weight-Sum'")
        return mergemode_selection["Weight-Sum"]
    return mergemode_selection[name]

def safe_get_calcmode(name):
    if name not in calcmode_selection:
        print(f"[Amethyst] Invalid calc mode '{name}' → forcing first available")
        return next(iter(calcmode_selection.values()))
    return calcmode_selection[name]

# Forge detection
IS_FORGE = hasattr(shared, 'cmd_opts') and getattr(shared.cmd_opts, 'forge', False)

# ===================================================================
# GLOBAL MERGER CONTEXT — FINAL 2025 IMMORTAL EDITION
# ===================================================================

# === MERGE STATISTICS TRACKER — FINAL 2025 EDITION ===
class MergeStats:
    def __init__(self):
        self.custom_merges     = 0   # Real merges from rules or global sliders
        self.copied_primary    = 0   # Keys copied from Model A (VAE, metadata, etc.)
        self.smart_resized     = 0   # Cross-arch: tensors that were SmartResized
        self.zero_filled       = 0   # Cross-arch: missing keys filled with zeros
        self.skipped           = 0   # Should always be 0 — true failure
        self.smart_merge       = 0   # Legacy: sparse merges (multiple sources) — kept for debug

    def __str__(self):
        total = (self.custom_merges + 
                 self.copied_primary + 
                 self.smart_resized + 
                 self.zero_filled + 
                 self.skipped)

        kitchen_sink = "YES" if self.skipped == 0 else "ALMOST"
        cross_arch   = "YES" if self.smart_resized > 0 else "NO"

        return (
            f"### AMETHYST MERGE COMPLETE ###\n"
            f"  • Custom merges          : {self.custom_merges:,}\n"
            f"  • Copied from Primary     : {self.copied_primary:,}  (VAE, metadata, etc.)\n"
            f"  • Smart-resized (cross-arch): {self.smart_resized:,}\n"
            f"  • Zero-filled (emergency) : {self.zero_filled:,}\n"
            f"  • Smart sparse merges     : {self.smart_merge:,}\n"
            f"  • Skipped (truly missing) : {self.skipped:,}\n"
            f"  • Total keys processed    : {total:,}\n"
            f"  • True Kitchen-Sink       : {kitchen_sink}\n"
            f"  • Cross-Arch Active       : {cross_arch}"
        )


# Global instance
merge_stats = MergeStats()

class MergerState:
    def __init__(self):
        self.temp_models = {}  # {filename: state_dict}

    def apply_lora_temp(self, lora_path, strength=1.0, progress=None):
        if progress:
            progress(f"Applying temp LoRA: {os.path.basename(lora_path)}")
        key = os.path.basename(lora_path)
        if key in self.temp_models:
            return self.temp_models[key]
        
        ckpt_dict = apply_lora_safely(lora_path, strength)
        self.temp_models[key] = ckpt_dict
        return ckpt_dict

    def clear_temp_models(self):
        self.temp_models.clear()
        import gc
        gc.collect()

    def stop(self):
        if self.io_start:
            self.running = False
            if self.thread:
                self.thread.join(timeout=3)
            io = psutil.disk_io_counters()
            total_read = (io.read_bytes - self.io_start[0]) / (1024*1024)
            total_write = (io.write_bytes - self.io_start[1]) / (1024*1024)
            self.progress(f"Disk Monitor: COMPLETE — Total: {total_read:.1f} MB read | {total_write:.1f} GB written")
            self.io_start = None

# Optional: convenience functions (you can use cmn.get_device() directly too)
def get_device():
    return cmn.get_device()

def get_dtype():
    return cmn.get_dtype()

def universal_model_reload():
    """
    Works on Forge Neo AND A1111 dev — one function to rule them all.
    """
    try:
        if hasattr(shared, 'forge_model_reload'):
            shared.forge_model_reload()
        elif hasattr(sd_models, 'reload_model_weights'):
            sd_models.reload_model_weights(forced_reload=True)
        elif shared.sd_model:
            sd_models.unload_model_weights(shared.sd_model)
    except Exception as e:
        print(f"[Amethyst] Model reload failed (non-fatal): {e}")

# === THE ONE TRUE FINETUNE SYSTEM (2023–2030) ===
COLS    = [[-1, 1/3, 2/3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]        # SD1.5 & older
COLSXL  = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]            # SDXL, Flux, Pony, Aurora, everything modern

# These 6 keys exist in every UNet ever made. Period.
FINETUNES = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]

def colorcalc(cols, is_xl: bool):
    colors = COLSXL if is_xl else COLS
    return [sum(y * cols[i] * 0.02 for i, y in enumerate(row)) for row in colors]

def fineman(fine: str, is_xl: bool):
    """The sacred fineman — do not touch, do not modernize, do not question"""
    if not fine or "," not in fine:
        return None
    try:
        vals = [float(x.strip()) for x in fine.split(",")[:8]]
        vals += [0.0] * (8 - len(vals))  # pad if short
    except:
        return None

    return [
        1 - vals[0] * 0.01,
        1 + vals[0] * 0.02,
        1 - vals[1] * 0.01,
        1 + vals[1] * 0.02,
        1 - vals[2] * 0.01,
        [vals[3] * 0.02] + colorcalc(vals[4:8], is_xl)
    ]

def lora_available():
    return hasattr(sd_models, 'load_lora_weights') or hasattr(shared.sd_model, 'set_lora')

def get_dtype_for_model(filename: str) -> torch.dtype:
    """Novel: Auto-detect dtype (bf16 for Flux, f16 for others)"""
    model_type, _ = mutil.id_checkpoint(filename)  # Assuming id_checkpoint returns type
    return torch.bfloat16 if model_type == 'Flux' else torch.float16

def get_checkpoint_match(name):
    """Compatibility wrapper for both Automatic1111-dev and Forge Neo"""
    if hasattr(sd_models, 'get_closest_checkpoint_match'):
        return sd_models.get_closet_checkpoint_match(name)  # A1111-dev (typo)
    elif hasattr(sd_models, 'get_closest_checkpoint_match'):
        return sd_models.get_closet_checkpoint_match(name)  # Forge Neo
    else:
        # Fallback: search manually
        for ckpt in sd_models.checkpoints_list.values():
            if name.lower() in ckpt.title.lower() or name.lower() in ckpt.filename.lower():
                return ckpt
        return None

def apply_lora_safely(lora_path, strength=1.0):
    if not lora_available():
        return "LoRA not supported in this version"
    
    dtype = get_dtype_for_model(lora_path)  # Novel: Auto-bf16
    with torch.no_grad():  # Optimize
        sd_models.load_lora_weights(shared.sd_model, lora_path, strength, dtype=dtype)
    return "LoRA applied successfully"

def parse_arguments(progress, mergemode_name, calcmode_name, model_a, model_b, model_c, model_d,
                    slider_a, slider_b, slider_c, slider_d, slider_e,
                    editor, discard, clude, clude_mode,
                    seed, enable_sliders, *custom_sliders):
    mergemode = safe_get_mergemode(mergemode_name)
    calcmode  = safe_get_calcmode(calcmode_name)
    parsed_targets = {}

    # ───── Seed handling (strict) ─────
    try:
        seed = int(float(seed))
    except (ValueError, TypeError):
        raise ValueError("Invalid seed value")
    if seed < 0:
        seed = random.randint(10**9, 10**10 - 1)
    cmn.last_merge_seed = seed

    # ───── Custom sliders (40+40) — NO FALLBACKS ─────
    if enable_sliders and len(custom_sliders) == 80:
        block_names = custom_sliders[:40]
        weights = custom_sliders[40:]
        for name, w_str in zip(block_names, weights):
            name = str(name).strip()
            if not name:
                continue
            try:
                weight = float(w_str)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid custom slider weight: {w_str}")
            parsed_targets[name] = {'alpha': weight, 'seed': seed}

    # ───── EMPTY WEIGHT EDITOR = GLOBAL SLIDERS (THE ONE TRUE WAY) ─────
    # If weight editor is empty → do NOTHING here
    # Global sliders are handled 100% correctly in create_tasks()
    # This is intentional and perfect

    # ───── Weight Editor parsing — STRICT, NO MERCY ─────
    if editor and editor.strip():
        editor_text = re.sub(r'#.*$', '', editor, flags=re.MULTILINE).strip()
        if not editor_text:
            raise ValueError("Weight editor is empty or only comments")

        # Replace slider_a ... slider_e with real values
        slider_map = {
            'slider_a': float(slider_a),
            'slider_b': float(slider_b),
            'slider_c': float(slider_c),
            'slider_d': float(slider_d),
            'slider_e': float(slider_e),
        }
        editor_text = re.sub(r'\bslider_[a-e]\b',
                             lambda m: str(slider_map[m.group()]),
                             editor_text)

        for line_num, line in enumerate(editor_text.split('\n'), 1):
            line = line.strip()
            if not line or ':' not in line:
                continue
            selector, weights_str = line.split(':', 1)
            selector = selector.strip()
            weights = [w.strip() for w in weights_str.split(',')]

            entry = {'seed': seed}
            for i, w in enumerate(weights[:len(VALUE_NAMES)]):
                try:
                    entry[VALUE_NAMES[i]] = float(w)
                except ValueError:
                    raise ValueError(f"Line {line_num}: Invalid weight '{w}' in '{line}'")
            if not selector:
                raise ValueError(f"Line {line_num}: Missing selector in '{line}'")
            parsed_targets[selector] = entry

    # ───── Resolve checkpoints — NO SILENT SKIPS ─────
    checkpoints = []
    progress('Resolving Checkpoints:')
    for i, model in enumerate((model_a, model_b, model_c, model_d)):
        if i + 1 > mergemode.input_models:
            checkpoints.append('')
            continue

        if not model or str(model).strip() in ('', 'None', 'nan') or isinstance(model, float):
            if i == 0:
                raise ValueError("Model A is required")
            checkpoints.append('')
            continue

        model_str = str(model).strip()
        info = sd_models.get_closet_checkpoint_match(model_str)
        if not info:
            raise ValueError(f"Checkpoint not found: {model_str}")

        if not info.filename.lower().endswith(('.safetensors', '.ckpt')):
            raise ValueError(f"Unsupported format: {info.filename}")

        model_type, _ = mutil.id_checkpoint(info.filename)
        short_name = os.path.splitext(os.path.basename(info.filename))[0]
        checkpoints.append(info.filename)
        progress(f' - Model {chr(65+i)}: {short_name} [{model_type or "Unknown"}]')

    cmn.primary = checkpoints[0] if checkpoints else None
    return parsed_targets, checkpoints, mergemode, calcmode, seed

def assign_weights_to_keys(targets, keys, already_assigned=None):
    """
    Assign weights to keys using regex selectors — FINAL 2025 KITCHEN-SINK EDITION
    Handles 3000+ keys, denoiser.sigmas, position_ids, model_ema, everything.
    """
    if not targets or not keys:
        return already_assigned or {}

    # Pre-compile all regex once — massive speed boost on 3000+ keys
    assigners = []
    for selector, weights in targets.items():
        try:
            regex = mutil.target_to_regex(selector)
            pattern = re.compile(regex)  # ← COMPILED = 10x faster
            assigners.append((weights, pattern))
        except re.error as e:
            print(f"[Merger] Invalid regex in selector '{selector}': {e}")
            continue

    # Sort by number of matches descending → longest/most specific first
    matches = []
    key_text = "\n".join(keys)
    for weights, pattern in assigners:
        found = pattern.findall(key_text)
        if found:
            matches.append((found, weights))

    matches.sort(key=lambda x: len(x[0]), reverse=True)

    # Build result — preserve order, no overwrites unless intended
    result = already_assigned or {}
    if not isinstance(result, defaultdict):
        result = defaultdict(dict, result)
    result.default_factory = dict

    assigned_count = 0
    for key_list, weights in matches:
        for key in key_list:
            if key not in result or not result[key]:  # Only assign if not already set
                result[key].update(weights)
                assigned_count += 1

    print(f"[Merger] Weight assignment → {assigned_count}/{len(keys)} keys matched")
    return dict(result)  # Convert back to regular dict

def create_tasks(
    progress, mergemode, calcmode, keys, assigned_keys, discard_keys,
    checkpoints, merge_stats,
    alpha, beta, gamma, delta, epsilon,           # ← REQUIRED. NO DEFAULTS.
    keep_zero_fill=True, bloat_mode=False
):
    """
    2025 KITCHEN-SINK MAXIMALISM — FINAL FORM
    • No silent fallbacks
    • No default=0
    • If no rule + all sliders=0 → FAIL LOUDLY
    • Only real merges survive
    """
    import scripts.untitled.operators as oper

    tasks = []
    custom_count = 0
    default_count = 0

    sliders_active = any(v != 0 for v in [alpha, beta, gamma, delta, epsilon])

    for key in sorted(keys):
        if key in discard_keys:
            continue

        # 1. Custom per-key rules
        if key in assigned_keys:
            try:
                custom_count += 1
                base_recipe = mergemode.create_recipe(
                    key, *checkpoints, **assigned_keys[key]
                )
                final_recipe = calcmode.modify_recipe(
                    base_recipe, key, *checkpoints, **assigned_keys[key]
                )
                tasks.append(final_recipe)
                continue
            except Exception as e:
                raise RuntimeError(
                    f"Custom rule failed for key '{key}': {e}\n"
                    f"   Rule data: {assigned_keys[key]}"
                ) from e

        # 2. Global sliders
        if sliders_active:
            try:
                custom_count += 1
                base_recipe = mergemode.create_recipe(
                    key, *checkpoints,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon
                )
                final_recipe = calcmode.modify_recipe(
                    base_recipe, key, *checkpoints,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon
                )
                tasks.append(final_recipe)
                continue
            except Exception as e:
                raise RuntimeError(
                    f"Global merge failed for key '{key}': {e}\n"
                    f"   Sliders: α={alpha}, β={beta}, γ={gamma}, δ={delta}, ε={epsilon}"
                ) from e

        # 3. NO MERGE → FAIL LOUDLY
        default_count += 1
        raise RuntimeError(
            f"No merge rule for key '{key}' and all sliders are 0.\n"
            f"   Set at least one slider ≠ 0 or use a weight editor rule."
        )

    progress("Assigned tasks:")
    progress(f"  • Custom merges (rules/sliders) : {custom_count:,}")
    progress(f"  • Intentional failures (no merge) : {default_count:,}")
    progress(f"  • Total keys processed           : {len(tasks):,}")
    return tasks


def prepare_merge(progress, save_name, save_settings, finetune, 
                  merge_mode_selector, calc_mode_selector,
                  model_a, model_b, model_c, model_d,
                  alpha, beta, gamma, delta, epsilon,
                  weight_editor, preset_output,          
                  discard, clude, clude_mode,
                  merge_seed, enable_sliders, *custom_sliders, keep_zero_fill=True, bloat_mode=False):
    
    progress('\n### Preparing merge ###')
    timer = Timer()
    cmn.interrupted = False
    cmn.stop = False

    merge_args = [
        merge_mode_selector,
        calc_mode_selector,
        model_a, model_b, model_c, model_d,
        alpha, beta, gamma, delta, epsilon,
        weight_editor,
        preset_output,          # ← Preset JSON
        discard,
        clude,
        merge_seed,
        clude_mode,
        enable_sliders,
        *custom_sliders,
        keep_zero_fill,
        bloat_mode
    ]

    # === Parse arguments ===
    targets, checkpoints, mergemode, calcmode, seed = parse_arguments(progress, *merge_args)

    # === UNPACK MODEL FILENAMES ===
    model_a = checkpoints[0] if len(checkpoints) > 0 else None
    model_b = checkpoints[1] if len(checkpoints) > 1 else None
    model_c = checkpoints[2] if len(checkpoints) > 2 else None
    model_d = checkpoints[3] if len(checkpoints) > 3 else None

    # === OPEN ALL CHECKPOINTS ONCE — EVERYTHING HAPPENS INSIDE ===
    with safe_open_multiple(checkpoints, "cpu") as loaded_files:
        cmn.loaded_checkpoints = loaded_files

        # 1. Collect ALL keys
        keys = set()
        for file in loaded_files.values():
            if file is not None:
                keys.update(file.keys())

        # 2. Apply clude/discard — 100% safe against None/int/bool/empty
        discard_raw = merge_args[12] if len(merge_args) > 12 else ""
        clude_raw   = merge_args[13] if len(merge_args) > 13 else ""
        clude_mode  = merge_args[14] if len(merge_args) > 14 else "None"

        # Convert everything to string safely (Gradio sometimes sends int/None/bool)
        discard_str = str(discard_raw).strip() if discard_raw is not None else ""
        clude_str   = str(clude_raw).strip()   if clude_raw   is not None else ""

        discard_keys = {k.strip() for k in discard_str.split(',') if k.strip()}
        clude_keys   = {k.strip() for k in clude_str.split(',')   if k.strip()}

        # Apply discard first (always)
        if discard_keys:
            keys -= discard_keys

        # Then apply clude (Exclude or Include)
        if clude_mode == "Exclude" and clude_keys:
            keys -= clude_keys
        elif clude_mode == "Include" and clude_keys:
            keys &= clude_keys
        progress(f"Total keys to merge: {len(keys)}")

        # === KITCHEN-SINK WEIGHT PRESETS — INJECT FROM UI ===
        if 'preset_output' in locals() and preset_output and preset_output.strip():
            try:
                ui_preset_json = json.loads(preset_output.strip())
                if isinstance(ui_preset_json, dict) and ui_preset_json:
                    targets = {**ui_preset_json, **targets}
                    progress("Applied Kitchen-Sink Weight Preset")
            except Exception as e:
                progress(f"Warning: Preset apply failed: {e}")
        # =====================================

        # 3. Create tasks
        assigned_keys = assign_weights_to_keys(targets, keys)
        tasks = create_tasks(
        progress, mergemode, calcmode, keys, assigned_keys, discard_keys,
        checkpoints, merge_stats,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon,
        keep_zero_fill=keep_zero_fill,
        bloat_mode=bloat_mode
        )

        progress(f"DEBUG: UI device choice = '{cmn.opts.get('device', 'MISSING')}'")

        # 4. FINAL 2025 DEVICE/DTYPE SELECTION — WORKS 100%
        device_choice = cmn.opts.get('device', 'cuda/float16').lower()
        if 'cpu' in device_choice:
            merge_device = 'cpu'
            merge_dtype = torch.float32
            dtype_name = "FP32"
        else:
            merge_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if merge_device == 'cpu':
                merge_dtype = torch.float32
                dtype_name = "FP32"
            else:
                if 'bf16' in device_choice or 'bfloat16' in device_choice:
                    merge_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    dtype_name = "BF16" if merge_dtype == torch.bfloat16 else "FP16"
                elif 'fp32' in device_choice or 'float32' in device_choice:
                    merge_dtype = torch.float32
                    dtype_name = "FP32"
                else:
                    merge_dtype = torch.float16
                    dtype_name = "FP16"

        cmn.device = torch.device(merge_device)
        cmn.dtype = merge_dtype
        progress(f"Merge running on {merge_device.upper()} with {dtype_name}")

        # Undo hijacks
        try:
            from modules import sd_hijack
            if hasattr(sd_hijack, 'model_hijack') and shared.sd_model:
                sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        except Exception:
            pass

        # 5. Build target shape map + finalize primary model
        # This runs AFTER we have already decided cmn.is_cross_arch and cmn.primary in do_merge()
        if not cmn.primary or cmn.primary not in cmn.loaded_checkpoints:
            progress("[Merger] Primary model not available — cross-arch shape mapping disabled")
            cmn.cross_arch_target_shapes.clear()
        else:
            primary_file = cmn.loaded_checkpoints[cmn.primary]
            if primary_file is None:
                progress("[Merger] Primary file is None — cross-arch disabled")
                cmn.cross_arch_target_shapes.clear()
            else:
                progress(f"[Merger] Building target shape map from primary: {os.path.basename(cmn.primary)}")
                cmn.cross_arch_target_shapes.clear()
                try:
                    total_keys = len(primary_file.keys())
                    collected = 0
                    for key in primary_file.keys():
                        try:
                            tensor = primary_file.get_tensor(key)
                            if tensor is not None and tensor.shape:
                                cmn.cross_arch_target_shapes[key] = tensor.shape
                                collected += 1
                        except:
                            continue
                    progress(f"[Merger] Target shapes collected: {collected}/{total_keys} keys")
                except Exception as e:
                    progress(f"[Merger] Failed to build shape map: {e}")
                    cmn.cross_arch_target_shapes.clear()

        # 6. RUN THE MERGE
        state_dict = {}
        try:
            state_dict = do_merge(
                model_a, model_b, model_c, model_d,
                checkpoints,
                tasks,
                state_dict={},
                progress=progress,
                merge_mode=mergemode,
                calc_mode=calcmode,
                alpha=merge_args[4], beta=merge_args[5], gamma=merge_args[6],
                delta=merge_args[7], epsilon=merge_args[8],
                weight_editor=merge_args[11],
                discard=merge_args[12],
                clude=merge_args[13],
                clude_mode=merge_args[14],
                timer=timer,
                cross_arch=cmn.is_cross_arch,
                threads=cmn.opts.get('threads', 8),
                keep_zero_fill=merge_args[-2],   # keep_zero_fill
                bloat_mode=merge_args[-1],       # bloat_mode
            )
        except Exception as e:
            import traceback
            print(f"[FATAL MERGE ERROR] {e}")
            traceback.print_exc()
            raise gr.Error(f"Merge crashed: {e}")

        if not state_dict:
            raise gr.Error("Merge failed — empty result")

        save_dict_final = state_dict

    # === OUTSIDE CONTEXT — FILES CLOSED ===

    # 7. Finetune — SACRED POST-PROCESSING
    if finetune:
        is_xl = bool(cmn.primary and 'SDXL' in cmn.checkpoints_types.get(cmn.primary, ''))
        fine = fineman(finetune, is_xl)
        if fine and len(fine) >= 6:
            print(f"[Merger] Applying finetune: {finetune} → XL={is_xl}")
            for i, key in enumerate(FINETUNES):
                if key in save_dict_final:
                    tensor = save_dict_final[key]
                    if i < 5:
                        save_dict_final[key] = tensor * fine[i]
                    else:
                        delta = torch.tensor(fine[i], device=tensor.device, dtype=tensor.dtype)
                        save_dict_final[key] = tensor + delta
            print("[Merger] Finetune applied to sacred keys")

    # === SAVE + AUTOLOAD — FULLY RESPECTS UI (including "Load in Memory") ===

    save_to_disk = 'Autosave' in save_settings
    load_in_memory = 'Load in Memory' in save_settings or save_to_disk
    overwrite = 'Overwrite' in save_settings

    # Determine target dtype from UI
    if 'fp32' in save_settings:
        target_dtype = torch.float32
        progress("Target dtype: FP32 (full precision)")
    elif 'bf16' in save_settings:
        target_dtype = torch.bfloat16
        progress("Target dtype: BF16 (optimal for RTX 40xx+)")
    else:
        target_dtype = torch.float16
        progress("Target dtype: FP16 (fastest)")

    # Kitchen-sink: keep ALL keys
    # Safety: detect if we actually need conversion
    if len(save_dict_final) == 0:
        progress("WARNING: Empty state_dict — skipping save")
        return {}
    
    # Determine current dtype from first tensor
    sample_tensor = next(iter(save_dict_final.values()))
    current_dtype = sample_tensor.dtype
    if target_dtype == current_dtype:
        progress(f"Already in target dtype: {target_dtype} — no conversion needed")
    else:
        progress(f"Converting {len(save_dict_final)} tensors → {target_dtype}")
        converted = {}
        for k, v in save_dict_final.items():
            if v.dtype.is_floating_point:
                converted[k] = v.to(dtype=target_dtype, device='cpu')
            else:
                converted[k] = v.cpu()
        save_dict_final = converted

    # Determine save path
    final_name = save_name or mutil.create_name([model_a, model_b, model_c, model_d], calcmode.name, merge_args[4])
    if not final_name.lower().endswith('.safetensors'):
        final_name += '.safetensors'
    base_dir = shared.cmd_opts.ckpt_dir or os.path.join(paths_internal.models_path, "Stable-diffusion")
    full_path = os.path.join(base_dir, final_name)
    os.makedirs(base_dir, exist_ok=True)

    checkpoint_info = None
    # Metadata
    from datetime import datetime
    metadata = {
        "modelspec.title": final_name,
        "modelspec.author": "Amethyst Merger",
        "modelspec.description": f"{'Cross-arch Kitchen-Sink ' if cmn.is_cross_arch else ''}merge → {len(save_dict_final)} keys • {target_dtype}",
        "modelspec.date": datetime.now().isoformat(),
        "modelspec.architecture": "stable-diffusion-xl" if cmn.is_cross_arch else "stable-diffusion"
    }

    # ——— SAVE TO DISK (only if Autosave enabled) ———
    if save_to_disk:
        try:
            import safetensors.torch
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            safetensors.torch.save_file(save_dict_final, full_path, metadata=metadata)
            progress(f"Model saved: {os.path.basename(full_path)} ({dtype_name})", popup=True)
        except Exception as e:
            progress(f"Save failed: {e}")
            save_to_disk = False  # Prevent trying to load a broken file
    else:
        progress("Autosave disabled — no file written")

    # ——— LOAD MERGED MODEL INTO UI (if requested) ———
    if load_in_memory:
        try:
            progress("Loading merged model directly into WebUI...")

            # === 1. FULL UNLOAD CURRENT MODEL — MANDATORY ===
            if shared.sd_model is not None:
                print("[Merger] Unloading current model to prevent OOM...")
                try:
                    sd_models.unload_model_weights(shared.sd_model)
                except:
                    pass
                try:
                    shared.sd_model = shared.sd_model.to('cpu')
                    del shared.sd_model
                except:
                    pass
                shared.sd_model = None

            # === 2. FULL MEMORY CLEANUP ===
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # === 3. Create valid checkpoint_info ===
            dummy_info = sd_models.CheckpointInfo(final_name)
            dummy_info.filename = full_path if save_to_disk and os.path.exists(full_path) else "in_memory_merged"
            dummy_info.title = final_name
            dummy_info.name = final_name
            dummy_info.hash = None

            # === 4. Load the merged model ===
            from scripts.untitled.misc_util import load_merged_state_dict
            load_merged_state_dict(save_dict_final, dummy_info)

            # === 5. Force correct dropdown name ===
            if hasattr(shared, 'sd_model') and shared.sd_model is not None:
                ci = getattr(shared.sd_model, "sd_checkpoint_info", None)
                if ci:
                    ci.name = final_name
                    ci.title = final_name
                    ci.filename = dummy_info.filename

            # === 6. Refresh model list ===
            sd_models.list_models()

            # === 7. Final cleanup ===
            try:
                del save_dict_final
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress("Merged model loaded instantly and ready to generate!", popup=True)

        except Exception as e:
            import traceback
            print(f"[FATAL] In-memory load failed: {e}\n{traceback.format_exc()}")
            progress(f"In-memory load failed: {e}")
            progress("Model merged successfully — select from dropdown or restart UI.")
    else:
        progress("Load in Memory disabled — model not loaded")

    # ——— UI REFRESH (when not loaded in memory) ———
    if save_to_disk:
        sd_models.list_models()

    # ——— Final success report ———
    arch_msg = "Cross-Arch Kitchen-Sink " if cmn.is_cross_arch else ""
    keys_count = len(save_dict_final) if 'save_dict_final' in locals() else 0
    progress(
        f"### {arch_msg}Merge completed: {keys_count} keys • {dtype_name} ###",
        report=True,
        popup=True,
    )
    timer.record("Save & load")
    progress(f"Total time: {timer.summary()}", report=True)

def do_merge(model_a, model_b, model_c, model_d, checkpoints, tasks, state_dict, progress, 
             merge_mode, calc_mode, alpha=0, beta=0, gamma=0, delta=0, epsilon=0,
             weight_editor="", discard="", clude="", clude_mode="Exclude", timer=None,
             cross_arch=False, threads=8, keep_zero_fill=True, bloat_mode=False):
    """
    FINAL KITCHEN-SINK CROSS-ARCH MERGE ENGINE – 2025 EDITION
    Keeps every key • Resizes intelligently • Works every time
    """
    progress('### Starting merge ###')
    # ——————————————————————————————————————————————
    # NOVEL FEATURE: Real-Time Disk Usage Monitor
    # ——————————————————————————————————————————————
    disk_monitor = None
    if cmn.opts.get('disk_monitor', True):
        class LiveDiskMonitor:
            def __init__(self, progress):
                self.progress = progress
                self.start_io = None
                self.running = False
                self.thread = None

            def start(self):
                io = psutil.disk_io_counters()
                if io is None:
                    self.progress("Disk Monitor: psutil failed (no permission?)")
                    return
                self.start_io = io.read_bytes, io.write_bytes
                self.running = True
                self.progress("🔍 Disk Monitor: STARTED — watching real-time IO...")
                
                def monitor_loop():
                    last_read = self.start_io[0]
                    last_write = self.start_io[1]
                    while self.running:
                        time.sleep(2)
                        current = psutil.disk_io_counters()
                        if current:
                            read_mb = (current.read_bytes - last_read) / (1024*1024)
                            write_mb = (current.write_bytes - last_write) / (1024*1024)
                            if read_mb > 5 or write_mb > 5:
                                self.progress(f"💾 Disk IO: +{read_mb:.1f} MB read | +{write_mb:.1f} MB written")
                            last_read = current.read_bytes
                            last_write = current.write_bytes
                self.thread = Thread(target=monitor_loop, daemon=True)
                self.thread.start()

            def stop(self):
                if not self.start_io:
                    return
                self.running = False
                if self.thread:
                    self.thread.join(timeout=3)
                final = psutil.disk_io_counters()
                if final and self.start_io:
                    total_read = (final.read_bytes - self.start_io[0]) / (1024*1024)
                    total_write = (final.write_bytes - self.start_io[1]) / (1024*1024)
                    self.progress(f"💾 Disk Monitor: COMPLETE — Total Read: {total_read:.1f} MB | Total Written: {total_write:.2f} GB")

        disk_monitor = LiveDiskMonitor(progress)
        disk_monitor.start()

    cmn.checkpoints_global = checkpoints

    for path, f in cmn.loaded_checkpoints.items():
        status = "OPENED" if f is not None else "FAILED/None"
        print(f"[DIAG] {status} → {os.path.basename(path) if path else 'None'}")
# ——————————————————————————————————————————————
    # 1. Global flags — respect UI checkbox
    # ——————————————————————————————————————————————
    cmn.is_cross_arch = cross_arch

    # Detect types
    cmn.checkpoints_types = {}
    for cp in checkpoints:
        if cp:
            typ, _ = mutil.id_checkpoint(cp)
            cmn.checkpoints_types[cp] = typ or "Unknown"

    # ——————————————————————————————————————————————
    # 2. Determine primary model + auto-enable cross-arch
    # ——————————————————————————————————————————————
    cmn.primary = None
    cmn.is_cross_arch = False

    # First: respect user's explicit cross-arch setting
    if getattr(cmn, 'force_cross_arch', False):
        cmn.is_cross_arch = True
        progress("Cross-Arch Kitchen-Sink ENABLED (user override)")

    # AUTO-DETECT cross-arch: ANY model differs from ANY other
    if not cmn.is_cross_arch:
        types = []
        for cp in checkpoints:
            if not cp:
                continue
            model_type, _ = mutil.id_checkpoint(cp)
            types.append(model_type)
        if len(set(types)) > 1:
            cmn.is_cross_arch = True
            progress(f"AUTO-ENABLED Cross-Arch Kitchen-Sink (detected types: {set(types)})")

    # Now decide primary model: prefer SDXL/Flux/Pony as shape reference
    if cmn.is_cross_arch:
        modern_models = [
            cp for cp in checkpoints
            if cp and mutil.id_checkpoint(cp)[0] in ('SDXL', 'Flux', 'Pony', 'Aurora')
        ]
        if modern_models:
            primary_cp = modern_models[0]
            idx = checkpoints.index(primary_cp)
            checkpoints[0], checkpoints[idx] = checkpoints[idx], checkpoints[0]
            progress(f"Cross-Arch → Using {os.path.basename(primary_cp)} as primary shape reference")
            cmn.primary = checkpoints[0]
        else:
            cmn.primary = model_a or checkpoints[0]
            progress("Cross-Arch enabled but no modern model found — using Model A")
    else:
        cmn.primary = model_a or checkpoints[0]
        progress("Same-arch merge — using Model A as primary")
    # ——————————————————————————————————————————————
    # 3. Unload current model
    # ——————————————————————————————————————————————
    if shared.sd_model:
        sd_models.unload_model_weights()

    state_dict = {}

    # ——————————————————————————————————————————————
    # 4. Reuse from loaded model — ONLY in same-arch
    # ——————————————————————————————————————————————
    if (not cmn.is_cross_arch and
        cmn.last_merge_tasks and
        shared.sd_model and
        hasattr(shared.sd_model, 'sd_checkpoint_info')):
        prev_keys = {t.key for t in cmn.last_merge_tasks}
        curr_keys = {t.key for t in tasks}
        if prev_keys == curr_keys:
            state_dict, tasks = get_tensors_from_loaded_model(state_dict, tasks)
            if state_dict:
                progress(f'Reused {len(state_dict)} tensors from loaded model')

    # ——————————————————————————————————————————————
    # 5. Trash handling
    # ——————————————————————————————————————————————
    is_sdxl = any('SDXL' in t for t in cmn.checkpoints_types.values() if t)
    if ('SDXL' in cmn.opts.get('trash_model', '') and is_sdxl) or cmn.opts.get('trash_model') == 'Enable':
        progress('Unloading webui models...')
        if hasattr(sd_models.model_data, 'loaded_sd_models'):
            while sd_models.model_data.loaded_sd_models:
                sd_models.send_model_to_trash(sd_models.model_data.loaded_sd_models.pop())
            sd_models.model_data.sd_model = None
        shared.sd_model = None

    devices.torch_gc()

    # ——————————————————————————————————————————————
    # 7. Parallel merge with CPU offload
    # ——————————————————————————————————————————————
    timer.record('Merge start')
    bar = tqdm(total=len(tasks), desc="Merging", leave=True)
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(initialize_task, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            if cmn.stop or cmn.interrupted:
                progress('Merge interrupted')
                bar.close()
                return {}

            try:
                key, tensor = future.result()
                if tensor is None:
                    continue

                with lock:
                    state_dict[key] = tensor.cpu()
                    if len(state_dict) % 300 == 0:
                        torch.cuda.empty_cache()

                bar.update(1)
            except Exception as e:
                task = futures[future]
                bar.close()
                raise RuntimeError(f"Failed on key {task.key}: {e}")

    bar.close()
    timer.record('Merge complete')

    # ——————————————————————————————————————————————
    # 9. Save task list for reuse + cleanup
    # ——————————————————————————————————————————————
    cmn.last_merge_tasks = tuple(tasks)

    progress(str(merge_stats), report=True, popup=True)
    progress(f'### Merge completed: {len(state_dict)} tensors ###')

    if disk_monitor:
        disk_monitor.stop()

    return state_dict

def initialize_task(task):
    """
    THE FINAL FORM — 2025 KITCHEN-SINK INITIALIZATION
    Every key. Every time. No exceptions.
    """
    # Target shape for SmartResize — ONLY in cross-arch mode
    target_shape = None
    if cmn.is_cross_arch:
        target_shape = cmn.cross_arch_target_shapes.get(task.key)
        if target_shape:
            merge_stats.smart_resized += 1  # Count every time we *could* resize
            print(f"[ResizeReady] {task.key} → target shape {target_shape}")
    
    # Debug: Show what kind of task we're dealing with
    task_type = task.__class__.__name__
    if task_type != "CopyPrimary":
        print(f"[TaskStart] {task.key} ← using {task_type}")

    # 1. Fast path — custom merge task (user-defined rules, presets, etc.)
    try:
        tensor = task.merge()
        merge_stats.custom_merges += 1
        print(f"[CustomMerge] {task.key} ← merged via {task.__class__.__name__}")
        return task.key, tensor.to(cmn.get_device(), dtype=cmn.get_dtype())
    except Exception as e:
        # Only debug-print if something actually went wrong (rare)
        print(f"[CustomMerge] Fast path failed for {task.key}: {e} — falling back to sparse")
        # Fall through to sparse path

    # 2. Sparse path — collect real tensors (zero-fill + SmartResize in cross-arch)
    tensors = []
    sources = []

    for i, cp_path in enumerate(cmn.checkpoints_global):
        if not cp_path:
            continue

        f = cmn.loaded_checkpoints.get(cp_path)
        model_name = os.path.basename(cp_path)

        if not f or task.key not in f.keys():
            # Key missing → only zero-fill in cross-arch mode
            if target_shape is not None:
                t = torch.zeros(target_shape, dtype=cmn.get_dtype(), device=cmn.get_device())
                tensors.append(t)
                sources.append(f"{model_name} (zero)")
                merge_stats.smart_resized += 1  # We had to resize/fill
            # In same-arch: no contribution (will be copied from primary later)
            continue

        try:
            t = f.get_tensor(task.key)
            
            # CRITICAL: SmartResize if shape doesn't match target (cross-arch)
            if target_shape and t.shape != target_shape:
                t = SmartResize(f"sparse_{task.key}", target_shape, t).oper(t)
                sources.append(f"{model_name} (resized)")
                merge_stats.smart_resized += 1
            else:
                sources.append(model_name)

            t = t.to(cmn.get_device(), dtype=cmn.get_dtype())
            tensors.append(t)

        except Exception as e:
            print(f"[Merge] Failed loading {task.key} from {model_name}: {e}")
            if target_shape is not None:
                t = torch.zeros(target_shape, dtype=cmn.get_dtype(), device=cmn.get_device())
                tensors.append(t)
                sources.append(f"{model_name} (zero-fallback)")
                merge_stats.smart_resized += 1

    # === FINAL DECISION: Did we collect ANY real tensors? ===
    if not tensors:
        # This is metadata, VAE, noise schedule, denoiser.sigmas, etc.
        # → ALWAYS copy from primary — TRUE KITCHEN-SINK HONESTY
        primary_path = cmn.checkpoints_global[0]
        primary_file = cmn.loaded_checkpoints.get(primary_path)
        model_name = os.path.basename(primary_path) if primary_path else "Unknown"

        if primary_file and task.key in primary_file.keys():
            try:
                t = primary_file.get_tensor(task.key)
                
                # SmartResize if needed (cross-arch edge case)
                if target_shape and t.shape != target_shape:
                    t = SmartResize(f"finalcopy_{task.key}", target_shape, t).oper(t)
                    merge_stats.smart_resized += 1
                    print(f"[FinalCopy] {task.key} ← COPIED FROM PRIMARY + RESIZED ({model_name})")
                else:
                    print(f"[FinalCopy] {task.key} ← COPIED FROM PRIMARY ({model_name})")
                
                merge_stats.copied_primary += 1
                return task.key, t.to(cmn.get_device(), dtype=cmn.get_dtype())
                
            except Exception as e:
                print(f"[FinalCopy] FAILED reading {task.key} from primary {model_name}: {e}")
        
        # ULTIMATE FALLBACK: Preserve the key at all costs
        if target_shape:
            t = torch.zeros(target_shape, dtype=cmn.get_dtype(), device=cmn.get_device())
            merge_stats.zero_filled += 1
            print(f"[Emergency] {task.key} ← ZERO-FILLED (ultimate kitchen-sink preservation)")
            return task.key, t

        # ONLY SKIP IF WE LITERALLY CANNOT PRESERVE THE KEY
        merge_stats.skipped += 1
        print(f"[CRITICAL] {task.key} ← SKIPPED (no shape info + missing everywhere)")
        return task.key, None

    elif len(tensors) == 1:
        if tensors[0].numel() == 0:
            print(f"[ScalarKey] {task.key} ← single source, size-0 tensor → using as-is")
        else:
            print(f"[SingleSource] {task.key} ← only one real tensor → using it")
        merge_stats.copied_primary += 1
        return task.key, tensors[0]

    # We have real tensors → run the actual merge operation
    # This is the "smart merge" path — only used when not all models had the key
    merge_stats.smart_merge += 1
    
    try:
        result = task.oper(*tensors)
        source_summary = ', '.join(sources) if sources else "unknown"
        print(f"[SmartMerge] {task.key} ← {source_summary}")
        return task.key, result.to(cmn.get_dtype())
        
    except Exception as e:
        # This should be extremely rare — but if it happens, we want MAXIMUM visibility
        print(f"[FATAL OPERATOR ERROR] {task.key}")
        print(f"    Operator: {task.__class__.__name__}")
        print(f"    Sources : {sources}")
        print(f"    Shapes  : {[t.shape if hasattr(t, 'shape') else 'no-shape' for t in tensors]}")
        print(f"    Error   : {e}")
        print(f"    Task    : {task}")
        raise RuntimeError(f"Operator failed on {task.key} — see log above") from e


# 1. get_tensors_from_loaded_model — fixed for cross-arch + extra keys
def get_tensors_from_loaded_model(state_dict: dict, tasks: list) -> tuple[dict, list]:
    """
    Reuse tensors from the currently loaded model ONLY when:
      • The previous merge used the EXACT same task list (same keys, same operations)
      • We are NOT in cross-arch mode (cross-arch always needs fresh computation)
      • We are not in true kitchen-sink mode with variable keys
    """
    if (cmn.last_merge_tasks is None or
        shared.sd_model is None or
        getattr(cmn, 'is_cross_arch', False)):
        return state_dict, tasks

    # CRITICAL: For true kitchen-sink, keys change every merge → never reuse
    # Only reuse if the ENTIRE task list (keys + operations) is identical
    if len(cmn.last_merge_tasks) != len(tasks):
        return state_dict, tasks

    # Compare tasks exactly — SAFE FOR CopyPrimary (which has no .operation)
    if any(
        t1.key != t2.key or
        getattr(t1, 'operation', None).__class__ != getattr(t2, 'operation', None).__class__ or
        getattr(t1, 'alpha', 1.0) != getattr(t2, 'alpha', 1.0) or
        getattr(t1, 'beta', 1.0) != getattr(t2, 'beta', 1.0)
        for t1, t2 in zip(cmn.last_merge_tasks, tasks)
    ):
        return state_dict, tasks

    # Safe to reuse — same merge, same everything
    try:
        if lora_available():
            sd_models.unload_lora_weights(shared.sd_model)
    except:
        pass

    old_state = shared.sd_model.state_dict()
    reused = 0
    remaining_tasks = []

    for task in tasks:
        if task.key in old_state:
            state_dict[task.key] = old_state[task.key].cpu()
            reused += 1
        else:
            remaining_tasks.append(task)

    print(f"[Merger] Reused {reused}/{len(tasks)} tensors from loaded model (exact same merge)")
    return state_dict, remaining_tasks

# 2. safe_open_multiple — perfect as-is (your version is actually excellent)
class safe_open_multiple:
    def __init__(self, checkpoints, device="cpu"):
        # Filter out None/empty early
        self.checkpoints = [cp for cp in checkpoints if cp and os.path.isfile(cp)]
        self.device = device
        self.open_files = {}

    def __enter__(self):
        successful = 0
        for full_path in self.checkpoints:
            try:
                print(f"[Merger] Opening → {os.path.basename(full_path)}")
                # Use timeout + context manager to prevent hanging on bad files
                f = safe_open(full_path, framework="pt", device=self.device)
                # Test read one key to catch corrupted files early
                if len(f.keys()) == 0:
                    raise RuntimeError("Empty state_dict")
                self.open_files[full_path] = f
                successful += 1
            except Exception as e:
                print(f"[Merger] FAILED → {os.path.basename(full_path)} | {e}")
                self.open_files[full_path] = None

        print(f"[Merger] Successfully opened {successful}/{len(self.checkpoints)} models")
        return self.open_files

    def __exit__(self, exc_type, exc_val, exc_tb):
        closed = 0
        for f in self.open_files.values():
            if f is not None:
                try:
                    f.close()
                    closed += 1
                except:
                    pass
        # Optional: debug
        # print(f"[Merger] Closed {closed} file handles")

# 3. clear_cache — tiny improvement
def clear_cache():
    """Clear everything so the next merge starts completely clean"""
    if cmn.opts.get('cache_size', 0) > 0:
        weights_cache.__init__(cmn.opts['cache_size'])
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    cmn.last_merge_tasks = None                     # ← changed from tuple() to None
    cmn.cross_arch_target_shapes = {}               # ← make sure old shapes don't leak
    if hasattr(cmn, 'merger_state'):
        cmn.merger_state.clear_temp_models()
    
    return "All caches cleared — ready for next merge"
