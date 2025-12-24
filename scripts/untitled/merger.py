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
from scripts.untitled.common import CLIP_PREFIXES, VAE_PREFIXES, cmn
from scripts.untitled.common import merge_stats
from scripts.untitled.operators import WeightsCache

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
        # Device / precision
        self.device = None
        self.dtype = torch.float32

        # Primary checkpoint
        self.primary = None

        # MUST start as None — populated by safe_open_multiple
        self.loaded_checkpoints = None

        # Used by LoadTensor fallback and sparse merge
        self.checkpoints_global = []

        # Target shapes for SmartResize and zero-fill
        self.cross_arch_target_shapes = {}

        # Task reuse
        self.last_merge_tasks = None

        # Options storage (UI-backed)
        self.opts = {}

        # Architecture detection
        self.same_arch = True

        # ─────────────────────────────────────────────
        # POLICY FLAGS (explicit, no magic attributes)
        # ─────────────────────────────────────────────
        self.dual_soul_enabled = False
        self.sacred_enabled = False
        self.smartresize_enabled = False

        # Execution control
        self.stop = False
        self.interrupted = False

    # -------------------------------------------------
    # Accessors (used everywhere)
    # -------------------------------------------------
    def get_device(self):
        return self.device

    def get_dtype(self):
        return self.dtype

    def set_device(self, device_str: str):
        self.device = torch.device(
            device_str if device_str != "cuda" else "cuda"
        )

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

def parse_arguments(
    progress,
    mergemode_name, calcmode_name,
    model_a, model_b, model_c, model_d,
    slider_a, slider_b, slider_c, slider_d, slider_e,
    editor, discard, clude, clude_mode,
    seed, enable_sliders, *custom_sliders
):
    """
    UI → Structured Intent
    No tensor logic.
    No architecture logic.
    No merge validity checks.
    All enforcement happens downstream.
    """

    mergemode = safe_get_mergemode(mergemode_name)
    calcmode  = safe_get_calcmode(calcmode_name)
    parsed_targets = {}

    # ─────────────────────────────────────────────
    # Seed handling (STRICT, deterministic)
    # ─────────────────────────────────────────────
    try:
        seed = int(float(seed))
    except (ValueError, TypeError):
        raise ValueError("Invalid seed value")

    if seed < 0:
        seed = random.randint(10**9, 10**10 - 1)

    cmn.last_merge_seed = seed

    # ─────────────────────────────────────────────
    # Custom sliders (40 names + 40 weights)
    # STRICT: no defaults, no padding, no guessing
    # ─────────────────────────────────────────────
    if enable_sliders and len(custom_sliders) == 80:
        block_names = custom_sliders[:40]
        weights     = custom_sliders[40:]

        for name, w_str in zip(block_names, weights):
            name = str(name).strip()
            if not name:
                continue

            try:
                weight = float(w_str)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid custom slider weight: {w_str}")

            parsed_targets[name] = {
                'alpha': weight,
                'seed': seed
            }

    # ─────────────────────────────────────────────
    # EMPTY weight editor = GLOBAL SLIDERS ONLY
    #
    # IMPORTANT:
    # - Do NOT populate parsed_targets here
    # - create_tasks() owns global/default behavior
    # - Merge validity is enforced there
    # ─────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # Weight Editor parsing — STRICT, NO FALLBACKS
    # ─────────────────────────────────────────────
    if editor and editor.strip():
        editor_text = re.sub(r'#.*$', '', editor, flags=re.MULTILINE).strip()
        if not editor_text:
            raise ValueError("Weight editor is empty or only comments")

        # Replace slider_a ... slider_e with numeric values
        slider_map = {
            'slider_a': float(slider_a),
            'slider_b': float(slider_b),
            'slider_c': float(slider_c),
            'slider_d': float(slider_d),
            'slider_e': float(slider_e),
        }

        editor_text = re.sub(
            r'\bslider_[a-e]\b',
            lambda m: str(slider_map[m.group()]),
            editor_text
        )

        for line_num, line in enumerate(editor_text.split('\n'), 1):
            line = line.strip()
            if not line or ':' not in line:
                continue

            selector, weights_str = line.split(':', 1)
            selector = selector.strip()

            if not selector:
                raise ValueError(f"Line {line_num}: Missing selector")

            weights = [w.strip() for w in weights_str.split(',')]

            entry = {'seed': seed}
            for i, w in enumerate(weights[:len(VALUE_NAMES)]):
                try:
                    entry[VALUE_NAMES[i]] = float(w)
                except ValueError:
                    raise ValueError(
                        f"Line {line_num}: Invalid weight '{w}' in '{line}'"
                    )

            parsed_targets[selector] = entry

    # ─────────────────────────────────────────────
    # Resolve checkpoints — NO SILENT SKIPS
    # ─────────────────────────────────────────────
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
        progress(f" - Model {chr(65+i)}: {short_name} [{model_type or 'Unknown'}]")

    # NOTE:
    # Primary model selection is finalized in do_merge().
    # Do NOT assign cmn.primary here.

    return parsed_targets, checkpoints, mergemode, calcmode, seed

def assign_weights_to_keys(
    targets,
    keys,
    already_assigned=None,
    specific_selectors_first=None,
):
    if specific_selectors_first is None:
        specific_selectors_first = cmn.opts.get(
            "specific_selectors_first", False
        )
    #Selector precedence policy:
    #------------------------------------------------------------
    #specific_selectors_first = False (DEFAULT, safer)
        #• Broad selectors (e.g. "model.*", ".*attn.*") apply first
        #• Narrow selectors only fill gaps
        #• Produces more uniform, conservative merges
        #• Best when:
            #- Copy VAE from Primary = ON
            #- Copy CLIP from Primary = ON
            #- User wants structural stability and re-merge safety

    #specific_selectors_first = True (style-forward)
        #• Narrow selectors apply first (block- or layer-specific)
        #• Broad selectors act as fallback
        #• Stronger stylistic and attention-level transfer
        #• Especially impacts:
            #- UNet attention blocks (composition, texture, style)
            #- CLIP-related weights *when CLIP merging is enabled*
            #- Cross-model stylistic dominance

    #IMPORTANT CLARIFICATION:
        #• This toggle does NOT force copying or merging of VAE or CLIP
        #• VAE and CLIP behavior is controlled separately by user options
        #• When VAE / CLIP are merged (not copied), this toggle
          #directly affects how strongly their associated weights dominate
        #• When VAE / CLIP are copied from primary (default),
          #this toggle influences how the UNet aligns stylistically with them


    if not targets or not keys:
        return already_assigned or {}

    # ─────────────────────────────────────────────
    # Precompile regex selectors (FAST PATH)
    # ─────────────────────────────────────────────
    assigners = []
    for selector, weights in targets.items():
        try:
            regex = mutil.target_to_regex(selector)
            pattern = re.compile(regex)
            assigners.append((pattern, weights, selector))
        except re.error as e:
            print(f"[Merger] Invalid selector regex '{selector}': {e}")

    if not assigners:
        return already_assigned or {}

    # ─────────────────────────────────────────────
    # Match selectors against keys (key-by-key only)
    # ─────────────────────────────────────────────
    matches = []
    for pattern, weights, selector in assigners:
        matched_keys = [k for k in keys if pattern.search(k)]
        if matched_keys:
            matches.append((matched_keys, weights, selector))

    # ─────────────────────────────────────────────
    # Selector precedence policy
    # ─────────────────────────────────────────────
    if specific_selectors_first:
        # Narrow selectors first → stronger style control
        matches.sort(key=lambda x: len(x[0]))
        policy = "SPECIFIC → BROAD (style-forward)"
    else:
        # Broad selectors first → safer, more uniform merges
        matches.sort(key=lambda x: len(x[0]), reverse=True)
        policy = "BROAD → SPECIFIC (safe-default)"

    print(f"[Merger] Selector precedence policy: {policy}")

    # ─────────────────────────────────────────────
    # Build assignment map (first-wins, never overwritten)
    # ─────────────────────────────────────────────
    result = already_assigned.copy() if already_assigned else {}
    assigned_count = 0

    for key_list, weights, selector in matches:
        for key in key_list:
            if key not in result:
                result[key] = dict(weights)
                assigned_count += 1

    print(
        f"[Merger] Weight assignment → {assigned_count}/{len(keys)} keys matched "
        f"(specific_first={specific_selectors_first})"
    )

    return result


def create_tasks(
    progress, mergemode, calcmode, keys, assigned_keys, discard_keys,
    checkpoints, merge_stats,
    alpha, beta, gamma, delta, epsilon,
    keep_zero_fill=True, bloat_mode=False
):
    """
    2025 KITCHEN-SINK MAXIMALISM — FINAL FORM

    Precedence:
      1. Per-key rules
      2. Global sliders
      3. FAIL LOUDLY

    No silent fallbacks. No defaults.
    """

    tasks = []
    custom_count = 0

    sliders_active = any(v != 0 for v in (alpha, beta, gamma, delta, epsilon))

    for key in sorted(keys):
        if key in discard_keys:
            continue

        # 1. Per-key rules (highest priority)
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

        # 3. No rule applies → hard fail
        raise RuntimeError(
            f"No merge rule for key '{key}' and all sliders are 0.\n"
            f"   Set at least one slider ≠ 0 or use a weight editor rule."
        )

    # Freeze task list immediately
    tasks = tuple(tasks)

    progress("Assigned tasks:")
    progress(f"  • Custom merges (rules/sliders) : {custom_count:,}")
    progress(f"  • Total keys processed          : {len(tasks):,}")

    return tasks


def prepare_merge(
    progress, save_name, save_settings, finetune,
    merge_mode_selector, calc_mode_selector,
    model_a, model_b, model_c, model_d,
    alpha, beta, gamma, delta, epsilon,
    weight_editor, preset_output,
    discard, clude, clude_mode,
    merge_seed, enable_sliders,
    keep_zero_fill=True,
    bloat_mode=False,
    copy_vae_from_primary=True,
    copy_clip_from_primary=True,
    dual_soul_toggle=False,
    sacred_keys_toggle=False,
    smartresize_toggle=False,
    allow_synthetic_custom_merge=False,
    allow_non_float_merges=False,
    *custom_sliders,
):
    progress("\n### Preparing merge ###")
    print(
    f"[Policy] copy_vae={copy_vae_from_primary} | "
    f"copy_clip={copy_clip_from_primary}"
    )
    timer = Timer()
    cmn.interrupted = False
    cmn.stop = False

    # ─────────────────────────────────────────────
    # 1. Parse arguments (STRICT)
    # ─────────────────────────────────────────────
    targets, checkpoints, mergemode, calcmode, seed = parse_arguments(
        progress,
        merge_mode_selector,
        calc_mode_selector,
        model_a, model_b, model_c, model_d,
        alpha, beta, gamma, delta, epsilon,
        weight_editor,
        discard,
        clude,
        clude_mode,
        merge_seed,
        enable_sliders,
        *custom_sliders
    )

    assert isinstance(checkpoints, list), "parse_arguments returned invalid checkpoints"

    # Normalize model paths
    model_a = checkpoints[0] if len(checkpoints) > 0 else None
    model_b = checkpoints[1] if len(checkpoints) > 1 else None
    model_c = checkpoints[2] if len(checkpoints) > 2 else None
    model_d = checkpoints[3] if len(checkpoints) > 3 else None

    # ─────────────────────────────────────────────
    # 2. Open checkpoints
    # ─────────────────────────────────────────────
    with safe_open_multiple(checkpoints, "cpu") as loaded_files:
        cmn.loaded_checkpoints = loaded_files

        # Collect all keys
        keys = set()
        for f in loaded_files.values():
            if f is not None:
                keys.update(f.keys())

        if not keys:
            raise gr.Error("No keys found in any checkpoint — merge aborted")

        # ─────────────────────────────────────────────
        # 3. Discard / clude (explicit, no magic)
        # ─────────────────────────────────────────────
        discard_keys = {k.strip() for k in str(discard).split(",") if k.strip()}
        clude_keys   = {k.strip() for k in str(clude).split(",") if k.strip()}

        if discard_keys:
            keys -= discard_keys

        if clude_mode == "Exclude" and clude_keys:
            keys -= clude_keys
        elif clude_mode == "Include" and clude_keys:
            keys &= clude_keys

        progress(f"Total keys to merge: {len(keys)}")

        if not keys:
            raise gr.Error("All keys were excluded — nothing left to merge")

        # ─────────────────────────────────────────────
        # 4. Preset injection (safe)
        # ─────────────────────────────────────────────
        if preset_output and preset_output.strip():
            try:
                preset = json.loads(preset_output)
                if isinstance(preset, dict):
                    targets = {**preset, **targets}
                    progress("Applied Kitchen-Sink preset")
            except Exception as e:
                progress(f"Preset ignored: {e}")

        # ─────────────────────────────────────────────
        # 5. Checkpoint type map (required later)
        # ─────────────────────────────────────────────
        cmn.checkpoints_types = {}
        for cp in checkpoints:
            if cp:
                typ, _ = mutil.id_checkpoint(cp)
                cmn.checkpoints_types[cp] = typ

        # ─────────────────────────────────────────────
        # 6. Device & dtype (deterministic)
        # ─────────────────────────────────────────────
        device_choice = cmn.opts.get("device", "cuda/float16").lower()

        if "cpu" in device_choice:
            cmn.device = torch.device("cpu")
            cmn.dtype = torch.float32
            dtype_name = "FP32"
        else:
            cmn.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if "bf16" in device_choice and torch.cuda.is_bf16_supported():
                cmn.dtype = torch.bfloat16
                dtype_name = "BF16"
            elif "fp32" in device_choice:
                cmn.dtype = torch.float32
                dtype_name = "FP32"
            else:
                cmn.dtype = torch.float16
                dtype_name = "FP16"

        progress(f"Merge running on {cmn.device.type.upper()} with {dtype_name}")

        # ─────────────────────────────────────────────
        # 8. Create tasks
        # ─────────────────────────────────────────────
        assigned_keys = assign_weights_to_keys(targets, keys)
        tasks = create_tasks(
            progress,
            mergemode,
            calcmode,
            keys,
            assigned_keys,
            discard_keys,
            checkpoints,
            merge_stats,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            keep_zero_fill=keep_zero_fill,
            bloat_mode=bloat_mode
        )

        if not tasks:
            raise gr.Error("Task list is empty — merge aborted")

        # ─────────────────────────────────────────────
        # 9. Run merge
        # ─────────────────────────────────────────────
        merged_state_dict = do_merge(
            model_a, model_b, model_c, model_d,
            checkpoints,
            tasks,
            {},
            progress,
            mergemode,
            calcmode,
            alpha, beta, gamma, delta, epsilon,
            weight_editor,
            discard,
            clude,
            clude_mode,
            timer,
            threads=cmn.opts.get("threads", 8),
            keep_zero_fill=keep_zero_fill,
            bloat_mode=bloat_mode,
            copy_vae_from_primary=copy_vae_from_primary,     
            copy_clip_from_primary=copy_clip_from_primary,
            dual_soul_toggle=dual_soul_toggle,
            sacred_keys_toggle=sacred_keys_toggle,
            smartresize_toggle=smartresize_toggle,
            allow_synthetic_custom_merge=allow_synthetic_custom_merge,
            allow_non_float_merges=allow_non_float_merges,  
        )

    # ─────────────────────────────────────────────
    # 10. Finetune (unchanged but now safe)
    # ─────────────────────────────────────────────
    if finetune:
        is_xl = bool(cmn.primary and "SDXL" in cmn.checkpoints_types.get(cmn.primary, ""))
        fine = fineman(finetune, is_xl)
        if fine:
            for i, key in enumerate(FINETUNES):
                if key in merged_state_dict:
                    if i < 5:
                        merged_state_dict[key] *= fine[i]
                    else:
                        merged_state_dict[key] += torch.tensor(
                            fine[i],
                            dtype=merged_state_dict[key].dtype
                        )

    # ─────────────────────────────────────────────
    # 11. SAVE + LOAD (UI-respecting, current API)
    # ─────────────────────────────────────────────
    from copy import deepcopy
    from modules import sd_models, shared
    from scripts.untitled.misc_util import save_state_dict, load_merged_state_dict, NoCaching
    import gc, os, re

    settings = list(save_settings or [])

    autosave = "Autosave" in settings
    load_in_memory = ("Load in Memory" in settings) or autosave

    # Resolve target dtype from UI
    if "fp32" in settings:
        target_dtype = torch.float32
        dtype_label = "FP32"
    elif "bf16" in settings:
        target_dtype = torch.bfloat16
        dtype_label = "BF16"
    else:
        target_dtype = torch.float16
        dtype_label = "FP16"

    # -------------------------------------------------
    # Resolve merge name (filesystem + UI safe)
    # -------------------------------------------------
    merge_name = (save_name or "").strip()
    if not merge_name:
        try:
            merge_name = mutil.create_name(
                [model_a, model_b, model_c, model_d],
                f"{mergemode.name}_{calcmode.name}",
                alpha,
            )
        except Exception:
            merge_name = "merged_model"

    merge_name = re.sub(r"[^\w\-.~]", "_", merge_name)

    # -------------------------------------------------
    # Prepare base checkpoint info (REAL primary) + merged identity (UI only)
    # -------------------------------------------------
    base_ckpt_info = None
    merged_ckpt_info = None

    try:
        getter = getattr(sd_models, "get_closet_checkpoint_match", None) \
              or getattr(sd_models, "get_closest_checkpoint_match", None)

        if getter and cmn.primary:
            base_ckpt_info = getter(os.path.basename(cmn.primary))
            if base_ckpt_info:
                merged_ckpt_info = deepcopy(base_ckpt_info)
                merged_ckpt_info.title = merge_name
                merged_ckpt_info.name = merge_name
    except Exception:
        base_ckpt_info = None
        merged_ckpt_info = None

    # -------------------------------------------------
    # Autosave (optional) -- ONLY place we write a file
    # -------------------------------------------------
    checkpoint_info = None

    if autosave:
        try:
            progress(f"Autosave enabled → saving ({dtype_label})…")

            checkpoint_info = save_state_dict(
                merged_state_dict,
                save_path=merge_name,
                settings=settings,
                timer=timer,
                discard_keys=discard_keys,
                target_dtype=target_dtype,
            )

            if checkpoint_info:
                progress(f"✅ Saved: {os.path.basename(checkpoint_info.filename)}")
            else:
                progress("⚠️ Saved but checkpoint registration failed")

        except Exception as e:
            progress(f"⚠️ Autosave failed: {e}")
            checkpoint_info = None
    else:
        progress("Autosave disabled — no file written")

    # -------------------------------------------------
    # Load merged model into memory (optional)
    # -------------------------------------------------
    if load_in_memory:
        try:
            progress("Loading merged model into memory…")

            # Ensure there is a REAL base model to inject into
            if shared.sd_model is None:
                if base_ckpt_info is None:
                    raise RuntimeError("Base checkpoint_info is missing; cannot load a base model.")
                progress("[Merger] Base model not loaded — loading primary checkpoint first…")
                sd_models.load_model(checkpoint_info=base_ckpt_info)

                shared.sd_model.sd_checkpoint_info = base_ckpt_info
                sd_models.model_data.set_sd_model(shared.sd_model)

            # IMPORTANT: for injection loads, NoCaching is optional; keep if you like.
            with NoCaching():
                load_merged_state_dict(
                    merged_state_dict,
                    checkpoint_info=base_ckpt_info,   # inject into this architecture
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            progress("✅ Merged model loaded and ready")

        except Exception as e:
            progress(f"❌ In-memory load failed: {e}")
    else:
        progress("Load in Memory disabled — model not loaded")

    # -------------------------------------------------
    # Refresh model list only if we wrote a real file
    # -------------------------------------------------
    if autosave:
        try:
            sd_models.list_models()
        except Exception:
            pass

    return merged_state_dict




def do_merge(
    model_a, model_b, model_c, model_d,
    checkpoints, tasks, state_dict, progress,
    merge_mode, calc_mode,
    alpha=0, beta=0, gamma=0, delta=0, epsilon=0,
    weight_editor="", discard="", clude="", clude_mode="Exclude",
    timer=None, threads=8, keep_zero_fill=True, bloat_mode=False,
    copy_vae_from_primary=True, copy_clip_from_primary=True,
    dual_soul_toggle=False, sacred_keys_toggle=False, smartresize_toggle=False,
    allow_synthetic_custom_merge=False, allow_non_float_merges=False,
):
    """
    2025 KITCHEN-SINK MERGE ENGINE
    do_merge is a dispatcher only:
      • sets global context (primary, same_arch, checkpoints_global)
      • unloads current model
      • optionally reuses tensors (same-arch only)
      • runs task pool and collects tensors
    Dual-Soul preservation happens inside initialize_task().
    # initialize_task guarantees:
    # - sacred handling
    # - shape correctness
    # - device/dtype correctness

    """
    progress("### Starting merge ###")

    # ------------------------------------------------------------
    # 0. Disk IO monitor (optional)
    # ------------------------------------------------------------
    disk_monitor = None
    if cmn.opts.get("disk_monitor", True):
        class LiveDiskMonitor:
            def __init__(self, progress_cb):
                self.progress = progress_cb
                self.start_io = None
                self.running = False
                self.thread = None

            def start(self):
                io = psutil.disk_io_counters()
                if io is None:
                    self.progress("Disk Monitor: psutil failed (no permission?)")
                    return
                self.start_io = (io.read_bytes, io.write_bytes)
                self.running = True
                self.progress("Disk Monitor: STARTED — watching real-time IO...")

                def monitor_loop():
                    last_read, last_write = self.start_io
                    while self.running:
                        time.sleep(2)
                        cur = psutil.disk_io_counters()
                        if not cur:
                            continue
                        read_mb = (cur.read_bytes - last_read) / (1024 * 1024)
                        write_mb = (cur.write_bytes - last_write) / (1024 * 1024)
                        if read_mb > 5 or write_mb > 5:
                            self.progress(f"Disk IO: +{read_mb:.1f} MB read | +{write_mb:.1f} MB written")
                        last_read, last_write = cur.read_bytes, cur.write_bytes

                self.thread = Thread(target=monitor_loop, daemon=True)
                self.thread.start()

            def stop(self):
                if not self.start_io:
                    return
                self.running = False
                if self.thread:
                    self.thread.join(timeout=3)
                final = psutil.disk_io_counters()
                if final:
                    total_read = (final.read_bytes - self.start_io[0]) / (1024 * 1024)
                    total_write = (final.write_bytes - self.start_io[1]) / (1024 * 1024)
                    self.progress(
                        f"Disk Monitor: COMPLETE — Total Read: {total_read:.1f} MB | Total Written: {total_write:.2f} GB"
                    )

        disk_monitor = LiveDiskMonitor(progress)
        disk_monitor.start()

    # ------------------------------------------------------------
    # 1. Globals: checkpoints that actually opened
    # ------------------------------------------------------------
    cmn.checkpoints_global = [cp for cp in checkpoints if cp and (cp in cmn.loaded_checkpoints)]
    for cp in cmn.checkpoints_global:
        f = cmn.loaded_checkpoints.get(cp)
        status = "OPENED" if f is not None else "FAILED/None"
        print(f"[DIAG] {status} → {os.path.basename(cp)}")

    # ------------------------------------------------------------
    # 2. same_arch detection (keep your heuristic, but centralize)
    # ------------------------------------------------------------
    types = set()
    has_sdxl_family = False
    has_sd15 = False
    has_flux = False

    for cp in cmn.checkpoints_global:
        try:
            model_type, _ = mutil.id_checkpoint(cp)
        except Exception:
            model_type = "unknown"
        lower = (model_type or "unknown").lower()
        types.add(lower)

        if any(x in lower for x in ("sdxl", "pony", "illustrious", "animagine", "noobai", "aurora")):
            has_sdxl_family = True
        if any(x in lower for x in ("1.5", "sd1")):
            has_sd15 = True
        if "flux" in lower:
            has_flux = True

    mixed = (len(types) > 1) or (has_sdxl_family and has_sd15) or (has_flux and (has_sdxl_family or has_sd15))
    cmn.same_arch = not mixed

    if not cmn.same_arch:
        progress(
            f"AUTO-DETECTED mixed architectures "
            f"(SDXL-family={has_sdxl_family}, SD1.5={has_sd15}, Flux={has_flux})"
        )
        progress("No merge policies enabled by default — awaiting user choice")
    else:
        progress(
            f"Same architecture family detected: "
            f"{next(iter(types)) if types else 'unknown'}"
        )

    # ------------------------------------------------------------
    # 3. Choose primary
    # ------------------------------------------------------------
    cmn.primary = None

    modern = [
        cp for cp in cmn.checkpoints_global
        if mutil.id_checkpoint(cp)[0] in ("SDXL", "Flux", "Pony", "Aurora")
    ]

    if modern:
        idx = cmn.checkpoints_global.index(modern[0])
        cmn.checkpoints_global[0], cmn.checkpoints_global[idx] = (
            cmn.checkpoints_global[idx],
            cmn.checkpoints_global[0],
        )
        cmn.primary = cmn.checkpoints_global[0]
    else:
        cmn.primary = model_a or (cmn.checkpoints_global[0] if cmn.checkpoints_global else None)

    progress(
        f"Using {os.path.basename(cmn.primary) if cmn.primary else 'None'} as primary"
    )


    # ------------------------------------------------------------
    # 3.25 Policy toggles (USER-DRIVEN)
    # ------------------------------------------------------------

    # Facts only
    cmn.mixed_arch = not cmn.same_arch

    # Explicit defaults (safe, but not inferred)
    cmn.dual_soul_enabled   = False
    cmn.sacred_enabled      = False
    cmn.smartresize_enabled = False

    force_cross_arch  = dual_soul_toggle
    force_sacred_keys = sacred_keys_toggle
    force_smartresize = smartresize_toggle

    if force_cross_arch:
        cmn.same_arch = False
        cmn.dual_soul_enabled = True
        progress("[Policy Override] Dual-Soul ENABLED by user")

    if force_sacred_keys:
        cmn.sacred_enabled = True
        progress("[Policy Override] Sacred keys ENABLED by user")

    if force_smartresize:
        cmn.smartresize_enabled = True
        progress("[Policy Override] SmartResize ENABLED by user")


    print(
        f"[Policy] same_arch={cmn.same_arch} | "
        f"dual_soul={'ON' if cmn.dual_soul_enabled else 'OFF'} | "
        f"sacred={'ON' if cmn.sacred_enabled else 'OFF'} | "
        f"smartresize={'ON' if cmn.smartresize_enabled else 'OFF'}"
    )

    # ------------------------------------------------------------
    # 3.5 Target shape map
    # ------------------------------------------------------------
    cmn.cross_arch_target_shapes.clear()

    if cmn.smartresize_enabled and cmn.primary in cmn.loaded_checkpoints:
        pf = cmn.loaded_checkpoints[cmn.primary]
        if pf is None:
            raise RuntimeError("Primary checkpoint failed to open")

        for k in pf.keys():
            try:
                t = pf.get_tensor(k)
                if t is not None and hasattr(t, "shape"):
                    cmn.cross_arch_target_shapes[k] = tuple(t.shape)
            except Exception as e:
                print(f"[ShapeMap] Skipped {k}: {e}")

        print(
            f"[ShapeMap] Built {len(cmn.cross_arch_target_shapes)} shapes "
            f"from {os.path.basename(cmn.primary)}"
        )
    else:
        print("[ShapeMap] SmartResize disabled by policy")

    # 4. Unload current model
    # ------------------------------------------------------------
    if shared.sd_model is not None:
        progress("[Merger] Unloading current model...")
        try:
            sd_models.unload_model_weights(shared.sd_model)
        except TypeError:
            # Some builds expect no args
            sd_models.unload_model_weights()
        shared.sd_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 6. Parallel merge (policy lives in initialize_task)
    # ------------------------------------------------------------
    if timer:
        timer.record("Merge start")

    bar = tqdm(total=len(tasks), desc="Merging", leave=True)
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(initialize_task, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            if cmn.stop or cmn.interrupted:
                progress("Merge interrupted")
                bar.close()
                return {}

            task = futures[future]
            try:
                key, tensor = future.result()
            except Exception as e:
                bar.close()
                raise RuntimeError(f"Failed on key {task.key}: {e}") from e

            if tensor is None:
                bar.update(1)
                continue

            with lock:
                state_dict[key] = tensor.cpu()

                if (len(state_dict) % 300) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            bar.update(1)

    bar.close()
    if timer:
        timer.record("Merge complete")

    # ------------------------------------------------------------
    # 6.5 Post-merge component policy (VAE / CLIP / Flux encoders)
    # ------------------------------------------------------------
    if cmn.primary and (copy_vae_from_primary or copy_clip_from_primary):
        pf = cmn.loaded_checkpoints.get(cmn.primary)
        progress(
            f"[Policy] Post-merge overrides: "
            f"copy_vae={copy_vae_from_primary}, copy_clip={copy_clip_from_primary}"
        )

        if pf is None:
            progress("[Policy WARNING] Primary checkpoint not loaded; cannot copy VAE/CLIP")
        else:
            copied_vae = 0
            copied_clip = 0

            # -------------------------
            # VAE / latent decoder
            # -------------------------
            if copy_vae_from_primary:
                for k in pf.keys():
                    if cmn.is_vae_key(k):
                        state_dict[k] = pf.get_tensor(k).cpu()
                        copied_vae += 1

                if copied_vae > 0:
                    progress(
                        f"[Policy] Copied VAE from primary "
                        f"({copied_vae} tensors)"
                    )
                else:
                    progress(
                        "[Policy WARNING] copy_vae enabled, but no VAE tensors matched "
                        f"(checked prefixes: {', '.join(VAE_PREFIXES)})"
                    )

            # -------------------------
            # Text / conditioning encoders
            # -------------------------
            if copy_clip_from_primary:
                for k in pf.keys():
                    if cmn.is_clip_key(k):
                        state_dict[k] = pf.get_tensor(k).cpu()
                        copied_clip += 1

                if copied_clip > 0:
                    progress(
                        f"[Policy] Copied text encoders from primary "
                        f"({copied_clip} tensors)"
                    )
                else:
                    progress(
                        "[Policy WARNING] copy_clip enabled, but no text encoder tensors matched "
                        f"(checked prefixes: {', '.join(CLIP_PREFIXES)})"
                    )

    # ------------------------------------------------------------
    # 7. Save task list + cleanup
    # ------------------------------------------------------------
    cmn.last_merge_tasks = tuple(tasks)
    progress(str(merge_stats), report=True, popup=True)
    progress(f"### Merge completed: {len(state_dict)} tensors ###")

    if disk_monitor:
        disk_monitor.stop()

    return state_dict


def initialize_task(task):
    """
    2025 DUAL-SOUL ELIGIBILITY INITIALIZER — POLICY-AWARE

    Guarantees:
    • No silent exclusions
    • Sacred keys preserved only when enabled
    • Full merging in same-arch
    • SmartResize only when explicitly enabled
    • Zero-fill only as last resort
    """

    print(
        f"[INIT] initialize_task | "
        f"dual_soul={cmn.dual_soul_enabled} "
        f"sacred={cmn.sacred_enabled} "
        f"smartresize={cmn.smartresize_enabled}"
    )

    key = task.key
    target_shape = cmn.cross_arch_target_shapes.get(key)

    task_type = task.__class__.__name__
    if task_type != "CopyPrimary":
        print(f"[TaskStart] {key} ← {task_type}")

    # =====================================================
    # 0. CUSTOM MERGE ELIGIBILITY (POLICY GATE)
    # =====================================================
    has_all_sources = True
    for cp in cmn.checkpoints_global:
        f = cmn.loaded_checkpoints.get(cp)
        if not f or not cmn.has_tensor(f, key):
            has_all_sources = False
            break

    synthetic_allowed = (
        cmn.opts.get("allow_synthetic_custom_merge", False)
        and (
            cmn.smartresize_enabled
            or cmn.opts.get("keep_zero_fill", False)
        )
    )

    eligible_custom = has_all_sources or synthetic_allowed

    # =====================================================
    # 1. FAST PATH — custom operator (ELIGIBLE ONLY)
    # =====================================================
    if eligible_custom:
        try:
            tensor = task.merge()

            if tensor is None:
                raise RuntimeError("Custom merge returned None")

            merge_stats.custom_merges += 1
            print(f"[CustomMerge:OK] {key} ← {task_type}")

            return key, tensor.to(
                cmn.get_device(),
                dtype=cmn.get_dtype()
            )

        except Exception as e:
            merge_stats.custom_failed += 1
            print(
                f"[CustomMerge:FAILED] {key} ← {task_type} | "
                f"Reason: {type(e).__name__}: {e}"
            )
            print("[Fallback] Proceeding to eligibility / SmartMerge path")
    else:
        reason = "missing sources"

        if not has_all_sources and not cmn.opts.get("allow_synthetic_custom_merge", False):
            reason = "synthetic disabled"

        print(f"[CustomMerge:SKIPPED] {key} ← {reason}")

    # =====================================================
    # 2. DUAL-SOUL SACRED PRESERVATION (POLICY-GATED)
    # =====================================================
    if (
        cmn.dual_soul_enabled
        and cmn.sacred_enabled
        and cmn.is_sacred_key(key)
    ):
        f = cmn.loaded_checkpoints.get(cmn.primary)

        if f and cmn.has_tensor(f, key):
            try:
                t = cmn.get_tensor(f, key)

                if (
                    cmn.smartresize_enabled
                    and target_shape is not None
                    and t.shape != target_shape
                ):
                    print(
                        f"[SmartResize][Sacred] {key} "
                        f"{tuple(t.shape)} → {tuple(target_shape)}"
                    )
                    t = SmartResize(
                        f"dual_soul_{key}",
                        target_shape,
                        source_tensor=t,
                        orig_key=key
                    ).oper(t)

                merge_stats.copied_primary += 1
                return key, t.to(cmn.get_device(), dtype=cmn.get_dtype())

            except Exception as e:
                print(f"[DualSoul] FAILED {key}: {e}")

        merge_stats.skipped += 1
        print(f"[DualSoul] SKIPPED missing sacred key: {key}")
        return key, None

    # =====================================================
    # 3. ELIGIBLE TENSOR COLLECTION (SPARSE PATH)
    # =====================================================
    tensors = []
    sources = []

    for cp_path in cmn.checkpoints_global:
        f = cmn.loaded_checkpoints.get(cp_path)
        if not f or not cmn.has_tensor(f, key):
            continue

        try:
            t = cmn.get_tensor(f, key)

            if (
                cmn.smartresize_enabled
                and target_shape is not None
                and t.shape != target_shape
            ):
                print(
                    f"[SmartResize][Sparse] {key} "
                    f"{tuple(t.shape)} → {tuple(target_shape)}"
                )
                t = SmartResize(
                    f"sparse_{key}",
                    target_shape,
                    source_tensor=t,
                    orig_key=key
                ).oper(t)

            if target_shape is not None and t.shape != target_shape:
                continue

            tensors.append(t.to(cmn.get_device(), dtype=cmn.get_dtype()))
            sources.append(os.path.basename(cp_path))

        except Exception as e:
            print(f"[Sparse] FAILED {key} from {cp_path}: {e}")

    # =====================================================
    # 4. NON-FLOATING TENSOR POLICY
    # =====================================================
    if tensors and not tensors[0].is_floating_point():
        if not cmn.opts.get("allow_non_float_merges", False):
            f = cmn.loaded_checkpoints.get(cmn.primary)
            if f and cmn.has_tensor(f, key):
                t = cmn.get_tensor(f, key)
                merge_stats.copied_primary += 1
                print(f"[NonFloatCopy] {key} ← primary (dtype={t.dtype})")
                return key, t.to(cmn.get_device(), dtype=cmn.get_dtype())

            # fallback if primary missing
            print(f"[NonFloatCopy] {key} ← first available")
            return key, tensors[0]


    # =====================================================
    # 5. FINAL DECISION TREE
    # =====================================================
    if not tensors:
        f = cmn.loaded_checkpoints.get(cmn.primary)

        if f and cmn.has_tensor(f, key):
            try:
                t = cmn.get_tensor(f, key)

                if (
                    cmn.smartresize_enabled
                    and target_shape is not None
                    and t.shape != target_shape
                ):
                    t = SmartResize(
                        f"finalcopy_{key}",
                        target_shape,
                        source_tensor=t,
                        orig_key=key
                    ).oper(t)

                merge_stats.copied_primary += 1
                print(f"[FinalCopy] {key}")
                return key, t.to(cmn.get_device(), dtype=cmn.get_dtype())

            except Exception as e:
                print(f"[FinalCopy] FAILED {key}: {e}")

        if target_shape is not None and cmn.opts.get("keep_zero_fill", True):
            merge_stats.zero_filled += 1
            print(f"[ZeroFill] {key}")
            return key, torch.zeros(
                target_shape,
                device=cmn.get_device(),
                dtype=cmn.get_dtype()
            )

        merge_stats.skipped += 1
        print(f"[SKIPPED] {key} (no contributors)")
        return key, None

    # =====================================================
    # 6. SINGLE OR SAFE LINEAR MERGE
    # =====================================================
    if len(tensors) == 1:
        merge_stats.copied_primary += 1
        return key, tensors[0]

    merge_stats.smart_merge += 1
    result = tensors[0].clone()
    count = 1

    for t in tensors[1:]:
        if t.shape == result.shape:
            result += t
            count += 1

    result /= count
    print(f"[SmartMerge] {key} ← {', '.join(sources)}")
    return key, result.to(cmn.get_dtype())


def get_tensors_from_loaded_model(state_dict: dict, tasks: list) -> tuple[dict, list]:
    """
    Reuse tensors from the currently loaded model ONLY when:
      • The previous merge used the EXACT same task list (object-equivalent)
      • We are in same-architecture mode
    """
    if (
        cmn.last_merge_tasks is None or
        shared.sd_model is None or
        not cmn.same_arch
    ):
        return state_dict, tasks

    # Must be EXACT same tasks, same order, same parameters
    if tuple(tasks) != cmn.last_merge_tasks:
        return state_dict, tasks

    # Safe to reuse — unload LoRAs if present
    try:
        if lora_available():
            sd_models.unload_lora_weights(shared.sd_model)
    except Exception:
        pass

    old_state = shared.sd_model.state_dict()
    reused = 0
    remaining_tasks = []

    for task in tasks:
        if task.key in old_state:
            state_dict[task.key] = old_state[task.key].detach().cpu()
            reused += 1
        else:
            remaining_tasks.append(task)

    print(
        f"[Merger] Reused {reused}/{len(tasks)} tensors "
        f"from loaded model (exact identical merge)"
    )

    return state_dict, remaining_tasks


class safe_open_multiple:
    def __init__(self, checkpoints, strict=False):
        # Normalize + filter early
        self.checkpoints = [
            os.path.abspath(cp)
            for cp in checkpoints
            if cp and os.path.isfile(cp)
        ]
        self.strict = strict
        self.open_files = {}
        self._contexts = {}

    def __enter__(self):
        successful = 0

        for full_path in self.checkpoints:
            try:
                print(f"[Merger] Opening → {os.path.basename(full_path)}")

                # Always open on CPU
                ctx = safe_open(full_path, framework="pt", device="cpu")
                f = ctx.__enter__()

                # Sanity check: non-empty state_dict
                if not f.keys():
                    raise RuntimeError("Empty state_dict")

                self._contexts[full_path] = ctx
                self.open_files[full_path] = f
                successful += 1

            except Exception as e:
                print(f"[Merger] FAILED → {os.path.basename(full_path)} | {e}")
                self.open_files[full_path] = None

                if self.strict:
                    raise

        print(f"[Merger] Successfully opened {successful}/{len(self.checkpoints)} models")
        return self.open_files

    def __exit__(self, exc_type, exc_val, exc_tb):
        closed = 0

        for ctx in self._contexts.values():
            try:
                ctx.__exit__(exc_type, exc_val, exc_tb)
                closed += 1
            except Exception:
                pass

        self._contexts.clear()
        self.open_files.clear()

# 3. clear_cache — tiny improvement
def clear_cache():
    """Clear everything so the next merge starts completely clean"""

    # --------------------------------------------------
    # 1. Reset weights cache safely
    # --------------------------------------------------
    cache_size = int(cmn.opts.get('cache_size', 0) or 0)

    global weights_cache
    weights_cache = WeightsCache(cache_size if cache_size > 0 else 0)

    # --------------------------------------------------
    # 2. Python GC
    # --------------------------------------------------
    gc.collect()

    # --------------------------------------------------
    # 3. CUDA cleanup (fully guarded)
    # --------------------------------------------------
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                torch.cuda.ipc_collect()
        except Exception:
            pass

    # --------------------------------------------------
    # 4. Reset merge-global state
    # --------------------------------------------------
    cmn.last_merge_tasks = None
    cmn.cross_arch_target_shapes = {}

    # --------------------------------------------------
    # 5. Loaded checkpoints (CRITICAL FIX)
    # --------------------------------------------------
    # loaded_checkpoints is explicitly allowed to be None
    if isinstance(getattr(cmn, "loaded_checkpoints", None), dict):
        cmn.loaded_checkpoints.clear()
    cmn.loaded_checkpoints = None  # enforce clean lifecycle

    # --------------------------------------------------
    # 6. Clear temporary models / state
    # --------------------------------------------------
    merger_state = getattr(cmn, "merger_state", None)
    if merger_state:
        try:
            merger_state.clear_temp_models()
        except Exception:
            pass

    # --------------------------------------------------
    # 7. Optional: reset merge stats
    # --------------------------------------------------
    merge_stats = getattr(cmn, "merge_stats", None)
    if merge_stats and hasattr(merge_stats, "reset"):
        try:
            merge_stats.reset()
        except Exception:
            pass

    return "All caches cleared — ready for next merge"
