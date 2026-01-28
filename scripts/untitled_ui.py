import gradio as gr
import os
import re
import json
import shutil
import torch
import tempfile
import safetensors
import safetensors.torch
from modules.shared import opts  # For defaults
from datetime import datetime  # For cleaner temp naming
from time import time  # ✅ ADDED
from modules import processing,scripts,sd_models,script_callbacks,shared,ui_components,paths,sd_samplers,ui,call_queue
from modules.ui_common import plaintext_to_html, create_refresh_button
from modules.processing import StableDiffusionProcessingTxt2Img  # ← ADD THIS
from scripts.untitled import merger,misc_util
from scripts.untitled.operators import weights_cache
from scripts.untitled import lora_merge
from scripts.untitled.misc_util import inject_checkpoint_components, extract_checkpoint_components

from scripts.untitled.common import cmn

from scripts.untitled.calcmodes import MERGEMODES_LIST, CALCMODES_LIST

mergemode_selection = {obj.name: obj for obj in MERGEMODES_LIST}
calcmode_selection  = {obj.name: obj for obj in CALCMODES_LIST}

# basic constants
extension_path = scripts.basedir()
ext2abs = lambda *x: os.path.join(extension_path,*x)

sd_checkpoints_path = os.path.join(paths.models_path,'checklora')
options_filename = ext2abs('scripts','untitled','options.json')
custom_sliders_examples = ext2abs('scripts','untitled','sliders_examples.json')
custom_sliders_presets = ext2abs('scripts','untitled','custom_sliders_presets.json')
merge_presets_filename = ext2abs('scripts','untitled','merge_presets.json')
merge_history_filename = ext2abs('scripts','untitled','merge_history.json')
loaded_slider_presets = None

try:
    with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
        EXAMPLE = file.read()
except FileNotFoundError:
    EXAMPLE = ""

model_a_keys = []

# ---------------------------
# Progress & Options classes
# ---------------------------
class Progress:
    def __init__(self):
        self.ui_report = []
        self.merge_keys = 0
        self.start_time = None
        self.total_keys = 0

    def start_merge(self, total_keys):
        self.start_time = time()
        self.total_keys = total_keys
        self.merge_keys = 0

    def get_eta(self):
        if self.merge_keys == 0 or not self.start_time:
            return "Calculating..."

        elapsed = time() - self.start_time
        rate = self.merge_keys / elapsed if elapsed > 0 else 0
        remaining = (self.total_keys - self.merge_keys) / rate if rate > 0 else 0

        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}m {seconds}s remaining"

    def update_progress(self, current: int, total: int) -> dict:
        """
        Updates internal state and returns a dict with live progress info.
        Used to update the HTML progress bar and merge report in real time.
        """
        self.merge_keys = current
        self.total_keys = total

        if total <= 0:
            pct = 0.0
            eta = "Calculating..."
        else:
            pct = (current / total) * 100
            elapsed = time() - (self.start_time or time())
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (total - current) / rate if rate > 0 else 0
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            eta = f"{minutes}m {seconds}s remaining" if current > 0 else "Calculating..."

        progress_html = (
            f"<div style='font-family: monospace; font-size: 14px; padding: 10px; background: #1e1e1e; "
            f"border-radius: 8px; border-left: 4px solid #8f8;'>"
            f"Progress: <b>{pct:.1f}%</b> • {current}/{total} keys • ETA: <b>{eta}</b>"
            f"</div>"
        )

        return {
            "progress_html": progress_html,
            "pct": pct,
            "eta": eta,
            "current": current,
            "total": total
        }

    def __call__(self, message, v=None, popup=False, report=False):
        if v is not None:
            message = f" - {message:<25}: {v}"
        if report:
            self.ui_report.append(message)
        if popup:
            gr.Info(message)
        print(message)

    def interrupt(self, message, popup=True):
        message = f"Merge interrupted: {message}"
        if popup:
            gr.Warning(message)
        self.ui_report = [message]
        raise merger.MergeInterruptedError(message)

    def get_report(self):
        return "\n".join(self.ui_report)

class Options:
    def __init__(self,filename):
        self.filename = filename
        try:
            with open(filename,'r') as file:
                self.options = json.load(file)
        except FileNotFoundError:
            self.options = dict()

    def create_option(self,key,component,component_kwargs,default):
        value = self.options.get(key) or default
        opt_component = component(value = value,**component_kwargs)
        opt_component.do_not_save_to_config = True
        self.options[key] = value
        def opt_event(value): self.options[key] = value
        opt_component.change(fn=opt_event, inputs=opt_component)
        return opt_component

    def __getitem__(self,key):
        return self.options.get(key)

    def save(self):
        with open(self.filename,'w') as file:
            json.dump(self.options,file,indent=4)
        gr.Info('Options saved')

    def get(self, key, default=None):
        return self.options.get(key, default)

cmn.opts = Options(options_filename)

# ---------------------------
# Helper functions for UI logic
# ---------------------------
_DEFAULT_SLIDER_MARKER = (-1, 2, 0.01)  # This is perfect. Keep exactly like this.

def _choose_slider_configs(mergemode, calcmode):
    """
    Return slider configs + info strings for all 20 sliders (a-t).
    CalcMode overrides MergeMode only if it explicitly defines a slider config
    different from the sentinel value.
    """
    # If calc mode defines at least one custom slider → use calc mode priority
    use_calc = (
        hasattr(calcmode, 'slid_a_config') and
        getattr(calcmode, 'slid_a_config', None) != _DEFAULT_SLIDER_MARKER
    )

    sliders = []
    slider_letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
    
    for letter in slider_letters:
        cfg_attr = f'slid_{letter}_config'
        info_attr = f'slid_{letter}_info'
        
        if use_calc:
            cfg = getattr(calcmode, cfg_attr, getattr(mergemode, cfg_attr))
            info = getattr(calcmode, info_attr, getattr(mergemode, info_attr))
        else:
            cfg = getattr(mergemode, cfg_attr)
            info = getattr(mergemode, info_attr)
        
        sliders.extend([cfg, info])
    
    return tuple(sliders)


def _required_counts(mergemode, calcmode):
    """
    Safely calculate how many sliders and models are actually needed.
    Handles all edge cases and guarantees 1–20 sliders, 1–4 models.
    """
    base_sliders = getattr(mergemode, 'input_sliders', 20) or 20
    base_models  = getattr(mergemode, 'input_models', 4) or 4

    # Count how many sliders the CalcMode actually customizes
    custom_slider_attrs = ('slid_a_config', 'slid_b_config', 'slid_c_config',
                           'slid_d_config', 'slid_e_config', 'slid_f_config',
                           'slid_g_config', 'slid_h_config', 'slid_i_config',
                           'slid_j_config', 'slid_k_config', 'slid_l_config',
                           'slid_m_config', 'slid_n_config', 'slid_o_config',
                           'slid_p_config', 'slid_q_config', 'slid_r_config',
                           'slid_s_config', 'slid_t_config')
    custom_slider_count = sum(
        1 for attr in custom_slider_attrs
        if getattr(calcmode, attr, None) not in (None, _DEFAULT_SLIDER_MARKER)
    )

    # Explicit override from CalcMode (rare, but supported)
    explicit_sliders = getattr(calcmode, 'input_sliders', None)
    explicit_models  = getattr(calcmode, 'input_models', None)

    req_sliders = base_sliders
    if explicit_sliders is not None:
        try:
            req_sliders = max(req_sliders, int(explicit_sliders))
        except:
            pass
    req_sliders = max(req_sliders, custom_slider_count)

    req_models = explicit_models if explicit_models is not None else base_models
    try:
        req_models = int(req_models)
    except:
        req_models = base_models

    # Clamp to valid ranges
    req_sliders = max(1, min(int(req_sliders), 20))
    req_models  = max(1, min(int(req_models), 4))

    return req_sliders, req_models


def mode_changed(merge_mode_name, calc_mode_name):
    import inspect
    
    mergemode = merger.mergemode_selection[merge_mode_name]
    calcmode = merger.calcmode_selection[calc_mode_name]

    # Descriptions
    merge_desc = mergemode.description
    calc_desc = calcmode.description

    # Slider configs (20 supported)
    configs = _choose_slider_configs(mergemode, calcmode)
    
    # Unpack all 20 sliders (40 values total: 20 configs + 20 infos)
    sliders_data = []
    for i in range(20):
        cfg = configs[i*2]
        info = configs[i*2 + 1]
        sliders_data.append((cfg, info))

    # Required counts
    req_sliders, req_models = _required_counts(mergemode, calcmode)

    # Extract default slider values from calcmode.modify_recipe signature
    try:
        sig = inspect.signature(calcmode.modify_recipe)
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
    except:
        defaults = {}
    
    # Map slider letters to parameter names
    slider_params = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 
                     'iota', 'kappa', 'lambda_', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon']
    
    # Help text - show first 5 sliders in help
    header = f"{mergemode.name} • {calcmode.name}"
    slider_help_text = (
        f"<pre style='font-family: monospace; background: #111; padding: 12px; border-radius: 6px;'>"
        f"{header}\n\n"
        f"α (alpha)  : {sliders_data[0][1]}\n"
        f"β (beta)   : {sliders_data[1][1]}\n"
        f"γ (gamma)  : {sliders_data[2][1]}\n"
        f"δ (delta)  : {sliders_data[3][1]}\n"
        f"ε (epsilon): {sliders_data[4][1]}\n\n"
        f"Required: {req_sliders} slider(s), {req_models} model(s)"
        f"</pre>"
    )

    # Slider updates (dynamic min/max/label + hide unused)
    slider_updates = []
    slider_labels = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ']
    slider_letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']

    for i in range(20):
        cfg, info = sliders_data[i]
        label = f"slider_{slider_letters[i]} [{slider_labels[i]}] ({info})"
        
        # Get default value for this slider from calcmode defaults
        param_name = slider_params[i]
        default_value = defaults.get(param_name, 0.5)  # Fallback to 0.5 if not found

        if i < req_sliders:
            update = gr.update(
                minimum=cfg[0], maximum=cfg[1], step=cfg[2],
                label=label,
                value=default_value,  # Apply calcmode's default value
                interactive=True, visible=True
            )
        else:
            update = gr.update(
                value=0,
                interactive=False,
                visible=False  # Fully hides unused sliders
            )
        slider_updates.append(update)

    # Model dropdowns (hide/disable unused)
    model_updates = [
        gr.update(interactive=i < req_models, visible=i < req_models)
        for i in range(4)
    ]

    # Merge button compatibility
    compatible = ('all' in calcmode.compatible_modes or 
                  mergemode.name in calcmode.compatible_modes)
    merge_button_update = gr.update(
        interactive=compatible,
        variant='primary' if compatible else 'secondary'
    )

    return [
        gr.update(value=merge_desc),
        gr.update(value=calc_desc),
        *slider_updates,  # All 20 sliders
        gr.update(value=slider_help_text),
        *model_updates,   # All 4 models
        merge_button_update
    ]
# ---------------------------
# Utility UI helpers
# ---------------------------
def get_checkpoints_list(sort):
    checkpoints_list = [x.title for x in sd_models.checkpoints_list.values() if x.is_safetensors]
    if sort == 'Newest first':
        sort_func = lambda x: os.path.getctime(sd_models.get_closet_checkpoint_match(x).filename)
        checkpoints_list.sort(key=sort_func, reverse=True)
    return checkpoints_list


def get_lora_list_with_info():
    """
    Returns list of LoRAs for CheckboxGroup.
    Format expected by gr.CheckboxGroup: list[str] (full paths as values)
    Shows nice label with size + type hint.
    Handles Lora/, lora/, LyCORIS/, and user override.
    """
    # Respect user's custom LoRA directory first
    lora_dirs = []
    if hasattr(shared.opts, "lora_dir") and shared.opts.lora_dir and os.path.isdir(shared.opts.lora_dir):
        lora_dirs.append(shared.opts.lora_dir)

    # Standard locations
    base = paths.models_path
    lora_dirs.extend([
        os.path.join(base, "Lora"),
        os.path.join(base, "lora"),
        os.path.join(base, "LoRA"),
        os.path.join(base, "LyCORIS"),
        os.path.join(base, "lycoris"),
    ])

    seen_paths = set()
    loras = []

    for directory in lora_dirs:
        if not os.path.isdir(directory):
            continue
        for file in sorted(os.listdir(directory)):
            if not file.lower().endswith(('.safetensors', '.pt', '.ckpt')):
                continue

            full_path = os.path.join(directory, file)
            if full_path in seen_paths:
                continue  # true deduplication by full path
            seen_paths.add(full_path)

            try:
                size_mb = os.path.getsize(full_path) // (1024 * 1024)
                dtype_hint = ""
                model_hint = ""

                with safetensors.torch.safe_open(full_path, framework="pt", device="cpu") as f:
                    keys = f.keys()
                    if any("bf16" in k for k in keys):
                        dtype_hint = "bf16"
                    else:
                        dtype_hint = "f16"

                    # Very fast model type detection
                    if any(k.startswith("lora_unet_down_blocks_4_") for k in keys):
                        model_hint = "[SDXL]"
                    elif any(k.startswith("lora_unet_down_blocks_") and "_4_" not in k for k in keys):
                        model_hint = "[SD1.5]"
                    elif any("single_blocks" in k for k in keys):
                        model_hint = "[Flux]"
                    elif any("transformer_blocks" in k for k in keys):
                        model_hint = "[SD3/Pony]"

                # ← FIXED: missing closing quote + parenthesis
                label = f"{file} — {size_mb}MB • {dtype_hint} {model_hint}"
            except Exception:
                label = f"{file} — {size_mb}MB"

            # CheckboxGroup uses the full_path as value
            loras.append(full_path)

    return loras


def refresh_models(sort):
    sd_models.list_models()
    checkpoints_list = get_checkpoints_list(sort)
    return (
        gr.update(choices=checkpoints_list),
        gr.update(choices=checkpoints_list),
        gr.update(choices=checkpoints_list),
        gr.update(choices=checkpoints_list)
    )

def save_custom_sliders(name,*sliders):
    new_preset = {name:sliders}
    try:
        with open(custom_sliders_presets,'r') as file:
            sliders_presets = json.load(file)
    except FileNotFoundError:
        shutil.copy(custom_sliders_examples,custom_sliders_presets)
        with open(custom_sliders_presets,'r') as file:
            sliders_presets = json.load(file)
    sliders_presets.update(new_preset)
    with open(custom_sliders_presets,'w') as file:
        json.dump(sliders_presets,file,indent=0)
    gr.Info('Preset saved')

def get_slider_presets():
    global loaded_slider_presets
    try:
        with open(custom_sliders_presets,'r') as file:
            loaded_slider_presets = json.load(file)
    except FileNotFoundError:
        shutil.copy(custom_sliders_examples,custom_sliders_presets)
        with open(custom_sliders_presets,'r') as file:
            loaded_slider_presets = json.load(file)
    return sorted(list(loaded_slider_presets.keys()))

def load_slider_preset(name):
    global loaded_slider_presets
    preset = loaded_slider_presets.get(name, None)
    if preset is None: return [gr.update()]*26
    return [gr.update(value=x) for x in preset]


# ---------------------------
# UI: build tabs (updated with status definition)
# ---------------------------
def on_ui_tabs():
    # COMPATIBILITY WRAPPER FOR CHECKPOINT MATCHING
    def get_checkpoint_match(name):
        """Works with both Automatic1111-dev and Forge Neo"""
        if hasattr(sd_models, 'get_closest_checkpoint_match'):
            return sd_models.get_closet_checkpoint_match(name)  # A1111-dev (typo)
        elif hasattr(sd_models, 'get_closest_checkpoint_match'):
            return sd_models.get_closet_checkpoint_match(name)  # Forge Neo
        else:
            for ckpt in sd_models.checkpoints_list.values():
                if name.lower() in ckpt.title.lower() or name.lower() in ckpt.filename.lower():
                    return ckpt
            return None
    
    # CORRECT: Returns plain list of checkpoint names
    def refresh_models(sort_mode="Alphabetical"):
        from modules import sd_models
        import os
        
        sd_models.list_models()
        checkpoints = [cp.title for cp in sd_models.checkpoints_list.values()]
        
        if sort_mode == "Newest first":
            checkpoints.sort(
                key=lambda x: os.path.getmtime(get_checkpoint_match(x).filename),
                reverse=True
            )
        else:
            checkpoints.sort(key=str.lower)
        
        return checkpoints

    # Get initial list once
    initial_checkpoints = refresh_models("Alphabetical")

    # Wrapper for live updates (returns gr.update objects)
    def update_dropdowns(sort_mode):
        new_list = refresh_models(sort_mode)
        return (
            gr.update(choices=new_list),
            gr.update(choices=new_list),
            gr.update(choices=new_list),
            gr.update(choices=new_list)
        )

    # -------------------------------------------------
    # Main UI
    # -------------------------------------------------
    with gr.Blocks() as cmn.blocks:
        with gr.Tab("Merge"):
            dummy_component = gr.Textbox(visible=False, interactive=True)

            with ui_components.ResizeHandleRow():
                with gr.Column():
                    # === LOGS ===
                    with gr.Accordion("Merge Status & Logs", open=True, elem_classes="amethyst-log-section"):
                        merge_report = gr.Textbox(
                            label="Merge Report",
                            lines=12,
                            elem_id="amethyst_merge_report",
                            interactive=False,
                            show_copy_button=True,
                        )
                        progress_bar = gr.HTML("<div style='font-family: monospace; padding: 10px;'>Ready</div>")

                    with gr.Accordion("Detailed Logs", open=False):
                        detailed_logs = gr.Textbox(lines=20, interactive=False, show_copy_button=True)
                        gr.Button("Clear Logs").click(fn=lambda: "", outputs=detailed_logs)

                    # === MODEL SELECTION ===                    # === MODEL CHECKPOINT DROPDOWNS — FINAL 2025 IMMORTAL EDITION ===
                    # • Model A always defaults to currently loaded model (never None)
                    # • B/C/D default to "None" (safe for 2-model merges)
                    # • Refresh/Sort never wipes your selection
                    # • Works perfectly on A1111 dev + reForge + Forge Neo

                    current_model_title = shared.opts.sd_model_checkpoint or "None"

                    def get_checkpoint_choices():
                        return ["None"] + [cp.title for cp in sd_models.checkpoints_list.values()]

                    with gr.Row():
                        slider_scale = 8
                        with gr.Column(variant='compact', min_width=150, scale=slider_scale):
                            with gr.Row():
                                model_a = gr.Dropdown(
                                    choices=get_checkpoint_choices(),
                                    label="model_a [Primary]",
                                    value=current_model_title,           # ← NEVER None
                                    type="value",
                                    scale=slider_scale,
                                    elem_id="amethyst_model_a"
                                )
                                swap_models_AB = gr.Button('Swap', elem_classes=["tool"], scale=1)
                            model_a_info = gr.HTML(plaintext_to_html('None | None', classname='untitled_sd_version'))
                            model_a.change(fn=checkpoint_changed, inputs=model_a, outputs=model_a_info)

                        with gr.Column(variant='compact', min_width=150, scale=slider_scale):
                            with gr.Row():
                                model_b = gr.Dropdown(
                                    choices=get_checkpoint_choices(),
                                    label="model_b [Secondary]",
                                    value="None",                        # ← safe default
                                    type="value",
                                    scale=slider_scale
                                )
                                swap_models_BC = gr.Button('Swap', elem_classes=["tool"], scale=1)
                            model_b_info = gr.HTML(plaintext_to_html('None | None', classname='untitled_sd_version'))
                            model_b.change(fn=checkpoint_changed, inputs=model_b, outputs=model_b_info)

                        with gr.Column(variant='compact', min_width=150, scale=slider_scale):
                            with gr.Row():
                                model_c = gr.Dropdown(
                                    choices=get_checkpoint_choices(),
                                    label="model_c [Tertiary]",
                                    value="None",
                                    type="value",
                                    scale=slider_scale
                                )
                                swap_models_CD = gr.Button('Swap', elem_classes=["tool"], scale=1)
                            model_c_info = gr.HTML(plaintext_to_html('None | None', classname='untitled_sd_version'))
                            model_c.change(fn=checkpoint_changed, inputs=model_c, outputs=model_c_info)

                        with gr.Column(variant='compact', min_width=150, scale=slider_scale):
                            with gr.Row():
                                model_d = gr.Dropdown(
                                    choices=get_checkpoint_choices(),
                                    label="model_d [Supplementary]",
                                    value="None",
                                    type="value",
                                    scale=slider_scale
                                )
                                refresh_button = gr.Button('Refresh', elem_classes=["tool"], scale=1)
                            model_d_info = gr.HTML(plaintext_to_html('None | None', classname='untitled_sd_version'))
                            model_d.change(fn=checkpoint_changed, inputs=model_d, outputs=model_d_info)

                        # Sort dropdown
                        checkpoint_sort = gr.Dropdown(
                            choices=['Alphabetical', 'Newest first'],
                            value='Alphabetical',
                            label='Sort Models',
                            scale=1,
                            min_width=120
                        )

                    # === SWAP LOGIC (unchanged — perfect as-is) ===
                    def swapvalues(x, y):
                        return gr.update(value=y), gr.update(value=x)
                    swap_models_AB.click(fn=swapvalues, inputs=[model_a, model_b], outputs=[model_a, model_b])
                    swap_models_BC.click(fn=swapvalues, inputs=[model_b, model_c], outputs=[model_b, model_c])
                    swap_models_CD.click(fn=swapvalues, inputs=[model_c, model_d], outputs=[model_c, model_d])

                    # === REFRESH LOGIC — FINAL IMMORTAL VERSION (handles NaN, None, everything) ===
                    def update_dropdowns(sort_mode):
                        new_choices = get_checkpoint_choices()
                        # Model A: preserve current loaded model (or fall back to first real model)
                        safe_a = current_model_title
                        if not safe_a or safe_a == "None" or safe_a not in new_choices:
                            # Find first real checkpoint (not "None")
                            real_models = [c for c in new_choices if c != "None"]
                            safe_a = real_models[0] if real_models else "None"

                        return (
                            gr.update(choices=new_choices, value=safe_a),
                            gr.update(choices=new_choices, value="None"),
                            gr.update(choices=new_choices, value="None"),
                            gr.update(choices=new_choices, value="None")
                        )

                    refresh_button.click(
                        fn=update_dropdowns,
                        inputs=checkpoint_sort,
                        outputs=[model_a, model_b, model_c, model_d]
                    )
                    checkpoint_sort.change(
                        fn=update_dropdowns,
                        inputs=checkpoint_sort,
                        outputs=[model_a, model_b, model_c, model_d]
                    )
                    # === MODE SELECTION ===
                    with gr.Row():
                        # ─────────────────────────────────────────────────────────────
                        # MERGE MODE RADIO — FIXED FOR YOUR SETUP (Dec 2025, Radio Edition)
                        # ─────────────────────────────────────────────────────────────
                        saved_merge_mode = cmn.opts.get('merge_mode', 'Weight-Sum')
                        if saved_merge_mode not in mergemode_selection:
                            saved_merge_mode = 'Weight-Sum'
                            
                        merge_mode_selector = gr.Radio(
                            label="Merge Mode",
                            choices=list(mergemode_selection.keys()),  # Real modes only
                            value=saved_merge_mode,                    # Load from saved options
                            type="value",
                            interactive=True,
                            elem_id="amethyst_merge_mode_radio"
                        )

                        merge_mode_desc = gr.Textbox(
                            label="Merge Mode Description",
                            value=mergemode_selection[saved_merge_mode].description,
                            interactive=False,
                            lines=2
                        )

                        with gr.Row():
                            saved_calc_mode = cmn.opts.get('calc_mode', list(calcmode_selection.keys())[0])
                            if saved_calc_mode not in calcmode_selection:
                                saved_calc_mode = list(calcmode_selection.keys())[0]
                                
                            calc_mode_selector = gr.Radio(
                                label='Calculation Mode (how to execute):',
                                choices=list(calcmode_selection.keys()),
                                value=saved_calc_mode,                 # Load from saved options
                                scale=3
                            )

                        calc_mode_desc = gr.Textbox(
                            label="Calculation Mode Description",
                            value=calcmode_selection[saved_calc_mode].description,
                            interactive=False,
                            lines=2
                        )

                        with gr.Accordion("Merge Behavior Options", open=False):
                            with gr.Row():
                                keep_zero_fill = gr.Checkbox(
                                    label="Kitchen-Sink Mode (Preserve All Keys)",
                                    value=cmn.opts.get('keep_zero_fill', True),
                                    info="Zero-fill missing keys — true kitchen-sink for future merging"
                                )
                                bloat_mode = gr.Checkbox(
                                    label="Legacy Bloat Mode",
                                    value=cmn.opts.get('bloat_mode', False),
                                    info="Pad tensors for max file size (~7.5GB+) — like old mergers"
                                )
                                
                                dual_soul_toggle = gr.Checkbox(
                                    label="Dual-Soul Mode (Cross-Arch Safety)",
                                    value=cmn.opts.get('dual_soul_toggle', False),
                                    info="Force cross-architecture protection logic even if detection fails"
                                )

                                sacred_keys_toggle = gr.Checkbox(
                                    label="Preserve Sacred Keys",
                                    value=cmn.opts.get('sacred_keys_toggle', False),
                                    info="Force preservation of noise / timestep / input layers"
                                )

                            with gr.Row():
                                smartresize_toggle = gr.Checkbox(
                                    label="Force SmartResize",
                                    value=cmn.opts.get('smart_resize_toggle', True),
                                    info="Force tensor resizing to primary shapes when mismatched"
                                )
                                specific_selectors_first = gr.Checkbox(
                                    label="Style-First Weight Matching",
                                    value=cmn.opts.get('specific_selectors_first', True),
                                    info="Apply narrow regex rules before broad ones. Stronger style transfer; safer OFF by default."
                                )
                                allow_exact_key_fallback = gr.Checkbox(
                                    label="Allow Exact-Key Selector Fallback",
                                    value=cmn.opts.get('allow_exact_key_fallback', True),
                                    info=(
                                    "If a selector fails to compile as regex, treat it as an exact tensor key.\n"
                                    "Useful for copy-paste from logs or surgical fixes.\n"
                                    "Safe: affects only one tensor."
                                    )
                                )
                                allow_glob_fallback = gr.Checkbox(
                                    label="Allow Glob Selector Fallback (Expert)",
                                    value=cmn.opts.get('allow_glob_fallback', True),
                                    info=(
                                    "If regex AND exact-key matching fail, treat selector as a glob (* ? []).\n"
                                    "⚠️ Can match many tensors.\n"
                                    "For expert artistic workflows only."
                                    )
                                )

                            with gr.Row():
                                copy_vae_from_primary = gr.Checkbox(
                                    label="Copy VAE from Primary (Recommended)",
                                    value=cmn.opts.get('copy_vae_from_primary', False),
                                    info="Preserve decoding stability and color fidelity"
                                )
                                copy_clip_from_primary = gr.Checkbox(
                                    label="Copy CLIP from Primary (Recommended)",
                                    value=cmn.opts.get('copy_clip_from_primary', False),
                                    info="Preserve prompt semantics and text understanding"
                                )
                                allow_synthetic_custom_merge = gr.Checkbox(
                                    label="Allow Synthetic Custom Merges (Unsafe)",
                                    value=cmn.opts.get('allow_synthetic_custom_merge', True),
                                    info="May inject zero-filled or resized tensors into custom merge math."
                                )
                                allow_non_float_merges = gr.Checkbox(
                                    label="Allow Non-Floating Tensor Merges (CLIP / VAE Destructive)",
                                    value=cmn.opts.get('allow_non_float_merges', True),
                                    info="Allow numeric merging of integer / boolean tensors "
                                )
                                allow_scalar_merges = gr.Checkbox(
                                    label="Allow Scalar Tensor Merges (Advanced)",
                                    value=cmn.opts.get('allow_scalar_merges', True),
                                    info=(
                                    "Enable controlled merging of 0-D tensors (scalars) such as logit_scale.\n"
                                    "Only whitelisted keys are allowed. Expert use only."
                                    )

                                )
                        # === Auto-save all merge interface checkboxes ===
                        # These checkboxes are the single source of truth for these options
                        # They persist via auto-save callbacks and are passed directly to start_merge()
                        
                        def save_and_update_option(key):
                            """Factory for option save callbacks"""
                            def _save(value):
                                cmn.opts.options[key] = value
                                cmn.opts.save()
                            return _save
                        
                        keep_zero_fill.change(fn=save_and_update_option('keep_zero_fill'), inputs=keep_zero_fill, outputs=None)
                        bloat_mode.change(fn=save_and_update_option('bloat_mode'), inputs=bloat_mode, outputs=None)
                        dual_soul_toggle.change(fn=save_and_update_option('dual_soul_toggle'), inputs=dual_soul_toggle, outputs=None)
                        sacred_keys_toggle.change(fn=save_and_update_option('sacred_keys_toggle'), inputs=sacred_keys_toggle, outputs=None)
                        smartresize_toggle.change(fn=save_and_update_option('smart_resize_toggle'), inputs=smartresize_toggle, outputs=None)
                        specific_selectors_first.change(fn=save_and_update_option('specific_selectors_first'), inputs=specific_selectors_first, outputs=None)
                        allow_exact_key_fallback.change(fn=save_and_update_option('allow_exact_key_fallback'), inputs=allow_exact_key_fallback, outputs=None)
                        allow_glob_fallback.change(fn=save_and_update_option('allow_glob_fallback'), inputs=allow_glob_fallback, outputs=None)
                        copy_vae_from_primary.change(fn=save_and_update_option('copy_vae_from_primary'), inputs=copy_vae_from_primary, outputs=None)
                        copy_clip_from_primary.change(fn=save_and_update_option('copy_clip_from_primary'), inputs=copy_clip_from_primary, outputs=None)
                        allow_synthetic_custom_merge.change(fn=save_and_update_option('allow_synthetic_custom_merge'), inputs=allow_synthetic_custom_merge, outputs=None)
                        allow_scalar_merges.change(fn=save_and_update_option('allow_scalar_merges'), inputs=allow_scalar_merges, outputs=None)
                        allow_non_float_merges.change(fn=save_and_update_option('allow_non_float_merges'), inputs=allow_non_float_merges, outputs=None)
   

                    slider_help = gr.Textbox(label="Slider Meaning", value="", interactive=False, lines=6, placeholder="Slider help will appear here when you change merge/calc modes.")

                    # MAIN SLIDERS - ✅ Updated to 0.0000001 for 7 decimal places (float32 precision)
                    with gr.Row(equal_height=True):
                        alpha = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_a [α] (alpha)", info='model_a - model_b', value=0.5, elem_classes=['main_sliders'])
                        beta  = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_b [β] (beta)",  info='-', value=0.5, elem_classes=['main_sliders'])
                        gamma = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_c [γ] (gamma)", info='-', value=0.5, elem_classes=['main_sliders'])
                        delta = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_d [δ] (delta)", info='-', value=0.5, elem_classes=['main_sliders'])
                        epsilon = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_e [ε] (epsilon)", info='-', value=0.5, elem_classes=['main_sliders'])
                    with gr.Row(equal_height=True):
                        zeta = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_f [ζ] (zeta)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        eta = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_g [η] (eta)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        theta = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_h [θ] (theta)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        iota = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_i [ι] (iota)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        kappa = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_j [κ] (kappa)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                    with gr.Row(equal_height=True):
                        lambda_ = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_k [λ] (lambda)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        mu = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_l [μ] (mu)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        nu = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_m [ν] (nu)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        xi = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_n [ξ] (xi)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        omicron = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_o [ο] (omicron)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                    with gr.Row(equal_height=True):
                        pi = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_p [π] (pi)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        rho = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_q [ρ] (rho)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        sigma = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_r [σ] (sigma)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        tau = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_s [τ] (tau)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                        upsilon = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_t [υ] (upsilon)", info='-', value=0.5, elem_classes=['main_sliders'], visible=False)
                    # CUSTOM SLIDERS UI — Elite, Dynamic, 5-Slider Safe
                    with ui_components.InputAccordion(False, label="Custom Sliders") as enable_sliders:
                        with gr.Accordion("Custom Slider Presets", open=True):
                            with gr.Row(variant="compact"):
                                sliders_preset_dropdown = gr.Dropdown(
                                                            label="Preset",
                                                            choices=get_slider_presets(),
                                                            value="blocks",
                                                            allow_custom_value=True,
                                                            scale=5
                                                        )
                                slider_refresh_button = gr.Button("Refresh", scale=1)

                                sliders_preset_load = gr.Button("Load", variant="secondary", scale=2)
                                sliders_preset_save = gr.Button("Save As...", variant="secondary", scale=2)

                            # Dynamic slider count (supports up to 30+ custom sliders)
                            slider_slider = gr.Slider(
                                minimum=0,
                                maximum=30,        # ← NOW SUPPORTS EPSILON + MANY MORE
                                value=26,
                                step=1,
                                label="Number of Custom Sliders (0 = disabled)"
                            )

                        custom_sliders = []
                        slider_container = gr.Column(visible=False)

                        def load_slider_preset(preset_name):
                            if not preset_name or preset_name not in get_slider_presets():
                                preset_name = "blocks"
                            with open(custom_sliders_examples, 'r') as f:
                                data = json.load(f)
                            values = data.get(preset_name, data["blocks"])
                            return (
                                gr.update(choices=get_slider_presets(), value=preset_name),
                                gr.update(value=len(values)),
                                *[gr.update(value=v) for v in values]
                            )

                        # Wire preset loading
                        sliders_preset_load.click(
                            fn=load_slider_preset,
                            inputs=sliders_preset_dropdown,
                            outputs=[sliders_preset_dropdown, slider_slider] + custom_sliders
                        )

                        slider_refresh_button.click(
                            fn=lambda: gr.update(choices=get_slider_presets()),
                            outputs=sliders_preset_dropdown
                        )

                        # Dynamic creation of sliders
                        with slider_container:
                            for i in range(30):  # Max possible
                                with gr.Row(variant="compact", visible=False) as row:
                                    target = gr.Textbox(
                                        placeholder="target keys (e.g. model.*in00*)",
                                        label=f"Target {i+1}",
                                        scale=3
                                    )
                                    weight = gr.Slider(
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=1.0,
                                        step=0.001,
                                        label=f"Weight {i+1}",
                                        scale=7
                                    )
                                    custom_sliders.extend([target, weight])

                        # Show/hide based on slider count
                        def show_custom_sliders(count):
                            updates = []
                            for i in range(30):
                                visible = i < count
                                updates.extend([
                                    gr.update(visible=visible),  # row
                                    gr.update(visible=visible),  # target
                                    gr.update(visible=visible)   # weight
                                ])
                            return updates

                        slider_slider.change(
                            fn=show_custom_sliders,
                            inputs=slider_slider,
                            outputs=custom_sliders
                        )
                        enable_sliders.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=enable_sliders,
                            outputs=slider_container
                        )

                    # ─────────────────────────────────────────────
                    # Fallback Tuning (HybridCascadeLite)
                    # ─────────────────────────────────────────────
                    with gr.Tab("Fallback Tuning"):
                        gr.Markdown("### Hybrid Fallback Behavior")
                        gr.Markdown(
                            "Controls how the merger behaves when **custom math fails**.\n\n"
                            "Fallbacks are **key-aware and depth-biased**:\n"
                            "• Early layers favor stability\n"
                            "• Mid layers balance\n"
                            "• Late layers allow more expressive blending\n\n"
                            "These settings affect **automatic fallback merges only**."
                        )

                        # -------------------------------------------------
                        # UNet / General Layers
                        # -------------------------------------------------
                        with gr.Accordion("UNet / General Layers", open=True):
                            cmn.opts.create_option(
                                "fallback_lerp_mix",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "Fallback Strength",
                                    "info": "0 = ultra-stable (mean-like), 1 = expressive (detail-preserving)"
                                },
                                default=1.0
                            )

                            cmn.opts.create_option(
                                "fallback_confidence",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "Confidence",
                                    "info": "Trust agreement vs stabilize disagreement"
                                },
                                default=0.5
                            )

                        # -------------------------------------------------
                        # CLIP / VAE Layers (Safer Defaults)
                        # -------------------------------------------------
                        with gr.Accordion("CLIP / VAE Layers (Safer Defaults)", open=False):
                            cmn.opts.create_option(
                                "clip_vae_lerp_mix",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "CLIP / VAE Strength",
                                    "info": "Lower values preserve semantic & decoding stability"
                                },
                                default=0.35
                            )

                            cmn.opts.create_option(
                                "clip_vae_confidence",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "CLIP / VAE Confidence",
                                    "info": "Lower = more conservative blending"
                                },
                                default=0.20
                            )

                            cmn.opts.create_option(
                                "clip_vae_lerp_temp",
                                gr.Slider,
                                {
                                    "minimum": 1.0,
                                    "maximum": 4.0,
                                    "step": 0.1,
                                    "label": "CLIP / VAE Temperature",
                                    "info": "Higher = gentler, smoother weighting"
                                },
                                default=3.0
                            )

                        # -------------------------------------------------
                        # Noise / Timestep Layers (Ultra-Safe)
                        # -------------------------------------------------
                        with gr.Accordion("Noise / Timestep Layers (Ultra-Safe)", open=False):
                            cmn.opts.create_option(
                                "noise_lerp_mix",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "Noise Strength",
                                    "info": "Lower values strongly preserve primary noise behavior"
                                },
                                default=0.4
                            )

                            cmn.opts.create_option(
                                "noise_confidence",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "Noise Confidence",
                                    "info": "How much agreement is trusted in noise-critical layers"
                                },
                                default=0.25
                            )

                            cmn.opts.create_option(
                                "noise_lerp_temp",
                                gr.Slider,
                                {
                                    "minimum": 1.0,
                                    "maximum": 4.0,
                                    "step": 0.1,
                                    "label": "Noise Temperature",
                                    "info": "Higher = extremely conservative blending"
                                },
                                default=2.5
                            )

                        # -------------------------------------------------
                        # Depth Bias
                        # -------------------------------------------------
                        with gr.Accordion("Depth Bias (UNet-Aware)", open=False):
                            cmn.opts.create_option(
                                "fallback_depth_bias",
                                gr.Slider,
                                {
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "step": 0.01,
                                    "label": "Depth Bias Strength",
                                    "info": (
                                        "How strongly fallback behavior scales with UNet depth.\n"
                                        "0 = uniform behavior, 1 = early stable → late expressive"
                                    )
                                },
                                default=0.35
                            )

                        gr.Markdown(
                            "ℹ️ These settings are applied **only when fallback logic activates**.\n"
                            "Custom calc modes, sacred preservation, and explicit COPY rules are unaffected."
                        )



                    # Supermerger Adjust - ✅ Updated to 0.0000001 for 7 decimal places (float32 precision)
                    with gr.Accordion("Supermerger Adjust", open=False) as acc_ad:
                        with gr.Row(variant="compact"):
                            finetune = gr.Textbox(label="Adjust", show_label=False, info="Adjust IN,OUT,OUT2,Contrast,Brightness,COL1,COL2,COL3", visible=True, value="", lines=1)
                            finetune_write = gr.Button(value="↑", elem_classes=["tool"])
                            finetune_read = gr.Button(value="↓", elem_classes=["tool"])
                            finetune_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                detail1 = gr.Slider(label="IN", minimum=-6, maximum=6, step=0.0000001, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail2 = gr.Slider(label="OUT", minimum=-6, maximum=6, step=0.0000001, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail3 = gr.Slider(label="OUT2", minimum=-6, maximum=6, step=0.0000001, value=0, info="Detail/Noise")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                contrast = gr.Slider(label="Contrast", minimum=-10, maximum=10, step=0.0000001, value=0, info="Contrast/Detail")
                            with gr.Column(scale=1, min_width=100):
                                bri = gr.Slider(label="Brightness", minimum=-10, maximum=10, step=0.0000001, value=0, info="Dark(Minius)-Bright(Plus)")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                col1 = gr.Slider(label="Cyan-Red", minimum=-10, maximum=10, step=0.0000001, value=0, info="Cyan(Minius)-Red(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col2 = gr.Slider(label="Magenta-Green", minimum=-10, maximum=10, step=0.0000001, value=0, info="Magenta(Minius)-Green(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col3 = gr.Slider(label="Yellow-Blue", minimum=-10, maximum=10, step=0.0000001, value=0, info="Yellow(Minius)-Blue(Plus)")

                        finetune.change(fn=lambda x: gr.update(label=f"Supermerger Adjust : {x}" if x != "" and x != "0,0,0,0,0,0,0,0" else "Supermerger Adjust"),
                                        inputs=[finetune], outputs=[acc_ad])

                        def finetune_update(finetune, detail1, detail2, detail3, contrast, bri, col1, col2, col3):
                            arr = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                            tmp = ",".join(map(lambda x: str(int(x)) if x == 0.0 else str(x), arr))
                            if finetune != tmp:
                                return gr.update(value=tmp)
                            return gr.update()

                        def finetune_reader(finetune):
                            try:
                                tmp = [float(t) for t in finetune.split(",") if t]
                                assert len(tmp) == 8, f"expected 8 values, received {len(tmp)}."
                            except (ValueError, AssertionError) as err:
                                gr.Warning(str(err))
                            else:
                                return [gr.update(value=x) for x in tmp]
                            return [gr.update()] * 8

                        finetunes = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                        finetune_reset.click(fn=lambda: [gr.update(value="")] + [gr.update(value=0.0)] * 8,
                                            inputs=[], outputs=[finetune, *finetunes])
                        finetune_read.click(fn=finetune_reader, inputs=[finetune], outputs=[*finetunes])
                        finetune_write.click(fn=finetune_update, inputs=[finetune, *finetunes], outputs=[finetune])

                        for slider in finetunes:
                            slider.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)

                    # OPTIONS
                    with gr.Accordion(label='Options', open=False):
                        save_options_button = gr.Button(value='Save Options', variant='primary')
                        save_options_button.click(fn=cmn.opts.save)

                        cmn.opts.create_option('trash_model',
                            gr.Radio,
                            {
                                'choices': ['Disable', 'Enable for SDXL', 'Enable'],
                                'label': 'Clear loaded SD models from memory at start of merge:',
                                'info': 'Saves VRAM but increases load time'
                            },
                            default='Enable for SDXL'
                        )

                        # FINAL: Device selection with auto-save
                        device_radio = cmn.opts.create_option('device',
                            gr.Radio,
                            {
                                'choices': [
                                    'cuda/float16',
                                    'cuda/bfloat16',
                                    'cuda/float32',
                                    'cpu/float32'
                                ],
                                'label': 'Merge device & precision:',
                                'info': 'bfloat16 is usually the best choice on modern NVIDIA GPUs'
                            },
                            default='cuda/float16'
                        )
                        device_radio.change(
                            fn=lambda x: cmn.opts.save(),
                            inputs=device_radio,
                            outputs=None
                        )

                        cmn.opts.create_option('threads',
                            gr.Slider,
                            {
                                'step': 2, 'minimum': 2, 'maximum': 20,
                                'label': 'Worker thread count:',
                                'info': 'Relevant for both CUDA and CPU merging. Core count ±2 is ideal.'
                            },
                            default=8
                        )

                        cache_size_slider = cmn.opts.create_option('cache_size',
                            gr.Slider,
                            {
                                'step': 64, 'minimum': 0, 'maximum': 16384,
                                'label': 'Cache size (MB):',
                                'info': 'Intermediate results cache. 8192–16384 recommended for huge merges.'
                            },
                            default=8192
                        )

                        # Note: All checkbox options are now managed via the merge interface checkboxes above
                        # They are registered with cmn.opts.components for persistence and auto-save
                        # This eliminates duplication while maintaining the rich UI/UX of the merge interface

                        def update_cache_size(value=None):
                            if value is None:
                                value = cmn.opts.get('cache_size', 8192)
                            weights_cache.__init__(max(0, int(value)))
                            return f"Cache resized to {value} MB"


                        cache_size_slider.release(
                            fn=update_cache_size,
                            inputs=cache_size_slider,
                            outputs=None
                        )

                        # Initialize cache at startup with saved value
                        weights_cache.__init__(cmn.opts.get('cache_size', 8192))

                    # MERGE & SAVE UI
                    with gr.Row(equal_height=True):
                        with gr.Column(variant='panel'):
                            save_name = gr.Textbox(
                                max_lines=1,
                                label='Save checkpoint as:',
                                lines=1,
                                placeholder='Enter name...',
                                scale=2
                            )

                            with gr.Row():
                                save_settings = gr.CheckboxGroup(
                                    label="Save & Load Options",
                                    choices=[
                                        "Autosave",        # Save to disk
                                        "Overwrite",       # Overwrite existing file
                                        "Load in Memory",  # ← ← ALWAYS keep this: load merged model immediately (with or without save)
                                        "fp16",            # Save/load as float16
                                        "bf16",            # Save/load as bfloat16
                                        "fp32"             # Save/load as float32
                                    ],
                                    value=['fp16', 'Load in Memory'],  # ← smart default
                                    interactive=True
                                )

                                save_loaded = gr.Button(
                                    value='Save loaded checkpoint',
                                    size='sm',
                                    scale=1
                                )

                            # === STATUS BOX + CLEAR BUTTON ===
                            with gr.Column(scale=3):
                                status = gr.Textbox(
                                    label="Save / Merge Status",
                                    lines=4,
                                    interactive=False,
                                    show_copy_button=True,
                                    value="Ready",
                                    elem_classes="amethyst-status-box"
                                )
                                with gr.Row():
                                    clear_status = gr.Button("Clear Status", size="sm", variant="secondary")
                                    clear_status.click(fn=lambda: "", outputs=status)

                            # === Save button actions ===
                            save_loaded.click(
                                fn=misc_util.save_loaded_model,
                                inputs=[save_name, save_settings],
                                outputs=status
                            )
                            save_loaded.click(
                                fn=refresh_models,
                                inputs=checkpoint_sort,
                            )

                        with gr.Column():
                            merge_button = gr.Button(value='Merge', variant='primary')
                            with gr.Row():
                                empty_cache_button = gr.Button(value='Empty Cache')
                                empty_cache_button.click(fn=merger.clear_cache, outputs=status)

                                stop_button = gr.Button(value='Stop', variant="stop")
                                def stopfunc():
                                    cmn.stop = True
                                    shared.state.interrupt()
                                stop_button.click(fn=stopfunc)

                            with gr.Row():
                                merge_seed = gr.Number(
                                    label='Merge Seed',
                                    value=99,
                                    min_width=100,
                                    precision=0,
                                    scale=1
                                )
                                merge_random_seed = ui_components.ToolButton(
                                    ui.random_symbol,
                                    tooltip="Set seed to -1 (random each merge)"
                                )
                                merge_random_seed.click(fn=lambda: -1, outputs=merge_seed)

                                merge_reuse_seed = ui_components.ToolButton(
                                    ui.reuse_symbol,
                                    tooltip="Reuse seed from last merge"
                                )
                                merge_reuse_seed.click(fn=lambda: cmn.last_merge_seed, outputs=merge_seed)

                            # INCLUDE / EXCLUDE / DISCARD (still in main Merge tab)
                            with gr.Accordion(label='Include/Exclude/Discard', open=False):
                                with gr.Row():
                                    with gr.Column():
                                        clude = gr.Textbox(
                                            max_lines=4,
                                            label='Include/Exclude:',
                                            info="Entered targets will remain as model_a when set to 'Exclude', and will be the only ones to be merged if set to 'Include'. Separate with whitespace.",
                                            value='',
                                            lines=4,
                                            scale=4
                                        )
                                        clude_mode = gr.Radio(
                                            label="",
                                            info="",
                                            choices=["Exclude", ("Include exclusively", 'include')],
                                            value='Exclude',
                                            min_width=300,
                                            scale=1
                                        )
                                    with gr.Column():
                                        discard = gr.Textbox(
                                            max_lines=5,
                                            label='Discard:',
                                            info="Remove layers from final save (autosave only). Examples: 'model_ema', 'first_stage_model', or 'model_ema first_stage_model'. Leave empty to keep all layers.",
                                            value='',
                                            lines=5,
                                            scale=1
                                        )

                    # Hidden textbox for preset JSON — must exist before any .click() uses it
                    preset_output = gr.Textbox(visible=False)
                    # ── Your existing Weight Editor textbox (must come BEFORE the .click) ──
                    weight_editor = gr.Textbox(
                        label="Weight Editor (JSON)",
                        lines=10,
                        placeholder='all: slider_a, slider_b, slider_c, slider_d, slider_e',
                        value="all: slider_a, slider_b, slider_c, slider_d, slider_e"
                    )
                    # Validation output
                    validation_output = gr.Textbox(
                        label="Validation",
                        value="Ready",
                        interactive=False,
                        lines=3,
                        max_lines=3
                    )

                    # ===================================================================
                    # 2. NOW IT'S SAFE TO WIRE BUTTONS
                    # ===================================================================
                    # Validation on weight editor change
                    weight_editor.change(
                        fn=lambda we, ma, mb, mm: validate_merge_config(ma, mb, we, mm),
                        inputs=[weight_editor, model_a, model_b, model_c, model_d, merge_mode_selector],
                        outputs=validation_output,
                        show_progress=False
                    )


                    # Model keys test
                    with gr.Accordion("Model Keys Tester", open=False):
                        target_tester = gr.Textbox(
                            label="Check model_a keys with regex",
                            info="Use '*' wildcard. E.g., 'cond*' for CLIP, 'model.*out*4*tran*norm*weight' for UNet blocks",
                            interactive=True,
                            placeholder="model.*out*4*tran*norm*weight"
                        )
                        target_tester_display = gr.Textbox(
                            label="Matching Keys",
                            lines=40,
                            max_lines=40,
                            interactive=False
                        )
                        target_tester.change(
                            fn=test_regex,
                            inputs=[target_tester],
                            outputs=target_tester_display,
                            show_progress="minimal"
                        )

                    # Wire the mode_changed events (already correct for 5 sliders)
                    merge_mode_selector.change(
                        fn=mode_changed,
                        inputs=[merge_mode_selector, calc_mode_selector],
                        outputs=[
                            merge_mode_desc, calc_mode_desc,
                            alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda_, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon,
                            slider_help,
                            model_a, model_b, model_c, model_d,
                            merge_button
                        ],
                        show_progress="hidden"
                    )
                    # Auto-save merge mode selection
                    merge_mode_selector.change(
                        fn=lambda x: (cmn.opts.options.update({'merge_mode': x}), cmn.opts.save()),
                        inputs=merge_mode_selector,
                        outputs=None,
                        queue=False
                    )
                    
                    calc_mode_selector.change(
                        fn=mode_changed,
                        inputs=[merge_mode_selector, calc_mode_selector],
                        outputs=[
                            merge_mode_desc, calc_mode_desc,
                            alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda_, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon,
                            slider_help,
                            model_a, model_b, model_c, model_d,
                            merge_button
                        ],
                        show_progress="hidden"
                    )
                    # Auto-save calc mode selection
                    calc_mode_selector.change(
                        fn=lambda x: (cmn.opts.options.update({'calc_mode': x}), cmn.opts.save()),
                        inputs=calc_mode_selector,
                        outputs=None,
                        queue=False
                    )

                    # === INITIALIZE SLIDER CONFIGS ON PAGE LOAD ===
                    # This fires when the page first loads, applying saved calc/merge mode to sliders
                    cmn.blocks.load(
                        fn=mode_changed,
                        inputs=[merge_mode_selector, calc_mode_selector],
                        outputs=[
                            merge_mode_desc, calc_mode_desc,
                            alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda_, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon,
                            slider_help,
                            model_a, model_b, model_c, model_d,
                            merge_button
                        ],
                        show_progress="hidden"
                    )

                    # ===================================================================
                    # FINAL MERGE BUTTON — NO *merge_args, EXPLICIT INPUTS (2025 STANDARD)
                    # ===================================================================
                    merge_button.click(
                        fn=start_merge,
                        inputs=[
                            save_name,
                            save_settings,
                            finetune,
                            merge_mode_selector,
                            calc_mode_selector,
                            model_a, model_b, model_c, model_d,
                            alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda_, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon,
                            weight_editor,
                            preset_output,              # ← Preset JSON
                            discard,
                            clude,
                            clude_mode,
                            merge_seed,
                            enable_sliders,
                            copy_vae_from_primary,
                            copy_clip_from_primary,
                            keep_zero_fill,    
                            bloat_mode,
                            dual_soul_toggle,
                            sacred_keys_toggle,
                            smartresize_toggle,
                            specific_selectors_first,
                            allow_glob_fallback,
                            allow_exact_key_fallback,
                            allow_synthetic_custom_merge,
                            allow_non_float_merges,
                            allow_scalar_merges,
                            *custom_sliders,
                        ],
                        outputs=status
                    )

                    # Optional: Clear preset after merge (clean UX)
                    merge_button.click(
                        fn=lambda: "",
                        outputs=preset_output,
                        queue=False
                    ).then(
                        fn=lambda: "Ready for next merge",
                        outputs=status
                    )
                    # ===================================================================
                    # AMETHYST KITCHEN-SINK WEIGHT PRESETS — FINAL 2025 EDITION
                    # ===================================================================

                with gr.Accordion("Kitchen-Sink Weight Presets", open=True):
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Weight Preset",
                                choices=[
                                    "Global Sliders (Default)",
                                    "50/50 UNet + Keep CLIP",
                                    "UNet 50/50 + Noise Blend",
                                    "Keep CLIP/T5 + Full UNet Merge",
                                    "Full Kitchen Sink (All 50/50)",
                                    "Primary Dominance (A=1.0, Others=0.0)",
                                    "Secondary Dominance (B=1.0, Others=0.0)",
                                ],
                                value="Global Sliders (Default)",
                                info="Quick presets for true kitchen-sink merging"
                            )
                            apply_preset = gr.Button(value="Apply Preset", variant="primary")

                        gr.HTML("<small><b>Active when using Weight Editor or Block Weights</b></small>")

                        # ------------------------------------------------------------------
                        # Preset definitions
                        # ------------------------------------------------------------------
                        def apply_weight_preset(choice: str) -> str:
                            presets = {
                                "Global Sliders (Default)": json.dumps({
                                                        ".*": {
                                        "alpha": "alpha",
                                        "beta": "beta",
                                       "gamma": "gamma",
                                        "delta": "delta",
                                        "epsilon": "epsilon"
                                                           }
                                }, indent=2),

                                "50/50 UNet + Keep CLIP": json.dumps({
                                    "model.diffusion_model.*": {"alpha": 0.5, "beta": 0.5},
                                    "conditioner.*":           {"alpha": 1.0, "beta": 0.0},
                                    "denoiser.*":              {"alpha": 0.7, "beta": 0.3},
                                    "first_stage_model.*":     {"alpha": 0.6, "beta": 0.4}
                                }, indent=2),

                                "UNet 50/50 + Noise Blend": json.dumps({
                                    "model.diffusion_model.*": {"alpha": 0.5, "beta": 0.5},
                                    "denoiser.*":              {"alpha": 0.9, "beta": 0.1},
                                    "*":                       {"alpha": 0.5, "beta": 0.5}
                                }, indent=2),

                                "Keep CLIP/T5 + Full UNet Merge": json.dumps({
                                    "conditioner.*":           {"alpha": 1.0, "beta": 0.0},
                                    "model.diffusion_model.*": {"alpha": 0.5, "beta": 0.5},
                                    "first_stage_model.*":     {"alpha": 0.5, "beta": 0.5},
                                    "denoiser.*":              {"alpha": 0.5, "beta": 0.5}
                                }, indent=2),

                                "Full Kitchen Sink (All 50/50)": json.dumps({
                                    "*": {"alpha": 0.5, "beta": 0.5}
                                }, indent=2),

                                "Primary Dominance (A=1.0, Others=0.0)": json.dumps({
                                    "*": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
                                }, indent=2),

                                "Secondary Dominance (B=1.0, Others=0.0)": json.dumps({
                                    "*": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0}
                                }, indent=2),
                            }
                            return presets.get(choice, "{}")

                                            # Apply preset button → fills preset_output → then injects into weight_editor
                        apply_preset.click(
                                fn=apply_weight_preset,
                        inputs=preset_dropdown,
                        outputs=preset_output
                         ).then(
                        fn=lambda x: x,
                        inputs=preset_output,
                        outputs=weight_editor
                        ).then(
                        fn=lambda: "Preset applied — click Merge",
                        outputs=validation_output
                        )
        
            # ================================================
            # PREVIEW TAB — SuperMerger-style "Merge & Gen"
            # ================================================
            with gr.Tab("Preview Merge", elem_id="amethyst-preview"):
                gr.Markdown("### Merge in-memory and instantly preview the result (no save required)")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion("Merge Settings (same as main tab)", open=False):
                            preview_model_a = gr.Dropdown(choices=initial_checkpoints, label="Model A", value=model_a.value)
                            preview_model_b = gr.Dropdown(choices=initial_checkpoints, label="Model B", value=model_b.value)
                            preview_model_c = gr.Dropdown(choices=initial_checkpoints, label="Model C", value=model_c.value)
                            preview_model_d = gr.Dropdown(choices=initial_checkpoints, label="Model D", value=model_d.value)
                            preview_merge_mode = gr.Radio(
                                choices=list(merger.mergemode_selection.keys()),
                                value=merge_mode_selector.value,
                                label="Merge Mode"
                            )
                            preview_calc_mode = gr.Radio(
                                choices=list(merger.calcmode_selection.keys()),
                                value=calc_mode_selector.value,
                                label="Calc Mode"
                            )
                            preview_alpha   = gr.Slider(-1, 2, value=alpha.value,   step=0.01, label="α (alpha)")
                            preview_beta    = gr.Slider(-1, 2, value=beta.value,    step=0.01, label="β (beta)")
                            preview_gamma   = gr.Slider(-1, 2, value=gamma.value,   step=0.01, label="γ (gamma)")
                            preview_delta   = gr.Slider(-1, 2, value=delta.value,   step=0.01, label="δ (delta)")
                            preview_epsilon = gr.Slider(-1, 2, value=epsilon.value, step=0.01, label="ε (epsilon)")

                        with gr.Accordion("Generation Settings", open=True):
                            preview_prompt = gr.Textbox(
                                label="Prompt",
                                lines=3,
                                value="masterpiece, best quality, ultra-detailed, 8k"
                            )
                            preview_neg = gr.Textbox(label="Negative Prompt", lines=2, value="")
                            with gr.Row():
                                preview_steps = gr.Slider(1, 100, value=20, step=1, label="Steps")
                                preview_sampler = gr.Dropdown(
                                    choices=[s.name for s in sd_samplers.samplers],
                                    value="Euler a",
                                    label="Sampler"
                                )
                            with gr.Row():
                                preview_w = gr.Slider(256, 2048, value=512, step=64, label="Width")
                                preview_h = gr.Slider(256, 2048, value=512, step=64, label="Height")
                                preview_cfg = gr.Slider(1, 30, value=7.0, step=0.5, label="CFG Scale")
                                preview_seed = gr.Number(value=-1, label="Seed (-1 = random)")

                        preview_btn = gr.Button("Merge & Preview", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        preview_gallery = gr.Gallery(
                            label="Preview Result",
                            show_label=False,
                            columns=2,
                            height="auto",
                            preview=True
                        )
                        preview_info = gr.Textbox(label="Generation Info", lines=4, interactive=False)
                        preview_status = gr.Textbox(  # ← NEW: Live status feedback
                            label="Status",
                            value="Ready — click 'Merge & Preview' to begin",
                            lines=3,
                            interactive=False,
                            elem_classes=["preview-status"]
                        )
                        preview_log = gr.Textbox(label="Merge Log", lines=8, interactive=False)

                # === CLICK HANDLER WITH STATUS UPDATES ===
                preview_btn.click(
                    fn=preview_merged_model,
                    inputs=[
                        preview_model_a, preview_model_b, preview_model_c, preview_model_d,
                        preview_merge_mode, preview_calc_mode,
                        preview_alpha, preview_beta, preview_gamma, preview_delta, preview_epsilon,
                        preview_prompt, preview_neg,
                        preview_steps, preview_sampler,
                        preview_w, preview_h, preview_cfg, preview_seed,
                    ],
                    outputs=[preview_gallery, preview_info, preview_status, preview_log],  # ← Now 4 outputs!
                    show_progress=True
                )

        # ================================================
        # COMPARE TAB — Elite model intelligence
        # ================================================
        with gr.Tab("Compare Models", elem_id="amethyst-compare"):
            gr.Markdown("### Deep Comparison of Two Checkpoints")
            gr.Markdown("*Shows architecture, dtype, size, EMA, CLIP versions, Flux/SDXL detection, and key differences*")

            with gr.Row():
                compare_a = gr.Dropdown(
                    choices=initial_checkpoints,
                    value=initial_checkpoints[0] if initial_checkpoints else None,
                    label="Model A (Reference)",
                    scale=2
                )
                compare_b = gr.Dropdown(
                    choices=initial_checkpoints,
                    value=None,
                    label="Model B (Compare)",
                    scale=2
                )

            with gr.Row():
                compare_btn = gr.Button("Compare Models", variant="primary", size="lg")

            compare_output = gr.Textbox(
                label="Detailed Comparison Report",
                lines=25,
                interactive=False,
                elem_classes="compare-report"
            )

            def compare_models_deep(model_a_name, model_b_name):
                if not model_a_name or not model_b_name:
                    return "Please select both models."

                try:
                    info_a = sd_models.get_closet_checkpoint_match(model_a_name)
                    info_b = sd_models.get_closet_checkpoint_match(model_b_name)
                    
                    if not info_a or not info_b:
                        return "One or both models not found."

                    path_a, path_b = info_a.filename, info_b.filename
                    size_a, size_b = os.path.getsize(path_a), os.path.getsize(path_b)

                    def analyze_model(path, name):
                        try:
                            with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
                                keys = set(f.keys())
                                sample_key = next(iter(keys))
                                dtype = f.get_tensor(sample_key).dtype
                                has_ema = any("model_ema" in k for k in keys)
                                has_flux = any("single_blocks" in k for k in keys)
                                has_sdxl = any("conditioner.embedders." in k for k in keys)
                                has_clip_l = any("embedders.1" in k for k in keys)
                                has_clip_g = any("embedders.0" in k for k in keys)

                                # Detect architecture
                                if has_flux:
                                    arch = "Flux.1"
                                elif has_sdxl:
                                    arch = "SDXL" + (" + Refiner" if "refiner" in name.lower() else "")
                                elif "input_blocks.0.0.weight" in keys:
                                    channels = f.get_tensor("model.diffusion_model.input_blocks.0.0.weight").shape[1]
                                    if channels == 9:
                                        arch = "SD 1.5 Inpainting"
                                    elif channels == 8:
                                        arch = "InstructPix2Pix"
                                    else:
                                        arch = "SD 1.5"
                                else:
                                    arch = "Unknown / Custom"

                                return {
                                    "name": name,
                                    "path": os.path.basename(path),
                                    "size_gb": size_a / 1e9 if path == path_a else size_b / 1e9,
                                    "keys": len(keys),
                                    "dtype": str(dtype).split(".")[-1],
                                    "has_ema": has_ema,
                                    "arch": arch,
                                    "clip": ("OpenCLIP-G" if has_clip_g else "") + (" + CLIP-L" if has_clip_l else ""),
                                    "keys_set": keys
                                }
                        except Exception as e:
                            return {"error": f"Failed to read {name}: {e}"}

                    a = analyze_model(path_a, model_a_name)
                    b = analyze_model(path_b, model_b_name)

                    if "error" in a or "error" in b:
                        return f"Error reading model: {a.get('error','')} {b.get('error','')}"

                    removed = len(a["keys_set"] - b["keys_set"])
                    added = len(b["keys_set"] - a["keys_set"])
                    common = len(a["keys_set"] & b["keys_set"])

                    report = f"""MODEL COMPARISON REPORT
┌─────────────────────────────────────────────────────────────┐
│ Model A (Reference) │ Model B (Compare)
├─────────────────────────────────────────────────────────────┤
│ Name: {a['name']:<42} │ {b['name']}
│ File: {a['path']:<42} │ {b['path']}
│ Size: {a['size_gb']:.3f} GB{' ' * 28} │ {b['size_gb']:.3f} GB ({'+' if b['size_gb'] > a['size_gb'] else ''}{b['size_gb'] - a['size_gb']:+.3f} GB)
│ Keys: {a['keys']:,}{' ' * 35} │ {b['keys']:,} ({'+' if b['keys'] > a['keys'] else ''}{b['keys'] - a['keys']:+})
│ Dtype: {a['dtype']:<42} │ {b['dtype']}
│ Architecture: {a['arch']:<34} │ {b['arch']}
│ CLIP: {a['clip'] or 'Standard CLIP':<38} │ {b['clip'] or 'Standard CLIP'}
│ EMA Present: {'Yes' if a['has_ema'] else 'No':<37} │ {'Yes' if b['has_ema'] else 'No'}
└─────────────────────────────────────────────────────────────┘

KEY DIFFERENCES
───────────────────────────────────────────────────────────────
Keys in common : {common:,}
Keys removed   : {removed:,}  ← (A → B)
Keys added     : {added:,}    ← (B → A)

SUMMARY
→ Model B is {((b['size_gb'] - a['size_gb']) / a['size_gb'] * 100):+.1f}% the size of Model A
→ Likely a {'merge' if removed + added < 100 else 'different base'} (low key churn = merge, high = new model)
"""
                    return report

                except Exception as e:
                    return f"Unexpected error: {str(e)}"

            compare_btn.click(
                fn=compare_models_deep,
                inputs=[compare_a, compare_b],
                outputs=compare_output
            )
            
        # ================================================
        # LORA TAB — Elite LoRA Merging (Forge Neo + A1111 dev)
        # ================================================
        with gr.Tab("LoRA", elem_id="tab_lora"):
            gr.Markdown(
                "# LoRA Merging — Bake into Checkpoint or Merge LoRAs\n"
                "Fully supports **SD1.5 ↔ SDXL ↔ Pony ↔ Flux** • Mixed ranks • Block-specific weights"
            )

            # === ELITE LORA LIST WITH INFO (2025 Edition) ===
            def get_lora_list_with_info():
                lora_dirs = []
                if hasattr(shared.opts, "lora_dir") and shared.opts.lora_dir and os.path.isdir(shared.opts.lora_dir):
                    lora_dirs.append(shared.opts.lora_dir)

                base = paths.models_path
                lora_dirs.extend([
                    os.path.join(base, "Lora"),
                    os.path.join(base, "lora"),
                    os.path.join(base, "LoRA"),
                    os.path.join(base, "LyCORIS"),
                    os.path.join(base, "lycoris"),
                ])

                seen_paths = set()
                loras = []

                for directory in lora_dirs:
                    if not os.path.isdir(directory):
                        continue
                    for file in sorted(os.listdir(directory)):
                        if not file.lower().endswith(('.safetensors', '.pt', '.ckpt')):
                            continue

                        full_path = os.path.join(directory, file)
                        if full_path in seen_paths:
                            continue
                        seen_paths.add(full_path)

                        try:
                            size_mb = os.path.getsize(full_path) // (1024 * 1024)
                            dtype_hint = model_hint = ""

                            with safetensors.torch.safe_open(full_path, framework="pt", device="cpu") as f:
                                keys = f.keys()
                                if any("bf16" in k for k in keys):
                                    dtype_hint = "bf16"
                                else:
                                    dtype_hint = "f16"

                                if any(k.startswith("lora_unet_down_blocks_4_") for k in keys):
                                    model_hint = "[SDXL]"
                                elif any(k.startswith("lora_unet_down_blocks_") and "_4_" not in k for k in keys):
                                    model_hint = "[SD1.5]"
                                elif any("single_blocks" in k for k in keys):
                                    model_hint = "[Flux]"
                                elif any("transformer_blocks" in k for k in keys):
                                    model_hint = "[SD3/Pony]"

                            label = f"{file} — {size_mb}MB • {dtype_hint} {model_hint}"
                        except Exception:
                            size_mb = os.path.getsize(full_path) // (1024 * 1024)
                            label = f"{file} — {size_mb}MB"

                        loras.append(full_path)  # CheckboxGroup uses path as value

                return loras

            # === SECTION 1: LoRAs → Checkpoint (Elite Baker) ===
            with gr.Accordion("Bake LoRAs → Checkpoint", open=True):
                gr.Markdown("### Bake multiple LoRAs into a checkpoint with **individual or block-specific weights**")

                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        label="Base Checkpoint",
                        choices=[cp.title for cp in sd_models.checkpoints_list.values()],
                        value=lambda: next((cp.title for cp in sd_models.checkpoints_list.values()), None)
                    )
                    create_refresh_button(
                        checkpoint_dropdown,
                        lambda: None,
                        lambda: {"choices": [cp.title for cp in sd_models.checkpoints_list.values()]},
                        "refresh_checkpoint_lora"
                    )

                with gr.Row():
                    lora_checkbox_group = gr.CheckboxGroup(
                        label="Select LoRAs to Bake",
                        choices=get_lora_list_with_info(),
                        type="value",
                        interactive=True
                    )
                    create_refresh_button(
                        lora_checkbox_group,
                        lambda: None,
                        lambda: {"choices": get_lora_list_with_info()},
                        "refresh_loras_bake"
                    )

                gr.Markdown("**Individual Weights** — Use number (e.g. `1.2`) or **block-specific** (e.g. `unet:1.2, te:0.8`)")

                weights_inputs = []
                for i in range(6):
                    w = gr.Textbox(
                        value="1.0",
                        label=f"LoRA {i+1} Weight",
                        placeholder="1.0 or unet:1.2, te:0.8",
                        visible=False
                    )
                    weights_inputs.append(w)

                def update_weight_fields(selected):
                    updates = []
                    for i, path in enumerate(selected or []):
                        if i >= len(weights_inputs): break
                        name = os.path.basename(path).split('.')[0][:28]
                        updates.append(gr.update(visible=True, label=f"{name}"))
                    for j in range(len(selected or []), len(weights_inputs)):
                        updates.append(gr.update(visible=False))
                    return updates

                lora_checkbox_group.change(
                    fn=update_weight_fields,
                    inputs=lora_checkbox_group,
                    outputs=weights_inputs
                )

                with gr.Row():
                    output_name = gr.Textbox(
                        label="Output Model Name",
                        placeholder="MyUltimateModel_v1",
                        value="MyUltimateModel_v1"
                    )
                    save_model = gr.Checkbox(label="Save to Disk", value=True)

                bake_btn = gr.Button("BAKE LoRAs → CHECKPOINT", variant="primary", size="lg")
                bake_status = gr.Textbox(label="Status", lines=14, interactive=False, show_copy_button=True)

                def bake_loras_to_checkpoint(loras, ckpt_name, out_name, save, *weights_raw):
                    if not loras:
                        return "Error: No LoRAs selected"
                    if not ckpt_name:
                        return "Error: No base checkpoint selected"

                    ckpt_info = merger.get_checkpoint_match(ckpt_name)
                    if not ckpt_info:
                        return "Error: Checkpoint not found"

                    weights = []
                    for w in weights_raw:
                        if not w.strip():
                            weights.append(1.0)
                        elif ":" in w.strip() or "," in w.strip():
                            weights.append(w.strip())
                        else:
                            try:
                                weights.append(float(w.strip()))
                            except:
                                weights.append(1.0)
                    weights = weights[:len(loras)]

                    try:
                        result = lora_merge.merge_loras_to_checkpoint(
                            checkpoint_path=ckpt_info.filename,
                            lora_paths=loras,
                            output_path=os.path.join(sd_models.model_path, f"{out_name}.safetensors"),
                            individual_weights=weights,
                            save_model=save,
                            progress=gr.Progress(track_tqdm=True)
                        )
                        return f"SUCCESS! {result}"
                    except Exception as e:
                        import traceback
                        return f"BAKE FAILED:\n{str(e)}\n\n{traceback.format_exc()}"

                bake_btn.click(
                    fn=bake_loras_to_checkpoint,
                    inputs=[lora_checkbox_group, checkpoint_dropdown, output_name, save_model] + weights_inputs,
                    outputs=bake_status
                )

            # === SECTION 2: LoRA → LoRA Merge ===
            with gr.Accordion("Merge Multiple LoRAs → New LoRA", open=False):
                gr.Markdown("### Combine 2–6 LoRAs into a single new LoRA file")

                with gr.Row():
                    lora_merge_checkbox = gr.CheckboxGroup(
                        label="Select LoRAs to Merge",
                        choices=get_lora_list_with_info(),
                        type="value"
                    )
                    create_refresh_button(
                        lora_merge_checkbox,
                        lambda: None,
                        lambda: {"choices": get_lora_list_with_info()},
                        "refresh_lora_merge"
                    )

                merge_weights = []
                for i in range(6):
                    w = gr.Slider(0.0, 3.0, value=1.0, step=0.01, label=f"LoRA {i+1}", visible=False)
                    merge_weights.append(w)

                def update_merge_sliders(selected):
                    updates = []
                    for i, path in enumerate(selected or []):
                        if i >= len(merge_weights): break
                        name = os.path.basename(path).split('.')[0][:28]
                        updates.append(gr.update(visible=True, label=f"{name}"))
                    for j in range(len(selected or []), len(merge_weights)):
                        updates.append(gr.update(visible=False))
                    return updates

                lora_merge_checkbox.change(fn=update_merge_sliders, inputs=lora_merge_checkbox, outputs=merge_weights)

                with gr.Row():
                    lora_merge_name = gr.Textbox(label="Output LoRA Name", placeholder="EpicMerge_v1", value="EpicMerge_v1")
                    global_mult = gr.Slider(0.1, 3.0, 1.0, step=0.05, label="Global Multiplier")

                merge_lora_btn = gr.Button("MERGE LoRAs → NEW LORA", variant="primary")
                merge_lora_status = gr.Textbox(label="Status", lines=12, interactive=False, show_copy_button=True)

                def merge_loras_final(loras, name, mult, *weights):
                    if len(loras or []) < 2:
                        return "Error: Select at least 2 LoRAs"
                    w = [x * mult for x in (list(weights)[:len(loras)] or [1.0]*len(loras))]
                    try:
                        result = lora_merge.merge_loras_resilient(
                            lora_paths=loras,
                            output_path=os.path.join(paths.models_path, "Lora", f"{name}.safetensors"),
                            weights=w,
                            progress=gr.Progress(track_tqdm=True)
                        )
                        return f"SUCCESS! {result}\n\nRefreshing LoRA list..."
                    except Exception as e:
                        import traceback
                        return f"Merge failed:\n{str(e)}\n\n{traceback.format_exc()}"

                merge_lora_btn.click(
                    fn=merge_loras_final,
                    inputs=[lora_merge_checkbox, lora_merge_name, global_mult] + merge_weights,
                    outputs=merge_lora_status
                ).then(
                    fn=lambda: (gr.update(choices=get_lora_list_with_info()), gr.update(choices=get_lora_list_with_info())),
                    outputs=[lora_checkbox_group, lora_merge_checkbox]
                )
                # === SECTION 3: Checkpoint Components (Extract / Inject) ===
                with gr.Accordion("Checkpoint Components (VAE / CLIP)", open=False):
                    gr.Markdown(
                        "### Extract or reuse VAE / CLIP components\n"
                        "Export components as standalone `.safetensors` for reuse or inspection.\n"
                        "**Does not modify the original checkpoint.**"
                    )

                    # ---------- EXTRACT ----------
                    with gr.Row():
                        extract_checkpoint = gr.Dropdown(
                            label="Source Checkpoint",
                            choices=[cp.title for cp in sd_models.checkpoints_list.values()],
                            value=lambda: next(
                                (cp.title for cp in sd_models.checkpoints_list.values()), None
                            ),
                        )
                        create_refresh_button(
                            extract_checkpoint,
                            lambda: None,
                            lambda: {"choices": [cp.title for cp in sd_models.checkpoints_list.values()]},
                            "refresh_checkpoint_extract",
                        )

                    with gr.Row():
                        extract_vae = gr.Checkbox(
                            label="Extract VAE",
                            value=True,
                            info="Latent decoder (color, contrast, stability)",
                        )
                        extract_clip = gr.Checkbox(
                            label="Extract CLIP / Text Encoders",
                            value=False,
                            info="Prompt understanding & conditioning",
                        )

                    with gr.Row():
                        extract_name = gr.Textbox(
                            label="Output Name",
                            placeholder="my_model_component",
                            value="extracted_component",
                        )
                        extract_dtype = gr.Radio(
                            label="Save Precision",
                            choices=["fp16", "bf16", "fp32"],
                            value="fp16",
                            horizontal=True,
                        )

                    extract_btn = gr.Button("EXTRACT COMPONENTS", variant="primary")
                    extract_status = gr.Textbox(
                        label="Status",
                        lines=10,
                        interactive=False,
                        show_copy_button=True,
                    )

                    extract_btn.click(
                        fn=extract_checkpoint_components,
                        inputs=[
                            extract_checkpoint,
                            extract_vae,
                            extract_clip,
                            extract_name,
                            extract_dtype,
                        ],
                        outputs=extract_status,
                    )

                    gr.Markdown("### Inject External VAE / CLIP into a Checkpoint")

                    # ---------- INJECT ----------
                    with gr.Row():
                        inject_checkpoint = gr.Dropdown(
                            label="Target Checkpoint",
                            choices=[cp.title for cp in sd_models.checkpoints_list.values()],
                            value=lambda: next(
                                (cp.title for cp in sd_models.checkpoints_list.values()), None
                            ),
                        )
                        create_refresh_button(
                            inject_checkpoint,
                            lambda: None,
                            lambda: {"choices": [cp.title for cp in sd_models.checkpoints_list.values()]},
                            "refresh_checkpoint_inject",
                        )

                    with gr.Row():
                        inject_vae = gr.Dropdown(
                            label="VAE to Inject (optional)",
                            choices=get_external_vae_list(),
                            value=None,
                            allow_custom_value=False,
                        )
                        create_refresh_button(
                            inject_vae,
                            lambda: None,
                            lambda: {"choices": get_external_vae_list()},
                            "refresh_external_vae",
                        )

                    with gr.Row():
                        inject_clip = gr.Dropdown(
                            label="CLIP / Text Encoder to Inject (optional)",
                            choices=get_external_clip_list(),
                            value=None,
                            allow_custom_value=False,
                        )
                        create_refresh_button(
                            inject_clip,
                            lambda: None,
                            lambda: {"choices": get_external_clip_list()},
                            "refresh_external_clip",
                        )

                    inject_replace = gr.Radio(
                        label="Injection Mode",
                        choices=["Replace existing", "Only if missing"],
                        value="Replace existing",
                        horizontal=True,
                    )

                    inject_btn = gr.Button("INJECT COMPONENTS", variant="primary")
                    inject_status = gr.Textbox(
                        label="Status",
                        lines=10,
                        interactive=False,
                        show_copy_button=True,
                    )

                    inject_btn.click(
                        fn=inject_checkpoint_components,
                        inputs=[
                            inject_checkpoint,
                            inject_vae,
                            inject_clip,
                            inject_replace,
                        ],
                        outputs=inject_status,
                    )

        # ================================================
        # PRESETS & HISTORY TAB — Elite UX (2025 Edition)
        # ================================================
        with gr.Tab("Presets & History", elem_id="amethyst-presets"):
            gr.Markdown("### Manage Saved Merge Presets & View History")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Saved Presets")

                    with gr.Row():
                        preset_selector = gr.Dropdown(
                            choices=get_merge_presets(),
                            label="Select Preset",
                            allow_custom_value=False,
                            scale=4
                        )
                        preset_refresh = gr.Button("Refresh", scale=1)

                    with gr.Row():
                        preset_load = gr.Button("Load Preset", variant="primary", size="sm")
                        preset_delete = gr.Button("Delete", variant="stop", size="sm")

                    preset_status = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                        lines=2
                    )

                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="Save Current Settings As",
                            placeholder="My Epic Merge v3",
                            scale=4
                        )
                        preset_save = gr.Button("Save Preset", variant="secondary", size="sm")

                with gr.Column(scale=3):
                    gr.Markdown("## Recent Merge History")

                    history_box = gr.Textbox(
                        label="Last 10 Merges",
                        value="\n".join(get_merge_history()) or "No history yet",
                        interactive=False,
                        lines=12,
                        max_lines=15,
                        elem_classes="history-box"
                    )

                    history_refresh = gr.Button("Refresh History", variant="secondary")

            # === PRESET REFRESH ===
            def refresh_presets():
                return gr.update(choices=get_merge_presets())

            preset_refresh.click(
                fn=refresh_presets,
                outputs=preset_selector,
                show_progress=False
            )

            # === SAVE PRESET (now includes epsilon!) ===
            preset_save.click(
                fn=save_merge_preset,
                inputs=[
                    preset_name_input,
                    model_a, model_b, model_c, model_d,
                    merge_mode_selector, calc_mode_selector,
                    alpha, beta, gamma, delta, epsilon,  # ← NOW INCLUDES EPSILON
                    weight_editor, discard, clude, clude_mode
                ],
                outputs=preset_status
            ).then(
                fn=refresh_presets,
                outputs=preset_selector
            )

            # === LOAD PRESET (includes epsilon) ===
            preset_load.click(
                fn=load_merge_preset,
                inputs=preset_selector,
                outputs=[
                    model_a, model_b, model_c, model_d,
                    merge_mode_selector, calc_mode_selector,
                    alpha, beta, gamma, delta, epsilon,  # ← EPSILON INCLUDED
                    weight_editor, discard, clude, clude_mode,
                    preset_status
                ]
            )

            # === DELETE PRESET ===
            preset_delete.click(
                fn=delete_merge_preset,
                inputs=preset_selector,
                outputs=[preset_status, preset_selector]
            )

            # === HISTORY REFRESH ===
            def refresh_history():
                return gr.update(value="\n".join(get_merge_history()) or "No history yet")

            history_refresh.click(
                fn=refresh_history,
                outputs=history_box,
                show_progress=False
            )
        return [(cmn.blocks, "Untitled merger", "untitled_merger")]

# register the tab
script_callbacks.on_ui_tabs(on_ui_tabs)

# ---------------------------
# Helper functions
# ---------------------------
def start_merge(
    save_name, save_settings, finetune,
    merge_mode_selector, calc_mode_selector,
    model_a, model_b, model_c, model_d,
    alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda_, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon,
    weight_editor, preset_output,
    discard, clude, clude_mode,
    merge_seed, enable_sliders,
    copy_vae_from_primary,
    copy_clip_from_primary,
    keep_zero_fill,
    bloat_mode,
    dual_soul_toggle,
    sacred_keys_toggle,
    smartresize_toggle,
    specific_selectors_first,
    allow_glob_fallback,
    allow_exact_key_fallback,
    allow_synthetic_custom_merge,
    allow_non_float_merges,
    allow_scalar_merges,
    *custom_sliders
):
    # DEBUG: Log what parameters start_merge() received from UI
    print(f"[DEBUG] start_merge() received from UI:")
    print(f"  copy_vae_from_primary={copy_vae_from_primary!r} (type: {type(copy_vae_from_primary).__name__})")
    print(f"  copy_clip_from_primary={copy_clip_from_primary!r} (type: {type(copy_clip_from_primary).__name__})")
    print(f"  keep_zero_fill={keep_zero_fill!r} (type: {type(keep_zero_fill).__name__})")
    print(f"  bloat_mode={bloat_mode!r}")
    print(f"  smart_resize_toggle={smartresize_toggle!r}")
    print(f"  dual_soul_toggle={dual_soul_toggle!r}")
    
    # DEBUG: Show what's in cmn.opts at merge time
    print(f"\n[DEBUG] Current options in cmn.opts.options:")
    print(f"  copy_vae_from_primary: {cmn.opts.get('copy_vae_from_primary')}")
    print(f"  keep_zero_fill: {cmn.opts.get('keep_zero_fill')}")
    print(f"  smart_resize_toggle: {cmn.opts.get('smart_resize_toggle')}")
    progress = Progress()

    # ------------------------------------------------------------
    # Policy: UI inputs win. opts are fallback only.
    # ------------------------------------------------------------
    def pick(ui_value, opt_key, default):
        """
        UI value wins if provided.
        Otherwise fall back to opts.
        Final coercion to bool happens at call site.
        """
        if ui_value is not None:
            return ui_value
        return cmn.opts.get(opt_key, default)

    # ------------------------------------------------------------
    # USER-DRIVEN policy toggles (strict bool normalization)
    # ------------------------------------------------------------
    keep_zero_fill = bool(
        pick(keep_zero_fill, "keep_zero_fill", True)
    )

    bloat_mode = bool(
        pick(bloat_mode, "bloat_mode", False)
    )

    dual_soul_toggle = bool(
        pick(dual_soul_toggle, "dual_soul_toggle", False)
    )

    sacred_keys_toggle = bool(
        pick(sacred_keys_toggle, "sacred_keys_toggle", False)
    )

    smartresize_toggle = bool(
        pick(smartresize_toggle, "smart_resize_toggle", True)
    )

    specific_selectors_first = bool(
        pick(specific_selectors_first, "specific_selectors_first", True)
    )

    allow_glob_fallback = bool(
        pick(allow_glob_fallback, "allow_glob_fallback", True)
    )

    allow_exact_key_fallback = bool(
        pick(allow_exact_key_fallback, "allow_exact_key_fallback", True)
    )

    # Hard safety clamp
    if allow_glob_fallback and not allow_exact_key_fallback:
        print("[Policy Clamp] Disabling glob fallback (exact-key fallback is OFF)")
        allow_glob_fallback = False

    copy_vae_from_primary = bool(
        pick(copy_vae_from_primary, "copy_vae_from_primary", False)
    )

    copy_clip_from_primary = bool(
        pick(copy_clip_from_primary, "copy_clip_from_primary", False)
    )

    allow_synthetic_custom_merge = bool(
        pick(allow_synthetic_custom_merge, "allow_synthetic_custom_merge", True)
    )

    allow_non_float_merges = bool(
        pick(allow_non_float_merges, "allow_non_float_merges", True)
    )

    allow_scalar_merges = bool(
        pick(allow_scalar_merges, "allow_scalar_merges", True)
    )

    assert isinstance(allow_scalar_merges, bool), \
        "allow_scalar_merges must be a boolean"

    print(
        f"[Policy] Scalar merges "
        f"{'ENABLED' if cmn.allow_scalar_merges else 'DISABLED'}"
    )

    cmn.keep_zero_fill = keep_zero_fill
    cmn.bloat_mode = bloat_mode

    cmn.dual_soul_toggle = dual_soul_toggle
    cmn.sacred_toggle = sacred_keys_toggle
    cmn.smartresize_toggle = smartresize_toggle

    cmn.allow_glob_fallback = allow_glob_fallback
    cmn.allow_exact_key_fallback = allow_exact_key_fallback
    cmn.allow_synthetic_custom_merge = allow_synthetic_custom_merge
    cmn.allow_non_float_merges = allow_non_float_merges

    cmn.allow_scalar_merges = allow_scalar_merges

    cmn.copy_vae_from_primary = copy_vae_from_primary
    cmn.copy_clip_from_primary = copy_clip_from_primary

    # ------------------------------------------------------------
    # Debug: prove who won (UI vs opts)
    # ------------------------------------------------------------
    print(
        "[UI→Merge] "
        f"keep_zero={keep_zero_fill} | "
        f"bloat={bloat_mode} | "
        f"dual_soul={dual_soul_toggle} | "
        f"sacred={sacred_keys_toggle} | "
        f"smartresize={smartresize_toggle} | "
        f"specific_first={specific_selectors_first} | "
        f"glob_fallback={allow_glob_fallback} | "
        f"exact_key_fallback={allow_exact_key_fallback} | "
        f"synthetic={allow_synthetic_custom_merge} | "
        f"non_float={allow_non_float_merges} | "
        f"scalars={allow_scalar_merges} | "
        f"copy_vae={copy_vae_from_primary} | "
        f"copy_clip={copy_clip_from_primary}"
    )

    # ------------------------------------------------------------
    # Seed normalization (UI-safe)
    # ------------------------------------------------------------
    if merge_seed in (None, "", "random", "auto"):
        seed = -1
    else:
        try:
            seed = int(merge_seed)
            if seed < -1:
                raise ValueError ("Seed must be -1 or a non-negative integer")
        except Exception:
            raise ValueError(f"Invalid seed value from UI: {merge_seed!r}")

    try:
        progress.start_merge(1000)

        merger.prepare_merge(
            progress,
            save_name,
            save_settings,
            finetune,
            merge_mode_selector,
            calc_mode_selector,
            model_a,
            model_b,
            model_c,
            model_d,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            zeta,
            eta,
            theta,
            iota,
            kappa,
            lambda_,
            mu,
            nu,
            xi,
            omicron,
            pi,
            rho,
            sigma,
            tau,
            upsilon,
            weight_editor,
            preset_output or "",
            discard or "",
            clude or "",
            clude_mode,
            seed,
            enable_sliders,
            *custom_sliders,

            # 🔐 keyword-only policy wall
            keep_zero_fill=keep_zero_fill,
            bloat_mode=bloat_mode,
            copy_vae_from_primary=copy_vae_from_primary,
            copy_clip_from_primary=copy_clip_from_primary,
            dual_soul_toggle=dual_soul_toggle,
            sacred_keys_toggle=sacred_keys_toggle,
            smartresize_toggle=smartresize_toggle,
            specific_selectors_first=specific_selectors_first,
            allow_glob_fallback=allow_glob_fallback,
            allow_exact_key_fallback=allow_exact_key_fallback,
            allow_synthetic_custom_merge=allow_synthetic_custom_merge,
            allow_non_float_merges=allow_non_float_merges,
            allow_scalar_merges=allow_scalar_merges
        )


        # Lock UI checkpoint to primary
        try:
            if cmn.primary:
                base_title = os.path.splitext(
                    os.path.basename(cmn.primary)
                )[0]
                shared.opts.sd_model_checkpoint = base_title
                print(f"[Amethyst] UI checkpoint locked → {base_title}")
        except Exception as e:
            print(f"[Amethyst] UI lock warning: {e}")

        save_to_history(
            {
                "models": str([model_a, model_b, model_c, model_d]),
                "modes": f"{finetune}+{save_settings}",
            },
            "Success",
        )


    except Exception as error:
        merger.clear_cache()
        universal_model_reload()

        error_msg = str(error)[:60] + ("..." if len(str(error)) > 60 else "")
        save_to_history({"status": "Failed"}, f"Failed: {error_msg}")

        if not isinstance(error, getattr(merger, "MergeInterruptedError", Exception)):
            raise

    return progress.get_report()


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

def test_regex(input):
    regex = misc_util.target_to_regex(input)
    
    # Use ALL keys from every loaded checkpoint — TRUE kitchen-sink
    all_keys = []
    for f in cmn.loaded_checkpoints.values():
        if f is not None:
            all_keys.extend(f.keys())
    
    # Match against the full key set
    selected_keys = [k for k in all_keys if re.match(regex, k)]
    joined = '\n'.join(selected_keys)
    
    return f'Matched keys: {len(selected_keys)}\n{joined}'

def update_model_a_keys(model_a):
    global model_a_keys
    try:
        path = sd_models.get_closet_checkpoint_match(model_a).filename
        with safetensors.torch.safe_open(path,framework='pt',device='cpu') as file:
            model_a_keys = file.keys()
    except Exception:
        model_a_keys = []

def checkpoint_changed(name):
    if not name:
        return gr.update(value=plaintext_to_html('None | None'))
    
    try:
        sdversion, dtype = misc_util.id_checkpoint(name)
        dtype_str = str(dtype).split('.')[-1].upper() if dtype else 'Unknown'
        info_text = f"{sdversion or 'Unknown'} | {dtype_str}"
        return gr.update(value=plaintext_to_html(info_text))
    except Exception as e:
        print(f"[Merger] Checkpoint info error: {e}")
        return gr.update(value=plaintext_to_html('Error loading info'))
    
# ✅ ENHANCEMENT 1: Merge Presets Management
def get_merge_presets():
    """Load all merge presets"""
    try:
        with open(merge_presets_filename, 'r') as f:
            return list(json.load(f).keys())
    except:
        return []

def save_merge_preset(preset_name, model_a_val, model_b_val, model_c_val, model_d_val,
                      merge_mode_val, calc_mode_val, alpha_val, beta_val, gamma_val, delta_val, epsilon_val,
                      weight_editor_val, discard_val, clude_val, clude_mode_val):
    """Save current merge configuration as preset"""
    try:
        with open(merge_presets_filename, 'r') as f:
            presets = json.load(f)
    except:
        presets = {}

    if not preset_name or not preset_name.strip():
        return "Preset name required"

    presets[preset_name] = {
        'model_a': model_a_val,
        'model_b': model_b_val,
        'model_c': model_c_val,
        'model_d': model_d_val,
        'merge_mode': merge_mode_val,
        'calc_mode': calc_mode_val,
        'sliders': [alpha_val, beta_val, gamma_val, delta_val, epsilon_val],  # ← NOW 5!
        'weight_editor': weight_editor_val,
        'discard': discard_val,
        'clude': clude_val,
        'clude_mode': clude_mode_val
    }

    with open(merge_presets_filename, 'w') as f:
        json.dump(presets, f, indent=2)

    return f"Preset '{preset_name}' saved"


def load_merge_preset(preset_name):
    """Load merge configuration from preset"""
    try:
        with open(merge_presets_filename, 'r') as f:
            presets = json.load(f)

        if preset_name not in presets:
            return [gr.update()] * 15 + ["Preset not found"]

        preset = presets[preset_name]
        sliders = preset.get('sliders', [0.5]*5)  # fallback to 5 values

        return [
            gr.update(value=preset.get('model_a', '')),
            gr.update(value=preset.get('model_b', '')),
            gr.update(value=preset.get('model_c', '')),
            gr.update(value=preset.get('model_d', '')),
            gr.update(value=preset.get('merge_mode')),
            gr.update(value=preset.get('calc_mode')),
            gr.update(value=sliders[0]),  # alpha
            gr.update(value=sliders[1]),  # beta
            gr.update(value=sliders[2]),  # gamma
            gr.update(value=sliders[3]),      # delta
            gr.update(value=sliders[4]),  # epsilon ← NOW INCLUDED!
            gr.update(value=preset.get('weight_editor', '')),
            gr.update(value=preset.get('discard', '')),
            gr.update(value=preset.get('clude', '')),
            gr.update(value=preset.get('clude_mode', 'Exclude')),
            f"Loaded: {preset_name}"
        ]
    except Exception as e:
        return [gr.update()] * 15 + [f"Error: {str(e)}"]

def delete_merge_preset(preset_name):
    """Delete a saved preset"""
    try:
        with open(merge_presets_filename, 'r') as f:
            presets = json.load(f)
        
        if preset_name in presets:
            del presets[preset_name]
            with open(merge_presets_filename, 'w') as f:
                json.dump(presets, f, indent=2)
            return f"✓ Deleted preset: {preset_name}", gr.update(choices=list(presets.keys()))
        else:
            return "❌ Preset not found", gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()

# ✅ ENHANCEMENT 2: Merge History Tracking
def save_to_history(merge_config, result_status):
    """Save successful merge to history"""
    try:
        with open(merge_history_filename, 'r') as f:
            history = json.load(f)
    except:
        history = []
    
    from datetime import datetime
    entry = {
        'timestamp': datetime.now().isoformat(),
        'config': merge_config,
        'status': result_status
    }
    
    history.append(entry)
    # Keep last 50 merges
    history = history[-50:]
    
    with open(merge_history_filename, 'w') as f:
        json.dump(history, f, indent=2)

def get_merge_history():
    """Get list of recent merges"""
    try:
        with open(merge_history_filename, 'r') as f:
            history = json.load(f)
        
        items = []
        for entry in history[-10:]:  # Last 10
            ts = entry.get('timestamp', '')[:16]
            status = '✓' if 'success' in entry.get('status', '').lower() else '✗'
            items.append(f"{status} {ts}")
        
        return list(reversed(items))
    except:
        return []

def preview_merged_model(
    model_a_val, model_b_val, model_c_val, model_d_val,
    merge_mode_val, calc_mode_val,
    alpha_val, beta_val, gamma_val, delta_val, epsilon_val,
    prompt_val="", negative_prompt_val="",
    steps=20, sampler_name="Euler a",
    width=512, height=512, cfg_scale=7.0, seed=-1,
    batch_count=1, batch_size=1,
    discard_val="", clude_val="", clude_mode_val="Exclude",
    weight_editor_val="",
    progress=gr.Progress(track_tqdm=True)
):
    """
    SuperMerger-style in-memory merge + live preview.
    Fully compatible with Forge Neo and Automatic1111 dev branch.
    Returns: (images, infotext, status_message, merge_report)
    """
    if not shared.sd_model:
        return [], "", "Error: No model is currently loaded in the WebUI.", ""

    progress(0, desc="Merging models in memory...")

    try:
        merged_state, merge_report = merger.do_merge(
            model_a_val, model_b_val, model_c_val, model_d_val,
            merge_mode_val, calc_mode_val,
            alpha_val, beta_val, gamma_val, delta_val, epsilon_val,
            weight_editor_val, discard_val, clude_val, clude_mode_val,
            seed=seed if seed >= 0 else -1,
            progress=progress
        )
        if not merged_state:
            return [], "", "Merge failed: No state dictionary returned.", merge_report or "No merge report provided."

    except Exception as e:
        import traceback
        error_msg = f"Merge error: {str(e)}\nDetails: {traceback.format_exc()}"
        print("[Amethyst Preview] ", error_msg)
        return [], "", f"Failed: {error_msg}", ""

    progress(60, desc="Loading merged weights into current model...")

    # Convert to half precision for inference
    merged_state_half = {k: v.half() for k, v in merged_state.items()}

    # Create dummy checkpoint info
    dummy_info = sd_models.CheckpointInfo("in_memory_merged_preview")
    dummy_info.filename = "in_memory"

    try:
        # A1111 dev: Fast reuse path (best performance)
        if (hasattr(sd_models, 'load_model_weights') and
                shared.sd_model is not None and
                hasattr(shared.sd_model, 'used_config')):
            sd_models.load_model_weights(
                shared.sd_model,
                dummy_info,
                merged_state_half
            )
            print("[Amethyst] Preview: Fast weight reuse (A1111 dev path)")
        # Forge Neo & fallback: Full reload (safe & universal)
        else:
            sd_models.load_model(
                checkpoint_info=dummy_info,
                already_loaded_state_dict=merged_state_half
            )
            print("[Amethyst] Preview: Full model load (Forge Neo / safe path)")

    except Exception as e:
        import traceback
        error_msg = f"Failed to load merged weights: {str(e)}\nDetails: {traceback.format_exc()}"
        print("[Amethyst Preview] ", error_msg)
        return [], "", f"Failed: {error_msg}", merge_report

    progress(80, desc="Generating preview image...")

    try:
        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=prompt_val or "masterpiece, best quality, highly detailed",
            negative_prompt=negative_prompt_val or "blurry, low quality",
            seed=seed if seed >= 0 else -1,
            sampler_name=sampler_name,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            batch_size=batch_size,
            n_iter=batch_count,
            do_not_save_samples=True,
            do_not_save_grid=True,
        )

        # Properly initialize scripts (supports ControlNet, ADetailer, etc.)
        p.scripts = scripts.scripts_txt2img
        p.script_args = [None] * len(getattr(scripts.scripts_txt2img, "alwayson_scripts", []))

        processed = processing.process_images(p)

    except Exception as e:
        import traceback
        error_msg = f"Image generation failed: {str(e)}\nDetails: {traceback.format_exc()}"
        print("[Amethyst Preview] ", error_msg)
        return [], "", f"Failed: {error_msg}", merge_report

    finally:
        # Critical cleanup to prevent memory leaks
        del merged_state_half
        del merged_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    infotext = processed.infotexts[0] if processed.infotexts else ""
    progress(100, desc="Preview complete!")

    return (
        processed.images,
        infotext,
        "Preview complete! Model merged and image generated successfully.",
        merge_report
    )

def validate_merge_config(model_a, model_b, weight_editor, merge_mode):
    """Validate merge configuration before execution"""
    if weight_editor.strip():
        try:
            # If it looks like valid JSON (starts with { and ends with }), assume it's from preset
            stripped = weight_editor.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                # Quick sanity check — does it contain alpha/beta?
                if any(k in stripped for k in ['"alpha"', '"beta"', '"gamma"', '"delta"']):
                    return True, "Preset applied — ready to merge"
        except:
            pass

    errors = []
    
    if not model_a or model_a == "":
        errors.append("❌ Model A (Primary) is required")
    if not model_b or model_b == "":
        errors.append("❌ Model B (Secondary) is required")
    
    if weight_editor.strip() == "":
        errors.append("⚠️ Weight editor is empty - only default-to-A merge will occur")
    
    try:
        lines = [l.strip() for l in weight_editor.split('\n') if l.strip() and not l.strip().startswith('#')]
        for line in lines:
            if ':' not in line:
                errors.append(f"❌ Invalid syntax: '{line}' (missing colon)")
    except:
        pass
    
    if errors:
        return False, '\n'.join(errors)
    return True, "✓ Configuration valid"


def get_external_vae_list():
    base = paths.models_path
    dirs = [os.path.join(base, "VAE"), os.path.join(base, "vae")]
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".safetensors"):
                out.append(os.path.join(d, f))
    return out


def get_external_clip_list():
    base = paths.models_path
    dirs = [
        os.path.join(base, "CLIP"),
        os.path.join(base, "TextEncoders"),
    ]
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".safetensors"):
                out.append(os.path.join(d, f))
    return out
