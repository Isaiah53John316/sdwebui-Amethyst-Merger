# Contains slider-help, soft-disable, corrected mode logic, LoRA tab fixed inside Blocks context - Fixed save with model_ema - Works now with Dev branch of Automatic1111 - needed for 50xx.
import gradio as gr
import os
import re
import json
import shutil
import torch
import safetensors
import safetensors.torch
import tempfile  # Add this import at the top of untitled_ui.py if not already present
from datetime import datetime  # For cleaner temp naming
from time import time  # ✅ ADDED
from modules import sd_models,script_callbacks,scripts,shared,ui_components,paths,sd_samplers,ui,call_queue
from modules.ui_common import plaintext_to_html, create_refresh_button
from scripts.untitled import merger,misc_util
from scripts.untitled.operators import weights_cache
from scripts.untitled import lora_merge
import scripts.untitled.common as cmn

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
    
    def update_progress(self, current, total):
        self.merge_keys = current
        self.total_keys = total
        eta = self.get_eta()
        progress_pct = (current / total * 100) if total > 0 else 0
        return f"Progress: {progress_pct:.1f}% - {eta}"

    def __call__(self,message, v=None, popup = False, report=False):
        if v:
            message = ' - '+ message + ' ' * (25-len(message)) + ': ' + str(v)
        if report:
            self.ui_report.append(message)
        if popup:
            gr.Info(message)
        print(message)

    def interrupt(self,message,popup=True):
        message = 'Merge interrupted:\t'+message
        if popup:
            gr.Warning(message)
        self.ui_report = [message]
        raise merger.MergeInterruptedError

    def get_report(self):
        return '\n'.join(self.ui_report)


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
_DEFAULT_SLIDER_MARKER = (-1, 2, 0.01)

def _get_mode_objects(mergemode_name, calcmode_name):
    mergemode = merger.mergemode_selection[mergemode_name]
    calcmode  = merger.calcmode_selection[calcmode_name]
    return mergemode, calcmode

def _choose_slider_configs(mergemode, calcmode):
    """
    Decide slider (a,b,c,d) configurations and info strings.
    CalcMode configs take priority if they are not equal to the default sentinel.
    Returns tuple: (a_cfg,a_info,b_cfg,b_info,c_cfg,c_info,d_cfg,d_info)
    """
    use_calc = hasattr(calcmode, 'slid_a_config') and getattr(calcmode, 'slid_a_config') != _DEFAULT_SLIDER_MARKER

    if use_calc:
        a_cfg = getattr(calcmode,'slid_a_config', mergemode.slid_a_config)
        a_info = getattr(calcmode,'slid_a_info', mergemode.slid_a_info)
        b_cfg = getattr(calcmode,'slid_b_config', mergemode.slid_b_config)
        b_info = getattr(calcmode,'slid_b_info', mergemode.slid_b_info)
        c_cfg = getattr(calcmode,'slid_c_config', mergemode.slid_c_config)
        c_info = getattr(calcmode,'slid_c_info', mergemode.slid_c_info)
        d_cfg = getattr(calcmode,'slid_d_config', mergemode.slid_d_config)
        d_info = getattr(calcmode,'slid_d_info', mergemode.slid_d_info)
    else:
        a_cfg, a_info = mergemode.slid_a_config, mergemode.slid_a_info
        b_cfg, b_info = mergemode.slid_b_config, mergemode.slid_b_info
        c_cfg, c_info = mergemode.slid_c_config, mergemode.slid_c_info
        d_cfg, d_info = mergemode.slid_d_config, mergemode.slid_d_info

    return (a_cfg, a_info, b_cfg, b_info, c_cfg, c_info, d_cfg, d_info)

def _required_counts(mergemode, calcmode):
    """
    Determine required sliders and required models.
    """
    base_sliders = getattr(mergemode, 'input_sliders', 4) or 4
    base_models  = getattr(mergemode, 'input_models', 4) or 4

    calc_sliders_attr = getattr(calcmode, 'input_sliders', None)
    calc_models_attr  = getattr(calcmode, 'input_models', None)

    calc_non_default_slider_count = 0
    for attr in ('slid_a_config','slid_b_config','slid_c_config','slid_d_config'):
        cfg = getattr(calcmode, attr, None)
        if cfg is not None and cfg != _DEFAULT_SLIDER_MARKER:
            calc_non_default_slider_count += 1

    req_sliders = base_sliders
    if calc_sliders_attr:
        try:
            req_sliders = max(req_sliders, int(calc_sliders_attr))
        except Exception:
            pass
    req_sliders = max(req_sliders, calc_non_default_slider_count)

    if calc_models_attr:
        try:
            req_models = int(calc_models_attr)
        except Exception:
            req_models = base_models
    else:
        req_models = base_models

    try:
        req_sliders = int(req_sliders)
    except Exception:
        req_sliders = 4
    try:
        req_models = int(req_models)
    except Exception:
        req_models = 4

    req_sliders = max(1, min(req_sliders, 4))
    req_models  = max(1, min(req_models, 4))

    return req_sliders, req_models

def _compatible(mergemode, calcmode):
    """Return True if calc mode is compatible with merge mode"""
    compat = getattr(calcmode, 'compatible_modes', ['all'])
    if 'all' in compat:
        return True
    compat_list = [c for c in compat if isinstance(c, str)]
    return (mergemode.name in compat_list) or (mergemode.name in compat)

def mode_changed(mergemode_name, calcmode_name):
    """
    Returns updates in this order:
    merge_desc_update, calc_desc_update,
    slider_a_update, slider_b_update, slider_c_update, slider_d_update,
    slider_help_update,
    model_a_update, model_b_update, model_c_update, model_d_update,
    merge_button_update
    """
    mergemode, calcmode = _get_mode_objects(mergemode_name, calcmode_name)

    a_cfg, a_info, b_cfg, b_info, c_cfg, c_info, d_cfg, d_info = _choose_slider_configs(mergemode, calcmode)

    slider_a_update = gr.update(minimum=a_cfg[0], maximum=a_cfg[1], step=a_cfg[2], info=a_info)
    slider_b_update = gr.update(minimum=b_cfg[0], maximum=b_cfg[1], step=b_cfg[2], info=b_info)
    slider_c_update = gr.update(minimum=c_cfg[0], maximum=c_cfg[1], step=c_cfg[2], info=c_info)
    slider_d_update = gr.update(minimum=d_cfg[0], maximum=d_cfg[1], step=d_cfg[2], info=d_info)

    req_sliders, req_models = _required_counts(mergemode, calcmode)

    # Soft-disable unused sliders (interactive False)
    def _sl_interactive(idx):
        return True if idx <= req_sliders else False

    # Merge update dicts with interactive flag
    slider_a_update = {**(slider_a_update if isinstance(slider_a_update, dict) else {}), 'interactive': _sl_interactive(1)}
    slider_b_update = {**(slider_b_update if isinstance(slider_b_update, dict) else {}), 'interactive': _sl_interactive(2)}
    slider_c_update = {**(slider_c_update if isinstance(slider_c_update, dict) else {}), 'interactive': _sl_interactive(3)}
    slider_d_update = {**(slider_d_update if isinstance(slider_d_update, dict) else {}), 'interactive': _sl_interactive(4)}

    # Slider help textbox content (textbox style)
    header = f"{mergemode.name} (merge) • {calcmode.name} (calc)"
    slider_help_text = (
        f"{header}\n\n"
        f"α (alpha): {a_info or '-'}\n"
        f"β (beta) : {b_info or '-'}\n"
        f"γ (gamma): {c_info or '-'}\n"
        f"δ (delta): {d_info or '-'}\n\n"
        f"(Note: unused sliders are visible but disabled.)"
    )
    slider_help_update = gr.update(value=slider_help_text)

    # Soft-disable model dropdowns
    model_a_update = gr.update(interactive=True)
    model_b_update = gr.update(interactive=True) if req_models >= 2 else gr.update(interactive=False)
    model_c_update = gr.update(interactive=True) if req_models >= 3 else gr.update(interactive=False)
    model_d_update = gr.update(interactive=True) if req_models >= 4 else gr.update(interactive=False)

    merge_desc_update = gr.update(value=mergemode.description)
    calc_desc_update  = gr.update(value=calcmode.description)

    merge_button_update = gr.update(interactive=_compatible(mergemode, calcmode))

    return (
        merge_desc_update, calc_desc_update,
        slider_a_update, slider_b_update, slider_c_update, slider_d_update,
        slider_help_update,
        model_a_update, model_b_update, model_c_update, model_d_update,
        merge_button_update
    )

# ---------------------------
# Utility UI helpers
# ---------------------------
def get_checkpoints_list(sort):
    checkpoints_list = [x.title for x in sd_models.checkpoints_list.values() if x.is_safetensors]
    if sort == 'Newest first':
        sort_func = lambda x: os.path.getctime(sd_models.get_closet_checkpoint_match(x).filename)
        checkpoints_list.sort(key=sort_func,reverse=True)
    return checkpoints_list

def get_lora_list():
    """Return deduplicated list of LoRA paths with filename only (no duplicates)"""
    lora_dirs = [
        os.path.join(paths.models_path, "Lora"),
        os.path.join(paths.models_path, "lora"),
        os.path.join(paths.models_path, "LyCORIS"),
    ]
    
    seen = set()
    loras = []
    
    for directory in lora_dirs:
        if not os.path.exists(directory):
            continue
        for file in os.listdir(directory):
            if file.lower().endswith(('.safetensors', '.pt')):
                filepath = os.path.join(directory, file)
                # Use just the filename as unique key
                if file not in seen:
                    seen.add(file)
                    # Show short name + (SDXL) or (SD1.5) hint if possible
                    display_name = file
                    try:
                        with safetensors.torch.safe_open(filepath, framework="pt", device="cpu") as f:
                            if "lora_unet_input_blocks_0_0" in f.keys():  # rough SD1.5 check
                                display_name += " [SD1.5]"
                            elif "lora_unet_input_blocks_4_1" in f.keys():  # rough SDXL check
                                display_name += " [SDXL]"
                    except:
                        pass
                    loras.append((display_name, filepath))
    
    return loras

def refresh_models(sort):
    sd_models.list_models()
    checkpoints_list = get_checkpoints_list(sort)
    return gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list)

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

def validate_merge_config(model_a, model_b, weight_editor, merge_mode):
    """Validate merge configuration before execution"""
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

# ---------------------------
# UI: build tabs
# ---------------------------
def on_ui_tabs():
    with gr.Blocks() as cmn.blocks:
        with gr.Tab("Merge"):
            dummy_component = gr.Textbox(visible=False,interactive=True)
            with ui_components.ResizeHandleRow():
                with gr.Column():
                    with gr.Group(label="Merge Status & Logs"):
                        status = gr.Textbox(
                            max_lines=4,
                            lines=4,
                            show_label=False,
                            interactive=False,
                            render=True,
                            elem_id="merge_status"
                        )
                        
                        with gr.Accordion("Detailed Logs", open=False):
                            detailed_logs = gr.Textbox(
                                max_lines=20,
                                lines=20,
                                interactive=False,
                                show_copy_button=True
                            )
                            
                            clear_logs_btn = gr.Button("Clear Logs")
                            clear_logs_btn.click(fn=lambda: gr.update(value=""), outputs=detailed_logs)

                    # MODEL SELECTION
                    with gr.Row():
                        slider_scale = 8
                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_a = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_a [Primary]",scale=slider_scale)
                                swap_models_AB = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                            model_a_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            # update model info when changed
                            model_a.change(fn=checkpoint_changed, inputs=model_a, outputs=model_a_info)
                            model_a.change(fn=update_model_a_keys, inputs=model_a)

                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_b = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_b [Secondary]",scale=slider_scale)
                                swap_models_BC = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                            model_b_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_b.change(fn=checkpoint_changed,inputs=model_b,outputs=model_b_info)

                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_c = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_c [Tertiary]",scale=slider_scale)
                                swap_models_CD = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                            model_c_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_c.change(fn=checkpoint_changed,inputs=model_c,outputs=model_c_info)

                        with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                            with gr.Row():
                                model_d = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="model_d [Supplementary]",scale=slider_scale)
                                refresh_button = gr.Button(value='🔄', elem_classes=["tool"],scale=1)
                            model_d_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                            model_d.change(fn=checkpoint_changed,inputs=model_d,outputs=model_d_info)

                        checkpoint_sort = gr.Dropdown(min_width=60,scale=1,visible=True,choices=['Alphabetical','Newest first'],value='Alphabetical',label='Sort')

                        def swapvalues(x,y): return gr.update(value=y), gr.update(value=x)
                        swap_models_AB.click(fn=swapvalues,inputs=[model_a,model_b],outputs=[model_a,model_b])
                        swap_models_BC.click(fn=swapvalues,inputs=[model_b,model_c],outputs=[model_b,model_c])
                        swap_models_CD.click(fn=swapvalues,inputs=[model_c,model_d],outputs=[model_c,model_d])
                        refresh_button.click(fn=refresh_models,inputs=checkpoint_sort, outputs=[model_a,model_b,model_c,model_d])
                        checkpoint_sort.change(fn=refresh_models,inputs=checkpoint_sort,outputs=[model_a,model_b,model_c,model_d])

                    # MODE SELECTION
                    with gr.Row():
                        merge_mode_selector = gr.Radio(
                            label='Merge Mode (formula structure):',
                            choices=list(merger.mergemode_selection.keys()),
                            value=list(merger.mergemode_selection.keys())[0],
                            scale=3
                        )
                    merge_mode_desc = gr.Textbox(
                        label="Merge Mode Description",
                        value=merger.mergemode_selection[list(merger.mergemode_selection.keys())[0]].description,
                        interactive=False,
                        lines=2
                    )

                    with gr.Row():
                        calc_mode_selector = gr.Radio(
                            label='Calculation Mode (how to execute):',
                            choices=list(merger.calcmode_selection.keys()),
                            value=list(merger.calcmode_selection.keys())[0],
                            scale=3
                        )
                    calc_mode_desc = gr.Textbox(
                        label="Calculation Mode Description",
                        value=merger.calcmode_selection[list(merger.calcmode_selection.keys())[0]].description,
                        interactive=False,
                        lines=2
                    )

                    with gr.Row():
                        cross_arch_mode = gr.Dropdown(
                            choices=["Standard (Same Arch)", "Cross-Arch (SD1.5/Flux to SDXL)"],
                            value="Standard (Same Arch)",
                            label="Merge Type (Architecture)",
                            info="Cross-Arch allows merging completely different model families"
                        )
                        thread_count = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=8,
                            step=1,
                            label="Thread count (speed)"
                        )
                        cache_size = gr.Slider(
                            minimum=512,
                            maximum=8192,
                            value=2048,
                            step=256,
                            label="Cache size (MB) – lower = less RAM"
                        )

                    # Connect UI controls to backend settings
                    cross_arch_mode.change(
                        fn=lambda x: setattr(cmn, 'cross_arch_enabled', x == "Cross-Arch (SD1.5/Flux to SDXL)"),
                        inputs=cross_arch_mode,
                        outputs=None
                    )

                    thread_count.change(
                        fn=lambda x: cmn.opts.options.update({'threads': int(x)}) or cmn.opts.save(),
                        inputs=thread_count,
                        outputs=None
                    )

                    slider_help = gr.Textbox(label="Slider Meaning", value="", interactive=False, lines=6, placeholder="Slider help will appear here when you change merge/calc modes.")

                    # MAIN SLIDERS - ✅ Updated to 0.0000001 for 7 decimal places (float32 precision)
                    with gr.Row(equal_height=True):
                        alpha = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_a [α] (alpha)", info='model_a - model_b', value=0.5, elem_classes=['main_sliders'])
                        beta  = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_b [β] (beta)",  info='-', value=0.5, elem_classes=['main_sliders'])
                        gamma = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_c [γ] (gamma)", info='-', value=0.25, elem_classes=['main_sliders'])
                        delta = gr.Slider(minimum=-1, step=0.0000001, maximum=2, label="slider_d [δ] (delta)", info='-', value=0.25, elem_classes=['main_sliders'])

                    # CUSTOM SLIDERS UI
                    with ui_components.InputAccordion(False, label='Custom sliders') as enable_sliders:
                        with gr.Accordion(label = 'Presets'):
                            with gr.Row(variant='compact'):
                                sliders_preset_dropdown = gr.Dropdown(label='Preset Name',allow_custom_value=True,choices=get_slider_presets(),value='blocks',scale=4)
                                slider_refresh_button = gr.Button(value='🔄', elem_classes=["tool"],scale=1,min_width=40)
                                slider_refresh_button.click(fn=lambda:gr.update(choices=get_slider_presets()),outputs=sliders_preset_dropdown)
                                sliders_preset_load = gr.Button(variant='secondary',value='Load presets',scale=2)
                                sliders_preset_save = gr.Button(variant='secondary',value='Save sliders as preset',scale=2)
                            with open(custom_sliders_examples,'r') as file:
                                presets = json.load(file)
                            slid_defaults = iter(presets['blocks'])
                            slider_slider = gr.Slider(step=2,maximum=26,value=slid_defaults.__next__(),label='Enabled Sliders')

                        custom_sliders = []
                        with gr.Row():
                            for w in [6,1,6]:
                                with gr.Column(scale=w,min_width=0):
                                    if w>1:
                                        for i in range(13):
                                            with gr.Row(variant='compact'):
                                                custom_sliders.append(gr.Textbox(show_label=False,visible=True,value=slid_defaults.__next__(),placeholder='target',min_width=100,scale=1,lines=1,max_lines=1))
                                                custom_sliders.append(gr.Slider(show_label=False,value=slid_defaults.__next__(),scale=6,minimum=0,maximum=1,step=0.0000001))

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

                            finetune.change(fn=lambda x:gr.update(label = f"Supermerger Adjust : {x}"if x != "" and x !="0,0,0,0,0,0,0,0" else "Supermerger Adjust"),inputs=[finetune],outputs = [acc_ad])

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
                            except ValueError as err: gr.Warning(str(err))
                            except AssertionError as err: gr.Warning(str(err))
                            else: return [gr.update(value=x) for x in tmp]
                            return [gr.update()]*8

                        finetunes = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                        finetune_reset.click(fn=lambda: [gr.update(value="")]+[gr.update(value=0.0)]*8, inputs=[], outputs=[finetune, *finetunes])
                        finetune_read.click(fn=finetune_reader, inputs=[finetune], outputs=[*finetunes])
                        finetune_write.click(fn=finetune_update, inputs=[finetune, *finetunes], outputs=[finetune])
                        detail1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        detail2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        detail3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        contrast.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        bri.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        col1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        col2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                        col3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)

                    # OPTIONS
                    with gr.Accordion(label='Options',open=False):
                        save_options_button = gr.Button(value = 'Save',variant='primary')
                        save_options_button.click(fn=cmn.opts.save)
                        cmn.opts.create_option('trash_model',
                                            gr.Radio,
                                            {'choices':['Disable','Enable for SDXL','Enable'],
                                                'label':'Clear loaded SD models from memory at the start of merge:',
                                                'info':'Saves some memory but increases loading times'},
                                                default='Enable for SDXL')

                        cmn.opts.create_option('device',
                                            gr.Radio,
                                            {'choices':['cuda/float16', 'cuda/float32', 'cpu/float32'],
                                                'label':'Preferred device/dtype for merging:'},
                                                default='cuda/float16')

                        cmn.opts.create_option('threads',
                                            gr.Slider,
                                            {'step':2,
                                                'minimum':2,
                                                'maximum':20,
                                                'label':'Worker thread count:',
                                                'info':'Relevant for both cuda and CPU merging. Using too many threads can harm performance. Your core-count +-2 is a good guideline.'},
                                                default=8)

                        cache_size_slider = cmn.opts.create_option('cache_size',
                                            gr.Slider,
                                            {'step':64,
                                                'minimum':0,
                                                'maximum':16384,
                                                'label':'Cache size (MB):',
                                                'info':'Stores the result of intermediate calculations, such as the difference between B and C in add-difference before its multiplied and added to A.'},
                                                default=4096)

                        cache_size_slider.release(fn=lambda x: weights_cache.__init__(x),inputs=cache_size_slider)
                        weights_cache.__init__(cmn.opts['cache_size'])

                    # MERGE & SAVE UI
                    with gr.Row(equal_height=True):
                        with gr.Column(variant='panel'):
                            save_name = gr.Textbox(max_lines=1,label='Save checkpoint as:',lines=1,placeholder='Enter name...',scale=2)
                            with gr.Row():
                                save_settings = gr.CheckboxGroup(label = " ",choices=["Autosave","Overwrite","fp16","bf16"],value=['fp16'],interactive=True,scale=2,min_width=100)
                                save_loaded = gr.Button(value='Save loaded checkpoint',size='sm',scale=1)
                                save_loaded.click(fn=misc_util.save_loaded_model, inputs=[save_name,save_settings], outputs=status)
                                save_loaded.click(fn=refresh_models, inputs=checkpoint_sort, outputs=[model_a,model_b,model_c,model_d])

                        with gr.Column():
                            merge_button = gr.Button(value='Merge',variant='primary')
                            with gr.Row():
                                empty_cache_button = gr.Button(value='Empty Cache')
                                empty_cache_button.click(fn=merger.clear_cache,outputs=status)

                                stop_button = gr.Button(value='Stop')
                                def stopfunc(): cmn.stop = True;shared.state.interrupt()
                                stop_button.click(fn=stopfunc)
                            with gr.Row():
                                merge_seed = gr.Number(label='Merge Seed', value=99,  min_width=100, precision=0,scale=1)
                                merge_random_seed = ui_components.ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                                merge_random_seed.click(fn=lambda:-1, outputs=merge_seed)
                                merge_reuse_seed = ui_components.ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation, mostly useful if it was randomized")
                                merge_reuse_seed.click(fn=lambda:cmn.last_merge_seed, outputs=merge_seed)

                    # INCLUDE / EXCLUDE
                    with gr.Accordion(label='Include/Exclude/Discard',open=False):
                        with gr.Row():
                            with gr.Column():
                                clude = gr.Textbox(
                                    max_lines=4,
                                    label='Include/Exclude:',
                                    info="Entered targets will remain as model_a when set to 'Exclude', and will be the only ones to be merged if set to 'Include'. Separate with whitespace.",
                                    value='clip',
                                    lines=4,
                                    scale=4
                                )

                                clude_mode = gr.Radio(label="", info="", choices=["Exclude", ("Include exclusively", 'include')], value='Exclude', min_width=300, scale=1)

                            with gr.Column():
                                discard = gr.Textbox(
                                    max_lines=5,
                                    label='Discard:',
                                    info="Remove layers from final save (autosave only). Examples: 'model_ema', 'first_stage_model', or 'model_ema first_stage_model'. Leave empty to keep all layers.",
                                    value='model_ema',
                                    lines=5,
                                    scale=1
                                )

                    # Weight editor
                    with gr.Accordion('Weight editor'):
                        weight_editor = gr.Code(value=EXAMPLE,lines=20,language='yaml',label='')

                    # Validation output (add this INSIDE on_ui_tabs)
                    validation_output = gr.Textbox(
                        max_lines=3,
                        label="Validation",
                        value="✓ Ready",
                        interactive=False,
                        lines=3
                    )

                    # Model keys test
                    with gr.Accordion('Model keys'):
                        target_tester = gr.Textbox(max_lines=1,label="Checks model_a keys using simple expression.",info="'*' is used as wildcard. Start expression with 'cond*' for clip. 'c*embedders.0*' for small clip. 'c*embedders.1*' for big clip. 'model.*' for unet and 'model_ema*' for ema unet",interactive=True,placeholder='model.*out*4*tran*norm*weight')
                        target_tester_display = gr.Textbox(max_lines=40,lines=40,label="Targeted keys:",info="",interactive=False)
                        target_tester.change(fn=test_regex,inputs=[target_tester],outputs=target_tester_display,show_progress='minimal')

                    # Wire the mode_changed events
                    merge_mode_selector.change(
                        fn=mode_changed,
                        inputs=[merge_mode_selector, calc_mode_selector],
                        outputs=[
                            merge_mode_desc, calc_mode_desc,
                            alpha, beta, gamma, delta,
                            slider_help,
                            model_a, model_b, model_c, model_d,
                            merge_button
                        ],
                        show_progress='hidden'
                    )
                    calc_mode_selector.change(
                        fn=mode_changed,
                        inputs=[merge_mode_selector, calc_mode_selector],
                        outputs=[
                            merge_mode_desc, calc_mode_desc,
                            alpha, beta, gamma, delta,
                            slider_help,
                            model_a, model_b, model_c, model_d,
                            merge_button
                        ],
                        show_progress='hidden'
                    )

                    # Wire validation
                    weight_editor.change(
                        fn=lambda we, ma, mb, mm: validate_merge_config(ma, mb, we, mm),
                        inputs=[weight_editor, model_a, model_b, merge_mode_selector],
                        outputs=validation_output,
                        show_progress=False
                    )

                    # Merge args and button wiring
                    merge_args = [
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
                        weight_editor,
                        discard,
                        clude,
                        clude_mode,
                        merge_seed,
                        enable_sliders,
                        slider_slider,
                        *custom_sliders
                    ]

                    merge_button.click(fn=start_merge,inputs=[save_name,save_settings,*merge_args],outputs=status)

        # Compare Tab
        with gr.Tab("Compare"):
            gr.Markdown("### Compare Two Checkpoints")
            
            with gr.Row():
                compare_original = gr.Dropdown(
                    get_checkpoints_list('Alphabetical'),
                    label="Original Model"
                )
                compare_merged = gr.Dropdown(
                    get_checkpoints_list('Alphabetical'),
                    label="Merged Model"
                )
            
            compare_output = gr.Textbox(
                label="Comparison Results",
                interactive=False,
                lines=8
            )
            
            def compare_models(orig, merged):
                """Compare two checkpoint files"""
                try:
                    with safetensors.torch.safe_open(
                        sd_models.get_closet_checkpoint_match(orig).filename,
                        framework='pt', device='cpu'
                    ) as f1:
                        orig_keys = set(f1.keys())
                        orig_size = os.path.getsize(
                            sd_models.get_closet_checkpoint_match(orig).filename
                        )
                    
                    with safetensors.torch.safe_open(
                        sd_models.get_closet_checkpoint_match(merged).filename,
                        framework='pt', device='cpu'
                    ) as f2:
                        merged_keys = set(f2.keys())
                        merged_size = os.path.getsize(
                            sd_models.get_closet_checkpoint_match(merged).filename
                        )
                    
                    result = f"""Original Model: {orig}
Size: {orig_size / 1e9:.2f} GB
Keys: {len(orig_keys)}

Merged Model: {merged}
Size: {merged_size / 1e9:.2f} GB
Keys: {len(merged_keys)}

Differences:
- Keys removed: {len(orig_keys - merged_keys)}
- Keys added: {len(merged_keys - orig_keys)}
- Size change: {(merged_size - orig_size) / 1e9:.2f} GB"""
                    return result
                except Exception as e:
                    return f"Error: {str(e)}"
            
            compare_button = gr.Button("Compare")
            compare_button.click(
                fn=compare_models,
                inputs=[compare_original, compare_merged],
                outputs=compare_output
            )
            
        # LoRA Tab
        with gr.Tab("LoRA", elem_id="tab_lora"):
            gr.Markdown("## LoRA Merging - LoRA to Checkpoint - LoRA to LoRA")

            with gr.Accordion("Merge LoRA(s) to Checkpoint", open=False):
                gr.Markdown("### Apply multiple LoRAs to a checkpoint — with individual weights!")

                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        label="Base Checkpoint",
                        choices=sorted(sd_models.checkpoints_list.keys()),
                        value=None
                    )
                    create_refresh_button(checkpoint_dropdown, sd_models.list_models, 
                                        lambda: {"choices": sorted(sd_models.checkpoints_list.keys())}, "refresh_checkpoints")

                with gr.Row():
                    lora_checkbox_group = gr.CheckboxGroup(
                        label="Select LoRAs to bake into checkpoint",
                        choices=[],
                        type="value",
                        interactive=True
                    )
                    create_refresh_button(
                        lora_checkbox_group,
                        lambda: None,
                        lambda: {'choices': get_lora_list()},
                        "refresh_loras_checkpoint"
                    )

                # Individual weight sliders (show/hide based on selection)
                gr.Markdown("**Individual LoRA Weights** (0.0 = skip, 1.0 = full)")
                ckpt_weight1 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 1", visible=False)
                ckpt_weight2 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 2", visible=False)
                ckpt_weight3 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 3", visible=False)
                ckpt_weight4 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 4", visible=False)
                ckpt_weight5 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 5", visible=False)
                ckpt_weight6 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 6", visible=False)

                def update_ckpt_labels(selected):
                    if not selected:
                        return [gr.update(visible=False)] * 6
                    updates = []
                    for i in range(6):
                        if i < len(selected):
                            name = os.path.basename(selected[i]).replace(".safetensors", "")[:25]
                            updates.append(gr.update(visible=True, label=f"{name}"))
                        else:
                            updates.append(gr.update(visible=False))
                    return updates

                lora_checkbox_group.change(
                    fn=update_ckpt_labels,
                    inputs=lora_checkbox_group,
                    outputs=[ckpt_weight1, ckpt_weight2, ckpt_weight3, ckpt_weight4, ckpt_weight5, ckpt_weight6]
                )

                with gr.Row():
                    output_name = gr.Textbox(label="Output Name (optional)", placeholder="my_perfect_model")
                    save_model = gr.Checkbox(label="Save merged checkpoint", value=True)

                with gr.Row():
                    merge_to_checkpoint_button = gr.Button("MERGE LoRAs → CHECKPOINT", variant="primary")

                merge_to_checkpoint_status = gr.Textbox(label="Status", lines=10, interactive=False, show_copy_button=True)

                def merge_with_individual_weights(selected_loras, w1,w2,w3,w4,w5,w6, checkpoint_name, out_name, save):
                    weights = [w for w in [w1,w2,w3,w4,w5,w6] if w is not None][:len(selected_loras)]
                    if len(weights) < len(selected_loras):
                        weights = [1.0] * len(selected_loras)
                    return merge_loras_to_checkpoint_ui(
                        checkpoint_name=checkpoint_name,
                        lora_paths=selected_loras,
                        output_name=out_name,
                        strength=1.0,  # We use individual weights instead
                        save_model=save,
                        individual_weights=weights  # Pass them here
                    )

                merge_to_checkpoint_button.click(
                    fn=merge_with_individual_weights,
                    inputs=[
                        lora_checkbox_group,
                        ckpt_weight1, ckpt_weight2, ckpt_weight3, ckpt_weight4, ckpt_weight5, ckpt_weight6,
                        checkpoint_dropdown, output_name, save_model
                    ],
                    outputs=merge_to_checkpoint_status
                )

            with gr.Accordion("Merge Multiple LoRAs", open=False):
                gr.Markdown("### Merge LoRAs with Individual Weights\n(Select 2–6 LoRAs — weights auto-adjust)")

                with gr.Row():
                    lora_merge_checkbox = gr.CheckboxGroup(
                        label="Select LoRAs to merge (with dtype info)",
                        choices=[],
                        type="value",
                        interactive=True
                    )
                    lora_merge_refresh = create_refresh_button(
                        lora_merge_checkbox,
                        lambda: None,
                        lambda: {'choices': get_lora_list()},
                        "refresh_lora_merge_list"
                    )

                gr.Markdown("**LoRA Weights** (0.0 = skip, 1.0 = full, >1.0 = amplify)")

                lora_merge_weight1 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 1", visible=False)
                lora_merge_weight2 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 2", visible=False)
                lora_merge_weight3 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 3", visible=False)
                lora_merge_weight4 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 4", visible=False)
                lora_merge_weight5 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 5", visible=False)
                lora_merge_weight6 = gr.Slider(0.0, 3.0, 1.0, step=0.01, label="LoRA 6", visible=False)

                def update_weight_labels(selected):
                    if len(selected) < 2:
                        return [gr.update(visible=False)] * 6
                    updates = []
                    for i in range(6):
                        if i < len(selected):
                            name = os.path.basename(selected[i]).replace(".safetensors", "")[:25]
                            updates.append(gr.update(visible=True, label=f"{name}"))
                        else:
                            updates.append(gr.update(visible=False))
                    return updates

                lora_merge_checkbox.change(
                    fn=update_weight_labels,
                    inputs=lora_merge_checkbox,
                    outputs=[
                        lora_merge_weight1, lora_merge_weight2, lora_merge_weight3,
                        lora_merge_weight4, lora_merge_weight5, lora_merge_weight6
                    ]
                )

                with gr.Row():
                    lora_merge_output_name = gr.Textbox(
                        label="Output LoRA Name",
                        placeholder="my_perfect_merge",
                        info="Saved as .safetensors"
                    )
                    lora_merge_global_strength = gr.Slider(
                        0.0, 3.0, 1.0, step=0.05, label="Global Multiplier"
                    )

                with gr.Row():
                    lora_merge_button = gr.Button("MERGE WITH WEIGHTS → NEW LORA", variant="primary")

                lora_merge_status = gr.Textbox(
                    label="Status", lines=12, interactive=False, show_copy_button=True
                )

                def merge_with_weights(selected, w1, w2, w3, w4, w5, w6, name, g):
                    weights = [w for w in [w1,w2,w3,w4,w5,w6] if w is not None][:len(selected)]
                    if len(weights) < len(selected):
                        weights = [1.0] * len(selected)
                    return merge_selected_loras_ui(selected, name, g, weights)

                lora_merge_button.click(
                    fn=merge_with_weights,
                    inputs=[
                        lora_merge_checkbox,
                        lora_merge_weight1, lora_merge_weight2, lora_merge_weight3,
                        lora_merge_weight4, lora_merge_weight5, lora_merge_weight6,
                        lora_merge_output_name,
                        lora_merge_global_strength
                    ],
                    outputs=lora_merge_status
                ).then(
                    fn=lambda: (gr.update(choices=get_lora_list()), gr.update(choices=get_lora_list())),
                    outputs=[lora_checkbox_group, lora_merge_checkbox]
                )

        # ✅ ENHANCEMENT 3: Merge Presets & History Tab
        with gr.Tab("Presets & History"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📋 Saved Merge Presets")
                    
                    with gr.Row():
                        preset_selector = gr.Dropdown(
                            choices=get_merge_presets(),
                            label="Select Preset",
                            scale=3
                        )
                        preset_refresh_btn = gr.Button("🔄", scale=1)
                    
                    with gr.Row():
                        preset_load_btn = gr.Button("Load Preset", variant="primary", scale=2)
                        preset_delete_btn = gr.Button("Delete", variant="stop", scale=1)
                    
                    preset_message = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Select a preset to load or delete"
                    )
                    
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="Save Current As",
                            placeholder="Enter preset name...",
                            scale=3
                        )
                        preset_save_btn = gr.Button("Save", variant="secondary", scale=1)
                
                with gr.Column():
                    gr.Markdown("## 📊 Merge History")
                    
                    history_display = gr.Textbox(
                        label="Recent Merges",
                        interactive=False,
                        lines=12,
                        max_lines=15,
                        value="\n".join(get_merge_history()) or "No merge history yet"
                    )
                    
                    history_refresh_btn = gr.Button("Refresh History")
            
            # Wire preset buttons
            preset_refresh_btn.click(
                fn=lambda: gr.update(choices=get_merge_presets()),
                outputs=preset_selector,
                show_progress=False
            )
            
            preset_save_btn.click(
                fn=save_merge_preset,
                inputs=[
                    preset_name_input, model_a, model_b, model_c, model_d,
                    merge_mode_selector, calc_mode_selector,
                    alpha, beta, gamma, delta,
                    weight_editor, discard, clude, clude_mode
                ],
                outputs=preset_message
            )
            
            preset_load_btn.click(
                fn=load_merge_preset,
                inputs=preset_selector,
                outputs=[
                    model_a, model_b, model_c, model_d,
                    merge_mode_selector, calc_mode_selector,
                    alpha, beta, gamma, delta,
                    weight_editor, discard, clude, clude_mode,
                    preset_message
                ]
            )
            
            preset_delete_btn.click(
                fn=delete_merge_preset,
                inputs=preset_selector,
                outputs=[preset_message, preset_selector]
            )
            
            history_refresh_btn.click(
                fn=lambda: gr.update(value="\n".join(get_merge_history()) or "No merge history yet"),
                outputs=history_display,
                show_progress=False
            )
            
    return [(cmn.blocks, "Untitled merger", "untitled_merger")]

# register the tab
script_callbacks.on_ui_tabs(on_ui_tabs)

# ---------------------------
# Helper functions
# ---------------------------
def start_merge(*args):
    progress = Progress()
    try:
        # ✅ ENHANCEMENT 2: Initialize real-time ETA tracking
        progress.start_merge(1000)  # Will be updated during merge
        
        merger.prepare_merge(progress, *args)
        
        # ✅ Save to history on success
        save_to_history({
            'models': str(args[3:7]),
            'modes': f"{args[2]}+{args[1]}"
        }, "✓ Success")
        
    except Exception as error:
        merger.clear_cache()
        if not shared.sd_model:
            sd_models.reload_model_weights(forced_reload=True)
        
        # Save failed merge to history
        save_to_history({'status': 'Failed'}, f"✗ {str(error)[:30]}")
        
        if not isinstance(error,merger.MergeInterruptedError):
            raise
    
    return progress.get_report()

def test_regex(input):
    regex = misc_util.target_to_regex(input)
    selected_keys = re.findall(regex,'\n'.join(model_a_keys),re.M)
    joined = '\n'.join(selected_keys)
    return  f'Matched keys: {len(selected_keys)}\n{joined}'

def update_model_a_keys(model_a):
    global model_a_keys
    try:
        path = sd_models.get_closet_checkpoint_match(model_a).filename
        with safetensors.torch.safe_open(path,framework='pt',device='cpu') as file:
            model_a_keys = file.keys()
    except Exception:
        model_a_keys = []

def checkpoint_changed(name):
    if name == "":
        return plaintext_to_html('None | None',classname='untitled_sd_version')
    sdversion, dtype = misc_util.id_checkpoint(name)
    return plaintext_to_html(f"{sdversion} | {str(dtype).split('.')[1]}",classname='untitled_sd_version')

def merge_loras_to_checkpoint_ui(checkpoint_name, lora_paths, output_name, strength, save_model, individual_weights=None, verbose=False):
    """UI wrapper for merging multiple LoRAs to a checkpoint – now 100% safe & fast"""
    progress = Progress()
    checkpoint_info = None

    try:
        if not checkpoint_name:
            return "Error: Please select a base checkpoint"
        if not lora_paths:
            return "Error: Please select at least one LoRA"
        if save_model and not output_name.strip():
            return "Error: Please provide an output name when saving"

        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
        if not checkpoint_info:
            return f"Error: Could not find checkpoint '{checkpoint_name}'"

        base_path = checkpoint_info.filename
        output_dir = os.path.dirname(base_path)

        progress(f"Loading base checkpoint: {os.path.basename(base_path)}")
        with safetensors.torch.safe_open(base_path, framework='pt', device='cpu') as f:
            merged_checkpoint_dict = {k: f.get_tensor(k) for k in f.keys()}

        all_summaries = []
        applied_count = 0

        for i, lora_path in enumerate(lora_paths):
            current_strength = strength
            if individual_weights and i < len(individual_weights):
                current_strength = individual_weights[i]

            if current_strength <= 0:
                progress(f"Skipping {os.path.basename(lora_path)} (strength = 0)")
                continue

            progress(f"Applying LoRA {applied_count + 1}: {os.path.basename(lora_path)} × {current_strength:.3f}")
            merged_checkpoint_dict, summary = lora_merge._apply_single_lora_to_dict(
                merged_checkpoint_dict,
                lora_path,
                strength=current_strength,
                progress=progress,
                verbose=verbose
            )
            all_summaries.append(summary)
            applied_count += 1

        final_report = "\n".join(all_summaries) if all_summaries else "No LoRAs applied"

        # ——————————————————————————————————————————————
        # SAFE & FAST MODEL LOADING (critical fix!)
        # ——————————————————————————————————————————————
        progress("Unloading current model...")
        if shared.sd_model:
            sd_models.unload_model_weights(shared.sd_model)
            shared.sd_model = None

        # Only reload base if it's not already loaded
        if (shared.sd_model is None or 
            not hasattr(shared.sd_model, 'sd_checkpoint_info') or 
            shared.sd_model.sd_checkpoint_info.filename != checkpoint_info.filename):
            progress("Loading base model into WebUI...")
            sd_models.load_model(checkpoint_info)
        else:
            progress("Base model already loaded – reusing")

        # Apply merged weights
        progress("Applying merged LoRA weights...")
        shared.sd_model.load_state_dict(merged_checkpoint_dict, strict=False)

        # Essential: move to GPU + re-apply optimizations
        shared.sd_model.to(devices.device)
        shared.sd_model.eval()

        # Re-apply attention, unet, hijacks – fixes black images & xformers
        sd_hijack.model_hijack.hijack(shared.sd_model)
        script_callbacks.model_loaded_callback(shared.sd_model)
        sd_unet.apply_unet("None")

        # ——————————————————————————————————————————————
        # Optional: Save to disk
        # ——————————————————————————————————————————————
        if save_model:
            safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', output_name.strip())
            if not safe_name.lower().endswith('.safetensors'):
                safe_name += '.safetensors'
            output_path = os.path.join(output_dir, safe_name)

            progress(f"Saving merged model → {safe_name}")
            safetensors.torch.save_file(merged_checkpoint_dict, output_path)

            # Refresh checkpoint list so new file appears
            sd_models.list_models()

            final_report += f"\nModel saved as: {safe_name}"
        else:
            final_report += f"\nModel loaded in memory (not saved)"

        progress(f"Successfully applied {applied_count} LoRA(s)!", popup=True)
        return progress.get_report() + "\n\n" + final_report

    except Exception as e:
        error_msg = f"LoRA merge failed: {str(e)}"
        progress(error_msg, popup=True)

        # Try to recover by reloading original checkpoint
        try:
            if checkpoint_info:
                sd_models.load_model(checkpoint_info)
                progress("Recovered: original checkpoint reloaded")
        except:
            pass

        return progress.get_report() + "\n" + error_msg
    
def merge_selected_loras_ui(selected_lora_paths, output_name, global_strength=1.0, individual_weights=None):
    progress = Progress()
    try:
        if not selected_lora_paths or len(selected_lora_paths) < 2:
            return "Error: Select at least 2 LoRAs"

        if individual_weights is None or len(individual_weights) != len(selected_lora_paths):
            individual_weights = [1.0] * len(selected_lora_paths)

        # Apply global multiplier
        weights = [w * global_strength for w in individual_weights]
        total = sum(weights)
        if total == 0:
            weights = [1.0 / len(weights)] * len(weights)  # prevent division by zero
        else:
            weights = [w / total for w in weights]

        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', output_name.strip())
        if not safe_name:
            safe_name = "merged_lora"
        if not safe_name.lower().endswith('.safetensors'):
            safe_name += '.safetensors'

        output_dir = os.path.dirname(selected_lora_paths[0])
        output_path = os.path.join(output_dir, safe_name)

        progress(f"Merging {len(selected_lora_paths)} LoRAs → {safe_name}")

        # Use resilient merge (handles SD1.5 + SDXL + Pony)
        result = lora_merge.merge_loras_resilient(
            lora_paths=selected_lora_paths,
            output_path=output_path,
            weights=weights,
            progress=progress
        )

        progress("GOD MERGE COMPLETE", popup=True)
        return progress.get_report() + f"\n\nSaved: {output_path}\n\n{result}"

    except Exception as e:
        error = f"Merge failed: {str(e)}"
        progress(error, popup=True)
        return progress.get_report() + "\n" + error
    
# ✅ ENHANCEMENT 1: Merge Presets Management
def get_merge_presets():
    """Load all merge presets"""
    try:
        with open(merge_presets_filename, 'r') as f:
            return list(json.load(f).keys())
    except:
        return []

def save_merge_preset(preset_name, model_a_val, model_b_val, model_c_val, model_d_val, 
                      merge_mode_val, calc_mode_val, alpha_val, beta_val, gamma_val, delta_val,
                      weight_editor_val, discard_val, clude_val, clude_mode_val):
    """Save current merge configuration as preset"""
    try:
        with open(merge_presets_filename, 'r') as f:
            presets = json.load(f)
    except:
        presets = {}
    
    if not preset_name or preset_name.strip() == '':
        return "❌ Preset name required"
    
    presets[preset_name] = {
        'model_a': model_a_val,
        'model_b': model_b_val,
        'model_c': model_c_val,
        'model_d': model_d_val,
        'merge_mode': merge_mode_val,
        'calc_mode': calc_mode_val,
        'sliders': [alpha_val, beta_val, gamma_val, delta_val],
        'weight_editor': weight_editor_val,
        'discard': discard_val,
        'clude': clude_val,
        'clude_mode': clude_mode_val
    }
    
    with open(merge_presets_filename, 'w') as f:
        json.dump(presets, f, indent=2)
    
    return f"✓ Preset '{preset_name}' saved"

def load_merge_preset(preset_name):
    """Load merge configuration from preset"""
    try:
        with open(merge_presets_filename, 'r') as f:
            presets = json.load(f)
        
        if preset_name not in presets:
            return [gr.update()] * 14 + ["❌ Preset not found"]
        
        preset = presets[preset_name]
        updates = [
            gr.update(value=preset.get('model_a', '')),
            gr.update(value=preset.get('model_b', '')),
            gr.update(value=preset.get('model_c', '')),
            gr.update(value=preset.get('model_d', '')),
            gr.update(value=preset.get('merge_mode')),
            gr.update(value=preset.get('calc_mode')),
            gr.update(value=preset['sliders'][0]),
            gr.update(value=preset['sliders'][1]),
            gr.update(value=preset['sliders'][2]),
            gr.update(value=preset['sliders'][3]),
            gr.update(value=preset.get('weight_editor', '')),
            gr.update(value=preset.get('discard', '')),
            gr.update(value=preset.get('clude', '')),
            gr.update(value=preset.get('clude_mode', 'Exclude')),
            f"✓ Loaded preset: {preset_name}"
        ]
        return updates
    except Exception as e:
        return [gr.update()] * 14 + [f"❌ Error loading preset: {str(e)}"]

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
