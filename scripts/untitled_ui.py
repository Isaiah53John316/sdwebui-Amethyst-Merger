# Contains slider-help, soft-disable, corrected mode logic, LoRA tab fixed inside Blocks context - Fixed save with model_ema - Works now with Dev branch of Automatic1111 - needed for 50xx.
import gradio as gr
import os
import re
import functools
import json
import shutil
import torch
import safetensors
import safetensors.torch
from modules import sd_models,script_callbacks,scripts,shared,ui_components,paths,sd_samplers,ui,call_queue
from modules.ui_common import create_output_panel,plaintext_to_html, create_refresh_button
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

cmn.opts = Options(options_filename)

# ---------------------------
# Helper functions for UI logic
# ---------------------------
# default marker used in calcmodes definitions for "no custom config"
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
    Strategy:
      - Start with mergemode.input_sliders / input_models
      - If calcmode defines input_sliders/input_models, consider them
      - Also count how many non-default slid_*_config entries the calc mode defines
        (this catches calc modes that define extra slider configs but don't set input_sliders)
    Returns (req_sliders, req_models) clamped to [1,4]
    """
    base_sliders = getattr(mergemode, 'input_sliders', 4) or 4
    base_models  = getattr(mergemode, 'input_models', 4) or 4

    calc_sliders_attr = getattr(calcmode, 'input_sliders', None)
    calc_models_attr  = getattr(calcmode, 'input_models', None)

    # Count non-default slider configs in calc mode
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

    # fallback clamps
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
    # compat can contain strings or other types; normalize
    compat_list = [c for c in compat if isinstance(c, str)]
    return (mergemode.name in compat_list) or (mergemode.name in compat)

# ---------------------------
# Main: mode_changed handler
# ---------------------------
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
# Utility UI helpers preserved from original file
# ---------------------------
def get_checkpoints_list(sort):
    checkpoints_list = [x.title for x in sd_models.checkpoints_list.values() if x.is_safetensors]
    if sort == 'Newest first':
        sort_func = lambda x: os.path.getctime(sd_models.get_closet_checkpoint_match(x).filename)
        checkpoints_list.sort(key=sort_func,reverse=True)
    return checkpoints_list

def get_lora_list():
    lora_choices = []
    possible_dirs = [
        os.path.join(paths.models_path, 'Lora'),
        os.path.join(paths.models_path, 'lora'),
    ]
    for lora_dir in possible_dirs:
        if os.path.exists(lora_dir):
            for root, dirs, files in os.walk(lora_dir):
                for file in files:
                    if file.endswith('.safetensors'):
                        full_path = os.path.join(root, file)
                        try:
                            with safetensors.torch.safe_open(full_path, framework='pt', device='cpu') as f:
                                keys = list(f.keys())
                                if len(keys) > 0:
                                    tensor = f.get_tensor(keys[0])
                                    dtype_str = str(tensor.dtype).split('.')[1]
                                    display_name = f"{file} [{dtype_str}]"
                                    lora_choices.append((display_name, full_path))
                        except:
                            lora_choices.append((file, full_path))
    lora_choices.sort(key=lambda x: x[0].lower())
    return [p for (_, p) in lora_choices]

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

# ---------------------------
# UI: build tabs
# ---------------------------
def on_ui_tabs():
    with gr.Blocks() as cmn.blocks:
        with gr.Tab("Merge"):
            dummy_component = gr.Textbox(visible=False,interactive=True)
            with ui_components.ResizeHandleRow():
                with gr.Column():
                    status = gr.Textbox(max_lines=4,lines=4,show_label=False,info="",interactive=False,render=True)

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
                        merge_mode_selector = gr.Radio(label='Merge Mode (formula structure):',choices=list(merger.mergemode_selection.keys()),value=list(merger.mergemode_selection.keys())[0],scale=3)
                    merge_mode_desc = gr.Textbox(label="Merge Mode Description", value=merger.mergemode_selection[list(merger.mergemode_selection.keys())[0]].description, interactive=False, lines=2)

                    with gr.Row():
                        calc_mode_selector = gr.Radio(label='Calculation Mode (how to execute):',choices=list(merger.calcmode_selection.keys()),value=list(merger.calcmode_selection.keys())[0],scale=3)
                    calc_mode_desc = gr.Textbox(label="Calculation Mode Description", value=merger.calcmode_selection[list(merger.calcmode_selection.keys())[0]].description, interactive=False, lines=2)

                    # Slider help textbox (visible panel)
                    slider_help = gr.Textbox(label="Slider Meaning", value="", interactive=False, lines=6, placeholder="Slider help will appear here when you change merge/calc modes.")

                    # MAIN SLIDERS
                    with gr.Row(equal_height=True):
                        alpha = gr.Slider(minimum=-1, step=0.01, maximum=2, label="slider_a [α] (alpha)", info='model_a - model_b', value=0.5, elem_classes=['main_sliders'])
                        beta  = gr.Slider(minimum=-1, step=0.01, maximum=2, label="slider_b [β] (beta)",  info='-', value=0.5, elem_classes=['main_sliders'])
                        gamma = gr.Slider(minimum=-1, step=0.01, maximum=2, label="slider_c [γ] (gamma)", info='-', value=0.25, elem_classes=['main_sliders'])
                        delta = gr.Slider(minimum=-1, step=0.01, maximum=2, label="slider_d [δ] (delta)", info='-', value=0.25, elem_classes=['main_sliders'])

                    # CUSTOM SLIDERS UI preserved
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
                                                custom_sliders.append(gr.Slider(show_label=False,value=slid_defaults.__next__(),scale=6,minimum=0,maximum=1,step=0.01))

                        def show_sliders(n):
                            n = int(n/2)
                            update_column = [gr.update(visible=True), gr.update(visible=True)]*n + [gr.update(visible=False), gr.update(visible=False)]*(13-n)
                            return update_column * 2

                        slider_slider.change(fn=show_sliders,inputs=slider_slider,outputs=custom_sliders,show_progress='hidden')
                        slider_slider.release(fn=show_sliders,inputs=slider_slider,outputs=custom_sliders,show_progress='hidden')

                        sliders_preset_save.click(fn=save_custom_sliders,inputs=[sliders_preset_dropdown,slider_slider,*custom_sliders])
                        sliders_preset_load.click(fn=load_slider_preset,inputs=[sliders_preset_dropdown],outputs=[slider_slider,*custom_sliders])

                    # Supermerger Adjust (Finetune) - restored block
                    with gr.Accordion("Supermerger Adjust", open=False) as acc_ad:
                        with gr.Row(variant="compact"):
                            finetune = gr.Textbox(label="Adjust", show_label=False, info="Adjust IN,OUT,OUT2,Contrast,Brightness,COL1,COL2,COL3", visible=True, value="", lines=1)
                            finetune_write = gr.Button(value="↑", elem_classes=["tool"])
                            finetune_read = gr.Button(value="↓", elem_classes=["tool"])
                            finetune_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                detail1 = gr.Slider(label="IN", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail2 = gr.Slider(label="OUT", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail3 = gr.Slider(label="OUT2", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                contrast = gr.Slider(label="Contrast", minimum=-10, maximum=10, step=0.01, value=0, info="Contrast/Detail")
                            with gr.Column(scale=1, min_width=100):
                                bri = gr.Slider(label="Brightness", minimum=-10, maximum=10, step=0.01, value=0, info="Dark(Minius)-Bright(Plus)")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                col1 = gr.Slider(label="Cyan-Red", minimum=-10, maximum=10, step=0.01, value=0, info="Cyan(Minius)-Red(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col2 = gr.Slider(label="Magenta-Green", minimum=-10, maximum=10, step=0.01, value=0, info="Magenta(Minius)-Green(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col3 = gr.Slider(label="Yellow-Blue", minimum=-10, maximum=10, step=0.01, value=0, info="Yellow(Minius)-Blue(Plus)")

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

                    # OPTIONS accordion (restore thread/cache & other options)
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
                                    info="Targets will be removed from the model, only applies to autosaved models. Separate with whitespace.",
                                    value='model_ema',
                                    lines=5,
                                    scale=1
                                )

                    # Weight editor / example YAML
                    with gr.Accordion('Weight editor'):
                        weight_editor = gr.Code(value=EXAMPLE,lines=20,language='yaml',label='')

                    # Model keys test
                    with gr.Accordion('Model keys'):
                        target_tester = gr.Textbox(max_lines=1,label="Checks model_a keys using simple expression.",info="'*' is used as wildcard. Start expression with 'cond*' for clip. 'c*embedders.0*' for small clip. 'c*embedders.1*' for big clip. 'model.*' for unet and 'model_ema*' for ema unet",interactive=True,placeholder='model.*out*4*tran*norm*weight')
                        target_tester_display = gr.Textbox(max_lines=40,lines=40,label="Targeted keys:",info="",interactive=False)
                        target_tester.change(fn=test_regex,inputs=[target_tester],outputs=target_tester_display,show_progress='minimal')

                    # Wire the mode_changed events to include merge_button now that it's defined
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

        # --- LO RA TAB (must stay inside the same Blocks context!) ---
        with gr.Tab("LoRA", elem_id="tab_lora"):
            gr.Markdown("## ⚠️ UNDER CONSTRUCTION\nLoRA merging is functional but experimental. Needs real-world testing and refinement. Use at your own risk!")
            lora_status = gr.Textbox(max_lines=4,lines=4,show_label=False,info="",interactive=False,render=True)

            with gr.Accordion("Merge LoRA(s) to Checkpoint", open=True):
                gr.Markdown("Bake one or multiple LoRAs into a checkpoint permanently")

                with gr.Row():
                    lora_to_ckpt_checkpoint = gr.Dropdown(get_checkpoints_list('Alphabetical'), label="Base Checkpoint", scale=4)
                    lora_refresh_ckpt = create_refresh_button(lora_to_ckpt_checkpoint, lambda: None, lambda: {'choices': get_checkpoints_list('Alphabetical')}, 'refresh_lora_ckpt')

                lora_ckpt_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))

                with gr.Row():
                    lora_checkbox_group = gr.CheckboxGroup(label="Select LoRAs to merge (with dtype info)", choices=[], type="value", interactive=True)
                    lora_refresh_list = create_refresh_button(lora_checkbox_group, lambda: None, lambda: {'choices': get_lora_list()}, 'refresh_lora_list')

                with gr.Row():
                    lora_to_ckpt_strength = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1.0, label="Global LoRA Strength", info="Apply this strength to all checked LoRAs (1.0 = normal)")
                    lora_to_ckpt_name = gr.Textbox(label="Output Name", placeholder="model_with_loras", info="Name for merged checkpoint")

                lora_to_ckpt_button = gr.Button("Merge LoRA(s) to Checkpoint", variant="primary")

            with gr.Accordion("Merge Multiple LoRAs", open=False):
                gr.Markdown("Combine 2-3 LoRA files into a single LoRA")
                with gr.Column():
                    with gr.Row():
                        lora_merge_lora1 = gr.Textbox(label="LoRA 1 Path", placeholder="/path/to/lora1.safetensors", scale=3)
                        lora_merge_weight1 = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.5, label="Weight", scale=1)

                    with gr.Row():
                        lora_merge_lora2 = gr.Textbox(label="LoRA 2 Path", placeholder="/path/to/lora2.safetensors", scale=3)
                        lora_merge_weight2 = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.5, label="Weight", scale=1)

                    with gr.Row():
                        lora_merge_lora3 = gr.Textbox(label="LoRA 3 Path (optional)", placeholder="/path/to/lora3.safetensors", scale=3)
                        lora_merge_weight3 = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.0, label="Weight", scale=1)

                gr.Markdown("*Weights will be normalized to sum to 1.0*")
                with gr.Row():
                    lora_merge_name = gr.Textbox(label="Output Name", placeholder="merged_lora", info="Name for merged LoRA file")
                    lora_merge_button = gr.Button("Merge LoRAs", variant="primary")

            # removed duplicate lora_status.render() call here

            # Wire up dtype detection
            lora_to_ckpt_checkpoint.change(fn=checkpoint_changed, inputs=lora_to_ckpt_checkpoint, outputs=lora_ckpt_info)

            # Wire up the LoRA merge buttons
            lora_to_ckpt_button.click(
                fn=merge_loras_to_checkpoint_ui if 'merge_loras_to_checkpoint_ui' in globals() else (lambda *args: "Not implemented"),
                inputs=[lora_to_ckpt_checkpoint, lora_checkbox_group, lora_to_ckpt_name, lora_to_ckpt_strength],
                outputs=lora_status
            )

            lora_merge_button.click(
                fn=merge_loras_ui if 'merge_loras_ui' in globals() else (lambda *args: "Not implemented"),
                inputs=[
                    lora_merge_lora1, lora_merge_weight1,
                    lora_merge_lora2, lora_merge_weight2,
                    lora_merge_lora3, lora_merge_weight3,
                    lora_merge_name
                ],
                outputs=lora_status
            )

    # initialize the slider_help and controls by calling mode_changed once (best effort)
    try:
        # compute initial updates (this does not apply them to UI automatically but ensures no error during load)
        _ = mode_changed(list(merger.mergemode_selection.keys())[0], list(merger.calcmode_selection.keys())[0])
    except Exception:
        pass

    return [(cmn.blocks, "Untitled merger", "untitled_merger")]

# register the tab
script_callbacks.on_ui_tabs(on_ui_tabs)

# ---------------------------
# Remaining helper functions (unchanged from original)
# ---------------------------
def start_merge(*args):
    progress = Progress()
    try:
        merger.prepare_merge(progress, *args)
    except Exception as error:
        merger.clear_cache()
        if not shared.sd_model:
            sd_models.reload_model_weights(forced_reload=True)
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

def lora_file_changed(lora_path):
    try:
        if not lora_path or not os.path.exists(lora_path):
            return plaintext_to_html('Invalid path',classname='untitled_sd_version')

        with safetensors.torch.safe_open(lora_path, framework='pt', device='cpu') as f:
            keys = list(f.keys())
            if len(keys) > 0:
                tensor = f.get_tensor(keys[0])
                dtype_str = str(tensor.dtype).split('.')[1]
                num_keys = len(keys)
                return plaintext_to_html(f'LoRA | {dtype_str} | {num_keys} keys',classname='untitled_sd_version')
            else:
                return plaintext_to_html('LoRA | empty file',classname='untitled_sd_version')
    except Exception as e:
        return plaintext_to_html(f'Error: {str(e)}',classname='untitled_sd_version')

# End of file
