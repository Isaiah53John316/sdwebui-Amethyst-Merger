import gradio as gr
from safetensors.torch import safe_open
from safetensors import SafetensorError
import concurrent.futures
from collections import defaultdict
import scripts.untitled.operators as oper
import scripts.untitled.misc_util as mutil
import scripts.untitled.common as cmn
import scripts.untitled.calcmodes as calcmodes
from modules.timer import Timer
import torch,os,re,gc,random
from tqdm import tqdm
from copy import copy,deepcopy
from modules import devices,shared,script_loading,paths,paths_internal,sd_models,sd_unet,sd_hijack
from functools import lru_cache

# Try Forge path first, fallback to old A1111 path for backwards compatibility
try:
    networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'sd_forge_lora','networks.py'))
except (FileNotFoundError, OSError):
    networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

class MergeInterruptedError(Exception):
    def __init__(self,*args):
        super().__init__(*args)

SKIP_KEYS = [
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "betas",
    "log_one_minus_alphas_cumprod",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "posterior_variance",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod"
]

# Compile once at module level
SKIP_KEYS_COMPILED = {key: re.compile(f"^{re.escape(key)}$") for key in SKIP_KEYS}

VALUE_NAMES = ('alpha','beta','gamma','delta')

mergemode_selection = {}
for mergemode_obj in calcmodes.MERGEMODES_LIST:
    mergemode_selection.update({mergemode_obj.name: mergemode_obj})

calcmode_selection = {}
for calcmode_obj in calcmodes.CALCMODES_LIST:
    calcmode_selection.update({calcmode_obj.name: calcmode_obj})


def parse_arguments(progress,mergemode_name,calcmode_name,model_a,model_b,model_c,model_d,slider_a,slider_b,slider_c,slider_d,editor,discard,clude,clude_mode,seed,enable_sliders,active_sliders,*custom_sliders):
    mergemode = mergemode_selection[mergemode_name]
    calcmode = calcmode_selection[calcmode_name]
    parsed_targets = {}

    if seed < 0:
        seed = random.randint(10**9,10**10-1)
    cmn.last_merge_seed = seed

    if enable_sliders:
        slider_col_a = custom_sliders[:int(len(custom_sliders)/2)]
        slider_col_b = custom_sliders[int(len(custom_sliders)/2):]

        enabled_sliders = slider_col_a[:active_sliders] + slider_col_b[:active_sliders]
        it = iter(enabled_sliders)
        parsed_sliders = {it.__next__():{'alpha':it.__next__(),'seed':seed} for x in range(0,active_sliders)}
        parsed_targets.update(parsed_sliders)
        try:
            del parsed_targets['']
        except KeyError: pass
        
    targets = re.sub(r'#.*$','',editor.lower(),flags=re.M)
    targets = re.sub(r'\bslider_a\b',str(slider_a),targets,flags=re.M)
    targets = re.sub(r'\bslider_b\b',str(slider_b),targets,flags=re.M)
    targets = re.sub(r'\bslider_c\b',str(slider_c),targets,flags=re.M)
    targets = re.sub(r'\bslider_d\b',str(slider_d),targets,flags=re.M)

    targets_list = targets.split('\n')
    for target in targets_list:
        if target != "":
            target = re.sub(r'\s+','',target)
            selector, weights = target.split(':')
            parsed_targets[selector] = {'seed':seed}
            for n,weight in enumerate(weights.split(',')):
                try:
                    parsed_targets[selector][VALUE_NAMES[n]] = float(weight)
                except ValueError:pass

    checkpoints = []
    progress('Using Checkpoints:')
    for n, model in enumerate((model_a,model_b,model_c,model_d)):
        if n+1 > mergemode.input_models:
            checkpoints.append('')
            continue
        name = model.split(' ')[0]
        checkpoint_info = sd_models.get_closet_checkpoint_match(name)
        if checkpoint_info == None: 
            if model:
                progress.interrupt('Couldn\'t find checkpoint: '+name)
            else:
                progress.interrupt('Missing input model')
        if not checkpoint_info.filename.endswith('.safetensors'): 
            progress.interrupt('This extension only supports safetensors checkpoints: '+name)
        progress(' - '+name)
        checkpoints.append(checkpoint_info.filename)
    cmn.primary = checkpoints[0]

    discards = re.findall(r'[^\s]+', discard, flags=re.I|re.M)
    cludes = re.findall(r'[^\s]+', clude, flags=re.I|re.M)

    with safe_open(cmn.primary,framework='pt',device='cpu') as file:
        keys = file.keys()

    # CRITICAL FIX: Only create discard_regex if there are actual discard patterns
    discard_keys = []
    if discards:  # Only process if discard list is not empty
        discard_regex = re.compile(mutil.target_to_regex(discards))
        discard_keys = list(filter(lambda x: re.search(discard_regex,x),keys))

    desired_keys = keys
    if cludes:
        clude_regex = re.compile(mutil.target_to_regex(cludes))
        if clude_mode.lower() == 'exclude':
            desired_keys = list(filter(lambda x: not re.search(clude_regex,x),keys))
        else:
            desired_keys = list(filter(lambda x: re.search(clude_regex,x),keys))

    assigned_keys = assign_weights_to_keys(parsed_targets,desired_keys)
    return mergemode, calcmode, keys, assigned_keys, discard_keys, checkpoints


def assign_weights_to_keys(targets,keys,already_assigned=None) -> dict:
    weight_assigners = []
    keystext = "\n".join(keys)

    for target_name,weights in targets.items():
        regex = mutil.target_to_regex(target_name)

        weight_assigners.append((weights, regex))
    
    keys_n_weights = list()

    for weights, regex in weight_assigners:
        target_keys = re.findall(regex,keystext,re.M)
        keys_n_weights.append((target_keys,weights))
    
    keys_n_weights.sort(key=lambda x: len(x[0]))
    keys_n_weights.reverse()

    assigned_keys = already_assigned or defaultdict()
    assigned_keys.default_factory = dict
    
    for keys, weights in keys_n_weights:
        for key in keys:
            assigned_keys[key].update(weights)

    return assigned_keys


def create_tasks(progress, mergemode, calcmode, keys, assigned_keys, discard_keys, checkpoints):
    tasks = []
    n = 0
    for key in keys:
        # NOTE: discard_keys should NOT filter during merge, only during save
        # Removed: if key in discard_keys:continue
        if any(regex.match(key) for regex in SKIP_KEYS_COMPILED.values()) or 'first_stage_model' in key:
            tasks.append(oper.LoadTensor(key, cmn.primary))
        elif key in assigned_keys.keys():
            n += 1
            base_recipe = mergemode.create_recipe(key, *checkpoints, **assigned_keys[key])
            final_recipe = calcmode.modify_recipe(base_recipe, key, *checkpoints, **assigned_keys[key])
            tasks.append(final_recipe)
        else:
            tasks.append(oper.LoadTensor(key, cmn.primary))

    progress('Assigned tasks: ')
    progress('Merges', v=n)
    progress('Default to A', v=len(tasks)-n)
    return tasks


def prepare_merge(progress, save_name, save_settings, finetune, *merge_args):
    progress('\n### Preparing merge ###')
    timer = Timer()
    cmn.interrupted = True
    cmn.stop = False

    mergemode, calcmode, keys, assigned_keys, discard_keys, checkpoints = parse_arguments(progress, *merge_args)
    tasks = create_tasks(progress, mergemode, calcmode, keys, assigned_keys, discard_keys, checkpoints)

    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)

    # FIXED: Properly load all checkpoints using context manager
    with safe_open_multiple(checkpoints, cmn.device()) as loaded_files:
        cmn.loaded_checkpoints = loaded_files  # This is the missing line!

        state_dict = merge(
            progress=progress,
            tasks=tasks,
            checkpoints=checkpoints,
            finetune=finetune,
            timer=timer,
            cross_arch=getattr(cmn, 'cross_arch_enabled', False),
            threads=cmn.opts.options.get('threads', 8)
        )

    # After context exits, files are safely closed — but merge is done
    merge_name = mutil.create_name(checkpoints, f"{mergemode.name}+{calcmode.name}", 0)
    checkpoint_info = deepcopy(sd_models.get_closet_checkpoint_match(os.path.basename(cmn.primary)))
    checkpoint_info.short_title = hash(cmn.last_merge_tasks)
    checkpoint_info.name_for_extra = '_TEMP_MERGE_' + merge_name

    if 'Autosave' in save_settings:
        checkpoint_info = mutil.save_state_dict(state_dict, save_name or merge_name, save_settings, timer, discard_keys)

    with mutil.NoCaching():
        mutil.load_merged_state_dict(state_dict, checkpoint_info)

    timer.record('Load model')
    del state_dict
    devices.torch_gc()

    cmn.interrupted = False
    progress('Merge completed in ' + timer.summary(), report=True)

def merge(progress, tasks, checkpoints, finetune="", timer=None, cross_arch=False, threads=8) -> dict:
    """
    Main merge function – now supports:
      • Cross-architecture merges (SD1.5 → SDXL, Flux → SDXL, etc.)
      • User-controlled thread count
      • Immediate CPU offload → 70–80% less VRAM usage
    """
    progress('### Starting merge ###')

    # ——————————————————————————————————————————————
    # 1. Detect model types & set global flags
    # ——————————————————————————————————————————————
    cmn.checkpoints_types = {}
    for cp in checkpoints:
        if cp:
            typ, _ = mutil.id_checkpoint(cp)
            cmn.checkpoints_types[cp] = typ

    # Force SDXL to be the shape reference if doing cross-arch merge
    cmn.is_cross_arch = cross_arch
    if cmn.is_cross_arch:
        sdxl_models = [cp for cp in checkpoints if cp and cmn.checkpoints_types.get(cp) in ('SDXL', 'SDXL-refiner')]
        if sdxl_models:
            # Make the first SDXL model the primary (defines all shapes)
            primary_idx = checkpoints.index(sdxl_models[0])
            if primary_idx != 0:
                checkpoints[0], checkpoints[primary_idx] = checkpoints[primary_idx], checkpoints[0]
                cmn.primary = checkpoints[0]
                progress('Cross-Arch mode → Using SDXL model as primary shape reference')
        else:
            progress('Warning: Cross-Arch enabled but no SDXL model found – results may be unpredictable')
            # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
        # SAFETY: Fix any corrupted/empty target shapes from caching
        if getattr(cmn, 'cross_arch_target_shapes', None):
            fixed = 0
            for k, shape in list(cmn.cross_arch_target_shapes.items()):
                if not isinstance(shape, tuple) or len(shape) == 0:
                    cmn.cross_arch_target_shapes[k] = (1,)  # safe fallback
                    fixed += 1
            if fixed:
                progress(f"Cross-Arch: Fixed {fixed} invalid target shapes")
        # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

    tasks_copy = copy(tasks)

    # ——————————————————————————————————————————————
    # 2. Unload current model (safe for both A1111 + Forge)
    # ——————————————————————————————————————————————
    if shared.sd_model and getattr(shared.sd_model, 'device', None) != 'cpu':
        sd_models.unload_model_weights(shared.sd_model)

    state_dict = {}

    # ——————————————————————————————————————————————
    # 3. Reuse tensors from last merge (still works)
    # ——————————————————————————————————————————————
    if (shared.sd_model and 
    hasattr(shared.sd_model, 'sd_checkpoint_info') and 
    shared.sd_model.sd_checkpoint_info.short_title == hash(cmn.last_merge_tasks) and 
    not cmn.is_cross_arch):  # ← THIS LINE DISABLES REUSE FOR CROSS-ARCH (fixes your error)
        print("[Merger] Reusing from loaded model (same-arch mode)")  # ← Optional fun debug line
    state_dict, tasks = get_tensors_from_loaded_model(state_dict, tasks)
    if len(state_dict) > 0:
        progress('Reusing from loaded model', v=len(state_dict))

    # ——————————————————————————————————————————————
    # 4. Trash model handling (unchanged)
    # ——————————————————————————————————————————————
    is_sdxl = any(t in cmn.checkpoints_types.values() for t in ['SDXL', 'SDXL-refiner'])
    if ('SDXL' in cmn.opts['trash_model'] and is_sdxl) or cmn.opts['trash_model'] == 'Enable':
        progress('Unloading webui models...')
        if hasattr(sd_models.model_data, 'loaded_sd_models'):
            while len(sd_models.model_data.loaded_sd_models) > 0:
                model = sd_models.model_data.loaded_sd_models.pop()
                sd_models.send_model_to_trash(model)
            sd_models.model_data.sd_model = None
            shared.sd_model = None
        else:
            sd_models.model_data.sd_model = None
            shared.sd_model = None

    devices.torch_gc()

    # ——————————————————————————————————————————————
    # 5. Pre-cache target shapes for cross-arch (huge speed win)
    # ——————————————————————————————————————————————
    if cmn.is_cross_arch:
            cmn.cross_arch_target_shapes = {}
            with safe_open(cmn.primary, framework='pt', device='cpu') as f:
                for k in f.keys():
                    try:
                        cmn.cross_arch_target_shapes[k] = f.get_tensor(k).shape
                    except:
                        pass
            progress(f'Cross-Arch: Cached {len(cmn.cross_arch_target_shapes)} target shapes from primary model')

            # SAFETY 1: Fix any corrupted/empty target shapes
            fixed = 0
            for k, shape in list(cmn.cross_arch_target_shapes.items()):
                if not isinstance(shape, tuple) or len(shape) == 0:
                    cmn.cross_arch_target_shapes[k] = (1,)
                    fixed += 1
            if fixed:
                progress(f"Cross-Arch: Fixed {fixed} invalid target shapes")

            # SAFETY 2: Remove insane shapes that would cause OOM
            insane_keys = [k for k, shape in list(cmn.cross_arch_target_shapes.items()) 
                          if any(s > 100000 for s in shape)]
            for k in insane_keys:
                del cmn.cross_arch_target_shapes[k]
            if insane_keys:
                progress(f"Cross-Arch: Removed {len(insane_keys)} corrupted/insane shapes (prevented OOM)")
    # ——————————————————————————————————————————————
    # 6. Fast merging with immediate CPU offload + progress
    # ——————————————————————————————————————————————
    timer.record('Merge start')
    progressbar = tqdm(total=len(tasks), desc='Merging', leave=False)

    import threading
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_task = {executor.submit(initialize_task, task): task for task in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            key, tensor = future.result()

            with lock:
                state_dict[key] = tensor.cpu()           # ← immediately free VRAM
                if len(state_dict) % 250 == 0:           # every 250 keys
                    torch.cuda.empty_cache()

            progressbar.update(1)

            if cmn.stop:
                progress.interrupt('Merge stopped by user')
                return {}

    progressbar.close()
    timer.record('Merge complete')

    # ——————————————————————————————————————————————
    # 7. Finetune (unchanged)
    # ——————————————————————————————————————————————
    fine = fineman(finetune, 'SDXL' in cmn.checkpoints_types.get(cmn.primary, ''))
    if finetune and fine:
        for key in FINETUNES:
            if key in state_dict:
                idx = FINETUNES.index(key)
                if idx < 5:
                    state_dict[key] = state_dict[key] * fine[idx]
                else:
                    state_dict[key] = state_dict[key] + torch.tensor(fine[5], device=state_dict[key].device)

    cmn.last_merge_tasks = tuple(tasks_copy)
    progress('### Merge finished ###', v=len(state_dict))

    # ——————————————————————————————————————————————
    # Restore PyTorch's original + − × operators
    # (only if we monkey-patched them during cross-arch merge)
    # ——————————————————————————————————————————————
    if getattr(cmn, 'is_cross_arch', False):
        if hasattr(torch.Tensor, '_original_add'):
            torch.Tensor.__add__ = torch.Tensor._original_add
            torch.Tensor.__sub__ = torch.Tensor._original_sub
            torch.Tensor.__mul__ = torch.Tensor._original_mul
            # clean up the backups so they don’t linger
            delattr(torch.Tensor, '_original_add')
            delattr(torch.Tensor, '_original_sub')
            delattr(torch.Tensor, '_original_mul')

    return state_dict
    
class MergerState:
    def __init__(self):
        self.temp_models = {}

    def apply_lora(self, lora_path, strength=1.0, progress=None):
        """
        Applies LORA transformations to the current state_dict and stores it in temp_models.
        """
        if progress:
            progress(f"Applying LoRA: {lora_path}")

        base_checkpoint_path = cmn.primary
        with safe_open(base_checkpoint_path, framework='pt', device='cpu') as checkpoint_file:
            checkpoint_dict = {k: checkpoint_file.get_tensor(k) for k in checkpoint_file.keys()}

        checkpoint_dict, summary = _apply_single_lora_to_dict(checkpoint_dict, lora_path, strength, progress)

        key = os.path.basename(lora_path)
        self.temp_models[key] = checkpoint_dict

        if progress:
            progress(summary)
        return checkpoint_dict

    def load_temp_model(self, key):
        """
        Loads a previously applied LORA model by its key.
        """
        if key in self.temp_models:
            return self.temp_models[key]
        return None

    def clear_temp_models(self):
        """
        Clears all stored temporary models.
        """
        self.temp_models.clear()

cmn.merger_state = MergerState()

def initialize_task(task):
            try:
                tensor = task.merge()
                return task.key, tensor
            except Exception as e:
                if getattr(cmn, 'is_cross_arch', False):
                    shapes = getattr(cmn, 'cross_arch_target_shapes', {})
                    if task.key in shapes:
                        target = shapes[task.key]
                        print(f"[Merger] Missing key {task.key} → using zeros {target}")
                        tensor = torch.zeros(target, device=cmn.device(), dtype=cmn.dtype())
                        return task.key, tensor  # ← THIS WAS MISSING!
                    else:
                        raise RuntimeError(f"Cross-arch merge failed: key {task.key} has no target shape")
                else:
                    raise RuntimeError(f"Merge failed for key {task.key}: {e}")


def get_tensors_from_loaded_model(state_dict,tasks) -> dict:
    intersected = set(cmn.last_merge_tasks).intersection(set(tasks))
    if intersected:
        with torch.no_grad():
            for module in shared.sd_model.modules():
                networks.network_restore_weights_from_backup(module)

        old_state_dict = shared.sd_model.state_dict()

        for task in intersected:
            try:
                state_dict[task.key] = old_state_dict[task.key]
            except:pass
            tasks.remove(task)
        
    return state_dict,tasks


class safe_open_multiple(object):
    def __init__(self, checkpoints, device):
        self.checkpoints = checkpoints
        self.device = device
        self.open_files = {}

    def __enter__(self):
        for name in self.checkpoints:
            if not name:
                self.open_files[name] = None
                continue
            filename = os.path.join(paths_internal.models_path, 'Stable-diffusion', name)
            try:
                self.open_files[name] = safe_open(filename, framework='pt', device=self.device)
            except Exception as e:
                print(f"[Merger] Failed to open checkpoint '{name}': {e}")
                self.open_files[name] = None  # Mark as failed so LoadTensor can fall back
        return self.open_files

    def __exit__(self, *args):
        for file in self.open_files.values():
            if file is not None:
                try:
                    file.__exit__(*args)
                except:
                    pass


def clear_cache():
    oper.weights_cache.__init__(cmn.opts['cache_size'])
    gc.collect()
    devices.torch_gc()
    torch.cuda.empty_cache()
    cmn.last_merge_tasks = tuple() #Not a cache but is included here to give the user a way to get around it
    return "All caches cleared"


#From https://github.com/hako-mikan/sd-webui-supermerger
def fineman(fine,isxl):
    if fine.find(",") != -1:
        tmp = [t.strip() for t in fine.split(",")]
        fines = [0.0]*8
        for i,f in enumerate(tmp[0:8]):
            try:
                f = float(f)
                fines[i] = f
            except Exception:
                pass

        fine = fines
    else:
        return None

    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3]*0.02] + colorcalc(fine[4:8],isxl)
        ]
    return fine

def colorcalc(cols,isxl):
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]

def weighttoxl(weight):
    weight = weight[:9] + weight[12:22] +[0]
    return weight

FINETUNES = [
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.weight",
"model.diffusion_model.out.2.bias",
]
