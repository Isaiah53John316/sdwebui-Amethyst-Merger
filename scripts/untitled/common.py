import torch
import json
import os
from datetime import datetime

blocks = None
opts = None
last_seed = 5318008

stop = False
interrupted = False

loaded_checkpoints = None
checkpoints_types = None
primary = ""

last_merge_tasks = tuple()
last_merge_seed = -1

class MergeHistory:
    def __init__(self, history_file):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def add_merge(self, models, merge_mode, calc_mode, sliders, discard, timestamp=None):
        """Record a successful merge"""
        entry = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'models': models,
            'merge_mode': merge_mode,
            'calc_mode': calc_mode,
            'sliders': sliders,
            'discard': discard
        }
        self.history.append(entry)
        self._save_history()
        return entry
    
    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history[-50:], f, indent=2)  # Keep last 50
    
    def get_recent(self, count=10):
        return self.history[-count:]

def device():
    device,dtype = opts['device'].split('/')
    return device 

def dtype():
    device,dtype = opts['device'].split('/')
    if dtype == 'float16': return torch.float16
    elif dtype == 'float8': return torch.float8_e4m3fn
    else: return torch.float32

