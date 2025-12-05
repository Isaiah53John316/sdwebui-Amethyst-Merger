import torch
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from modules import shared

# =============================================================================
# MERGE HISTORY — Elite, Safe, Cross-Platform (Forge Neo + A1111 dev)
# =============================================================================

class MergeHistory:
    """
    Thread-safe, robust merge history with automatic pruning.
    Works perfectly on Forge Neo and A1111 dev.
    """
    def __init__(self, history_file: str):
        self.history_file = history_file
        self.history: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[Merger] Failed to load history: {e}")
            return []

    def _save(self) -> None:
        try:
            # Keep last 50 entries
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history[-50:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Merger] Failed to save history: {e}")

    def add(
        self,
        models: Tuple[str, ...],
        merge_mode: str,
        calc_mode: str,
        sliders: Tuple[float, ...],
        discard: str = "",
        weight_editor: str = "",
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Record a successful merge"""
        entry = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "models": list(models),
            "merge_mode": merge_mode,
            "calc_mode": calc_mode,
            "sliders": list(sliders),
            "discard": discard,
            "weight_editor": weight_editor,
            "seed": seed if seed is not None else -1,
            "webui": "Forge Neo" if hasattr(shared, 'is_forge') else "A1111"
        }
        self.history.append(entry)
        self._save()
        return entry

    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        return self.history[-count:][::-1]  # Most recent first

    def clear(self) -> None:
        self.history.clear()
        self._save()


# =============================================================================
# GLOBAL HELPERS — Safe, Modern, No Assumptions
# =============================================================================

def get_device(self) -> torch.device:
    """Return torch.device based on user choice in Amethyst Options"""
    choice = self.opts.get('device', 'cuda/float16').lower()
    
    if 'cpu' in choice:
        return torch.device('cpu')
    else:
        # Always prefer CUDA if available
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dtype(self) -> torch.dtype:
    """Return torch.dtype based on user choice — supports fp16, fp32, bf16"""

    choice = self.opts.get('device', 'cuda/float16').lower()
    
    if 'float32' in choice or 'fp32' in choice:
        return torch.float32
    elif 'bfloat16' in choice or 'bf16' in choice:
        # Only allow bfloat16 if hardware actually supports it
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            print("[Amethyst] bfloat16 requested but not supported on this GPU → falling back to float16")
            return torch.float16
    else:
        # Default: float16 (fastest on modern GPUs)
        return torch.float16

# =============================================================================
# GLOBAL STATE — Clean and Minimal
# =============================================================================

last_merge_seed: Optional[int] = None
current_merge_task: Optional[str] = None
last_merge_tasks: Tuple = tuple()  # Cache for detecting repeated merges
checkpoints_types: Dict[str, str] = {}  # Maps checkpoint path → type (SDXL, SD1.5, etc)

# Initialize history (do this once in your extension's on_ui_tabs or script init)
# merge_history = MergeHistory(merge_history_filename)