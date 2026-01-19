import torch
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from modules import shared
from safetensors.torch import safe_open
 
SACRED_PATTERNS = (
    # ── Sampling-critical (noise / timestep math) ──
    "conv_in.",
    "input_blocks.0.0.",
    "time_embed.",
    "time_embedding",
    "timestep_embed",
    "timestep_embedding",
    "time_in.",
    "vector_in.",
    "img_in.",
    "offset_noise",
    "noise_offset",
    "noise_augmentor",
    "learned_sigma",
    "sigma_embed",

    # ── Input stems / patch embeddings ──
    "patch_embed",
    "patch_embedding",
    "input_proj",
    "input_projection",
    "stem.",
    "image_embed",
    "image_embedding",

    # ── Latent space (VAE scale & decode stability) ──
    "first_stage_model.",
    "vae.",
    "encoder.",
    "decoder.",
    "quant_conv.",
    "post_quant_conv.",

    # ── Semantic space (CLIP / text encoders) ──
    "cond_stage_model.",
    "conditioner.",
    "text_model.",
    "text_model.embeddings",
    "token_embedding",
    "position_embedding",
    "positional_embedding",
    "text_projection",
    "logit_scale",

    # ── Cross-modal gating & modulation ──
    "gate",
    "gating",
    "modulator",
    "modulation",
    "scale_shift",
    "scale_and_shift",
    "affine",

    # ── Adaptive / conditional normalization ──
    "adaptive_norm",
    "conditional_norm",
    "ada_norm",
    "ada_layernorm",
    "ada_ln",

    # ── Flux / SD3 / modern transformer stems ──
    "single_blocks",
    "double_blocks",
    "img_proj",
    "txt_proj",
    "x_embedder",
    "context_embedder",
    "t_embedder",

    # ── Positional embeddings (architecture-agnostic) ──
    "pos_embed",
    "position_emb",
    "position_ids",
)


# ============================================================
# MODEL COMPONENT PREFIXES
# ============================================================

VAE_PREFIXES = (
    "first_stage_model.",  # SD1.5
    "vae.",                # SDXL / Flux
)

CLIP_PREFIXES = (
    # SD1.5
    "cond_stage_model.",
    # SDXL
    "conditioner.",
    "text_model.",
    # Flux / SD3 / modern
    "txt_proj.",
    "context_embedder.",
    "x_embedder.",
    "t_embedder.",
)

def is_vae_key(key: str) -> bool:
    # VAE keys are structurally stable → startswith is safe
    return key.startswith(VAE_PREFIXES)

def is_clip_key(key: str) -> bool:
    # CLIP stacks are more fragmented across architectures
    return (
        key.startswith(CLIP_PREFIXES)
        or any(p in key for p in CLIP_PREFIXES)
    )


# === MERGE STATISTICS TRACKER — FINAL 2025 EDITION (HARDENED) ===
class MergeStats:
    def __init__(self):
        # ── Core merge execution accounting ─────────────────────
        self.custom_merges      = 0   # Successful custom / rule-based merges
        self.custom_failed      = 0   # Custom merges attempted but failed safely

        self.copied_primary     = 0   # Keys copied from primary (metadata, missing keys, etc.)
        self.smart_resized      = 0   # Tensors that were resized (SmartResize called)
        self.zero_filled        = 0   # Missing keys filled with zeros (kitchen-sink)
        self.smart_merge        = 0   # Sparse merges with multiple sources

        # ── Scalar-specific execution ──────────────────────────
        self.scalar_merges      = 0   # Explicit scalar merges (ScaleOnly / ScalarGuard paths)
        self.scalar_failed      = 0   # Scalar merge attempted but failed safely
        self.scalar_rejected    = 0   # Scalar handling rejected by policy / whitelist

        # ── Failure tracking ────────────────────────────────────
        self.skipped            = 0   # Truly missing keys (should be 0 in true kitchen-sink)

        # ── Selector resolution diagnostics (ROUTING, not merging) ──
        self.selector_regex     = 0   # Keys matched via regex selectors
        self.selector_exact     = 0   # Keys matched via exact-key fallback
        self.selector_glob      = 0   # Keys matched via glob fallback (dangerous / opt-in)
        self.selector_failed    = 0   # Scalar merge attempted but failed or aborted safely

    # ── Safety net: never crash due to missing counters ─────────
    def __getattr__(self, name):
        if (
            name.endswith("_failed")
            or name.endswith("_applied")
            or name.endswith("_attempted")
        ):
            setattr(self, name, 0)
            return 0
        raise AttributeError(name)

    # ── Execution-only accounting ──────────────────────────────
    def total_processed(self):
        """
        Total tensors actually handled by the merge engine.
        Selector routing and diagnostics are intentionally excluded.
        """
        return (
            self.custom_merges +
            self.custom_failed +
            self.scalar_merges +
            self.scalar_failed +
            self.copied_primary +
            self.smart_resized +
            self.zero_filled +
            self.smart_merge +
            self.skipped
        )

    def __str__(self):
        total = self.total_processed()

        kitchen_sink  = "YES" if self.skipped == 0 else "ALMOST"
        resize_active = "YES" if self.smart_resized > 0 else "NO"
        scalar_active = "YES" if self.scalar_merges > 0 else "NO"

        return (
            f"### AMETHYST MERGE COMPLETE ###\n"
            f"  • Custom merges            : {self.custom_merges:,}\n"
            f"  • Custom failed (safe)     : {self.custom_failed:,}\n"
            f"  • Scalar merges            : {self.scalar_merges:,}  ({scalar_active})\n"
            f"  • Scalar failed (safe)     : {self.scalar_failed:,}\n"
            f"  • Scalar rejected (policy) : {self.scalar_rejected:,}\n"
            f"  • Copied from Primary      : {self.copied_primary:,}  (metadata, missing keys)\n"
            f"  • Smart-resized            : {self.smart_resized:,}  ({resize_active})\n"
            f"  • Zero-filled (kitchen-sink): {self.zero_filled:,}\n"
            f"  • Smart sparse merges      : {self.smart_merge:,}\n"
            f"  • Skipped (truly missing)  : {self.skipped:,}\n"
            f"  • Total keys processed     : {total:,}\n"
            f"  • True Kitchen-Sink        : {kitchen_sink}\n"
            f"\n"
            f"── Selector Resolution ─────────────────────────\n"
            f"  • Regex selectors matched  : {self.selector_regex:,}\n"
            f"  • Exact-key fallbacks      : {self.selector_exact:,}\n"
            f"  • Glob fallbacks (⚠️)       : {self.selector_glob:,}\n"
            f"  • Selectors unmatched      : {self.selector_failed:,}"
        )


# Global instance
merge_stats = MergeStats()


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
        self.smartresize_enabled = True

        # Execution control
        self.stop = False
        self.interrupted = False

    # -------------------------------------------------
    # Accessors
    # -------------------------------------------------
    def get_device(self):
        if self.device is None:
            raise RuntimeError("MergerContext.device not initialized")
        return self.device

    def get_dtype(self):
        return self.dtype

    def set_dtype(self, dtype_str: str):
        if dtype_str == "fp16":
            self.dtype = torch.float16
        elif dtype_str == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

    def has_tensor(self, file, key: str) -> bool:
        from scripts.untitled.common import has_tensor
        return has_tensor(file, key)

    def get_tensor(self, file, key: str):
        from scripts.untitled.common import get_tensor
        return get_tensor(file, key)

    def is_vae_key(self, key: str) -> bool:
        from scripts.untitled.common import is_vae_key
        return is_vae_key(key)

    def is_clip_key(self, key: str) -> bool:
        from scripts.untitled.common import is_clip_key
        return is_clip_key(key)

    def is_sacred_key(self, key: str) -> bool:
        from scripts.untitled.common import is_sacred_key
        return is_sacred_key(key)

    def is_mergeable_tensor(self, t: torch.Tensor) -> bool:
        from scripts.untitled.common import is_mergeable_tensor
        return is_mergeable_tensor(t)

    def get_opt(self, key, default=None):
        if isinstance(self.opts, dict):
            return self.opts.get(key, default)
        return default

    # -------------------------------------------------
    # Fallback helpers
    # -------------------------------------------------
    def _default_fallback_weights(self, n: int):
        """
        Primary-dominant participation weights.
        HybridCascadeLite decides *how* these are used.
        """
        if n <= 1:
            return [1.0]
        base = 0.7
        rem = (1.0 - base) / (n - 1)
        return [base] + [rem] * (n - 1)

    def hybrid_fallback_op(self, key: str, tensors):
        """
        Default fallback operator (HybridCascadeLite).

        Behavior:
          • Key-aware routing (CLIP / VAE / noise / UNet)
          • Depth-biased confidence & blending
          • COPY behavior embedded internally
          • Deterministic, low-memory, fallback-safe
          • Policy enforcement remains in initialize_task
        """
        if not tensors:
            raise RuntimeError(
                f"hybrid_fallback_op called with no tensors (key={key})"
            )

        from scripts.untitled.operators import HybridCascadeLite

        weights = self._default_fallback_weights(len(tensors))

        return HybridCascadeLite(
            key,
            weights,
            *tensors,

            # Global fallback personality
            base_mix=float(self.get_opt("fallback_lerp_mix", 1.0)),
            confidence=float(self.get_opt("fallback_confidence", 0.5)),

            # CLIP / VAE (safer)
            clip_mix=float(self.get_opt("clip_vae_lerp_mix", 0.6)),
            clip_conf=float(self.get_opt("clip_vae_confidence", 0.35)),
            clip_temp=float(self.get_opt("clip_vae_lerp_temp", 2.0)),

            # Noise / timestep (ultra-safe)
            noise_mix=float(self.get_opt("noise_lerp_mix", 0.4)),
            noise_conf=float(self.get_opt("noise_confidence", 0.25)),
            noise_temp=float(self.get_opt("noise_lerp_temp", 2.5)),

            # Depth behavior
            depth_bias=float(self.get_opt("fallback_depth_bias", 0.35)),
        )


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
    
def safe_apply(op, base, other, key=None):
    """
    Apply a binary tensor op safely.
    Prevents broadcasting, scalar propagation, and catastrophic allocation.
    """

    # Hard validation
    if not isinstance(base, torch.Tensor) or not isinstance(other, torch.Tensor):
        if key:
            print(f"[SafeOp] Non-tensor operand skipped: {key}")
        return base

    # 🚨 Scalars are forbidden
    if base.ndim == 0 or other.ndim == 0:
        if key:
            print(f"[SafeOp] Scalar operand blocked: {key}")
        return base

    # Shape enforcement
    if base.shape != other.shape:
        if key:
            print(
                f"[SafeOp] Shape mismatch skipped: {key} "
                f"{tuple(base.shape)} vs {tuple(other.shape)}"
            )
        return base

    return op(base, other)


def bake_component_into_state_dict(
    target_state_dict: dict,
    component_state_dict: dict,
    *,
    key_filter,                 # callable(key:str)->bool (ex: is_vae_key)
    progress=None,
    label="BAKE",
    overwrite=True,
    strict_shapes=True,
):
    """
    Bake a component (VAE/CLIP/etc) into an existing state_dict.

    - overwrite=True: replace existing keys if present
    - strict_shapes=True: skip mismatched shapes instead of injecting nonsense
    """
    if not target_state_dict:
        raise ValueError("target_state_dict is empty")
    if not component_state_dict:
        if progress:
            progress(f"[{label} WARNING] component_state_dict is empty")
        return target_state_dict

    injected = 0
    replaced = 0
    skipped_shape = 0
    skipped_missing = 0

    for k, v in component_state_dict.items():
        if not key_filter(k):
            continue

        if k in target_state_dict:
            if strict_shapes and hasattr(target_state_dict[k], "shape") and hasattr(v, "shape"):
                if target_state_dict[k].shape != v.shape:
                    skipped_shape += 1
                    continue

            if overwrite:
                target_state_dict[k] = v
                replaced += 1
            else:
                skipped_missing += 1
        else:
            # allow injecting missing keys if you want “baked VAE even if absent”
            target_state_dict[k] = v
            injected += 1

    if progress:
        progress(
            f"[{label}] replaced={replaced} injected={injected} "
            f"skipped_shape={skipped_shape}"
        )

    return target_state_dict

def load_safetensors_state_dict(path: str, *, device="cpu"):
    """
    Load a safetensors file into a plain state_dict + metadata.

    Guarantees:
      • All values are torch.Tensors
      • Metadata is always a dict
      • No silent partial reads
    """
    sd: dict[str, torch.Tensor] = {}

    with safe_open(path, framework="pt", device=device) as f:
        meta = f.metadata() or {}

        for k in f.keys():
            t = f.get_tensor(k)
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"[Safetensors ERROR] Key '{k}' did not yield a torch.Tensor "
                    f"(got {type(t).__name__})"
                )
            sd[k] = t

    return sd, meta


def has_tensor(file, key: str) -> bool:
    """
    Return True if the tensor source contains `key`.

    Supports:
      • safetensors safe_open handles
      • custom wrappers with has_tensor()
      • dict-like objects
    """
    if file is None:
        return False

    # Preferred: explicit method
    if hasattr(file, "has_tensor"):
        try:
            return bool(file.has_tensor(key))
        except Exception:
            return False

    # safetensors handle
    if hasattr(file, "keys"):
        try:
            return key in file.keys()
        except Exception:
            return False

    # dict-like fallback
    try:
        return key in file
    except Exception:
        return False



def get_tensor(file, key: str) -> torch.Tensor:
    """
    Retrieve tensor `key` from a tensor source.

    Guarantees:
      • Always returns a torch.Tensor
      • Raises loudly on failure
    """
    if file is None:
        raise KeyError(f"[get_tensor] Tensor source is None (key='{key}')")

    if hasattr(file, "get_tensor"):
        t = file.get_tensor(key)
    else:
        t = file[key]

    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"[get_tensor ERROR] Object for key '{key}' is not a torch.Tensor "
            f"(got {type(t).__name__})"
        )

    return t


def is_mergeable_tensor(t: torch.Tensor) -> bool:
    """
    Mergeable tensors must be floating-point torch.Tensors.
    """
    return isinstance(t, torch.Tensor) and t.is_floating_point()


def handle_non_mergeable_tensor(
    key: str,
    tensors: list[torch.Tensor],
    *,
    prefer: int = 0,
) -> torch.Tensor:
    """
    Policy for non-floating or non-mergeable tensors.

    Behavior:
      • Select preferred tensor (default: primary)
      • Clone to prevent aliasing
      • Fail loudly on invalid inputs
    """
    if not tensors:
        raise RuntimeError(
            f"[NonMergeable ERROR] No tensors available for key '{key}'"
        )

    if prefer < 0 or prefer >= len(tensors):
        prefer = 0

    t = tensors[prefer]

    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"[NonMergeable ERROR] Object for key '{key}' is not a torch.Tensor "
            f"(got {type(t).__name__})"
        )

    return t.clone()



def is_sacred_key(key: str) -> bool:
    """
    Return True if the key is considered sacred / non-mergeable
    by global policy.
    """
    return bool(SACRED_PATTERNS) and any(p in key for p in SACRED_PATTERNS)


def bake_vae_from_file(
    target_state_dict: dict,
    vae_path: str,
    *,
    key_filter,          # is_vae_key from common.py
    progress=None,
    label="BAKE:VAE",
    strict_shapes=True,
):
    vae_sd, vae_meta = load_safetensors_state_dict(vae_path, device="cpu")

    if progress:
        src = vae_meta.get("source_checkpoint", "Unknown")
        progress(f"[{label}] Loaded VAE file: {vae_path} (source={src})")

    return bake_component_into_state_dict(
        target_state_dict,
        vae_sd,
        key_filter=key_filter,
        progress=progress,
        label=label,
        overwrite=True,
        strict_shapes=strict_shapes,
    ), vae_meta


def extract_submodel(
    source_state_dict,
    *,
    key_filter,
    component_name,
    arch=None,
    source_checkpoint=None,
    progress=None,
):
    """
    Extract a submodel (VAE / CLIP / etc) from a state_dict.

    key_filter: callable(key: str) -> bool
    Returns: (extracted_state_dict, metadata_dict)
    """
    extracted = {}

    for k, v in source_state_dict.items():
        if key_filter(k):
            extracted[k] = v

    metadata = {
        "component": component_name,
        "arch": arch or "Unknown",
        "source_checkpoint": (
            os.path.basename(source_checkpoint)
            if source_checkpoint else "Unknown"
        ),
        "extracted_by": "Amethyst Merger",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tensor_count": len(extracted),
    }

    if progress:
        progress(
            f"[Extract:{component_name}] "
            f"{len(extracted)} tensors | arch={metadata['arch']}"
        )

    return extracted, metadata

# =============================================================================
# GLOBAL STATE — Clean and Minimal
# =============================================================================

last_merge_seed: Optional[int] = None
current_merge_task: Optional[str] = None
last_merge_tasks: Tuple = tuple()  # Cache for detecting repeated merges
checkpoints_types: Dict[str, str] = {}  # Maps checkpoint path → type (SDXL, SD1.5, etc)

# Initialize history (do this once in your extension's on_ui_tabs or script init)
# merge_history = MergeHistory(merge_history_filename)

cmn = MergerContext()