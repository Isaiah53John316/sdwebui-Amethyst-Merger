# Personal Edits and Fixes:
Contains slider-help, soft-disable, corrected mode logic, corrected slider descriptions, UI fixes.
LoRA tab fixed inside Blocks context WIP: Keep Autosave Toggle on, it does not temporarily load into memory yet when finished. Be sure to enter name for the checkpoint to be saved on Lora Tab. Multiple Loras can be merged at once now into the same checkpoint.
Fixed save with model_ema removed from Discard.
History and Preset Tab added.
Works now with Dev branch of Automatic1111 - needed for 50xx.

Feel free to use anything you see. Credit to all below for all their work.

# Batteries Included - Advanced Model Merging for Stable Diffusion

A powerful Stable Diffusion checkpoint merging extension with **15 merge algorithms**, **LoRA support**, and **intelligent caching.

Warning: LORA TAB IS UNHINGED and may need therapy. We've placed it under heavy supervision while we re-align it's morals. AKA: We've got it under construction because an LLM doesn't understand certain concepts.

## 🎉 Now Works on Forge!

This is an ongoing project to give diversity in the webUI economy... er by that we mean...

It works on forge now. We're actively maintaining this and finding new adventurous toys that we can shove into this that gives this that EXPLORATORY FEEL! .. not the suppository feel.

Batteries ARE included because some of us are still running on rechargeable batteries from 30+ years ago.

### ✨ What's New in This Fork

- ✅ **Forge Compatibility** - Works on both SD WebUI Forge AND A1111!
- ✅ **6 New Merge Modes** - Added Triple Sum, Quad Sum, Multiply Difference, Sum Twice, Self, and Tensor modes
- ✅ **LoRA Merging** - Merge LoRAs to checkpoints or combine multiple LoRAs
- ✅ **Bug Fixes** - Fixed DARE (Power-up) division by zero crash
- ✅ **4-Model Support** - Quad Sum mode finally uses all 4 model slots!

---

## 🚀 Features

### Advanced Merge Algorithms (15 Total!)

**Basic Modes:**
- **Weight-Sum** - Classic linear interpolation between two models
- **Self** - Multiply model weights by a scalar value

**Multi-Model Modes:**
- **Triple Sum** - Weighted blend of 3 models
- **Quad Sum** - Weighted blend of 4 models (NEW!)
- **Sum Twice** - Hierarchical two-stage merging

**Difference-Based Modes:**
- **Add Difference** - Add weighted difference: `A + (B-C) × α`
- **Multiply Difference** - Multiplicative difference: `A × ((B-C) × α + 1)` (NEW!)
- **Train Difference** - Treat difference as fine-tuning relative to base model

**Advanced Interpolation:**
- **Comparative Interp** - Adaptive interpolation based on value differences
- **Enhanced Manual Interp** - Manual threshold control for interpolation
- **Enhanced Auto Interp** - Automatic threshold calculation

**Research-Based Methods:**
- **Power-up (DARE)** - Drop And REscale from research papers (FIXED!)
- **Extract** - Merge common/uncommon features between models
- **Add Dissimilarities** - Add dissimilar features to base model

**Experimental:**
- **Tensor** - Swap entire tensors by probability instead of blending (NEW!)

### LoRA Support

- **Merge LoRA to Checkpoint** - Permanently bake a LoRA into a checkpoint with adjustable strength
- **Merge Multiple LoRAs** - Combine 2-3 LoRA files with custom weights

### Performance Features

- **Calculation Caching** - Reuses intermediate calculations to speed up subsequent merges
- **Multi-threading** - Configurable worker threads for parallel operations
- **Memory Efficient** - Smart caching with configurable size limits (0-16GB)

---

## 📋 Requirements

- Stable Diffusion WebUI (A1111) **OR** Stable Diffusion WebUI Forge
- Only supports `.safetensors` format checkpoints
- Python 3.10+ with torch, safetensors, scipy (usually pre-installed)

---

## 🎯 Compatibility

| Platform | Status | Tested Version |
|----------|--------|----------------|
| SD WebUI Forge | ✅ Working | Latest (2024) |
| SD WebUI A1111 | ✅ Working | 1.7.0 - 1.9.3+ |
| SD 1.5 Models | ✅ Fully Supported | All variants |
| SDXL Models | ✅ Generally Works | May have edge cases |

---

## 🙏 Credits & Inspiration

This extension builds upon the excellent work of:

- **Original Extension:** [groinge/sd-webui-untitledmerger](https://github.com/groinge/sd-webui-untitledmerger)
- **Silveroxides Fork** [silveroxides/sd-webui-untitledmerger](https://github.com/silveroxides/sd-webui-untitledmerger)
- **Our Fork** [ktiseos_nyx/sd-webui-supermerger](https://github.com/Ktiseos-Nyx/sd-webui-untitledmerger)
- **Supermerger:** [hako-mikan/sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger) - UI patterns and merge algorithms
- **DARE/Power-up:** [martyn/safetensors-merge-supermario](https://github.com/martyn/safetensors-merge-supermario) - Research implementation
- **MergeLM:** [yule-BUAA/MergeLM](https://github.com/yule-BUAA/MergeLM) - Theoretical foundations

---

## 🛠️ Installation

1. Navigate to your WebUI extensions folder:
   ```bash
   cd stable-diffusion-webui/extensions/
   # OR for Forge:
   cd stable-diffusion-webui-forge/extensions/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/sd-webui-untitledmerger.git
   ```

3. Restart your WebUI

4. Look for the **"Untitled merger"** tab!

---

## 📖 Usage

### Basic Checkpoint Merging

1. Select your merge mode from the dropdown
2. Choose base models (2-4 depending on mode)
3. Adjust sliders for merge weights
4. Configure save options (fp16/bf16, autosave, etc.)
5. Click **Merge**!

### Advanced Features

- **Custom Sliders** - Per-block weight control for fine-grained merging
- **Include/Exclude Filters** - Regex-based layer targeting
- **Supermerger Adjust** - Fine-tune detail, contrast, brightness, and color
- **YAML Weight Editor** - Batch operations via configuration files

### LoRA Merging

- **LoRA → Checkpoint:** Bake a LoRA into a model permanently
- **LoRA → LoRA:** Combine multiple LoRAs with custom ratios

---

## 🐛 Known Issues & Limitations

- Image generation tab intentionally removed (not needed for merging workflow)
- LoRA merging uses heuristic key matching (may need adjustment for exotic LoRAs)
- SDXL merging works but less tested than SD 1.5

---

## 📝 To-Do List

- [ ] XYZ plotting for parameter exploration
- [ ] State saving and merge history
- [ ] Improve LoRA key matching for better compatibility
- [ ] Block weight presets compatible with Supermerger format
- [ ] Merge history logger


---
## Fork history



Who forked it from this fork:

---

## 📜 License

Same as original - check [LICENSE](LICENSE) file for details.

---

## 💬 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing issues for solutions

---

**Happy Merging!** 🎨✨
