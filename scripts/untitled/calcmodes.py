from typing import Any
import scripts.untitled.operators as opr
import scripts.untitled.common as cmn   # ← ADD THIS LINE

MERGEMODES_LIST = []
CALCMODES_LIST = []

# ============================================================================
# BASE CLASSES
# ============================================================================

class MergeMode:
    """Defines the STRUCTURE of the merge formula (what models combine and how)"""
    name = 'mergemode'
    description = 'description'
    input_models = 4
    input_sliders = 3

    slid_a_info = '-'
    slid_a_config = (-1, 2, 0.01) #minimum,maximum,step

    slid_b_info = '-'
    slid_b_config = (-1, 2, 0.01)

    slid_c_info = '-'
    slid_c_config = (-1, 2, 0.01)

    slid_d_info = '-'
    slid_d_config = (-1, 2, 0.01)

    def create_recipe(self, key, model_a, model_b, model_c, model_d, seed=False, alpha=0, beta=0, gamma=0, delta=0) -> opr.Operation:
        """Create the base operation tree structure"""
        raise NotImplementedError


class CalcMode:
    """Defines HOW operations are calculated (modifies merge mode execution)"""
    name = 'calcmode'
    description = 'description'
    compatible_modes = ['all']  # Which merge modes this works with ('all' or list of mode names)

    def modify_recipe(self, recipe, key, model_a, model_b, model_c, model_d, **kwargs) -> opr.Operation:
        """Modify the recipe created by a MergeMode. Default: return unchanged"""
        return recipe


# ============================================================================
# MERGE MODES (Formula Structures)
# ============================================================================

class WeightSum(MergeMode):
    name = 'Weight-Sum'
    description = 'model_a * (1 - alpha) + model_b * alpha'
    input_models = 2
    input_sliders = 1
    slid_a_info = "model_a - model_b"
    slid_a_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        if alpha >= 1:
            return b
        elif alpha <= 0:
            return a

        c = opr.Multiply(key, 1-alpha, a)
        d = opr.Multiply(key, alpha, b)

        res = opr.Add(key, c, d)
        return res

MERGEMODES_LIST.append(WeightSum)


class AddDifference(MergeMode):
    name = 'Add Difference'
    description = 'model_a + (model_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "addition multiplier"
    slid_a_config = (-1, 2, 0.01)
    slid_b_info = "smooth (slow)"
    slid_b_config = (0, 1, 1)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        diff = opr.Sub(key, b, c)
        if beta == 1:
            diff = opr.Smooth(key, diff)
        diff.cache()
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

MERGEMODES_LIST.append(AddDifference)


class TripleSum(MergeMode):
    name = 'Triple Sum'
    description = 'model_a * alpha + model_b * beta + model_c * gamma'
    input_models = 3
    input_sliders = 3
    slid_a_info = "model_a weight"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "model_b weight"
    slid_b_config = (0, 1, 0.01)
    slid_c_info = "model_c weight"
    slid_c_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0.33, beta=0.33, gamma=0.34, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        a_weighted = opr.Multiply(key, alpha, a)
        b_weighted = opr.Multiply(key, beta, b)
        c_weighted = opr.Multiply(key, gamma, c)

        ab = opr.Add(key, a_weighted, b_weighted)
        res = opr.Add(key, ab, c_weighted)
        return res

MERGEMODES_LIST.append(TripleSum)


class SumTwice(MergeMode):
    name = 'Sum Twice'
    description = '(1-beta)*((1-alpha)*model_a + alpha*model_b) + beta*model_c'
    input_models = 3
    input_sliders = 2
    slid_a_info = "model_a - model_b"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "result - model_c"
    slid_b_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0.5, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        # First merge: (1-alpha)*A + alpha*B
        a_weighted = opr.Multiply(key, 1-alpha, a)
        b_weighted = opr.Multiply(key, alpha, b)
        first_merge = opr.Add(key, a_weighted, b_weighted)

        # Second merge: (1-beta)*first_merge + beta*C
        first_weighted = opr.Multiply(key, 1-beta, first_merge)
        c_weighted = opr.Multiply(key, beta, c)

        res = opr.Add(key, first_weighted, c_weighted)
        return res

MERGEMODES_LIST.append(SumTwice)


class QuadSum(MergeMode):
    name = 'Quad Sum'
    description = 'model_a * alpha + model_b * beta + model_c * gamma + model_d * delta (EXPERIMENTAL - may produce artifacts)'
    input_models = 4
    input_sliders = 4
    slid_a_info = "model_a weight"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "model_b weight"
    slid_b_config = (0, 1, 0.01)
    slid_c_info = "model_c weight"
    slid_c_config = (0, 1, 0.01)
    slid_d_info = "model_d weight"
    slid_d_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)
        d = opr.LoadTensor(key,model_d)

        a_weighted = opr.Multiply(key, alpha, a)
        b_weighted = opr.Multiply(key, beta, b)
        c_weighted = opr.Multiply(key, gamma, c)
        d_weighted = opr.Multiply(key, delta, d)

        ab = opr.Add(key, a_weighted, b_weighted)
        abc = opr.Add(key, ab, c_weighted)
        res = opr.Add(key, abc, d_weighted)
        return res

MERGEMODES_LIST.append(QuadSum)


# ============================================================================
# CALCULATION MODES (How Operations Are Computed)
# ============================================================================

class Normal(CalcMode):
    """Standard arithmetic - no modifications"""
    name = 'normal'
    description = 'Standard calculation (no modifications)'
    compatible_modes = ['all']
    input_models = 4  # Supports all merge modes (max 4 models)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, **kwargs):
        return recipe

CALCMODES_LIST.append(Normal)


class TrainDifferenceCalc(CalcMode):
    name = 'trainDifference'
    description = 'Treats difference as fine-tuning with adaptive scaling'
    compatible_modes = ['Add Difference', 'Triple Sum', 'Sum Twice']
    input_models = 3

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        diff = opr.TrainDiff(key, a, b, c)
        diff.cache()
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(TrainDifferenceCalc)

class TrainDifferenceCalc(CalcMode):
    name = 'trainDifference'
    description = 'True task-arithmetic difference (b - c) with zero-mean & norm preservation'
    compatible_modes = ['Add Difference', 'Triple Sum', 'Sum Twice']
    input_models = 3

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        diff = opr.TrainDiff(key, a, b, c)
        diff.cache()
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(TrainDifferenceCalc)


class ExtractCalc(CalcMode):
    """Similarity-based feature extraction"""
    name = 'extract'
    description = 'Adds (dis)similar features between models using cosine similarity'
    compatible_modes = ['Add Difference']
    input_models = 3  # Uses A, B, C

    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)
    slid_b_info = 'similarity - dissimilarity'
    slid_b_config = (0, 1, 0.01)
    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)
    slid_d_info = 'addition multiplier'
    slid_d_config = (-1, 4, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=1, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.Extract(key, alpha, beta, gamma*15, a, b, c)
        extracted.cache()

        multiplied = opr.Multiply(key, delta, extracted)
        res = opr.Add(key, a, multiplied)
        return res

CALCMODES_LIST.append(ExtractCalc)


class TensorCalc(CalcMode):
    """Exchanges entire tensors by probability"""
    name = 'tensor'
    description = 'Swaps entire tensors from A or B based on probability (not weighted blend)'
    compatible_modes = ['Weight-Sum']
    input_models = 2  # Uses A, B

    slid_a_info = "probability of using model_b"
    slid_a_config = (0, 1, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, seed=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        res = opr.TensorExchange(key, alpha, seed, a, b)
        return res

CALCMODES_LIST.append(TensorCalc)


class SelfCalc(CalcMode):
    """Multiply model weights by scalar"""
    name = 'self'
    description = 'Multiply model weights by scalar value (single model operation)'
    compatible_modes = ['Weight-Sum']  # Works as modification of WeightSum with one model
    input_models = 2  # Compatible with 2-model mode, but only uses A (loads B but ignores)

    slid_a_info = "weight multiplier"
    slid_a_config = (0, 2, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=1.0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        res = opr.Multiply(key, alpha, a)
        return res

CALCMODES_LIST.append(SelfCalc)


class InterpDifferenceCalc(CalcMode):
    """Comparative interpolation based on value differences"""
    name = 'Comparative Interp'
    description = 'Interpolates between values depending on their difference relative to other values'
    compatible_modes = ['Weight-Sum']
    input_models = 2  # Uses A, B

    slid_a_info = "concave - convex"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "similarity - difference"
    slid_b_config = (0, 1, 1)
    slid_c_info = "binomial - linear"
    slid_c_config = (0, 1, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        # Skip embeddings
        if key.startswith('cond_stage_model.transformer.text_model.embeddings') or key.startswith('conditioner.embedders.0.transformer.text_model.embeddings') or key.startswith('conditioner.embedders.1.model.token_embedding') or key.startswith('conditioner.embedders.1.model.positional_embedding'):
            return a
        b = opr.LoadTensor(key,model_b)
        return opr.InterpolateDifference(key, alpha, beta, gamma, seed, a ,b)

CALCMODES_LIST.append(InterpDifferenceCalc)


class ManEnhInterpDifferenceCalc(CalcMode):
    """Enhanced interpolation with manual threshold control"""
    name = 'Enhanced Man Interp'
    description = 'Enhanced interpolation with manual threshold control'
    compatible_modes = ['Weight-Sum']
    input_models = 2  # Uses A, B

    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "lower mean threshold"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "upper mean threshold"
    slid_c_config = (0, 1, 0.001)
    slid_d_info = "smoothness factor"
    slid_d_config = (0, 1, 0.001)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key,model_b)
        return opr.ManualEnhancedInterpolateDifference(key, alpha, beta, gamma, delta, seed, a ,b)

CALCMODES_LIST.append(ManEnhInterpDifferenceCalc)


class AutoEnhInterpDifferenceCalc(CalcMode):
    """Enhanced interpolation with automatic threshold calculation"""
    name = 'Enhanced Auto Interp'
    description = 'Interpolates with automatic threshold calculation'
    compatible_modes = ['Weight-Sum']
    input_models = 2  # Uses A, B

    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "threshold adjustment factor"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "smoothness factor"
    slid_c_config = (0, 1, 0.001)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key,model_b)
        return opr.AutoEnhancedInterpolateDifference(key, alpha, beta, gamma, seed, a ,b)

CALCMODES_LIST.append(AutoEnhInterpDifferenceCalc)


class DARECalc(CalcMode):
    name = 'DARE (2025)'
    description = 'Dropout-Aware REweighting — current SOTA for interference-free merges'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Density (keep %) — 0.10–0.25"
    slid_a_config = (0.01, 0.5, 0.01)

    slid_b_info = "Dropout probability — 0.2–0.5"
    slid_b_config = (0.0, 0.7, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.15, beta=0.3, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        density = alpha
        dropout_p = beta
        seed = cmn.last_merge_seed or 42
        return opr.DARE(key, density=density, dropout_p=dropout_p, seed=seed, a=a, b=b)

CALCMODES_LIST.append(DARECalc)


class SmoothMixCalc(CalcMode):
    name = 'Smooth Mix (legacy)'
    description = 'Old beloved smooth mixing behavior (b - a instead of b - c)'
    compatible_modes = ['Add Difference']
    input_models = 3

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        diff = opr.Sub(key, b, a)
        diff = opr.Smooth(key, diff)
        diff.cache()
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(SmoothMixCalc)

class SmoothTrainDiffCalc(CalcMode):
    name = 'Smooth TrainDifference (3-model)'
    description = 'True (B - C) difference, smoothed, added to A — the holy grail'
    compatible_modes = ['Add Difference']
    input_models = 3

    slid_a_info = "Strength multiplier"
    slid_a_config = (0.0, 2.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=1.0, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        diff = opr.Sub(key, b, c)           # True difference: B - C
        diff = opr.Smooth(key, diff)        # Apply the beloved smoothing
        diff.cache()
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(SmoothTrainDiffCalc)

class AddDissimilarityCalc(CalcMode):
    """Add dissimilar features between models"""
    name = 'Add Dissimilarities'
    description = 'Adds dissimilar features between model_b and model_c to model_a'
    compatible_modes = ['Add Difference']
    input_models = 3

    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)
    slid_b_info = 'addition multiplier'
    slid_b_config = (-1, 4, 0.01)
    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)

    # ← NO self! This fork calls statically!
    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        extracted = opr.Similarities(key, alpha, 1, gamma*15, b, c)
        extracted.cache()

        multiplied = opr.Multiply(key, beta, extracted)
        return opr.Add(key, a, multiplied)

CALCMODES_LIST.append(AddDissimilarityCalc)


class TIESCalc(CalcMode):
    name = 'TIES-Merging'
    description = 'State-of-the-art merging: Trim + Sign resolution + Disjoint (2024 paper)'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3
    slid_a_info = "Density (keep %) – try 0.10–0.25"
    slid_a_config = (0.01, 0.5, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        density = alpha
        seed = cmn.last_merge_seed or 42
        return opr.TIES(key, density=density, seed=seed, a=a, b=b)

CALCMODES_LIST.append(TIESCalc)


class SLERPCalc(CalcMode):
    name = 'SLERP (Spherical)'
    description = 'True spherical linear interpolation — best for cross-family merges'
    compatible_modes = ['Weight-Sum', 'Add Difference']  # works best with Weight-Sum
    input_models = 2

    slid_a_info = "Blend ratio (0 = Model A, 1 = Model B)"
    slid_a_config = (0.0, 1.0, 0.01)

    # NO self — this fork calls statically!
    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        return opr.SLERP(key, alpha=alpha, a=a, b=b)

CALCMODES_LIST.append(SLERPCalc)


class ReBasinCalc(CalcMode):
    name = 'Git Re-Basin'
    description = 'Permutation-aware merging — cleaner cross-family results'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Merge ratio"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        return opr.ReBasin(key, alpha, a, b)

CALCMODES_LIST.append(ReBasinCalc)


class DeMeCalc(CalcMode):
    name = 'DeMe (Decouple-Merge)'
    description = 'Timestep-decoupled merge — sharper diffusion'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Decouple strength"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        return opr.DeMe(key, alpha, a, b)

CALCMODES_LIST.append(DeMeCalc)


class BlockWeightedCalc(CalcMode):
    name = 'Block-Weighted Merge'
    description = 'Layer-specific alphas (input/mid/output)'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Global alpha"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        return opr.BlockWeighted(key, [alpha] * 12, a, b)

CALCMODES_LIST.append(BlockWeightedCalc)


class ToMeCalc(CalcMode):
    name = 'ToMe (Token Merge)'
    description = 'Inference speedup — up to 60% faster generation'
    compatible_modes = ['all']
    input_models = 1
    slid_a_info = "Merge ratio (0.4 = 60% speedup)"
    slid_a_config = (0.0, 0.8, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.6, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        return opr.ToMe(key, alpha, a)

CALCMODES_LIST.append(ToMeCalc)


class AttentionMergeCalc(CalcMode):
    name = 'Attention-Only Merge'
    description = 'Merge only attention layers — pure style transfer'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Attention alpha"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.7, beta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        return opr.AttentionMerge(key, alpha, a, b)

CALCMODES_LIST.append(AttentionMergeCalc)