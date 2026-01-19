from typing import Any
import scripts.untitled.operators as opr
import scripts.untitled.common as cmn   

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
    input_sliders = 5 

    slid_a_info = '-'
    slid_a_config = (-1, 2, 0.01)
    slid_b_info = '-'
    slid_b_config = (-1, 2, 0.01)
    slid_c_info = '-'
    slid_c_config = (-1, 2, 0.01)
    slid_d_info = '-'
    slid_d_config = (-1, 2, 0.01)
    slid_e_info = '-' 
    slid_e_config = (-1, 2, 0.01)

    def create_recipe(self, key, model_a, model_b, model_c, model_d, seed=False, alpha=0, beta=0, gamma=0, delta=0, epsilon=0) -> opr.Operation:
        """Create the base operation tree structure"""
        raise NotImplementedError


class CalcMode:
    """Defines HOW operations are calculated (modifies merge mode execution)"""
    name = 'calcmode'
    description = 'description'
    compatible_modes = ['all']  # Which merge modes this works with ('all' or list of mode names)

    # NO self — this fork calls modify_recipe statically!
    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, **kwargs) -> opr.Operation:
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
    name = 'normal'
    description = 'Standard calculation (no modifications) — pure merge'
    compatible_modes = ['all']
    input_models = 4

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, **kwargs):
        return recipe

CALCMODES_LIST.append(Normal)


class TrainDifferenceCalc(CalcMode):
    name = 'trainDifference'
    description = 'True task-arithmetic difference (B - C) → A — with zero-mean & norm preservation'
    compatible_modes = ['Add Difference', 'Triple Sum', 'Sum Twice']
    input_models = 3
    slid_a_info = "Strength multiplier"
    slid_a_config = (0.0, 2.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=1.0, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        diff = opr.TrainDiff(key, a, b, c)
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(TrainDifferenceCalc)


class ExtractCalc(CalcMode):
    name = 'extract'
    description = 'Adds (dis)similar features between models using cosine similarity'
    compatible_modes = ['Add Difference']
    input_models = 3
    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)
    slid_b_info = 'similarity - dissimilarity'
    slid_b_config = (0, 1, 0.01)
    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)
    slid_d_info = 'addition multiplier'
    slid_d_config = (-1, 4, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=1, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        extracted = opr.Extract(key, alpha, beta, gamma*15, a, b, c)
        multiplied = opr.Multiply(key, delta, extracted)
        return opr.Add(key, a, multiplied)

CALCMODES_LIST.append(ExtractCalc)


class TensorCalc(CalcMode):
    name = 'tensor'
    description = 'Swaps entire tensors from A or B based on probability (not weighted blend)'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "probability of using model_b"
    slid_a_config = (0, 1, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        seed = cmn.last_merge_seed or 42
        return opr.TensorExchange(key, alpha, seed, a, b)

CALCMODES_LIST.append(TensorCalc)


class SelfCalc(CalcMode):
    name = 'self'
    description = 'Multiply model weights by scalar value (single model operation)'
    compatible_modes = ['Weight-Sum']
    input_models = 2  # Compatible with 2-model UI, but only uses A
    slid_a_info = "weight multiplier"
    slid_a_config = (0, 2, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=1.0, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        return opr.Multiply(key, alpha, a)

CALCMODES_LIST.append(SelfCalc)


class InterpDifferenceCalc(CalcMode):
    name = 'Comparative Interp'
    description = 'Interpolates between values depending on their difference relative to other values'
    compatible_modes = ['All']
    input_models = 2
    slid_a_info = "concave - convex"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "similarity - difference"
    slid_b_config = (0, 1, 1)
    slid_c_info = "binomial - linear"
    slid_c_config = (0, 1, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        # Skip embeddings
        if key.startswith('cond_stage_model.transformer.text_model.embeddings') or \
           key.startswith('conditioner.embedders.0.transformer.text_model.embeddings') or \
           key.startswith('conditioner.embedders.1.model.token_embedding') or \
           key.startswith('conditioner.embedders.1.model.positional_embedding'):
            return a
        b = opr.LoadTensor(key, model_b)
        return opr.InterpolateDifference(key, alpha, beta, gamma, cmn.last_merge_seed or 42, a, b)

CALCMODES_LIST.append(InterpDifferenceCalc)


class ManEnhInterpDifferenceCalc(CalcMode):
    name = 'Enhanced Man Interp'
    description = 'Enhanced interpolation with manual threshold control'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "lower mean threshold"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "upper mean threshold"
    slid_c_config = (0, 1, 0.001)
    slid_d_info = "smoothness factor"
    slid_d_config = (0, 1, 0.001)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key, model_b)
        return opr.ManualEnhancedInterpolateDifference(key, alpha, beta, gamma, delta, cmn.last_merge_seed or 42, a, b)

CALCMODES_LIST.append(ManEnhInterpDifferenceCalc)


class AutoEnhInterpDifferenceCalc(CalcMode):
    name = 'Enhanced Auto Interp'
    description = 'Interpolates with automatic threshold calculation'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "threshold adjustment factor"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "smoothness factor"
    slid_c_config = (0, 1, 0.001)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key, model_b)
        return opr.AutoEnhancedInterpolateDifference(key, alpha, beta, gamma, cmn.last_merge_seed or 42, a, b)

CALCMODES_LIST.append(AutoEnhInterpDifferenceCalc)


class WISECalc(CalcMode):
    name = 'WISE'
    description = 'Winner-Indexed Sparse Energy merge'
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
        return opr.WISE(key, density, dropout_p, seed, a, b)

CALCMODES_LIST.append(WISECalc)


class WISE3Calc(CalcMode):
    name = 'WISE (3-model)'
    description = '3-model WISE'
    compatible_modes = ['Weight-Sum']
    input_models = 3
    slid_a_info = "Density for Model B (0.10–0.30)"
    slid_a_config = (0.01, 0.5, 0.01)
    slid_b_info = "Dropout for Model B (0.2–0.5)"
    slid_b_config = (0.0, 0.7, 0.05)
    slid_c_info = "Density for Model C (0.10–0.30)"
    slid_c_config = (0.01, 0.5, 0.01)
    slid_d_info = "Dropout for Model C (0.2–0.5)"
    slid_d_config = (0.0, 0.7, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.18, beta=0.35, gamma=0.15, delta=0.40, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        seed = cmn.last_merge_seed or 42
        intermediate = opr.WISE(key, alpha, beta, seed, a, b)
        result = opr.WISE(key, gamma, delta, seed + 1, intermediate, c)
        return result

CALCMODES_LIST.append(WISE3Calc)

class SmoothMixCalc(CalcMode):
    name = 'Smooth Mix (legacy)'
    description = 'Old beloved smooth mixing behavior (b - a instead of b - c)'
    compatible_modes = ['Add Difference']
    input_models = 3
    slid_a_info = "Strength"
    slid_a_config = (0.0, 2.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        diff = opr.Sub(key, b, a)
        diff = opr.Smooth(key, diff)
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(SmoothMixCalc)

class SmoothTrainDiffCalc(CalcMode):
    name = 'Smooth TrainDifference (3-model)'
    description = 'True (B - C) difference, smoothed, added to A — holy grail'
    compatible_modes = ['Add Difference']
    input_models = 3
    slid_a_info = "Strength multiplier"
    slid_a_config = (0.0, 2.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=1.0, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        diff = opr.Sub(key, b, c)
        diff = opr.Smooth(key, diff)
        diffm = opr.Multiply(key, alpha, diff)
        return opr.Add(key, a, diffm)

CALCMODES_LIST.append(SmoothTrainDiffCalc)


class AddDissimilarityCalc(CalcMode):
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

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        extracted = opr.Similarities(key, alpha, 1, gamma*15, b, c)
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
        return opr.TIES(key, density, seed, a, b)

CALCMODES_LIST.append(TIESCalc)


class SLERPCalc(CalcMode):
    name = 'SLERP (Spherical)'
    description = 'True spherical linear interpolation — best for cross-family merges'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Blend ratio (0 = Model A, 1 = Model B)"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5, beta=0, gamma=0, **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        # weights = [Model B], base = Model A
        return opr.SLERP(key, [alpha], a, b)

CALCMODES_LIST.append(SLERPCalc)



class SLERP3Calc(CalcMode):
    name = 'SLERP (3-model Spherical)'
    description = 'True 3-point spherical interpolation — perfect for triple fusions'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Weight for Model B (0–1)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Weight for Model C (0–1, total B+C ≤ 1)"
    slid_b_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5, beta=0.3, gamma=0, **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        # weights = [Model B, Model C], base = Model A
        return opr.SLERP(key, [alpha, beta], a, b, c)

CALCMODES_LIST.append(SLERP3Calc)


class LERPCalc(CalcMode):
    name = 'LERP (Linear)'
    description = 'Linear interpolation — stable, magnitude-preserving, safest blend'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Blend ratio (0 = Model A, 1 = Model B)"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5, beta=0, gamma=0, **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        # Linear blend: A*(1-alpha) + B*alpha
        return opr.LERP(key, [1.0 - alpha, alpha], a, b)

CALCMODES_LIST.append(LERPCalc)

class LERP3Calc(CalcMode):
    name = 'LERP (3-model Linear)'
    description = '3-way linear interpolation — smooth and predictable fusion'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Weight for Model B (0–1)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Weight for Model C (0–1, total B+C ≤ 1)"
    slid_b_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5, beta=0.3, gamma=0, **kwargs
    ):
        # Clamp to keep A >= 0
        beta = min(beta, 1.0 - alpha)
        base_weight = 1.0 - alpha - beta

        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        # Linear blend: A*base + B*alpha + C*beta
        return opr.LERP(key, [base_weight, alpha, beta], a, b, c)

CALCMODES_LIST.append(LERP3Calc)

class LERPMEANCalc(CalcMode):
    name = 'LERP/MEAN Hybrid'
    description = 'Linear blend stabilized by mean — wide-coverage, safe-by-design fallback'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Blend ratio (0 = Model A, 1 = Model B)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Stability mix (0 = MEAN, 1 = LERP)"
    slid_b_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5, beta=1.0, gamma=0, **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        # weights: A*(1-alpha) + B*alpha
        weights = [1.0 - alpha, alpha]

        return opr.LERPMEAN(
            key,
            weights,
            a,
            b,
            mix=beta,          # beta = how much LERP vs MEAN
        )

CALCMODES_LIST.append(LERPMEANCalc)

class LERPMEAN3Calc(CalcMode):
    name = 'LERP/MEAN Hybrid (3-model)'
    description = '3-way linear blend with mean stabilization'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Weight for Model B (0–1)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Weight for Model C (0–1, total B+C ≤ 1)"
    slid_b_config = (0.0, 1.0, 0.01)

    slid_c_info = "Stability mix (0 = MEAN, 1 = LERP)"
    slid_c_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5, beta=0.3, gamma=1.0, **kwargs
    ):
        # Ensure A >= 0
        beta = min(beta, 1.0 - alpha)
        base_weight = 1.0 - alpha - beta

        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        weights = [base_weight, alpha, beta]

        return opr.LERPMEAN(
            key,
            weights,
            a,
            b,
            c,
            mix=gamma,
        )

CALCMODES_LIST.append(LERPMEAN3Calc)

class AdaptiveLERPCalc(CalcMode):
    name = 'Adaptive LERP'
    description = (
        'Adaptive blend that preserves structure where models disagree '
        'and blends freely where they agree'
    )
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Blend (0 = Model A, 1 = Model B)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Style strength (0 = very stable, 1 = very expressive)"
    slid_b_config = (0.0, 1.0, 0.01)

    slid_c_info = "Confidence (how much agreement is trusted)"
    slid_c_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5,   # blend
        beta=1.0,    # style strength
        gamma=0.5,   # confidence
        **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        weights = [1.0 - alpha, alpha]

        return opr.AdaptiveLERP(
            key,
            weights,
            a,
            b,
            base_mix=beta,
            confidence=gamma,
        )


CALCMODES_LIST.append(AdaptiveLERPCalc)


class AdaptiveLERP3Calc(CalcMode):
    name = 'Adaptive LERP (3-model)'
    description = '3-way adaptive blend with automatic per-channel stabilization'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Weight for Model B"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Weight for Model C (total ≤ 1)"
    slid_b_config = (0.0, 1.0, 0.01)

    slid_c_info = "Style strength (0 = stable, 1 = expressive)"
    slid_c_config = (0.0, 1.0, 0.01)

    slid_d_info = "Confidence (trust agreement vs stability)"
    slid_d_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5,
        beta=0.3,
        gamma=1.0,
        delta=0.5,
        **kwargs
    ):
        beta = min(beta, 1.0 - alpha)
        base_weight = 1.0 - alpha - beta

        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        weights = [base_weight, alpha, beta]

        return opr.AdaptiveLERP(
            key,
            weights,
            a,
            b,
            c,
            base_mix=gamma,
            confidence=delta,
        )


CALCMODES_LIST.append(AdaptiveLERP3Calc)


class COPYCalc(CalcMode):
    name = 'COPY (Select Model)'
    description = 'Preserve weights from a single model verbatim'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Model to copy (0 = Model A, 1 = Model B)"
    slid_a_config = (0, 1, 1)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0, beta=0, gamma=0, **kwargs
    ):
        idx = int(alpha)

        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        return opr.COPY(key, a, b, prefer=idx)
    
CALCMODES_LIST.append(COPYCalc)

class COPY3Calc(CalcMode):
    name = 'COPY (3-model Select)'
    description = 'Preserve weights from one of three models'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Model index to copy (0 = A, 1 = B, 2 = C)"
    slid_a_config = (0, 2, 1)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0, beta=0, gamma=0, **kwargs
    ):
        idx = int(alpha)

        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        return opr.COPY(key, a, b, c, prefer=idx)

CALCMODES_LIST.append(COPY3Calc)

class ReBasinCalc(CalcMode):
    name = 'Git Re-Basin'
    description = 'Permutation-aware merging — cleaner cross-family results'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Merge ratio"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, gamma=0, **kwargs):
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

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, gamma=0, **kwargs):
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

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0, gamma=0, **kwargs):
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
    slid_b_info = "Unused"
    slid_b_config = (0.0, 0.0, 0.01)
    slid_c_info = "Unused"
    slid_c_config = (0.0, 0.0, 0.01)
    slid_d_info = "Unused"
    slid_d_config = (0.0, 0.0, 0.01)

    @staticmethod
    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                     alpha=0.6, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        return opr.ToMe(key, alpha, a)

CALCMODES_LIST.append(ToMeCalc)

class SmoothConvCalc(CalcMode):
    name = 'SmoothConv (2D+1D)'
    description = '2D Gaussian on Conv weights, 1D on Linear/Attention — cleans noise perfectly'
    compatible_modes = ['all']
    input_models = 1

    slid_a_info = "Smoothing sigma (strength)"
    slid_a_config = (0.0, 3.0, 0.05)
    slid_b_info = "Kernel size (0 = auto)"
    slid_b_config = (0, 15, 1)
    slid_c_info = "Unused"
    slid_c_config = (0.0, 0.0, 0.01)
    slid_d_info = "Unused"
    slid_d_config = (0.0, 0.0, 0.01)

    @staticmethod
    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                     alpha=1.0, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)

        # ← CRITICAL FIX: DO NOT CALL .numel() ON LoadTensor
        # Skip critical text encoder keys
        if any(skip in key for skip in [
            'cond_stage_model.transformer.text_model.embeddings',
            'conditioner.embedders.0.transformer.text_model.embeddings',
            'conditioner.embedders.1.model.token_embedding',
            'position_ids',
            'position_embedding'
        ]):
            return a

        kernel_size = int(beta) if beta > 0 else None
        return opr.SmoothConv(key, sigma=alpha, kernel_size=kernel_size, tensor=a)

CALCMODES_LIST.append(SmoothConvCalc)


class Smooth1DCalc(CalcMode):
    name = 'Smooth (1D Classic)'
    description = 'Classic 1D Gaussian smoothing — great for attention & linear layers'
    compatible_modes = ['all']
    input_models = 1

    slid_a_info = "Smoothing strength"
    slid_a_config = (0.0, 3.0, 0.05)
    slid_b_info = "Unused"
    slid_b_config = (0.0, 0.0, 0.01)
    slid_c_info = "Unused"
    slid_c_config = (0.0, 0.0, 0.01)
    slid_d_info = "Unused"
    slid_d_config = (0.0, 0.0, 0.01)

    @staticmethod
    def modify_recipe(self, recipe, key, model_a, model_b, model_c, model_d,
                     alpha=1.0, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        return opr.Smooth(key, tensor=a)

CALCMODES_LIST.append(Smooth1DCalc)

class AttentionMergeCalc(CalcMode):
    name = 'Attention-Only Merge'
    description = 'Merge only attention layers — pure style transfer'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Attention alpha"
    slid_a_config = (0.0, 1.0, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.7, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        return opr.AttentionMerge(key, alpha, a, b)

CALCMODES_LIST.append(AttentionMergeCalc)

class SVDDeNoiseCalc(CalcMode):
    name = 'SVD DeNoise (Expert)'
    description = 'Nuclear-grade interference removal via SVD — slow but god-tier'
    compatible_modes = ['Weight-Sum']
    input_models = 2
    slid_a_info = "Singular value threshold (0.01–0.1)"
    slid_a_config = (0.001, 0.2, 0.001)
    slid_b_info = "Keep top % of values (0.1–0.5)"
    slid_b_config = (0.05, 0.5, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.02, beta=0.2, gamma=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        seed = cmn.last_merge_seed or 42
        return opr.SingularValueDeOperator(key, alpha, beta, seed, a, b)

CALCMODES_LIST.append(SVDDeNoiseCalc)

class DARECalc(CalcMode):
    name = 'DARE'
    description = 'Sparse delta merging with per-source density (DARE)'
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
        return opr.DARE_Nway(key, density, dropout_p, seed, a, b)

CALCMODES_LIST.append(DARECalc)


class DARE3Calc(CalcMode):
    name = 'DARE (3-model)'
    description = '3-model DARE'
    compatible_modes = ['Weight-Sum']
    input_models = 3
    slid_a_info = "Density for Model B (0.10–0.30)"
    slid_a_config = (0.01, 0.5, 0.01)
    slid_b_info = "Dropout for Model B (0.2–0.5)"
    slid_b_config = (0.0, 0.7, 0.05)
    slid_c_info = "Density for Model C (0.10–0.30)"
    slid_c_config = (0.01, 0.5, 0.01)
    slid_d_info = "Dropout for Model C (0.2–0.5)"
    slid_d_config = (0.0, 0.7, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.18, beta=0.35, gamma=0.15, delta=0.40, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        seed = cmn.last_merge_seed or 42
        intermediate = opr.DARE_Nway(key, alpha, beta, seed, a, b)
        result = opr.DARE_Nway(key, gamma, delta, seed + 1, intermediate, c)
        return result

CALCMODES_LIST.append(DARE3Calc)

class DAREWISECalc(CalcMode):
    name = 'DAREWISE'
    description = 'Hybrid merge: DARE (structure) + WISE (style)'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "DARE density (structure)"
    slid_a_config = (0.01, 0.5, 0.01)
    slid_b_info = "DARE dropout"
    slid_b_config = (0.0, 0.7, 0.05)
    slid_c_info = "WISE density (style)"
    slid_c_config = (0.01, 0.5, 0.01)
    slid_d_info = "WISE dropout"
    slid_d_config = (0.0, 0.7, 0.05)
    slid_e_info = "Style mix (0 = DARE, 1 = WISE)"
    slid_e_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.15, beta=0.3, gamma=0.15, delta=0.3, epsilon=0.5, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        seed = cmn.last_merge_seed or 42
        return opr.DAREWISE(key, alpha, beta, gamma, delta, epsilon, seed, a, b)

CALCMODES_LIST.append(DAREWISECalc)


class AdaptiveDAREWISECalc(CalcMode):
    name = 'DAREWISE (Adaptive)'
    description = 'Self-regulating hybrid merge — structure-safe by default'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Aggression bias (0 = safe, 1 = expressive)"
    slid_a_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.5, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        seed = cmn.last_merge_seed or 42
        return opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)

CALCMODES_LIST.append(AdaptiveDAREWISECalc)


class AdaptiveDAREWISE3Calc(CalcMode):
    name = 'DAREWISE (Adaptive, 3-model)'
    description = '3-model adaptive hybrid merge — structure-safe by default'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Aggression bias (0 = safe, 1 = expressive)"
    slid_a_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.5, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        seed = cmn.last_merge_seed or 42
        ab = opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)
        return opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed + 1, ab, c)

CALCMODES_LIST.append(AdaptiveDAREWISE3Calc)

class ProgressiveDAREWISEBalancedCalc(CalcMode):
    name = 'DAREWISE (Progressive – Balanced)'
    description = 'Structure → adaptive → balanced style (safe default)'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Style bias (global)"
    slid_a_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.5, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        seed = cmn.last_merge_seed or 42
        k = key.lower()

        if any(s in k for s in ('attn', 'attention', 'to_q', 'to_k', 'to_v', 'proj')):
            return opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        if 'down_blocks' in k or 'input_blocks' in k:
            return opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        if 'mid_block' in k:
            return opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)

        return opr.DAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)

CALCMODES_LIST.append(ProgressiveDAREWISEBalancedCalc)


class ProgressiveDAREWISEBalanced3Calc(CalcMode):
    name = 'DAREWISE (Progressive – Balanced, 3-model)'
    description = '3-model structure → adaptive → balanced style'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Style bias (global)"
    slid_a_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.5, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        seed = cmn.last_merge_seed or 42
        k = key.lower()

        if any(s in k for s in ('attn', 'attention', 'to_q', 'to_k', 'to_v', 'proj')):
            ab = opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        elif 'down_blocks' in k or 'input_blocks' in k:
            ab = opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        elif 'mid_block' in k:
            ab = opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)
        else:
            ab = opr.DAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)

        return opr.DAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed + 1, ab, c)

CALCMODES_LIST.append(ProgressiveDAREWISEBalanced3Calc)


class ProgressiveDAREWISEAggressiveCalc(CalcMode):
    name = 'DAREWISE (Progressive – Aggressive)'
    description = 'Structure → adaptive → pure style (maximum takeover)'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Style bias (global)"
    slid_a_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.5, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        seed = cmn.last_merge_seed or 42
        k = key.lower()

        if any(s in k for s in ('attn', 'attention', 'to_q', 'to_k', 'to_v', 'proj')):
            return opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        if 'down_blocks' in k or 'input_blocks' in k:
            return opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        if 'mid_block' in k:
            return opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)

        return opr.WISE(key, 0.18, 0.35, seed, a, b)

CALCMODES_LIST.append(ProgressiveDAREWISEAggressiveCalc)

class ProgressiveDAREWISEAggressive3Calc(CalcMode):
    name = 'DAREWISE (Progressive – Aggressive, 3-model)'
    description = '3-model structure → adaptive → pure style'
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Style bias (global)"
    slid_a_config = (0.0, 1.0, 0.05)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d,
                      alpha=0.5, beta=0, gamma=0, delta=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)
        seed = cmn.last_merge_seed or 42
        k = key.lower()

        if any(s in k for s in ('attn', 'attention', 'to_q', 'to_k', 'to_v', 'proj')):
            ab = opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        elif 'down_blocks' in k or 'input_blocks' in k:
            ab = opr.DARE_Nway(key, 0.15, 0.25, seed, a, b)
        elif 'mid_block' in k:
            ab = opr.AdaptiveDAREWISE(key, 0.15, 0.25, 0.18, 0.35, alpha, seed, a, b)
        else:
            ab = opr.WISE(key, 0.18, 0.35, seed, a, b)

        return opr.WISE(key, 0.18, 0.35, seed + 1, ab, c)

CALCMODES_LIST.append(ProgressiveDAREWISEAggressive3Calc)

class HybridCascadeSimpleCalc(CalcMode):
    name = 'Hybrid Cascade Simple'
    description = (
        'Automatically selects the best merge strategy per layer '
        '(COPY, AdaptiveLERP, DAREWISE, TrainDiff, TIES) '
        'based on tensor role and agreement'
    )
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 2

    slid_a_info = "Blend ratio (0 = Model A, 1 = Model B)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Confidence (0 = ultra-stable, 1 = aggressive hybridization)"
    slid_b_config = (0.0, 1.0, 0.01)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5,
        beta=0.5,
        **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        # Linear participation weights (who contributes)
        weights = [1.0 - alpha, alpha]

        return opr.HybridCascadeSimple(
            key,
            weights,
            a,
            b,
            confidence=beta,
            seed=cmn.last_merge_seed or 42,
        )

CALCMODES_LIST.append(HybridCascadeSimpleCalc)

class HybridCascadeSimple3Calc(CalcMode):
    name = 'Hybrid Cascade Simple (3-model)'
    description = (
        '3-way intelligent hybrid merge with automatic per-layer operator selection'
    )
    compatible_modes = ['Weight-Sum', 'Add Difference']
    input_models = 3

    slid_a_info = "Weight for Model B (0–1)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Weight for Model C (0–1, total ≤ 1)"
    slid_b_config = (0.0, 1.0, 0.01)

    slid_c_info = "Confidence (0 = stable, 1 = aggressive hybrid)"
    slid_c_config = (0.0, 1.0, 0.01)
 
    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.3,
        beta=0.3,
        gamma=0.5,
        **kwargs
    ):
        # Ensure A >= 0
        beta = min(beta, 1.0 - alpha)
        base_weight = 1.0 - alpha - beta

        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        c = opr.LoadTensor(key, model_c)

        weights = [base_weight, alpha, beta]

        return opr.HybridCascadeSimple(
            key,
            weights,
            a,
            b,
            c,
            confidence=gamma,
            seed=cmn.last_merge_seed or 42,
        )

CALCMODES_LIST.append(HybridCascadeSimple3Calc)

class HybridCascadeCalc(CalcMode):
    name = "HybridCascade"
    description = (
        "Block-aware hybrid merge that automatically selects "
        "the safest or most expressive operator per layer "
        "(AdaptiveLERP / DAREWISE / TrainDiff / TIES)"
    )
    compatible_modes = ["Weight-Sum", "Add Difference", "QuadSum"]
    input_models = 4  # works with 2–4

    # ----------------------------------
    # Slider definitions (A–E)
    # ----------------------------------

    slid_a_info = "Global Confidence (0 = stable, 1 = expressive)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Depth Bias Strength (early stable → late expressive)"
    slid_b_config = (0.0, 1.0, 0.01)

    slid_c_info = "TrainDiff Strength (training-style delta injection)"
    slid_c_config = (0.0, 1.0, 0.01)

    slid_d_info = "DARE/WISE Density (detail vs sparsity)"
    slid_d_config = (0.05, 0.6, 0.01)

    slid_e_info = "TIES Pre-sparsity (0 = off)"
    slid_e_config = (0.0, 0.6, 0.01)

    # slid_f_info = "Weight Temperature (advanced)"      # ← intentionally disabled
    # slid_f_config = (0.5, 3.0, 0.05)

    # ----------------------------------
    # Recipe builder
    # ----------------------------------
    def modify_recipe(
        recipe,
        key,
        model_a,
        model_b,
        model_c,
        model_d,
        alpha=0.55,   # confidence
        beta=0.45,   # depth bias
        gamma=0.3,   # TrainDiff strength
        delta=0.3,  # DARE/WISE density
        epsilon=0.0, # TIES density
        **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        tensors = [a, b]
        weights = [1.0, 1.0]

        if model_c is not None:
            c = opr.LoadTensor(key, model_c)
            tensors.append(c)
            weights.append(1.0)

        if model_d is not None:
            d = opr.LoadTensor(key, model_d)
            tensors.append(d)
            weights.append(1.0)

        # Normalize weights (primary-dominant handled inside HybridCascade)
        total = sum(weights)
        weights = [w / total for w in weights]

        return opr.HybridCascade(
            key,
            weights,
            *tensors,

            # ---- global personality ----
            confidence=alpha,

            # ---- depth bias ----
            depth_bias_enabled=True,
            depth_conf_strength=beta,
            depth_mix_strength=beta * 0.75,
            depth_traindiff_strength=beta,

            # ---- TrainDiff ----
            use_traindiff=gamma > 0.01,
            traindiff_strength=gamma,

            # ---- DARE / WISE ----
            dare_density=delta,
            wise_density=delta,
            dare_dropout=0.10,
            wise_dropout=0.30,

            # ---- optional TIES ----
            use_ties=epsilon > 0.01,
            ties_density=epsilon,

            seed=cmn.last_merge_seed or 42,
        )


CALCMODES_LIST.append(HybridCascadeCalc)

class HybridCascadeLiteCalc(CalcMode):
    name = "Hybrid Cascade Lite"
    description = "Key-aware, depth-biased adaptive merge (fallback-safe)"
    compatible_modes = ["Weight-Sum", "Add Difference",]
    input_models = 2

    slid_a_info = "Blend ratio (0 = A, 1 = B)"
    slid_a_config = (0.0, 1.0, 0.01)

    slid_b_info = "Aggression (LERP vs MEAN)"
    slid_b_config = (0.0, 1.0, 0.01)

    slid_c_info = "Confidence (agreement trust)"
    slid_c_config = (0.0, 1.0, 0.01)

    slid_d_info = "Depth bias strength"
    slid_d_config = (0.0, 1.0, 0.01)

    slid_e_info = "Weight temperature (safety)"
    slid_e_config = (0.5, 3.0, 0.05)

    def modify_recipe(
        recipe, key,
        model_a, model_b, model_c, model_d,
        alpha=0.5,
        beta=1.0,
        gamma=0.5,
        delta=0.35,
        epsilon=1.0,
        **kwargs
    ):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)

        weights = [1.0 - alpha, alpha]

        return opr.HybridCascadeLite(
            key,
            weights,
            a,
            b,
            base_mix=beta,
            confidence=gamma,
            depth_bias=delta,
        )

CALCMODES_LIST.append(HybridCascadeLiteCalc)
