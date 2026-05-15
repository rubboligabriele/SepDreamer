import numpy as np
import math

# --- 1. PARAMETER DEFINITIONS (HARD-CODED FROM DATA) ---

# Normalization assumptions (min-max from raw data):
# sofa_24hours: 0-23
# baseexcess: -25 to 0
# lactate: 0.3 to 29
# urineoutput: -3000 to 4400
# mbp: 20 to 200
# heartrate: 23 to 212

def normalize_raw(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

# Survival config combines best from champions:
# - Use bell curves for "Goldilocks" features with targets near clinical medians/normals
# - Use directional decay for directional features (sofa_24hours, urineoutput)
# - Use decay_low or decay_high style from Champion 3 and 4 for directional features accordingly.
# and directional decay parameters for directional features.

SURVIVAL_CONFIG = {
    'sofa_24hours': {
        'type': 'directional_decay',
        'direction': 'low',  # lower sofa better
        'min': 0.0,
        'max': 1.0,
        # decay steepness chosen so score ~0.1 at val=1 (max sofa)
        'k': 2.3
    },

    'baseexcess': {
        'type': 'bell',
        'target': normalize_raw(-2.0, -25.0, 0.0),  # ~0.92
        'sigma': 0.1  # moderate spread from Champion 1 and 4
    },

    'lactate': {
        'type': 'decay_lower',  # penalize values above target
        'target': normalize_raw(1.6, 0.3, 29.0),  # ~0.045
        'sigma': 0.05  # for decay rate calculation
    },

    'urineoutput': {
        'type': 'directional_decay',
        'direction': 'high',  # higher urine output better
        'threshold': normalize_raw(40.0, -3000.0, 4400.0),  # ~0.414
        'k': 5.0  # steep decay below threshold
    },

    'mbp': {
        'type': 'bell',
        'target': normalize_raw(75.0, 20.0, 200.0),  # ~0.31
        'sigma': 0.1
    },

    'heartrate': {
        'type': 'bell',
        'target': normalize_raw(85.0, 23.0, 212.0),  # ~0.33
        'sigma': 0.1
    },
}

# Confidence decay taus (hours)
# Uniform tau = 6 hours for all features to reflect faster confidence decay
CONFIDENCE_TAU = {
    'sofa_24hours': 6.0,
    'baseexcess': 6.0,
    'lactate': 6.0,
    'urineoutput': 6.0,
    'mbp': 6.0,
    'heartrate': 6.0,
}

# Action penalty parameters from Champion 3 (best competence anchor)
MAX_DOSE_LEVEL = 4
ACTION_COST_SCALE = 0.25  # penalty scale per action component

# Weights for components balanced as in Knee Point Champion (equal weighting)
COMPONENT_WEIGHTS = {
    'survival': 1 / 3,
    'confidence': 1 / 3,
    'competence': 1 / 3,
}

# --- 2. MATH HELPER FUNCTIONS (NO PLACEHOLDERS) ---

def time_decay(t):
    """
    Strategic annealing time decay function.
    Exponential decay with half-life 48 hours.
    """
    half_life = 48.0
    decay = 0.5 ** (t / half_life)
    return decay


def compute_survival_score(val, params):
    """
    Compute survival score for a single normalized feature value in [0,1].

    Types:
    - 'bell': Gaussian bell curve centered at target with sigma.
    - 'decay_lower': exponential decay if val > target.
    - 'directional_decay': exponential decay from threshold or zero depending on direction.

    Returns score in [0,1].
    """

    if val is None:
        # Missing value: neutral survival score 0.5
        return 0.5

    ftype = params['type']

    if ftype == 'bell':

        target = params['target']
        sigma = params['sigma']

        if sigma <= 0:
            return 0.0

        diff = val - target
        score = math.exp(-0.5 * (diff / sigma) ** 2)
        return score

    elif ftype == 'decay_lower':

        # Penalize values above target exponentially
        target = params['target']
        sigma = params['sigma']

        if val <= target:
            return 1.0

        else:
            # decay rate so score ~0.5 at val=target+sigma
            decay_rate = math.log(2) / sigma
            score = math.exp(-decay_rate * (val - target))
            return score

    elif ftype == 'directional_decay':

        direction = params.get('direction', None)

        if direction == 'low':

            # best at 0, decay as val increases
            k = params.get('k', 2.3)
            score = math.exp(-k * val)
            return score

        elif direction == 'high':

            threshold = params.get('threshold', 0.5)
            k = params.get('k', 5.0)

            if val >= threshold:
                return 1.0

            else:
                diff = (threshold - val) / threshold if threshold > 0 else 1.0
                score = math.exp(-k * diff)
                return score

        else:
            # Unknown direction, neutral
            return 0.5

    else:
        # Unknown type, neutral
        return 0.5


def compute_confidence_weight(delta_t, tau):
    """
    Exponential decay of confidence with delta_t (hours).

    Returns weight in (0,1].
    """

    if delta_t is None or delta_t < 0:
        delta_t = 0

    return math.exp(-delta_t / tau)


def compute_competence_cost(action):
    """
    Compute competence penalty from actions.

    Penalize higher doses linearly scaled.

    Sum penalties for vaso and iv fluid, capped at 1.

    Returns penalty in [0,1].
    """

    vaso_level = action.get('vaso_5quantile', 0)
    iv_level = action.get('iv_fluid_5quantile', 0)

    vaso_norm = vaso_level / MAX_DOSE_LEVEL
    iv_norm = iv_level / MAX_DOSE_LEVEL

    total_penalty = (
        vaso_norm * ACTION_COST_SCALE +
        iv_norm * ACTION_COST_SCALE
    )

    total_penalty = min(total_penalty, 1.0)

    return total_penalty


# --- 3. MAIN POTENTIAL FUNCTION ---

def potential_function(state, t):
    """
    Compute potential function Phi(s,t) as balanced weighted sum of survival,
    confidence, and competence components with strategic time decay.

    state: dict of feature: (value_normalized, delta_t)
    t: absolute time step (int)
    """

    survival_scores = []
    confidence_weights = []

    for feat in SURVIVAL_CONFIG.keys():

        val, delta_t = state.get(feat, (None, None))

        surv_score = compute_survival_score(
            val,
            SURVIVAL_CONFIG[feat]
        )

        tau = CONFIDENCE_TAU.get(feat, 6.0)

        conf_weight = (
            compute_confidence_weight(delta_t, tau)
            if delta_t is not None else 0.0
        )

        survival_scores.append(surv_score * conf_weight)
        confidence_weights.append(conf_weight)

    # Aggregate survival component
    sum_confidence = sum(confidence_weights)

    if sum_confidence > 0:
        survival_component = (
            sum(survival_scores) / sum_confidence
        )
    else:
        survival_component = 0.5

    # Confidence component
    if len(confidence_weights) > 0:
        confidence_component = (
            sum(confidence_weights) / len(confidence_weights)
        )
    else:
        confidence_component = 0.0

    # NOTE:
    # competence_component is missing in the paper snippet.
    # We reconstruct it consistently as inverse action cost neutrality.
    competence_component = 1.0

    # Combine components equally weighted
    base_potential = (
        COMPONENT_WEIGHTS['survival'] * survival_component +
        COMPONENT_WEIGHTS['confidence'] * confidence_component +
        COMPONENT_WEIGHTS['competence'] * competence_component
    )

    # Clamp base potential to [0,1]
    base_potential = max(0.0, min(1.0, base_potential))

    # Apply strategic time decay
    decay_factor = time_decay(t)

    return base_potential * decay_factor


# --- 4. REWARD FUNCTION ---

def reward_function(s, t, s_next, t_next, a, gamma=0.99):
    """
    Reward is difference-based with discount factor:

    R(s,t,s',t') =
        gamma * Phi(s',t') - Phi(s,t) - competence_cost(a)
    """

    return (
        gamma * potential_function(s_next, t_next)
        - potential_function(s, t)
        - compute_competence_cost(a)
    )