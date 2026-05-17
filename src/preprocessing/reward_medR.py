import numpy as np
import pandas as pd
import math
from src.preprocessing.columns import *

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

MEDR_FEATURE_MAP = {
    "sofa_24hours": C_SOFA,
    "baseexcess": C_ARTERIAL_BE,
    "lactate": C_ARTERIAL_LACTATE,
    "urineoutput": C_URINE_OUTPUT,
    "mbp": C_MEANBP,
    "heartrate": C_HR,
}

MEDR_RAW_RANGES = {
    "sofa_24hours": (0.0, 23.0),
    "baseexcess": (-25.0, 25.0),
    "lactate": (0.3, 29.0),
    "urineoutput": (0.0, 800.0),
    "mbp": (40.0, 140.0),
    "heartrate": (40.0, 160.0),
}

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
        'target': normalize_raw(-2.0, *MEDR_RAW_RANGES["baseexcess"]),
        'sigma': 0.1,
    },

    'lactate': {
        'type': 'decay_lower',
        'target': normalize_raw(1.6, *MEDR_RAW_RANGES["lactate"]),
        'sigma': 0.05,
    },

    'urineoutput': {
        'type': 'directional_decay',
        'direction': 'high',
        'threshold': normalize_raw(40.0, *MEDR_RAW_RANGES["urineoutput"]),
        'k': 5.0,
    },

    'mbp': {
        'type': 'bell',
        'target': normalize_raw(75.0, *MEDR_RAW_RANGES["mbp"]),
        'sigma': 0.1,
    },

    'heartrate': {
        'type': 'bell',
        'target': normalize_raw(85.0, *MEDR_RAW_RANGES["heartrate"]),
        'sigma': 0.1,
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


def normalize_feature_value(feat_name, raw_value):
    if raw_value is None or pd.isna(raw_value):
        return None

    lo, hi = MEDR_RAW_RANGES[feat_name]
    val = normalize_raw(float(raw_value), lo, hi)

    # medR expects normalized values in [0, 1]
    return float(np.clip(val, 0.0, 1.0))


def build_medr_state(row, delta_row):
    """
    Build medR state:
        state = {
            medr_feature_name: (normalized_value, freshness_delta_hours)
        }

    delta_row should come from delta_fresh_df, not MedDreamer delta.
    """
    state = {}

    for medr_name, col_name in MEDR_FEATURE_MAP.items():
        raw_value = row[col_name] if col_name in row.index else None
        delta_t = delta_row[col_name] if col_name in delta_row.index else None

        state[medr_name] = (
            normalize_feature_value(medr_name, raw_value),
            None if pd.isna(delta_t) else float(delta_t),
        )

    return state


def build_medr_action(action_row):
    """
    medR action uses already-binned action levels:
        vaso_5quantile: 0..4
        iv_fluid_5quantile: 0..4
    """
    return {
        "iv_fluid_5quantile": int(action_row["iv_fluid_5quantile"]),
        "vaso_5quantile": int(action_row["vaso_5quantile"]),
    }


def print_medr_feature_statistics(states_df):
    print("\n===== FEATURE RANGE STATISTICS =====\n")

    for name, col in MEDR_FEATURE_MAP.items():
        if col not in states_df.columns:
            print(f"[WARNING] Missing column: {col}")
            continue

        x = states_df[col].dropna()
        if len(x) == 0:
            print(f"[WARNING] Column {col} is empty")
            continue

        print(f"Feature: {name}")
        print(f"  Column : {col}")
        print(f"  Min    : {x.min():.4f}")
        print(f"  P1     : {x.quantile(0.01):.4f}")
        print(f"  P5     : {x.quantile(0.05):.4f}")
        print(f"  Median : {x.median():.4f}")
        print(f"  P95    : {x.quantile(0.95):.4f}")
        print(f"  P99    : {x.quantile(0.99):.4f}")
        print(f"  Max    : {x.max():.4f}")
        print("-" * 50)


def add_medr_reward_to_dataframe(
    states_df,
    actions_df,
    delta_fresh_df,
    reward_col=C_REWARD,
    gamma=0.99,
):
    """
    Computes medR transition reward t -> t+1 and stores it in row t+1.

    Requirements:
    - states_df has patient states after forward fill
    - delta_fresh_df has freshness delta per feature
    - actions_df has outgoing action a_t, with columns:
        iv_fluid_5quantile, vaso_5quantile
    """

    print_medr_feature_statistics(states_df)

    required_action_cols = [
        C_ICUSTAYID,
        C_TIMESTEP,
        "iv_fluid_5quantile",
        "vaso_5quantile",
    ]

    for col in MEDR_FEATURE_MAP.values():
        if col not in states_df.columns:
            raise ValueError(f"Missing medR state feature column: {col}")
        if col not in delta_fresh_df.columns:
            raise ValueError(f"Missing medR freshness delta column: {col}")

    for c in required_action_cols:
        if c not in actions_df.columns:
            raise ValueError(f"Missing required action column: {c}")

    df = states_df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()
    delta_fresh_df = delta_fresh_df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()
    actions_df = actions_df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()

    rewards = np.zeros(len(df), dtype=np.float32)

    # Faster lookup by key
    delta_lookup = delta_fresh_df.set_index([C_ICUSTAYID, C_TIMESTEP])
    action_lookup = actions_df.set_index([C_ICUSTAYID, C_TIMESTEP])

    for stay_id, g in df.groupby(C_ICUSTAYID, sort=False):
        idx = g.index.to_list()

        if len(idx) == 0:
            continue

        # First row has no previous transition
        rewards[df.index.get_loc(idx[0])] = 0.0

        first_time = float(df.loc[idx[0], C_TIMESTEP])

        for k in range(1, len(idx)):
            prev_i = idx[k - 1]
            cur_i = idx[k]

            prev_row = df.loc[prev_i]
            cur_row = df.loc[cur_i]

            prev_key = (prev_row[C_ICUSTAYID], prev_row[C_TIMESTEP])
            cur_key = (cur_row[C_ICUSTAYID], cur_row[C_TIMESTEP])

            if prev_key not in delta_lookup.index:
                raise ValueError(f"Missing freshness delta for previous key: {prev_key}")

            if cur_key not in delta_lookup.index:
                raise ValueError(f"Missing freshness delta for current key: {cur_key}")

            if prev_key not in action_lookup.index:
                raise ValueError(f"Missing action for transition starting at key: {prev_key}")

            prev_delta = delta_lookup.loc[prev_key]
            cur_delta = delta_lookup.loc[cur_key]
            action_row = action_lookup.loc[prev_key]

            s = build_medr_state(prev_row, prev_delta)
            s_next = build_medr_state(cur_row, cur_delta)
            a = build_medr_action(action_row)

            # t should be relative episode time, not Unix timestamp
            t = (float(prev_row[C_TIMESTEP]) - first_time) / 3600.0
            t_next = (float(cur_row[C_TIMESTEP]) - first_time) / 3600.0

            rewards[df.index.get_loc(cur_i)] = reward_function(
                s=s,
                t=t,
                s_next=s_next,
                t_next=t_next,
                a=a,
                gamma=gamma,
            )

    df[reward_col] = rewards
    return df

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

    if val is None or pd.isna(val):
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