import numpy as np
import pandas as pd
import math
from preprocessing.utils.columns import *

# =============================================================================
# 1. PARAMETER DEFINITIONS
# =============================================================================

def normalize_raw(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


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
    # urine_output is now in ml/h (rate). Typical ICU range 0–500 ml/h.
    # Oliguria threshold: ~40 ml/h (0.5 ml/kg/h × 80 kg).
    "urineoutput": (0.0, 500.0),
    "mbp": (40.0, 140.0),
    "heartrate": (40.0, 160.0),
}

# Parameters from medr_cand_0062 (best candidate, stable across evaluation runs):
#   action_cost_scale=0.05, half_life=48, confidence_tau=24,
#   sigma_scale=1.25, k_scale=1.0, survival_weight=0.8, confidence_weight=0.2
_K_SCALE = 1.0
_SIGMA_SCALE = 1.25

SURVIVAL_CONFIG = {
    "sofa_24hours": {
        "type": "directional_decay",
        "direction": "low",
        "k": 2.3 * _K_SCALE,
    },
    "baseexcess": {
        "type": "bell",
        "target": normalize_raw(-2.0, *MEDR_RAW_RANGES["baseexcess"]),
        "sigma": 0.1 * _SIGMA_SCALE,
    },
    "lactate": {
        "type": "decay_lower",
        "target": normalize_raw(1.6, *MEDR_RAW_RANGES["lactate"]),
        "sigma": 0.05 * _SIGMA_SCALE,
    },
    "urineoutput": {
        "type": "directional_decay",
        "direction": "high",
        "threshold": normalize_raw(40.0, *MEDR_RAW_RANGES["urineoutput"]),
        "k": 5.0 * _K_SCALE,
    },
    "mbp": {
        "type": "bell",
        "target": normalize_raw(75.0, *MEDR_RAW_RANGES["mbp"]),
        "sigma": 0.1 * _SIGMA_SCALE,
    },
    "heartrate": {
        "type": "bell",
        "target": normalize_raw(85.0, *MEDR_RAW_RANGES["heartrate"]),
        "sigma": 0.1 * _SIGMA_SCALE,
    },
}

# confidence_tau=24h from cand0062 — confidence decays over 24h of feature staleness
CONFIDENCE_TAU = {
    "sofa_24hours": 24.0,
    "baseexcess": 24.0,
    "lactate": 24.0,
    "urineoutput": 24.0,
    "mbp": 24.0,
    "heartrate": 24.0,
}

MAX_DOSE_LEVEL = 4

# cand0062: action_cost_scale=0.05, survival_weight=0.8, confidence_weight=0.2
ACTION_COST_SCALE = 0.05
POTENTIAL_DIFF_SCALE = 20.0
USE_TIME_DECAY = False

COMPONENT_WEIGHTS = {
    "survival": 0.8,
    "confidence": 0.2,
}


# =============================================================================
# 2. DATAFRAME HELPERS
# =============================================================================

def normalize_feature_value(feat_name, raw_value):
    if raw_value is None or pd.isna(raw_value):
        return None

    lo, hi = MEDR_RAW_RANGES[feat_name]
    val = normalize_raw(float(raw_value), lo, hi)
    return float(np.clip(val, 0.0, 1.0))


def build_medr_state(row, delta_row):
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

    for col in required_action_cols:
        if col not in actions_df.columns:
            raise ValueError(f"Missing required action column: {col}")

    df = states_df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()
    delta_fresh_df = delta_fresh_df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()
    actions_df = actions_df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()

    rewards = np.zeros(len(df), dtype=np.float32)

    delta_lookup = delta_fresh_df.set_index([C_ICUSTAYID, C_TIMESTEP])
    action_lookup = actions_df.set_index([C_ICUSTAYID, C_TIMESTEP])

    for stay_id, g in df.groupby(C_ICUSTAYID, sort=False):
        idx = g.index.to_list()

        if len(idx) == 0:
            continue

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
            # This is the outgoing clinical action a_t that caused transition s_t -> s_{t+1}.
            # Later, episode construction shifts actions right, so episode action[t] becomes a_{t-1}.
            action_row = action_lookup.loc[prev_key]

            s = build_medr_state(prev_row, prev_delta)
            s_next = build_medr_state(cur_row, cur_delta)
            a = build_medr_action(action_row)

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


# =============================================================================
# 3. MATH HELPERS
# =============================================================================

def time_decay(t):
    if not USE_TIME_DECAY:
        return 1.0

    half_life = 48.0
    return 0.5 ** (t / half_life)


def compute_survival_score(val, params):
    if val is None or pd.isna(val):
        return 0.5

    ftype = params["type"]

    if ftype == "bell":
        target = params["target"]
        sigma = params["sigma"]

        if sigma <= 0:
            return 0.0

        diff = val - target
        return math.exp(-0.5 * (diff / sigma) ** 2)

    if ftype == "decay_lower":
        target = params["target"]
        sigma = params["sigma"]

        if val <= target:
            return 1.0

        decay_rate = math.log(2) / sigma
        return math.exp(-decay_rate * (val - target))

    if ftype == "directional_decay":
        direction = params.get("direction", None)

        if direction == "low":
            k = params.get("k", 2.3)
            return math.exp(-k * val)

        if direction == "high":
            threshold = params.get("threshold", 0.5)
            k = params.get("k", 5.0)

            if val >= threshold:
                return 1.0

            diff = (threshold - val) / threshold if threshold > 0 else 1.0
            return math.exp(-k * diff)

    return 0.5


def compute_confidence_weight(delta_t, tau):
    if delta_t is None or pd.isna(delta_t) or delta_t < 0:
        delta_t = 0.0

    return math.exp(-float(delta_t) / tau)


def compute_competence_cost(action):
    vaso_level = action.get("vaso_5quantile", 0)
    iv_level = action.get("iv_fluid_5quantile", 0)

    vaso_norm = vaso_level / MAX_DOSE_LEVEL
    iv_norm = iv_level / MAX_DOSE_LEVEL

    total_penalty = (
        vaso_norm * ACTION_COST_SCALE +
        iv_norm * ACTION_COST_SCALE
    )

    return min(total_penalty, 1.0)


# =============================================================================
# 4. POTENTIAL AND REWARD
# =============================================================================

def potential_function(state, t):
    survival_scores = []
    confidence_weights = []

    for feat in SURVIVAL_CONFIG.keys():
        val, delta_t = state.get(feat, (None, None))

        surv_score = compute_survival_score(
            val,
            SURVIVAL_CONFIG[feat],
        )

        tau = CONFIDENCE_TAU.get(feat, 6.0)

        conf_weight = (
            compute_confidence_weight(delta_t, tau)
            if delta_t is not None
            else 0.0
        )

        survival_scores.append(surv_score * conf_weight)
        confidence_weights.append(conf_weight)

    sum_confidence = sum(confidence_weights)

    if sum_confidence > 0:
        survival_component = sum(survival_scores) / sum_confidence
    else:
        survival_component = 0.5

    if len(confidence_weights) > 0:
        confidence_component = sum(confidence_weights) / len(confidence_weights)
    else:
        confidence_component = 0.0    

    base_potential = (
        COMPONENT_WEIGHTS["survival"] * survival_component
        + COMPONENT_WEIGHTS["confidence"] * confidence_component
    )

    base_potential = max(0.0, min(1.0, base_potential))

    return base_potential * time_decay(t)


def reward_function(s, t, s_next, t_next, a, gamma=0.99):
    phi_now = potential_function(s, t)
    phi_next = potential_function(s_next, t_next)

    potential_reward = POTENTIAL_DIFF_SCALE * (
        gamma * phi_next - phi_now
    )

    action_cost = compute_competence_cost(a)

    return potential_reward - action_cost