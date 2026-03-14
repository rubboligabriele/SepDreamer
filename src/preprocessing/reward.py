import numpy as np
import pandas as pd

from src.preprocessing.columns import *

# Constants taken from the paper "Deep Reinforcement Learning for Sepsis Treatment"
DEFAULT_C0 = -0.025
DEFAULT_C1 = -0.125
DEFAULT_C2 = -2.0
DEFAULT_R_TERMINAL = 15.0


def compute_sofa_term(sofa_t, sofa_tp1, c0=DEFAULT_C0, c1=DEFAULT_C1):
    """
    Compute the SOFA-based part of the intermediate reward.

    Reward:
        c0 * I[(sofa_tp1 == sofa_t) and (sofa_tp1 > 0)]
      + c1 * (sofa_tp1 - sofa_t)
    """
    same_nonzero = float((sofa_tp1 == sofa_t) and (sofa_tp1 > 0))
    delta_sofa = sofa_tp1 - sofa_t
    return c0 * same_nonzero + c1 * delta_sofa


def compute_lactate_term(lactate_t, lactate_tp1, c2=DEFAULT_C2):
    """
    Compute the lactate-based part of the intermediate reward.

    Reward:
        c2 * tanh(lactate_tp1 - lactate_t)
    """
    delta_lactate = lactate_tp1 - lactate_t
    return c2 * np.tanh(delta_lactate)


def compute_intermediate_reward(
    sofa_t,
    sofa_tp1,
    lactate_t,
    lactate_tp1,
    c0=DEFAULT_C0,
    c1=DEFAULT_C1,
    c2=DEFAULT_C2,
    missing_strategy=None,
):
    """
    Compute the intermediate reward between two consecutive timesteps.

    Missing values are intentionally not resolved yet.
    For now:
    - if any required value is missing, return np.nan
    - missing_strategy is reserved for future implementations
    """
    values = [sofa_t, sofa_tp1, lactate_t, lactate_tp1]
    if any(pd.isna(v) for v in values):
        if missing_strategy is None:
            return np.nan
        raise NotImplementedError(
            f"Missing strategy '{missing_strategy}' is not implemented yet."
        )

    sofa_term = compute_sofa_term(sofa_t, sofa_tp1, c0=c0, c1=c1)
    lactate_term = compute_lactate_term(lactate_t, lactate_tp1, c2=c2)
    return float(sofa_term + lactate_term)


def compute_terminal_reward(died, r_terminal=DEFAULT_R_TERMINAL):
    """
    Compute terminal reward.

    If patient died:
        -r_terminal
    else:
        +r_terminal
    """
    return -float(r_terminal) if died else float(r_terminal)


def compute_transition_reward(
    row_t,
    row_tp1=None,
    is_terminal=False,
    outcome_col=C_MORTA_90,
    sofa_col=C_SOFA,
    lactate_col=C_ARTERIAL_LACTATE,
    c0=DEFAULT_C0,
    c1=DEFAULT_C1,
    c2=DEFAULT_C2,
    r_terminal=DEFAULT_R_TERMINAL,
    missing_strategy=None,
):
    """
    Compute reward for one transition.

    If is_terminal is False:
        reward is computed from row_t -> row_tp1

    If is_terminal is True:
        reward is computed only from the terminal outcome stored in row_t[outcome_col]
    """
    if is_terminal:
        if pd.isna(row_t[outcome_col]):
            if missing_strategy is None:
                return np.nan
            raise NotImplementedError(
                f"Missing strategy '{missing_strategy}' is not implemented yet."
            )
        died = bool(row_t[outcome_col])
        return compute_terminal_reward(died=died, r_terminal=r_terminal)

    if row_tp1 is None:
        raise ValueError("row_tp1 must be provided for non-terminal transitions.")

    return compute_intermediate_reward(
        sofa_t=row_t[sofa_col],
        sofa_tp1=row_tp1[sofa_col],
        lactate_t=row_t[lactate_col],
        lactate_tp1=row_tp1[lactate_col],
        c0=c0,
        c1=c1,
        c2=c2,
        missing_strategy=missing_strategy,
    )


def add_reward_to_dataframe(
    df,
    outcome_col=C_MORTA_90,
    sofa_col=C_SOFA,
    lactate_col=C_ARTERIAL_LACTATE,
    reward_col="reward",
    c0=DEFAULT_C0,
    c1=DEFAULT_C1,
    c2=DEFAULT_C2,
    r_terminal=DEFAULT_R_TERMINAL,
    missing_strategy=None,
):
    """
    Add a reward column to a dataframe.

    Assumptions:
    - each ICU stay is one episode
    - rows are transitions ordered by timestep
    - terminal reward is assigned to the last row of each ICU stay
    """
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()
    rewards = np.full(len(df), np.nan, dtype=np.float32)

    for _, g in df.groupby(C_ICUSTAYID, sort=False):
        idx = g.index.to_list()

        if len(idx) == 1:
            rewards[df.index.get_loc(idx[0])] = compute_transition_reward(
                row_t=df.loc[idx[0]],
                is_terminal=True,
                outcome_col=outcome_col,
                sofa_col=sofa_col,
                lactate_col=lactate_col,
                c0=c0,
                c1=c1,
                c2=c2,
                r_terminal=r_terminal,
                missing_strategy=missing_strategy,
            )
            continue

        for k in range(len(idx) - 1):
            i = idx[k]
            j = idx[k + 1]
            rewards[df.index.get_loc(i)] = compute_transition_reward(
                row_t=df.loc[i],
                row_tp1=df.loc[j],
                is_terminal=False,
                outcome_col=outcome_col,
                sofa_col=sofa_col,
                lactate_col=lactate_col,
                c0=c0,
                c1=c1,
                c2=c2,
                r_terminal=r_terminal,
                missing_strategy=missing_strategy,
            )

        last_i = idx[-1]
        rewards[df.index.get_loc(last_i)] = compute_transition_reward(
            row_t=df.loc[last_i],
            is_terminal=True,
            outcome_col=outcome_col,
            sofa_col=sofa_col,
            lactate_col=lactate_col,
            c0=c0,
            c1=c1,
            c2=c2,
            r_terminal=r_terminal,
            missing_strategy=missing_strategy,
        )

    df[reward_col] = rewards
    return df