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
):
    reward = 0.0
    has_signal = False

    # ---- SOFA term ----
    if not (pd.isna(sofa_t) or pd.isna(sofa_tp1)):
        reward += compute_sofa_term(
            sofa_t,
            sofa_tp1,
            c0=c0,
            c1=c1,
        )
        has_signal = True

    # ---- Lactate term ----
    if not (pd.isna(lactate_t) or pd.isna(lactate_tp1)):
        reward += compute_lactate_term(
            lactate_t,
            lactate_tp1,
            c2=c2,
        )
        has_signal = True

    # ---- If nothing observed ----
    if not has_signal:
        return 0.0

    return float(reward)


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
            raise ValueError("Missing terminal outcome")
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
):
    """
    RL-consistent reward construction.

    For each ICU stay with raw states s0 ... s_{T-1}:

    - we KEEP only rows 0 ... T-2
    - reward[k] =
        intermediate reward for k < T-2
        terminal reward for k = T-2

    The final raw state s_{T-1} is dropped.
    """

    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).copy()

    kept_rows = []
    rewards = []

    for _, g in df.groupby(C_ICUSTAYID, sort=False):

        g = g.sort_values(C_TIMESTEP)

        if len(g) < 2:
            # cannot build a transition → skip stay
            continue

        rows = g.to_dict("records")

        T = len(rows)

        # we will keep rows 0 ... T-2
        for k in range(T - 1):

            row_t = rows[k]

            # last RL step → terminal reward
            if k == T - 2:

                r = compute_terminal_reward(
                    died=bool(row_t[outcome_col]),
                    r_terminal=r_terminal,
                )

            else:

                row_tp1 = rows[k + 1]

                r = compute_intermediate_reward(
                    sofa_t=row_t[sofa_col],
                    sofa_tp1=row_tp1[sofa_col],
                    lactate_t=row_t[lactate_col],
                    lactate_tp1=row_tp1[lactate_col],
                    c0=c0,
                    c1=c1,
                    c2=c2,
                )

            kept_rows.append(row_t)
            rewards.append(r)

    out = pd.DataFrame(kept_rows)
    out[reward_col] = np.array(rewards, dtype=np.float32)

    return out