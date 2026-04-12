import os
import argparse
import numpy as np
import pandas as pd

from src.preprocessing.columns import *
from src.preprocessing.utils import load_csv


DEFAULT_REWARD_COL = "reward"


def is_close(x, value, tol=1e-6):
    return np.isfinite(x) & (np.abs(x - value) < tol)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reward distribution in patient states dataframe."
    )
    parser.add_argument("input", type=str, help="Path to patient_states_with_reward.csv")
    parser.add_argument(
        "--reward-col",
        dest="reward_col",
        type=str,
        default=DEFAULT_REWARD_COL,
        help=f"Reward column name (default: {DEFAULT_REWARD_COL})"
    )
    parser.add_argument(
        "--sofa-col",
        dest="sofa_col",
        type=str,
        default=C_SOFA,
        help=f"SOFA column name (default: {C_SOFA})"
    )
    parser.add_argument(
        "--lactate-col",
        dest="lactate_col",
        type=str,
        default=C_ARTERIAL_LACTATE,
        help=f"Lactate column name (default: {C_ARTERIAL_LACTATE})"
    )
    args = parser.parse_args()

    print("Reading dataframe...")
    df = load_csv(args.input)
    df = df.sort_values([C_ICUSTAYID, C_TIMESTEP]).reset_index(drop=True)

    reward_col = args.reward_col
    sofa_col = args.sofa_col
    lactate_col = args.lactate_col

    if reward_col not in df.columns:
        raise ValueError(f"Column '{reward_col}' not found in dataframe.")

    rewards = df[reward_col].to_numpy(dtype=float)

    print("\n=== GLOBAL REWARD STATS ===")
    print(f"Rows: {len(df)}")
    print(f"NaN rewards: {np.isnan(rewards).sum()}")
    print(f"Min reward: {np.nanmin(rewards):.6f}")
    print(f"Max reward: {np.nanmax(rewards):.6f}")
    print(f"Mean reward: {np.nanmean(rewards):.6f}")
    print(f"Std reward: {np.nanstd(rewards):.6f}")

    print("\nReward quantiles:")
    for q in [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]:
        print(f"  q={q:>4}: {np.nanquantile(rewards, q):.6f}")

    # terminal rows = last timestep per icustayid
    last_idx = df.groupby(C_ICUSTAYID, sort=False).tail(1).index
    terminal_mask = df.index.isin(last_idx)
    interm_mask = ~terminal_mask

    terminal_rewards = df.loc[terminal_mask, reward_col].to_numpy(dtype=float)
    interm_rewards = df.loc[interm_mask, reward_col].to_numpy(dtype=float)

    print("\n=== TERMINAL VS INTERMEDIATE ===")
    print(f"Intermediate rows: {interm_mask.sum()}")
    print(f"Terminal rows: {terminal_mask.sum()}")

    print("\nIntermediate reward stats:")
    if len(interm_rewards) > 0:
        print(f"  min:  {np.nanmin(interm_rewards):.6f}")
        print(f"  max:  {np.nanmax(interm_rewards):.6f}")
        print(f"  mean: {np.nanmean(interm_rewards):.6f}")
        print(f"  std:  {np.nanstd(interm_rewards):.6f}")

    print("\nTerminal reward stats:")
    if len(terminal_rewards) > 0:
        print(f"  unique values: {np.unique(terminal_rewards)}")
        print(f"  mean: {np.nanmean(terminal_rewards):.6f}")

    print("\n=== REWARD VALUE COUNTS ===")
    print(f"reward == 0:       {is_close(rewards, 0.0).sum()}")
    print(f"reward == -0.025:  {is_close(rewards, -0.025).sum()}")
    print(f"reward == -0.125:  {is_close(rewards, -0.125).sum()}")
    print(f"reward == +15:     {is_close(rewards, 15.0).sum()}")
    print(f"reward == -15:     {is_close(rewards, -15.0).sum()}")
    print(f"reward > 0:        {(rewards > 0).sum()}")
    print(f"reward < 0:        {(rewards < 0).sum()}")

    # Analyze transitions
    sofa_same_nonzero = 0
    sofa_changed = 0
    sofa_missing_transition = 0

    lactate_both_present = 0
    lactate_missing_transition = 0

    total_nonterminal = 0

    for _, g in df.groupby(C_ICUSTAYID, sort=False):
        g = g.sort_values(C_TIMESTEP)
        if len(g) < 2:
            continue

        sofa_vals = g[sofa_col].to_numpy()
        lact_vals = g[lactate_col].to_numpy()

        for i in range(len(g) - 1):
            total_nonterminal += 1

            s0 = sofa_vals[i]
            s1 = sofa_vals[i + 1]
            l0 = lact_vals[i]
            l1 = lact_vals[i + 1]

            if pd.isna(s0) or pd.isna(s1):
                sofa_missing_transition += 1
            else:
                if s0 == s1 and s1 > 0:
                    sofa_same_nonzero += 1
                if s0 != s1:
                    sofa_changed += 1

            if pd.isna(l0) or pd.isna(l1):
                lactate_missing_transition += 1
            else:
                lactate_both_present += 1

    print("\n=== TRANSITION SIGNAL ANALYSIS ===")
    print(f"Total non-terminal transitions: {total_nonterminal}")

    print("\nSOFA transitions:")
    print(f"  both SOFA present:      {total_nonterminal - sofa_missing_transition}")
    print(f"  missing SOFA:           {sofa_missing_transition}")
    print(f"  same SOFA and > 0:      {sofa_same_nonzero}")
    print(f"  changed SOFA:           {sofa_changed}")

    print("\nLactate transitions:")
    print(f"  both lactate present:   {lactate_both_present}")
    print(f"  missing lactate:        {lactate_missing_transition}")

    print("\n=== TOP 20 MOST COMMON INTERMEDIATE REWARDS ===")
    if len(interm_rewards) > 0:
        vc = pd.Series(np.round(interm_rewards, 6)).value_counts().head(20)
        print(vc.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()